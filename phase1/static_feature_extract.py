import argparse
import csv
import json
import multiprocessing
import os
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from androguard.misc import AnalyzeAPK

"""
APK 静态特征提取。

输出结构兼容原始 Static_analysis 字段，并额外补充：
- API families: 面向安全语义的 API 家族计数，减少原始 API 过稀疏的问题。
- Suspicious strings: URL/IP/命令/动态加载等字符串线索。
- Obfuscation: 类名、组件名、包名层面的混淆启发式统计。
"""


API_FAMILY_PATTERNS = {
    "sms": [
        r"android\.telephony\.SmsManager",
        r"sendTextMessage",
        r"sendMultipartTextMessage",
    ],
    "telephony": [
        r"android\.telephony",
        r"getDeviceId",
        r"getSubscriberId",
        r"getLine1Number",
    ],
    "location": [
        r"android\.location",
        r"getLastKnownLocation",
        r"requestLocationUpdates",
    ],
    "contacts": [
        r"ContactsContract",
        r"READ_CONTACTS",
    ],
    "account": [
        r"android\.accounts",
        r"AccountManager",
    ],
    "package_manager": [
        r"android\.content\.pm",
        r"getInstalledPackages",
        r"getInstalledApplications",
    ],
    "reflection": [
        r"java\.lang\.reflect",
        r"Class\.forName",
        r"getDeclaredMethod",
        r"invoke",
    ],
    "dynamic_loading": [
        r"dalvik\.system\.DexClassLoader",
        r"dalvik\.system\.PathClassLoader",
        r"loadClass",
        r"loadLibrary",
    ],
    "crypto": [
        r"javax\.crypto",
        r"java\.security",
        r"MessageDigest",
        r"Cipher",
    ],
    "network": [
        r"java\.net",
        r"okhttp",
        r"retrofit",
        r"HttpURLConnection",
        r"Socket",
    ],
    "file_io": [
        r"java\.io",
        r"android\.os\.Environment",
        r"openFileOutput",
    ],
    "shell": [
        r"java\.lang\.Runtime\.exec",
        r"java\.lang\.ProcessBuilder",
    ],
    "accessibility": [
        r"android\.accessibilityservice",
        r"AccessibilityService",
    ],
    "device_admin": [
        r"DeviceAdminReceiver",
        r"DevicePolicyManager",
    ],
}

SUSPICIOUS_STRING_PATTERNS = {
    "url": re.compile(r"https?://", re.IGNORECASE),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "apk": re.compile(r"\.apk\b", re.IGNORECASE),
    "dex": re.compile(r"\.dex\b|classes\d*\.dex", re.IGNORECASE),
    "so": re.compile(r"\.so\b", re.IGNORECASE),
    "base64_hint": re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})"),
    "shell_path": re.compile(r"/system/bin/|/system/xbin/|/data/local/tmp", re.IGNORECASE),
}

COMMON_SYSTEM_COMMANDS = {
    "su",
    "sh",
    "chmod",
    "chown",
    "mount",
    "exec",
    "ping",
    "netstat",
    "logcat",
    "pm",
    "am",
    "getprop",
    "setprop",
    "iptables",
    "ifconfig",
    "busybox",
}

VALID_SPLITS = {"train", "val", "test"}
LABEL_DIR_ALIASES = {
    1: ("mal", "malware", "1"),
    0: ("benign", "leg", "normal", "0"),
}


def clean_class_name(name: Optional[str]) -> str:
    if not name:
        return ""
    while name.startswith("["):
        name = name[1:]
    if name.startswith("L") and name.endswith(";"):
        name = name[1:-1]
    return name.replace("/", ".")


def package_name(class_name: str) -> str:
    parts = class_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else class_name


def is_obfuscated_name(name: str) -> bool:
    """保守的混淆启发式：短包段过多、低信息类名、匿名类密集等。"""
    if not name:
        return False
    clean = name.replace("/", ".").replace("$", ".")
    segments = [seg for seg in clean.split(".") if seg]
    if not segments:
        return False
    short_segments = sum(1 for seg in segments if len(seg) <= 2)
    single_letter_segments = sum(1 for seg in segments if len(seg) == 1)
    numeric_segments = sum(1 for seg in segments if seg.isdigit())
    last = segments[-1]
    return (
        single_letter_segments >= 2
        or short_segments >= max(3, len(segments) // 2)
        or numeric_segments >= 1
        or bool(re.fullmatch(r"[a-zA-Z]{1,2}\d*", last))
    )


def classify_api_family(api_call: str) -> List[str]:
    families = []
    for family, patterns in API_FAMILY_PATTERNS.items():
        if any(re.search(pattern, api_call, re.IGNORECASE) for pattern in patterns):
            families.append(family)
    return families


class StageOnePreprocessor:
    def __init__(self, apk_path: Path, output_dir: Path, label: int, split: Optional[str] = None):
        self.apk_path = Path(apk_path)
        self.apk_name = self.apk_path.name
        self.label = int(label)
        self.split = split
        self.output_dir = Path(output_dir) / self.apk_path.stem

        self.static_analysis = {
            "Permissions": [],
            "Opcodes": {},
            "API calls": {},
            "System commands": {},
            "API packages": {},
            "API families": {},
            "Activities": {},
            "Services": {},
            "Receivers": {},
            "Intents": {},
            "Suspicious strings": {},
            "Obfuscation": {},
        }

        self.report = {
            "apk_name": self.apk_name,
            "apk_path": str(self.apk_path),
            "label": self.label,
            "split": self.split,
            "belong": self.split,
            "Static_analysis": self.static_analysis,
        }

    def run_pipeline(self) -> None:
        self._extract_features()
        self._save_report()

    def _extract_manifest_features(self, apk_obj) -> None:
        self.static_analysis["Permissions"] = sorted(set(apk_obj.get_permissions()))

        xml = apk_obj.get_android_manifest_xml()
        intents_counter = Counter()
        component_names: List[str] = []

        if xml is None:
            return

        for tag, target_key in [
            ("activity", "Activities"),
            ("service", "Services"),
            ("receiver", "Receivers"),
        ]:
            target_dict = self.static_analysis[target_key]
            for item in xml.findall(f".//{tag}"):
                name = item.get("{http://schemas.android.com/apk/res/android}name")
                if not name:
                    continue

                component_names.append(name)
                actions = []
                for action in item.findall(".//action"):
                    act_name = action.get("{http://schemas.android.com/apk/res/android}name")
                    if act_name:
                        actions.append(act_name)
                        intents_counter[act_name] += 1

                for category in item.findall(".//category"):
                    cat_name = category.get("{http://schemas.android.com/apk/res/android}name")
                    if cat_name:
                        actions.append(cat_name)
                        intents_counter[cat_name] += 1

                target_dict[name] = sorted(set(actions))

        self.static_analysis["Intents"] = dict(sorted(intents_counter.items()))
        self._add_component_obfuscation(component_names, apk_obj.get_package())

    def _add_component_obfuscation(self, component_names: Iterable[str], pkg: Optional[str]) -> None:
        names = list(component_names)
        obfuscated = sum(1 for name in names if is_obfuscated_name(name))
        package_obfuscated = int(is_obfuscated_name(pkg or ""))
        self.static_analysis["Obfuscation"].update(
            {
                "component_count": len(names),
                "obfuscated_component_count": obfuscated,
                "obfuscated_component_ratio": round(obfuscated / max(1, len(names)), 6),
                "package_name_obfuscated": package_obfuscated,
            }
        )

    def _extract_dex_features(self, dx) -> None:
        opcodes_counter = Counter()
        api_calls_counter = Counter()
        api_packages_counter = Counter()
        api_families_counter = Counter()
        internal_classes = set()
        internal_methods = 0
        obfuscated_classes = 0

        for meth_analysis in dx.get_methods():
            if meth_analysis.is_external():
                continue

            internal_methods += 1
            meth = meth_analysis.get_method()
            class_name = clean_class_name(meth.get_class_name() if meth else "")
            if class_name:
                if class_name not in internal_classes and is_obfuscated_name(class_name):
                    obfuscated_classes += 1
                internal_classes.add(class_name)

            if meth and hasattr(meth, "get_instructions"):
                for ins in meth.get_instructions():
                    opcodes_counter[ins.get_name()] += 1

            for _, call, _ in meth_analysis.get_xref_to():
                target_method = call.get_method()
                if not target_method:
                    continue

                target_class = clean_class_name(target_method.get_class_name())
                method_name = target_method.get_name()
                if not target_class:
                    continue

                api_call = f"{target_class}.{method_name}"
                api_calls_counter[api_call] += 1
                api_packages_counter[package_name(target_class)] += 1
                for family in classify_api_family(api_call):
                    api_families_counter[family] += 1

        self.static_analysis["Opcodes"] = dict(sorted(opcodes_counter.items()))
        self.static_analysis["API calls"] = dict(sorted(api_calls_counter.items()))
        self.static_analysis["API packages"] = dict(sorted(api_packages_counter.items()))
        self.static_analysis["API families"] = dict(sorted(api_families_counter.items()))
        self.static_analysis["Obfuscation"].update(
            {
                "internal_class_count": len(internal_classes),
                "internal_method_count": internal_methods,
                "obfuscated_class_count": obfuscated_classes,
                "obfuscated_class_ratio": round(
                    obfuscated_classes / max(1, len(internal_classes)), 6
                ),
            }
        )

    def _extract_string_features(self, dx) -> None:
        command_counter = Counter()
        suspicious_counter = Counter()

        for string_obj in dx.get_strings():
            val = string_obj.get_value()
            if not isinstance(val, str):
                continue

            text = val.strip()
            if text in COMMON_SYSTEM_COMMANDS:
                command_counter[text] += 1

            for key, pattern in SUSPICIOUS_STRING_PATTERNS.items():
                if pattern.search(text):
                    suspicious_counter[key] += 1

        self.static_analysis["System commands"] = dict(sorted(command_counter.items()))
        self.static_analysis["Suspicious strings"] = dict(sorted(suspicious_counter.items()))

    def _extract_features(self) -> None:
        try:
            apk_obj, _, dx = AnalyzeAPK(str(self.apk_path))
            self._extract_manifest_features(apk_obj)
            self._extract_dex_features(dx)
            self._extract_string_features(dx)
        except Exception as exc:
            self.report["error"] = f"Analysis Failed: {exc}"

    def make_json_safe(self, obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, dict):
            return {self.make_json_safe(k): self.make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, set, tuple)):
            return [self.make_json_safe(x) for x in obj]
        return obj

    def _save_report(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "static_features_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(self.make_json_safe(self.report), handle, indent=2, ensure_ascii=False)


def worker_task(task_args: Tuple[str, str, int, Optional[str]]) -> Tuple[bool, str, str]:
    apk_path, output_dir, label, split = task_args
    processor = StageOnePreprocessor(Path(apk_path), Path(output_dir), label, split)

    try:
        processor.run_pipeline()
        if "error" in processor.report:
            return False, processor.apk_name, processor.report["error"]
        return True, processor.apk_name, "Success"
    except Exception as exc:
        return False, processor.apk_name, str(exc)


def iter_apks(sample_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in sample_dir.rglob("*.apk") if path.is_file())


def normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None or split == "":
        return None
    value = split.strip().lower()
    if value not in VALID_SPLITS:
        raise ValueError(f"belong/split 只能是 {sorted(VALID_SPLITS)}，当前为: {split}")
    return value


def label_name(label: int) -> str:
    return "mal" if int(label) == 1 else "benign"


def infer_label_dir(dataset_root: Path, split: str, label: int) -> Optional[Path]:
    for alias in LABEL_DIR_ALIASES[int(label)]:
        candidate = dataset_root / split / alias
        if candidate.exists():
            return candidate
    return None


def load_manifest(manifest_path: Path, default_split: Optional[str]) -> List[Tuple[Path, int, Optional[str]]]:
    tasks = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"apk_path", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("manifest CSV 必须包含 apk_path,label 两列")
        for row in reader:
            row_split = row.get("split") or row.get("belong") or default_split
            tasks.append((Path(row["apk_path"]), int(row["label"]), normalize_split(row_split)))
    return tasks


def collect_samples(args: argparse.Namespace) -> List[Tuple[Path, int, Optional[str]]]:
    pairs: List[Tuple[Path, int, Optional[str]]] = []
    default_split = normalize_split(args.belong)
    if args.manifest:
        pairs.extend(load_manifest(Path(args.manifest), default_split))
    if args.dataset_root:
        if default_split is None:
            raise ValueError("使用 --dataset-root 时必须传入 --belong train|val|test")
        dataset_root = Path(args.dataset_root)
        for label in [1, 0]:
            sample_dir = infer_label_dir(dataset_root, default_split, label)
            if sample_dir is None:
                print(f"[warning] 未找到 {dataset_root / default_split} 下 label={label} 的目录")
                continue
            pairs.extend((apk_path, label, default_split) for apk_path in iter_apks(sample_dir))
    for spec in args.input:
        if "=" not in spec:
            raise ValueError("--input 格式应为 label=目录，例如 1=/data/mal")
        label_text, dir_text = spec.split("=", 1)
        label = int(label_text)
        sample_dir = Path(dir_text)
        if not sample_dir.exists():
            print(f"[warning] 目录不存在，跳过: {sample_dir}")
            continue
        pairs.extend((apk_path, label, default_split) for apk_path in iter_apks(sample_dir))
    return pairs


def output_parent(output_root: Path, split: Optional[str], label: int) -> Path:
    if split:
        return output_root / split / label_name(label)
    return output_root / label_name(label)


def build_tasks(args: argparse.Namespace) -> List[Tuple[str, str, int, Optional[str]]]:
    pairs = collect_samples(args)
    tasks = []
    output_root = Path(args.output)
    for apk_path, label, split in pairs:
        if not apk_path.exists():
            print(f"[warning] APK 不存在，跳过: {apk_path}")
            continue
        if not zipfile.is_zipfile(apk_path):
            print(f"[warning] 非法 APK/ZIP，跳过: {apk_path}")
            continue
        parent = output_parent(output_root, split, label)
        expected = parent / apk_path.stem / "static_features_report.json"
        if expected.exists() and not args.overwrite:
            continue
        tasks.append((str(apk_path), str(parent), int(label), split))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量提取 APK 静态特征")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="样本目录，格式 label=目录；可重复传入，例如 --input 1=/data/mal --input 0=/data/benign",
    )
    parser.add_argument("--dataset-root", help="原始数据根目录，自动读取 dataset_root/belong/mal 与 dataset_root/belong/benign")
    parser.add_argument("--belong", choices=sorted(VALID_SPLITS), help="当前处理的数据划分: train、val 或 test")
    parser.add_argument("--manifest", help="CSV，包含 apk_path,label 两列；可选 split 或 belong 列")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 static_features_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)

    print(f"[*] 待处理 APK: {len(tasks)}")
    if not tasks:
        print("[*] 没有需要处理的新任务。")
        return

    success_count = 0
    fail_count = 0
    workers = max(1, min(args.workers, len(tasks)))
    print(f"[*] 启动 worker: {workers}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            success, apk_name, message = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"[{index}/{len(tasks)}] [-] {apk_name}: {message}")
            if index % 25 == 0 or index == len(tasks):
                print(f"[{index}/{len(tasks)}] success={success_count}, failed={fail_count}")

    print("=" * 42)
    print(f"完成: total={len(tasks)}, success={success_count}, failed={fail_count}")


if __name__ == "__main__":
    main()
