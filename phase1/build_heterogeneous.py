import argparse
import csv
import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from androguard.misc import AnalyzeAPK

"""
构建 APK 级异构图。

图分类模型在 phase2/train_hetero_hgt.py 中实现。这里产出 PyTorch Geometric
HeteroData 的 .pt 文件，以及用于可解释性/排错的 meta.json。
当前 `build_heterogeneous.py` 每个 APK 会生成一张图，保存为：
```text
hetero_graph.pt
```

图里面主要由三部分组成：

1. **节点特征**
2. **异构边**
3. **图级标签**

**节点类型**

当前有这些节点类型：

```text
app
class
method
api
package
permission
component
intent
opcode
api_family
```

含义大概是：

- `app`：当前 APK 本身，每张图一个 app 节点
- `class`：DEX 中的类
- `method`：DEX 中的方法
- `api`：方法调用到的外部 API
- `package`：API 所属包名
- `permission`：Manifest 里的权限
- `component`：activity、service、receiver、provider
- `intent`：组件声明的 action/category
- `opcode`：方法中的 opcode
- `api_family`：风险 API 家族，比如 sms、network、crypto、shell 等

每个节点都有 `x` 特征，来自 `stable_hash_features()`，不是 one-hot，而是哈希文本特征 + 一些统计特征，比如名称长度、是否混淆、是否 android/java 包、出现次数等。

**边类型**

有，而且每条边还会自动加反向边。

正向边包括：

```text
(app, uses_permission, permission)
(app, declares_component, component)
(component, handles_intent, intent)
(app, has_class, class)
(class, defines_method, method)
(method, uses_opcode, opcode)
(method, calls_api, api)
(api, belongs_to_package, package)
(api, has_family, api_family)
```

同时自动生成反向边，例如：

```text
(permission, rev_uses_permission, app)
(component, rev_declares_component, app)
(intent, rev_handles_intent, component)
(class, rev_has_class, app)
(method, rev_defines_method, class)
(opcode, rev_uses_opcode, method)
(api, rev_calls_api, method)
(package, rev_belongs_to_package, api)
(api_family, rev_has_family, api)
```

所以不是只有节点，**边是完整保存进 `HeteroData` 的**：

```python
data[edge_type].edge_index = edge_index
```

**图级标签**

每张图还有：

```python
data.y = torch.tensor([label])
data.apk_name = apk_name
data.split = split
```

也就是说图模型训练时用的是 APK 级图分类，不是节点分类。每个 APK 一张异构图，最后预测这个 APK 是恶意还是良性。
"""


RISK_API_HINTS = {
    "sms": re.compile(r"SmsManager|sendTextMessage|sendMultipartTextMessage", re.I),
    "telephony": re.compile(r"TelephonyManager|getDeviceId|getSubscriberId|getLine1Number", re.I),
    "location": re.compile(r"LocationManager|requestLocationUpdates|getLastKnownLocation", re.I),
    "reflection": re.compile(r"java\.lang\.reflect|Class\.forName|getDeclaredMethod|invoke", re.I),
    "dynamic_loading": re.compile(r"DexClassLoader|PathClassLoader|loadClass|loadLibrary", re.I),
    "crypto": re.compile(r"javax\.crypto|java\.security|MessageDigest|Cipher", re.I),
    "network": re.compile(r"java\.net|HttpURLConnection|Socket|okhttp|retrofit", re.I),
    "shell": re.compile(r"Runtime\.exec|ProcessBuilder", re.I),
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
    if not name:
        return False
    clean = name.replace("/", ".").replace("$", ".")
    segments = [seg for seg in clean.split(".") if seg]
    if not segments:
        return False
    short_segments = sum(1 for seg in segments if len(seg) <= 2)
    single_letter_segments = sum(1 for seg in segments if len(seg) == 1)
    last = segments[-1]
    return single_letter_segments >= 2 or short_segments >= max(3, len(segments) // 2) or bool(
        re.fullmatch(r"[a-zA-Z]{1,2}\d*", last)
    )


def label_name(label: int) -> str:
    return "mal" if int(label) == 1 else "benign"


def normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None or split == "":
        return None
    value = split.strip().lower()
    if value not in VALID_SPLITS:
        raise ValueError(f"belong/split 只能是 {sorted(VALID_SPLITS)}，当前为: {split}")
    return value


def output_parent(output_root: Path, split: Optional[str], label: int) -> Path:
    if split:
        return output_root / split / label_name(label)
    return output_root / label_name(label)


def stable_hash_features(text: str, count: float, hash_dim: int) -> np.ndarray:
    vec = np.zeros(hash_dim + 12, dtype=np.float32)
    encoded = text.encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(encoded, digest_size=32).digest()
    for index in range(0, len(digest), 2):
        bucket = digest[index] % hash_dim
        sign = 1.0 if digest[index + 1] % 2 == 0 else -1.0
        vec[bucket] += sign

    stripped = text.strip()
    length = max(1, len(stripped))
    digits = sum(ch.isdigit() for ch in stripped)
    upper = sum(ch.isupper() for ch in stripped)
    separators = sum(ch in "./$_-" for ch in stripped)
    segments = [seg for seg in re.split(r"[./$_-]+", stripped) if seg]

    offset = hash_dim
    vec[offset + 0] = min(math.log1p(length) / 8.0, 1.0)
    vec[offset + 1] = min(len(segments) / 20.0, 1.0)
    vec[offset + 2] = digits / length
    vec[offset + 3] = upper / length
    vec[offset + 4] = separators / length
    vec[offset + 5] = float(is_obfuscated_name(stripped))
    vec[offset + 6] = float(stripped.startswith("android.") or ".android." in stripped)
    vec[offset + 7] = float(stripped.startswith("java.") or stripped.startswith("javax."))
    vec[offset + 8] = float("permission" in stripped.lower())
    vec[offset + 9] = float(any(pattern.search(stripped) for pattern in RISK_API_HINTS.values()))
    vec[offset + 10] = min(math.log1p(max(count, 1.0)) / 10.0, 1.0)
    vec[offset + 11] = 1.0
    return vec


class HeteroGraphBuilder:
    def __init__(self, hash_dim: int):
        self.hash_dim = hash_dim
        self.node_ids: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.node_features: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.node_counts: Dict[str, Counter] = defaultdict(Counter)
        self.edges: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)

    def add_node(self, node_type: str, name: str, count: float = 1.0) -> int:
        name = name or "<empty>"
        if name in self.node_ids[node_type]:
            self.node_counts[node_type][name] += count
            return self.node_ids[node_type][name]

        node_id = len(self.node_ids[node_type])
        self.node_ids[node_type][name] = node_id
        self.node_counts[node_type][name] = count
        self.node_features[node_type].append(stable_hash_features(name, count, self.hash_dim))
        return node_id

    def add_edge(
        self,
        src_type: str,
        src_id: int,
        relation: str,
        dst_type: str,
        dst_id: int,
        add_reverse: bool = True,
    ) -> None:
        self.edges[(src_type, relation, dst_type)].append((src_id, dst_id))
        if add_reverse:
            self.edges[(dst_type, f"rev_{relation}", src_type)].append((dst_id, src_id))

    def to_heterodata(self, label: int, apk_name: str, split: Optional[str]):
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError as exc:
            raise ImportError(
                "构建异构图需要 torch 和 torch-geometric，请先安装 requirements.txt 中的图模型依赖"
            ) from exc

        data = HeteroData()
        for node_type, features in self.node_features.items():
            data[node_type].x = torch.tensor(np.vstack(features), dtype=torch.float32)

        for edge_type, pairs in self.edges.items():
            if not pairs:
                continue
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            data[edge_type].edge_index = edge_index

        data.y = torch.tensor([int(label)], dtype=torch.long)
        data.apk_name = apk_name
        data.split = split
        return data

    def meta(self) -> Dict[str, object]:
        return {
            "node_names": {node_type: list(name_to_id.keys()) for node_type, name_to_id in self.node_ids.items()},
            "node_counts": {
                node_type: dict(counter) for node_type, counter in self.node_counts.items()
            },
            "edge_counts": {"/".join(edge_type): len(pairs) for edge_type, pairs in self.edges.items()},
        }


def extract_manifest(apk_obj, builder: HeteroGraphBuilder, app_id: int) -> None:
    for permission in sorted(set(apk_obj.get_permissions())):
        permission_id = builder.add_node("permission", permission)
        builder.add_edge("app", app_id, "uses_permission", "permission", permission_id)

    xml = apk_obj.get_android_manifest_xml()
    if xml is None:
        return

    for tag in ["activity", "service", "receiver", "provider"]:
        for item in xml.findall(f".//{tag}"):
            name = item.get("{http://schemas.android.com/apk/res/android}name")
            if not name:
                continue
            component_id = builder.add_node("component", f"{tag}:{name}")
            builder.add_edge("app", app_id, "declares_component", "component", component_id)

            for action in item.findall(".//action"):
                action_name = action.get("{http://schemas.android.com/apk/res/android}name")
                if action_name:
                    intent_id = builder.add_node("intent", action_name)
                    builder.add_edge("component", component_id, "handles_intent", "intent", intent_id)

            for category in item.findall(".//category"):
                category_name = category.get("{http://schemas.android.com/apk/res/android}name")
                if category_name:
                    intent_id = builder.add_node("intent", category_name)
                    builder.add_edge("component", component_id, "handles_intent", "intent", intent_id)


def collect_method_records(dx) -> Tuple[List[Dict[str, object]], Counter, Counter]:
    records: List[Dict[str, object]] = []
    api_counter = Counter()
    opcode_counter = Counter()

    for meth_analysis in dx.get_methods():
        if meth_analysis.is_external():
            continue
        meth = meth_analysis.get_method()
        if not meth:
            continue

        class_name = clean_class_name(meth.get_class_name())
        method_name = meth.get_name()
        descriptor = meth.get_descriptor()
        method_full_name = f"{class_name}.{method_name}{descriptor}"

        opcodes = Counter()
        if hasattr(meth, "get_instructions"):
            for ins in meth.get_instructions():
                opcode = ins.get_name()
                opcodes[opcode] += 1
                opcode_counter[opcode] += 1

        calls = []
        for _, call, _ in meth_analysis.get_xref_to():
            target_method = call.get_method()
            if not target_method:
                continue
            target_class = clean_class_name(target_method.get_class_name())
            target_name = target_method.get_name()
            if not target_class:
                continue
            api_call = f"{target_class}.{target_name}"
            api_counter[api_call] += 1
            calls.append(api_call)

        records.append(
            {
                "class": class_name,
                "method": method_full_name,
                "opcodes": opcodes,
                "calls": calls,
                "score": len(calls) + sum(opcodes.values()),
            }
        )

    return records, api_counter, opcode_counter


def build_graph_for_apk(
    apk_path: Path,
    label: int,
    hash_dim: int,
    max_methods: int,
    max_api_nodes: int,
    max_opcode_nodes: int,
):
    apk_obj, _, dx = AnalyzeAPK(str(apk_path))
    builder = HeteroGraphBuilder(hash_dim=hash_dim)
    package = apk_obj.get_package() or apk_path.stem
    app_id = builder.add_node("app", f"{apk_path.stem}|{package}", count=1)

    extract_manifest(apk_obj, builder, app_id)

    records, api_counter, opcode_counter = collect_method_records(dx)
    if max_methods > 0:
        records = sorted(records, key=lambda item: item["score"], reverse=True)[:max_methods]

    top_apis = {name for name, _ in api_counter.most_common(max_api_nodes)} if max_api_nodes > 0 else set(api_counter)
    top_opcodes = (
        {name for name, _ in opcode_counter.most_common(max_opcode_nodes)}
        if max_opcode_nodes > 0
        else set(opcode_counter)
    )

    for record in records:
        class_name = str(record["class"])
        method_name = str(record["method"])
        class_id = builder.add_node("class", class_name)
        method_id = builder.add_node("method", method_name, count=float(record["score"]))

        builder.add_edge("app", app_id, "has_class", "class", class_id)
        builder.add_edge("class", class_id, "defines_method", "method", method_id)

        for opcode, count in Counter(record["opcodes"]).items():
            if opcode not in top_opcodes:
                continue
            opcode_id = builder.add_node("opcode", opcode, count=float(count))
            builder.add_edge("method", method_id, "uses_opcode", "opcode", opcode_id)

        for api_call in record["calls"]:
            if api_call not in top_apis:
                continue
            api_id = builder.add_node("api", api_call, count=float(api_counter[api_call]))
            pkg = package_name(api_call.rsplit(".", 1)[0])
            package_id = builder.add_node("package", pkg, count=float(api_counter[api_call]))
            builder.add_edge("method", method_id, "calls_api", "api", api_id)
            builder.add_edge("api", api_id, "belongs_to_package", "package", package_id)

            for family, pattern in RISK_API_HINTS.items():
                if pattern.search(api_call):
                    family_id = builder.add_node("api_family", family, count=float(api_counter[api_call]))
                    builder.add_edge("api", api_id, "has_family", "api_family", family_id)

    return builder


def worker_task(task: Tuple[str, int, Optional[str], str, bool, int, int, int, int]) -> Dict[str, object]:
    apk_text, label, split, output_text, overwrite, hash_dim, max_methods, max_api_nodes, max_opcode_nodes = task
    apk_path = Path(apk_text)
    sample_dir = output_parent(Path(output_text), split, label) / apk_path.stem
    graph_path = sample_dir / "hetero_graph.pt"
    meta_path = sample_dir / "hetero_graph_meta.json"

    row = {
        "apk_name": apk_path.name,
        "apk_path": str(apk_path),
        "label": int(label),
        "split": split,
        "belong": split,
        "graph_path": str(graph_path),
        "meta_path": str(meta_path),
        "status": "ok",
        "error": "",
        "node_count": 0,
        "edge_count": 0,
    }

    try:
        if graph_path.exists() and meta_path.exists() and not overwrite:
            row["status"] = "skipped"
            return row

        sample_dir.mkdir(parents=True, exist_ok=True)
        builder = build_graph_for_apk(
            apk_path=apk_path,
            label=label,
            hash_dim=hash_dim,
            max_methods=max_methods,
            max_api_nodes=max_api_nodes,
            max_opcode_nodes=max_opcode_nodes,
        )
        data = builder.to_heterodata(label=label, apk_name=apk_path.name, split=split)

        import torch

        torch.save(data, graph_path)
        meta = builder.meta()
        meta.update(
            {
                "apk_name": apk_path.name,
                "apk_path": str(apk_path),
                "label": int(label),
                "split": split,
                "belong": split,
                "hash_dim": hash_dim,
                "max_methods": max_methods,
                "max_api_nodes": max_api_nodes,
                "max_opcode_nodes": max_opcode_nodes,
            }
        )
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        row["node_count"] = sum(len(names) for names in builder.node_ids.values())
        row["edge_count"] = sum(len(pairs) for pairs in builder.edges.values())
        return row
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        return row


def iter_apks(sample_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in sample_dir.rglob("*.apk") if path.is_file())


def infer_label_dir(dataset_root: Path, split: str, label: int) -> Optional[Path]:
    for alias in LABEL_DIR_ALIASES[int(label)]:
        candidate = dataset_root / split / alias
        if candidate.exists():
            return candidate
    return None


def load_manifest(manifest_path: Path, default_split: Optional[str]) -> List[Tuple[Path, int, Optional[str]]]:
    samples = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"apk_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("manifest CSV 必须包含 apk_path,label 两列")
        for row in reader:
            row_split = row.get("split") or row.get("belong") or default_split
            samples.append((Path(row["apk_path"]), int(row["label"]), normalize_split(row_split)))
    return samples


def collect_samples(args: argparse.Namespace) -> List[Tuple[Path, int, Optional[str]]]:
    samples: List[Tuple[Path, int, Optional[str]]] = []
    default_split = normalize_split(args.belong)
    if args.manifest:
        samples.extend(load_manifest(Path(args.manifest), default_split))
    if args.dataset_root:
        if default_split is None:
            raise ValueError("使用 --dataset-root 时必须传入 --belong train|val|test")
        dataset_root = Path(args.dataset_root)
        for label in [1, 0]:
            sample_dir = infer_label_dir(dataset_root, default_split, label)
            if sample_dir is None:
                print(f"[warning] 未找到 {dataset_root / default_split} 下 label={label} 的目录")
                continue
            samples.extend((apk_path, label, default_split) for apk_path in iter_apks(sample_dir))
    for spec in args.input:
        if "=" not in spec:
            raise ValueError("--input 格式应为 label=目录，例如 1=/data/mal")
        label_text, dir_text = spec.split("=", 1)
        sample_dir = Path(dir_text)
        if not sample_dir.exists():
            print(f"[warning] 目录不存在，跳过: {sample_dir}")
            continue
        samples.extend((apk_path, int(label_text), default_split) for apk_path in iter_apks(sample_dir))
    return samples


def write_index(rows: List[Dict[str, object]], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "apk_name",
        "apk_path",
        "label",
        "split",
        "belong",
        "graph_path",
        "meta_path",
        "status",
        "error",
        "node_count",
        "edge_count",
    ]
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 APK 异构图数据集")
    parser.add_argument("--input", action="append", default=[], help="label=目录，可重复传入")
    parser.add_argument("--dataset-root", help="原始数据根目录，自动读取 dataset_root/belong/mal 与 dataset_root/belong/benign")
    parser.add_argument("--belong", choices=sorted(VALID_SPLITS), help="当前处理的数据划分: train、val 或 test")
    parser.add_argument("--manifest", help="CSV，包含 apk_path,label 两列；可选 split 或 belong 列")
    parser.add_argument("--output", required=True, help="图输出目录")
    parser.add_argument("--index", default=None, help="index.csv 路径，默认写到 output/index.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-dim", type=int, default=64)
    parser.add_argument("--max-methods", type=int, default=3000)
    parser.add_argument("--max-api-nodes", type=int, default=2500)
    parser.add_argument("--max-opcode-nodes", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = [
        (apk_path, label, split)
        for apk_path, label, split in collect_samples(args)
        if apk_path.exists() and zipfile.is_zipfile(apk_path)
    ]
    print(f"[*] 待构图 APK: {len(samples)}")
    if not samples:
        return

    tasks = [
        (
            str(apk_path),
            int(label),
            split,
            args.output,
            args.overwrite,
            args.hash_dim,
            args.max_methods,
            args.max_api_nodes,
            args.max_opcode_nodes,
        )
        for apk_path, label, split in samples
    ]

    rows: List[Dict[str, object]] = []
    workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if row["status"] == "failed":
                print(f"[{index}/{len(tasks)}] [-] {row['apk_name']}: {row['error']}")
            if index % 10 == 0 or index == len(tasks):
                ok = sum(1 for item in rows if item["status"] in {"ok", "skipped"})
                failed = sum(1 for item in rows if item["status"] == "failed")
                print(f"[{index}/{len(tasks)}] ok={ok}, failed={failed}")

    if args.index:
        index_path = Path(args.index)
    elif args.belong:
        index_path = Path(args.output) / args.belong / "index.csv"
    else:
        index_path = Path(args.output) / "index.csv"
    write_index(sorted(rows, key=lambda item: str(item["apk_name"])), index_path)
    print(f"[*] index written: {index_path}")


if __name__ == "__main__":
    main()
