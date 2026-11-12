"""Discover failed APKs from phase1 outputs and remove their source and derived data."""

import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


SPLITS = {"train", "val", "test"}
LABELS = {"0": "benign", "1": "mal", "benign": "benign", "mal": "mal"}


@dataclass(frozen=True)
class Target:
    split: str
    label: str
    apk_name: str

    @property
    def stem(self) -> str:
        return self.apk_name[:-4]


def fail(message: str) -> None:
    raise SystemExit(message)


def target_from_fields(split: str, label: str, apk_name: str) -> Target:
    split = str(split).strip().lower()
    label = str(label).strip().lower()
    apk_name = str(apk_name).strip()
    if split not in SPLITS:
        fail(f"无效划分 {split!r}，应为 train、val 或 test")
    if label not in LABELS:
        fail(f"无效标签 {label!r}，应为 mal/benign 或 1/0")
    if not apk_name or apk_name in {".", ".."} or Path(apk_name).name != apk_name:
        fail(f"APK 名称必须是单个文件名: {apk_name!r}")
    if "/" in apk_name or "\\" in apk_name or ":" in apk_name:
        fail(f"APK 名称不能包含路径: {apk_name!r}")
    if not apk_name.lower().endswith(".apk"):
        apk_name += ".apk"
    if not apk_name[:-4] or apk_name[:-4] in {".", ".."}:
        fail(f"无效 APK 名称: {apk_name!r}")
    return Target(split, LABELS[label], apk_name)


def collect_targets(args: argparse.Namespace) -> list[Target]:
    targets = []
    if args.sample_id:
        if not args.split or not args.label:
            fail("使用 --sample-id 时必须同时指定 --split 和 --label")
        targets.extend(target_from_fields(args.split, args.label, value)
                       for group in args.sample_id for value in group)
    if args.manifest:
        with Path(args.manifest).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"split", "label", "apk_name"}.issubset(reader.fieldnames or []):
                fail("清理清单 CSV 必须包含 split,label,apk_name 三列")
            targets.extend(target_from_fields(row["split"], row["label"], row["apk_name"])
                           for row in reader)
    unique = list(dict.fromkeys(targets))
    directories = [(target.split, target.label, target.stem.casefold()) for target in unique]
    if len(directories) != len(set(directories)):
        fail("多个目标会写入同一个提取目录，请检查 APK 名称大小写及扩展名")
    return unique


def discover_failed_targets(apk_root: Path, static_root: Path, graph_root: Path,
                            split_filter: Optional[str], label_filter: Optional[str]) -> dict[Target, set[str]]:
    """Use explicit extraction errors and invalid APK archives, never missing outputs alone."""
    discovered: dict[Target, set[str]] = {}

    def mark(split: str, label: str, apk_name: str, reason: str) -> None:
        target = target_from_fields(split, label, apk_name)
        discovered.setdefault(target, set()).add(reason)

    splits = [split_filter] if split_filter else sorted(SPLITS)
    labels = [LABELS[label_filter]] if label_filter else ["benign", "mal"]
    for split in splits:
        for label in labels:
            apk_dir = checked_child(apk_root, split, label)
            if apk_dir.exists() and not apk_dir.is_dir():
                fail(f"原始 APK 分类路径不是目录: {apk_dir}")
            if apk_dir.is_dir():
                for apk in sorted(apk_dir.iterdir()):
                    if apk.is_symlink():
                        fail(f"原始 APK 目录含符号链接: {apk}")
                    if apk.is_file() and apk.suffix.lower() == ".apk" and not zipfile.is_zipfile(apk):
                        mark(split, label, apk.name, "APK 不是有效 ZIP")

            static_dir = checked_child(static_root, split, label)
            if static_dir.exists() and not static_dir.is_dir():
                fail(f"静态结果分类路径不是目录: {static_dir}")
            if static_dir.is_dir():
                for sample_dir in sorted(static_dir.iterdir()):
                    if sample_dir.is_symlink():
                        fail(f"静态结果目录含符号链接: {sample_dir}")
                    report_path = sample_dir / "static_features_report.json"
                    if not sample_dir.is_dir() or not report_path.exists():
                        continue
                    if report_path.is_symlink():
                        fail(f"静态报告是符号链接: {report_path}")
                    try:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        mark(split, label, sample_dir.name + ".apk", f"静态报告无法读取: {exc}")
                        continue
                    if not isinstance(report, dict):
                        mark(split, label, sample_dir.name + ".apk", "静态报告不是 JSON 对象")
                    elif report.get("error"):
                        mark(split, label, sample_dir.name + ".apk", f"静态提取失败: {report['error']}")

        index = checked_child(graph_root, split, "index.csv")
        if index.exists():
            if not index.is_file():
                fail(f"图索引路径不是文件: {index}")
            _, rows = read_index(index)
            for row in rows:
                label = LABELS.get(str(row.get("label", "")).strip().lower())
                if label is None or label not in labels:
                    continue
                row_split = (row.get("split") or row.get("belong") or "").strip().lower()
                if row_split and row_split != split:
                    fail(f"图索引行的划分与所在目录不符: {index}: {row.get('apk_name')}")
                if str(row.get("status") or "").strip().lower() == "failed" or str(row.get("error") or "").strip():
                    mark(split, label, row.get("apk_name") or "", f"图构建失败: {row.get('error') or row.get('status')}")
    return discovered


def checked_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        fail(f"目录不存在或是符号链接: {path}")
    return path.resolve()


def checked_child(root: Path, *parts: str) -> Path:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            fail(f"拒绝处理符号链接: {current}")
    if current.resolve().parent != (root.joinpath(*parts[:-1])).resolve():
        fail(f"路径越界: {current}")
    return current


def read_index(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.is_symlink():
        fail(f"拒绝处理符号链接索引: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {"apk_name", "label"}.issubset(fields):
            fail(f"图索引缺少 apk_name 或 label: {path}")
        return fields, list(reader)


def index_matches(row: dict[str, str], target: Target, index: Path) -> bool:
    row_label = LABELS.get(str(row.get("label", "")).strip().lower())
    if row.get("apk_name") != target.apk_name or row_label != target.label:
        return False
    row_split = (row.get("split") or row.get("belong") or "").strip().lower()
    if row_split and row_split != target.split:
        fail(f"图索引中的样本划分与目录不一致: {index}: {target.apk_name}")
    return True


def make_plan(targets: list[Target], apk_root: Path, static_root: Path, graph_root: Path,
              allow_missing_apk: bool = False):
    files = []
    directories = []
    index_updates = {}
    for target in targets:
        apk = checked_child(apk_root, target.split, target.label, target.apk_name)
        static_dir = checked_child(static_root, target.split, target.label, target.stem)
        graph_dir = checked_child(graph_root, target.split, target.label, target.stem)
        if not apk.is_file() and not allow_missing_apk:
            fail(f"找不到目标 APK，停止清理: {apk}")
        if apk.is_file():
            files.append(apk)
        for directory in (static_dir, graph_dir):
            if directory.exists():
                if not directory.is_dir():
                    fail(f"提取结果路径不是目录: {directory}")
                # rmtree would follow nested junctions/symlinks on some platforms.
                resolved_directory = directory.resolve()
                for path in directory.rglob("*"):
                    if path.is_symlink() or resolved_directory not in path.resolve().parents:
                        fail(f"提取目录含链接或越界路径，停止清理: {path}")
                directories.append(directory)

        index = checked_child(graph_root, target.split, "index.csv")
        if index.exists() and not index.is_file():
            fail(f"图索引路径不是文件: {index}")
        if index.is_file():
            if index not in index_updates:
                index_updates[index] = read_index(index)
            fields, rows = index_updates[index]
            matched = [row for row in rows if index_matches(row, target, index)]
            if len(matched) > 1:
                fail(f"图索引有重复样本行，停止清理: {index}: {target.apk_name}")
            index_updates[index] = (fields, [row for row in rows if row not in matched])
    return files, directories, index_updates


def rewrite_index(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent,
                                         prefix=".remove-bad-apks-", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk-root", type=Path, required=True, help="原始 APK 数据根目录")
    parser.add_argument("--static-root", type=Path, required=True, help="静态特征报告根目录")
    parser.add_argument("--graph-root", type=Path, required=True, help="图数据根目录")
    parser.add_argument("--split", choices=sorted(SPLITS), help="自动扫描的划分；省略则扫描全部划分")
    parser.add_argument("--label", choices=sorted(LABELS), help="自动扫描的标签；省略则扫描两类")
    parser.add_argument("--sample-id", nargs="+", action="append", default=[], help="APK 文件名或不带 .apk 的 ID；可重复指定")
    parser.add_argument("--manifest", type=Path, help="清理清单 CSV：split,label,apk_name；可跨划分")
    parser.add_argument("--audit-dir", type=Path, help="保存清理清单和原图索引的全新目录")
    parser.add_argument("--apply", action="store_true", help="执行删除；默认仅预览")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apk_root, static_root, graph_root = (checked_root(root) for root in
                                         (args.apk_root, args.static_root, args.graph_root))
    roots = (apk_root, static_root, graph_root)
    if len(set(roots)) != 3 or any(a in b.parents for a in roots for b in roots if a != b):
        fail("APK、静态数据、图数据根目录必须彼此独立")
    automatic = not args.sample_id and not args.manifest
    if automatic:
        reasons = discover_failed_targets(apk_root, static_root, graph_root, args.split, args.label)
        targets = sorted(reasons, key=lambda item: (item.split, item.label, item.apk_name))
    else:
        targets = collect_targets(args)
        if not targets:
            fail("指定的清理清单为空")
        reasons = {target: {"手动指定"} for target in targets}
    directories_by_name = [(target.split, target.label, target.stem.casefold()) for target in targets]
    if len(directories_by_name) != len(set(directories_by_name)):
        fail("多个错误记录对应同一个提取目录，请检查大小写冲突")
    if not targets:
        print("未发现明确的错误记录或无效 APK，未修改数据。")
        return
    files, directories, index_updates = make_plan(targets, apk_root, static_root, graph_root,
                                                    allow_missing_apk=automatic)
    for target in targets:
        apk = checked_child(apk_root, target.split, target.label, target.apk_name)
        print(f"{target.split}/{target.label}: {apk}；原因: {'；'.join(sorted(reasons[target]))}")
    for directory in directories:
        print(f"  删除提取目录: {directory}")
    for index, (_, rows) in index_updates.items():
        print(f"  更新图索引: {index}，保留 {len(rows)} 行")
    if not args.apply:
        print("预览结束，未修改数据。确认路径后添加 --apply。")
        return

    audit = args.audit_dir or Path("deduplicate_audit") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if audit.exists() or audit.is_symlink():
        fail(f"审计目录已存在: {audit}")
    audit = audit.resolve()
    if any(audit == root or root in audit.parents or audit in root.parents for root in roots):
        fail("审计目录必须位于三个数据根目录之外")
    audit.mkdir(parents=True)
    (audit / "targets.json").write_text(json.dumps(
        [{"split": t.split, "label": t.label, "apk_name": t.apk_name,
          "reasons": sorted(reasons[t])} for t in targets],
        ensure_ascii=False, indent=2), encoding="utf-8")
    for index in index_updates:
        shutil.copy2(index, audit / f"{index.parent.name}_index_before.csv")
    try:
        for index, (fields, rows) in index_updates.items():
            rewrite_index(index, fields, rows)
        for file in files:
            file.unlink()
        for directory in directories:
            shutil.rmtree(directory)
    except OSError as exc:
        fail(f"清理中断，可能已部分执行；原图索引备份位于 {audit}: {exc}")
    print(f"完成：处理 {len(targets)} 个错误样本，删除 {len(files)} 个 APK、{len(directories)} 个提取目录；审计记录: {audit}")


if __name__ == "__main__":
    main()
