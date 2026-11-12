"""Read-only comparison of source APKs, static reports, and graph outputs."""

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


GRAPH_FILES = ("hetero_graph.pt", "hetero_graph_data.pt")


def read_index(path: Path):
    rows = defaultdict(list)
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = {"0": "benign", "1": "mal"}.get(str(row.get("label", "")))
            name = row.get("apk_name")
            if label and name:
                rows[(label, Path(name).stem)].append(row)
    return rows


def audit_split(apk_root: Path, static_root: Path, graph_root: Path, split: str) -> int:
    index = read_index(graph_root / split / "index.csv")
    differences = 0
    for label in ("benign", "mal"):
        apk_dir = apk_root / split / label
        static_dir = static_root / split / label
        graph_dir = graph_root / split / label
        apks = defaultdict(list)
        if apk_dir.is_dir():
            for path in sorted(apk_dir.rglob("*.apk")):
                if path.is_file():
                    apks[path.stem].append(path)
        static_ids = {path.parent.name for path in static_dir.glob("*/static_features_report.json")} if static_dir.is_dir() else set()
        graph_ids = {path.parent.name for filename in GRAPH_FILES for path in graph_dir.glob(f"*/{filename}")} if graph_dir.is_dir() else set()
        problems = []
        for sample_id, paths in sorted(apks.items()):
            issues = []
            if len(paths) != 1:
                issues.append(f"同名 APK 有 {len(paths)} 个")
            for path in paths:
                if not zipfile.is_zipfile(path):
                    issues.append("APK 不是有效 ZIP")
            report_path = static_dir / sample_id / "static_features_report.json"
            if not report_path.is_file():
                issues.append("无静态报告")
            else:
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if not isinstance(report, dict):
                        issues.append("静态报告不是 JSON 对象")
                    elif "error" in report:
                        issues.append(f"静态报告含 error 字段（加载器会跳过）: {report['error']}")
                except (OSError, ValueError) as exc:
                    issues.append(f"静态报告无法读取: {exc}")
            if sample_id not in graph_ids:
                issues.append("无图数据")
            rows = index.get((label, sample_id), [])
            if not rows:
                issues.append("图索引无记录")
            for row in rows:
                if row.get("status") not in {"ok", "skipped", ""} or row.get("error"):
                    issues.append(f"图索引状态或错误: {row.get('error') or row.get('status')}")
            if issues:
                problems.append((sample_id, paths, issues))
        for sample_id in sorted((static_ids | graph_ids | {name for row_label, name in index if row_label == label}) - apks.keys()):
            problems.append((sample_id, [], ["有提取结果或索引，但无原始 APK"]))
        differences += len(problems)
        print(f"\n[{split}/{label}] APK={sum(map(len, apks.values()))}，"
              f"静态报告={len(static_ids)}，图文件={len(graph_ids)}，"
              f"图索引记录={sum(len(index[key]) for key in index if key[0] == label)}")
        if not problems:
            print("  未发现按样本 ID 的差异")
        for sample_id, paths, issues in problems:
            print(f"  {sample_id}: {'；'.join(issues)}")
            for path in paths:
                print(f"    APK: {path}")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    for root in (args.apk_root, args.static_root, args.graph_root):
        if not root.is_dir():
            parser.error(f"目录不存在: {root}")
    total = audit_split(args.apk_root, args.static_root, args.graph_root, args.split)
    print(f"\n发现 {total} 个需核查的样本；脚本只读，未修改文件。")


if __name__ == "__main__":
    main()
