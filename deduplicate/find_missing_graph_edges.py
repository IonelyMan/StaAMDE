"""扫描缺少 edge_index 的图，生成供 remove_bad_apks 使用的清单。"""

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deduplicate.remove_bad_apks import SPLITS, checked_child, checked_root, target_from_fields
from phase2.graph.data import GRAPH_DATA_FILENAMES, load_graph, to_homogeneous_graph


def scan(root: Path, split_filter: Optional[str] = None):
    """Only inspect the graph files and directory layout used by the graph loader."""
    root = checked_root(root)
    targets = []
    inspected = 0
    for split in ([split_filter] if split_filter else sorted(SPLITS)):
        for label in ("benign", "mal"):
            directory = checked_child(root, split, label)
            if not directory.exists():
                continue
            if not directory.is_dir():
                raise ValueError(f"图分类路径不是目录: {directory}")
            for sample_dir in sorted(directory.iterdir()):
                if sample_dir.is_symlink():
                    raise ValueError(f"拒绝处理符号链接: {sample_dir}")
                if not sample_dir.is_dir():
                    continue
                graph_path = None
                for filename in GRAPH_DATA_FILENAMES:
                    candidate = sample_dir / filename
                    if candidate.is_symlink():
                        raise ValueError(f"拒绝处理符号链接: {candidate}")
                    if candidate.exists():
                        if not candidate.is_file():
                            raise ValueError(f"图数据路径不是文件: {candidate}")
                        if graph_path is None:
                            graph_path = candidate
                if graph_path is None:
                    continue
                inspected += 1
                try:
                    data = to_homogeneous_graph(load_graph(graph_path))
                    missing = getattr(data, "edge_index", None) is None
                except Exception as exc:
                    raise RuntimeError(f"无法加载或转换图 {graph_path}: {exc}") from exc
                if missing:
                    target = target_from_fields(split, label, sample_dir.name + ".apk")
                    targets.append(target)
                    print(f"缺少 edge_index: {graph_path}")
    if not inspected:
        raise ValueError(f"未找到可扫描的图文件: {root}")
    return targets, inspected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), help="省略则扫描全部划分")
    parser.add_argument("--output", type=Path, required=True, help="新建清单文件，不能覆盖已有文件")
    args = parser.parse_args()
    try:
        targets, inspected = scan(args.graph_root, args.split)
        content = io.StringIO(newline="")
        writer = csv.writer(content)
        writer.writerow(["split", "label", "apk_name"])
        writer.writerows((t.split, t.label, t.apk_name) for t in targets)
        # Exclusive creation prevents overwriting a prior list and happens only after a full scan.
        handle = args.output.open("x", encoding="utf-8-sig", newline="")
        try:
            with handle:
                handle.write(content.getvalue())
        except Exception:
            args.output.unlink(missing_ok=True)
            raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"扫描失败，未生成完整清单: {exc}") from exc
    print(f"扫描 {inspected} 个图，发现 {len(targets)} 个缺少 edge_index；清单: {args.output}")


if __name__ == "__main__":
    main()
