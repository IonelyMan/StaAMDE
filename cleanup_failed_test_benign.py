#!/usr/bin/env python3
"""Remove the eleven known failed benign test samples. Dry run by default."""

"""
python3 cleanup_failed_test_benign.py \
  --apk-benign-dir ~/7T/lzw/datasets/android_zoo/android_apks/test/benign

预览会列出每个待删除路径。确认输出符合你的数据后，执行：
python3 cleanup_failed_test_benign.py \
  --apk-benign-dir ~/7T/lzw/datasets/android_zoo/android_apks/test/benign \
  --apply
"""


import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


FAILED_IDS = {
    "com.happy.mobilepop", "com.kica.android.app", "com.lottehotels.global",
    "com.mobilestudio.mycreditmanager", "com.sk.vas.tshare", "com.skt.sh",
    "com.skt.tjuniorparents", "com.sktelecom.tid",
    "com.wooribank.smart.zeropaybiz", "kr.ac.sejong.smartcampus", "kr.go.safeapp.app",
}


def fail(message):
    raise SystemExit(f"核验失败，停止执行: {message}")


def direct_directory(path):
    if path.is_symlink() or not path.is_dir():
        fail(f"目录不存在或是符号链接: {path}")
    return path.resolve()


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_exists(index_path, graph_text):
    # Match phase2.graph.data.normalize_index_graph_path.
    graph_path = Path(graph_text)
    candidate = index_path.parent / graph_path
    for path in (graph_path, candidate):
        if path.exists():
            return True
    for filename in ("hetero_graph_data.pt", "hetero_graph.pt"):
        for path in (graph_path.with_name(filename), candidate.with_name(filename)):
            if path.exists():
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk-benign-dir", type=Path, required=True,
                        help="原始 APK 的 test/benign 目录")
    parser.add_argument("--static-benign-dir", type=Path, default=Path(
        "/home/linux/7T/lzw/datasets/android_zoo/android_static/test/benign"))
    parser.add_argument("--graph-test-dir", type=Path, default=Path(
        "/home/linux/7T/lzw/datasets/android_zoo/android_graphs/test"))
    parser.add_argument("--audit-dir", type=Path, default=Path("cleanup_audit"),
                        help="保存索引备份和失败报告的目录（必须不存在）")
    parser.add_argument("--apply", action="store_true", help="执行不可逆删除；默认仅预览")
    args = parser.parse_args()

    apk_root = direct_directory(args.apk_benign_dir)
    static_root = direct_directory(args.static_benign_dir)
    graph_test = direct_directory(args.graph_test_dir)
    graph_root = direct_directory(graph_test / "benign")
    index = graph_test / "index.csv"
    if index.is_symlink() or not index.is_file():
        fail(f"图索引不存在或是符号链接: {index}")

    # Require one ordinary APK per ID and reject symlinked paths.
    apks = sorted(p for p in apk_root.rglob("*") if p.is_file() and p.suffix.lower() == ".apk")
    apk_counts = Counter(p.stem for p in apks)
    if len(apks) != 274 or len(apk_counts) != 274:
        fail(f"APK 应为 274 个且 ID 不重复，实际文件 {len(apks)}，ID {len(apk_counts)}")
    for p in apks:
        if p.is_symlink() or not inside(p.resolve(), apk_root):
            fail(f"APK 是符号链接或不在指定目录内: {p}")
    apk_by_id = {p.stem: p for p in apks}

    reports = sorted(static_root.rglob("static_features_report.json"))
    static_counts = Counter(p.parent.name for p in reports)
    if len(reports) != 274 or len(static_counts) != 274:
        fail(f"静态报告应为 274 个且 ID 不重复，实际文件 {len(reports)}，ID {len(static_counts)}")
    report_by_id = {}
    for p in reports:
        if p.is_symlink() or p.parent.is_symlink() or p.parent.parent != static_root:
            fail(f"静态报告必须位于普通目录 benign/<ID>/ 下: {p}")
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            fail(f"无法读取报告 {p}: {exc}")
        sid = p.parent.name
        if str(report.get("label")) != "0" or Path(str(report.get("apk_name") or (sid + ".apk"))).stem != sid:
            fail(f"静态报告标签或 APK 名称与目录不一致: {p}")
        if ("error" in report) != (sid in FAILED_IDS):
            fail(f"静态报告 error 与预期的 11 个失败 ID 不一致: {p}")
        report_by_id[sid] = p

    with index.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {"apk_name", "apk_path", "label", "status", "graph_path"}.issubset(fields):
        fail("图索引缺少必要字段")
    benign = [r for r in rows if r["label"] == "0"]
    index_counts = Counter(Path(r["apk_name"]).stem for r in benign)
    if len(benign) != 274 or len(index_counts) != 274:
        fail(f"图索引应有 274 条不重复的良性记录，实际 {len(benign)} 条，{len(index_counts)} 个 ID")
    apk_ids, static_ids, graph_ids = set(apk_by_id), set(report_by_id), set(index_counts)
    if apk_ids != static_ids or apk_ids != graph_ids:
        for label, difference in (
            ("APK 有、静态报告没有", apk_ids - static_ids),
            ("静态报告有、APK 没有", static_ids - apk_ids),
            ("图索引有、APK 没有", graph_ids - apk_ids),
            ("APK 有、图索引没有", apk_ids - graph_ids),
            ("静态报告有、图索引没有", static_ids - graph_ids),
            ("图索引有、静态报告没有", graph_ids - static_ids),
        ):
            print(f"{label}: {len(difference)} 个")
            for sid in sorted(difference):
                print(f"  {sid}")        
        fail("APK、静态报告与图索引的良性 ID 集合不一致")
    for r in benign:
        sid = Path(r["apk_name"]).stem
        if Path(r["apk_path"]).resolve() != apk_by_id[sid].resolve():
            fail(f"图索引的 APK 路径与指定目录不一致: {sid}: {r['apk_path']}")
        if not r["graph_path"]:
            fail(f"图路径为空: {sid}")
        exists = graph_exists(index, r["graph_path"])
        if sid in FAILED_IDS:
            if r["status"] != "failed" or exists:
                fail(f"预期失败且不存在图文件: {sid}")
        elif r["status"] not in {"ok", "skipped", ""} or not exists:
            fail(f"预期保留的图记录无效: {sid}")

    graph_dirs = []
    for sid in sorted(FAILED_IDS):
        p = graph_root / sid
        if p.is_symlink() or (p.exists() and not p.is_dir()):
            fail(f"图样本路径异常: {p}")
        if p.is_dir():
            graph_dirs.append(p)

    kept_rows = [r for r in rows if not (r["label"] == "0" and Path(r["apk_name"]).stem in FAILED_IDS)]
    if sum(r["label"] == "0" for r in kept_rows) != 263:
        fail("清理后图索引良性数量不是 263")
    print("待删除 11 个 APK、11 个静态样本目录、以下存在的图样本目录：")
    for sid in sorted(FAILED_IDS):
        print(f"  {sid}\n    APK: {apk_by_id[sid]}\n    静态: {report_by_id[sid].parent}\n    图: {graph_root / sid}")
    print(f"图索引: {index}；良性记录 274 -> 263；其他记录保留")
    if not args.apply:
        print("预览结束，未修改文件；确认路径和 ID 后添加 --apply 执行。")
        return

    audit = args.audit_dir.absolute()
    if audit.exists() or audit.is_symlink():
        fail(f"审计目录已存在，请指定新路径: {audit}")
    if any(inside(audit.resolve(), root) for root in (apk_root, static_root, graph_test)):
        fail("审计目录必须位于 APK、静态数据和图测试数据目录之外")
    # Audit is written before any deletion. It does not contain APKs.
    audit.mkdir(parents=True)
    shutil.copy2(index, audit / "index_before.csv")
    for sid in sorted(FAILED_IDS):
        shutil.copy2(report_by_id[sid], audit / f"{sid}.json")
    (audit / "manifest.json").write_text(json.dumps({
        "created": datetime.now().isoformat(),
        "apks": [{"sample_id": s, "path": str(apk_by_id[s]), "sha256": sha256_file(apk_by_id[s])}
                 for s in sorted(FAILED_IDS)],
        "static_root": str(static_root), "graph_test": str(graph_test),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=index.parent,
                                         prefix=".cleanup-index-", delete=False) as out:
            temp_name = out.name
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept_rows)
        # Replace the index before deleting files, so a failed replacement cannot
        # leave deleted APKs behind while retaining the old index.
        os.replace(temp_name, index)
        temp_name = None
        for sid in sorted(FAILED_IDS):
            apk_by_id[sid].unlink()
            shutil.rmtree(report_by_id[sid].parent)
        for path in graph_dirs:
            shutil.rmtree(path)
    except Exception as exc:
        print(f"清理中断，可能已部分删除；原图索引和失败报告备份在 {audit}。请勿直接重跑。")
        raise SystemExit(str(exc)) from exc
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()

    print(f"完成：已删除 11 个 APK 和静态样本目录、{len(graph_dirs)} 个图样本目录；图索引保留 263 条良性记录。")
    print(f"原索引和失败报告备份: {audit}")
    print("注意：旧的预测 CSV 不会自动更新，需要重新推理才能刷新。")


if __name__ == "__main__":
    main()
