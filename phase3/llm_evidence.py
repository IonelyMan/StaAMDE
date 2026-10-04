import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM evidence packet and prompt from SHAP and graph attention outputs.")
    parser.add_argument("--sample-id", default=None, help="APK sample id. If omitted, choose the highest-risk available prediction.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="llm_explain")
    parser.add_argument("--static-predictions", default="inference/lightgbm/predictions/test_predictions.csv")
    parser.add_argument("--graph-predictions", default="inference/tasgatv2/predictions/test_predictions.csv")
    parser.add_argument("--ensemble-predictions", default="ensemble/ensemble_predictions.csv")
    parser.add_argument("--ml-explain-dir", default="ml_explain", help="机器学习解释结果根目录")
    parser.add_argument("--top-static-positive", type=int, default=20)
    parser.add_argument("--top-static-negative", type=int, default=10)
    parser.add_argument("--top-global-static", type=int, default=20)
    parser.add_argument("--top-graph-nodes", type=int, default=25)
    parser.add_argument("--top-graph-edges", type=int, default=20)
    parser.add_argument("--max-name-len", type=int, default=320)
    return parser.parse_args()


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def compact_text(text: object, max_len: int) -> str:
    value = str(text or "")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= max_len:
        return value
    head = max_len // 2 - 3
    tail = max_len - head - 7
    return f"{value[:head]} ... {value[-tail:]}"


def safe_filename(text: str) -> str:
    text = Path(str(text)).name
    if text.lower().endswith(".apk"):
        text = text[:-4]
    keep = []
    for char in str(text):
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "sample"


def prediction_label(pred: Optional[int]) -> str:
    if pred is None:
        return "unknown"
    return "malware" if int(pred) == 1 else "benign"


def index_predictions(paths: Dict[str, Path]) -> Dict[str, Dict[str, Dict[str, object]]]:
    indexed: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name, path in paths.items():
        rows = read_csv(path)
        for row in rows:
            sample_id = row.get("sample_id")
            if not sample_id:
                sample_key = str(row.get("sample_key") or "")
                sample_id = sample_key.split(":")[-1] if sample_key else ""
            if not sample_id:
                continue
            indexed.setdefault(str(sample_id), {})[name] = {
                "source": str(path),
                "sample_key": row.get("sample_key"),
                # "sample_id": str(sample_id),
                "apk_name": row.get("apk_name") or f"{sample_id}.apk",
                "split": row.get("split"),
                # "y_true": to_int(row.get("y_true"), -1) if row.get("y_true") not in {None, ""} else None,
                "prob_malware": to_float(row.get("prob_malware")),
                "pred": to_int(row.get("pred"), 1 if to_float(row.get("prob_malware")) >= 0.5 else 0),
            }
            for key, value in row.items():
                if key.startswith("prob_"):
                    indexed[str(sample_id)][name][key] = to_float(value)
    return indexed


def choose_sample_id(indexed_predictions: Dict[str, Dict[str, Dict[str, object]]], requested: Optional[str]) -> str:
    if requested:
        return requested
    candidates = []
    for sample_id, modalities in indexed_predictions.items():
        row = modalities.get("ensemble") or modalities.get("static") or modalities.get("graph")
        if not row:
            continue
        prob = to_float(row.get("prob_malware"))
        pred = to_int(row.get("pred"), int(prob >= 0.5))
        candidates.append((pred, prob, sample_id))
    if not candidates:
        raise SystemExit("No prediction rows were found. Pass --sample-id and valid prediction CSV paths.")
    candidates.sort(reverse=True)
    return candidates[0][2]


def load_static_local_evidence(
    sample_dir: Path,
    sample_id: str,
    top_positive: int,
    top_negative: int,
    max_name_len: int,
) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    path = sample_dir / "shap_contributions.csv"
    if not path.exists():
        warnings.append(
            f"No local SHAP CSV found at {path}. Run phase2.lightgbm.explain with --sample-id."
        )
        return {"available": False, "top_positive_features": [], "top_negative_features": []}, warnings

    rows = [row for row in read_csv(path) if not row.get("sample_id") or row.get("sample_id") == sample_id]
    if not rows:
        warnings.append(f"SHAP CSV has no rows for sample_id={sample_id}: {path}")
        return {"available": False, "top_positive_features": [], "top_negative_features": []}, warnings
    rows = [row for row in rows if not str(row.get("feature", "")).startswith("other_")]
    positives = sorted([row for row in rows if to_float(row.get("shap_value")) > 0], key=lambda row: to_float(row.get("shap_value")), reverse=True)
    negatives = sorted([row for row in rows if to_float(row.get("shap_value")) < 0], key=lambda row: to_float(row.get("shap_value")))

    def convert(row: Dict[str, str]) -> Dict[str, object]:
        feature = row.get("feature") or ""
        return {
            "rank": to_int(row.get("rank"), 0),
            "feature": compact_text(feature, max_name_len),
            "feature_value": row.get("feature_value"),
            "shap_value": to_float(row.get("shap_value")),
            "abs_shap_value": to_float(row.get("abs_shap_value"), abs(to_float(row.get("shap_value")))),
            "direction": "pushes_toward_malware" if to_float(row.get("shap_value")) > 0 else "pushes_toward_benign",
        }

    sample_meta = rows[0] if rows else {}
    return (
        {
            "available": True,
            "source": str(path),
            # "sample_id": sample_meta.get("sample_id") or sample_id,
            "apk_name": sample_meta.get("apk_name"),
            # "y_true": to_int(sample_meta.get("y_true"), -1) if sample_meta.get("y_true") not in {None, ""} else None,
            "prob_malware": to_float(sample_meta.get("prob_malware")),
            "pred": to_int(sample_meta.get("pred"), -1) if sample_meta.get("pred") not in {None, ""} else None,
            "base_value": to_float(sample_meta.get("base_value")),
            "model_output": to_float(sample_meta.get("model_output")),
            "top_positive_features": [convert(row) for row in positives[:top_positive]],
            "top_negative_features": [convert(row) for row in negatives[:top_negative]],
        },
        warnings,
    )


def load_static_global_summary(ml_explain_dir: Path, split: str, top_k: int, max_name_len: int) -> Dict[str, object]:
    path = ml_explain_dir / "global" / "lightgbm" / f"{split}_shap_summary.csv"
    rows = read_csv(path)
    if not rows:
        return {"available": False, "features": []}
    rows = sorted(rows, key=lambda row: to_float(row.get("mean_abs_shap")), reverse=True)[:top_k]
    return {
        "available": True,
        "source": str(path),
        "features": [
            {
                "rank": to_int(row.get("rank"), index + 1),
                "feature": compact_text(row.get("feature"), max_name_len),
                "n_samples": to_int(row.get("n_samples")) if row.get("n_samples") not in {None, ""} else None,
                "mean_abs_shap": to_float(row.get("mean_abs_shap")),
                "mean_shap": to_float(row.get("mean_shap")),
            }
            for index, row in enumerate(rows)
        ],
    }


def load_graph_evidence(
    sample_dir: Path,
    sample_id: str,
    top_nodes: int,
    top_edges: int,
    max_name_len: int,
) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    node_path = sample_dir / "node_attention.csv"
    edge_path = sample_dir / "edge_attention.csv"
    node_rows = read_csv(node_path)
    edge_rows = read_csv(edge_path)
    node_rows = [row for row in node_rows if not row.get("sample_id") or row.get("sample_id") == sample_id]
    edge_rows = [row for row in edge_rows if not row.get("sample_id") or row.get("sample_id") == sample_id]
    if not node_rows:
        warnings.append(
            f"No graph node attention CSV found for sample_id={sample_id}. "
            "Run phase2.graph.explain with --sample-id to generate graph evidence for this APK."
        )
    if not edge_rows:
        warnings.append(
            f"No graph edge attention CSV found for sample_id={sample_id}. "
            "Run phase2.graph.explain with --sample-id to generate graph evidence for this APK."
        )

    node_rows = [row for row in node_rows if row.get("node_type") != "app"]
    # 排序注意力分数，降序
    node_rows = sorted(node_rows, key=lambda row: to_float(row.get("attention")), reverse=True)[:top_nodes]
    edge_rows = [row for row in edge_rows if row.get("source_node") != row.get("target_node")]
    edge_rows = sorted(edge_rows, key=lambda row: to_float(row.get("attention")), reverse=True)[:top_edges]

    def convert_node(row: Dict[str, str]) -> Dict[str, object]:
        name = row.get("node_name") or ""
        return {
            "rank": len(converted_nodes) + 1,
            "node_type": row.get("node_type"),
            "local_index": to_int(row.get("local_index"), -1),
            "node_name": compact_text(name, max_name_len),
            "attention": to_float(row.get("attention")),
            "normalized_attention": to_float(row.get("normalized_attention")),
        }

    converted_nodes: List[Dict[str, object]] = []
    for row in node_rows:
        converted_nodes.append(convert_node(row))

    converted_edges = []
    for index, row in enumerate(edge_rows, start=1):
        source = row.get("source_name") or ""
        target = row.get("target_name") or ""
        converted_edges.append(
            {
                "rank": index,
                "source_type": row.get("source_type"),
                "source_name": compact_text(source, max_name_len),
                "target_type": row.get("target_type"),
                "target_name": compact_text(target, max_name_len),
                "attention": to_float(row.get("attention")),
            }
        )

    sample_meta = node_rows[0] if node_rows else {}
    return (
        {
            "available": bool(node_rows or edge_rows),
            "node_attention_source": str(node_path),
            "edge_attention_source": str(edge_path),
            # "sample_id": sample_id,
            "apk_name": sample_meta.get("apk_name"),
            # "y_true": to_int(sample_meta.get("y_true"), -1) if sample_meta.get("y_true") not in {None, ""} else None,
            "prob_malware": to_float(sample_meta.get("prob_malware")),
            "pred": to_int(sample_meta.get("pred"), -1) if sample_meta.get("pred") not in {None, ""} else None,
            "top_nodes": converted_nodes,
            "top_edges": converted_edges,
        },
        warnings,
    )


def build_prediction_summary(sample_id: str, modalities: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    row = modalities.get("ensemble") or modalities.get("static") or modalities.get("graph") or {}
    summary = {
        # "sample_id": sample_id,
        "apk_name": row.get("apk_name") or f"{sample_id}.apk",
        "split": row.get("split"),
        # "y_true": row.get("y_true"),
        "final_prediction": prediction_label(row.get("pred")),
        "final_prob_malware": row.get("prob_malware"),
        "modalities": {},
    }
    for name in ["ensemble", "static", "graph"]:
        if name in modalities:
            summary["modalities"][name] = {
                "prob_malware": modalities[name].get("prob_malware"),
                "pred": prediction_label(modalities[name].get("pred")),
                "source": modalities[name].get("source"),
            }
            if name == "ensemble":
                for key, value in modalities[name].items():
                    if str(key).startswith("prob_"):
                        summary["modalities"][name][key] = value
    return summary


def compact_record(record: Dict[str, object]) -> Dict[str, object]:
    compact: Dict[str, object] = {}
    for key, value in record.items():
        if value is None:
            continue
        if value == "":
            continue
        compact[key] = value
    return compact


def prompt_feature_rows(rows: object, limit: int) -> List[Dict[str, object]]:
    result = []
    if not isinstance(rows, list):
        return result
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        result.append(
            compact_record(
                {
                    "rank": row.get("rank"),
                    "type": row.get("node_type"),
                    "name": row.get("feature") or row.get("node_name"),
                    "value": row.get("feature_value"),
                    "shap_value": row.get("shap_value"),
                    "abs_shap_value": row.get("abs_shap_value"),
                    "mean_abs_shap": row.get("mean_abs_shap"),
                    "mean_shap": row.get("mean_shap"),
                    "n_samples": row.get("n_samples"),
                    "attention": row.get("attention"),
                    "normalized_attention": row.get("normalized_attention"),
                    "direction": row.get("direction"),
                }
            )
        )
    return result


def prompt_edge_rows(rows: object, limit: int) -> List[Dict[str, object]]:
    result = []
    if not isinstance(rows, list):
        return result
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        result.append(
            compact_record(
                {
                    "rank": row.get("rank"),
                    "source_type": row.get("source_type"),
                    "source": row.get("source_name"),
                    "target_type": row.get("target_type"),
                    "target": row.get("target_name"),
                    "attention": row.get("attention"),
                }
            )
        )
    return result


def compact_prediction_for_prompt(prediction: Dict[str, object]) -> Dict[str, object]:
    modalities = {}
    for name, row in dict(prediction.get("modalities") or {}).items():
        if not isinstance(row, dict):
            continue
        compact = {"prob_malware": row.get("prob_malware"), "pred": row.get("pred")}
        for key in ["prob_static", "prob_graph"]:
            if key in row:
                compact[key] = row[key]
        modalities[name] = compact
    return {
        # "sample_id": prediction.get("sample_id"),
        "apk_name": prediction.get("apk_name"),
        # "y_true": prediction.get("y_true"),
        "final_prediction": prediction.get("final_prediction"),
        "final_prob_malware": prediction.get("final_prob_malware"),
        "modalities": modalities,
    }


def build_prompt(evidence: Dict[str, object]) -> str:
    static_local = evidence["static_local_shap"] if isinstance(evidence.get("static_local_shap"), dict) else {}
    static_global = evidence["static_global_shap"] if isinstance(evidence.get("static_global_shap"), dict) else {}
    graph_attention = evidence["graph_attention"] if isinstance(evidence.get("graph_attention"), dict) else {}
    compact_payload = {
        "prediction": compact_prediction_for_prompt(evidence["prediction"]),
        "evidence_semantics": {
            "positive_shap": "positive SHAP values increase the malware-class score",
            "negative_shap": "negative SHAP values decrease the malware-class score",
            "graph_attention": "higher attention means the graph model focused more on that node/edge",
        },
        "static_shap_local": {
            "top_positive": prompt_feature_rows(static_local.get("top_positive_features"), 10_000),
            "top_negative": prompt_feature_rows(static_local.get("top_negative_features"), 10_000),
        },
        "static_shap_global_top": prompt_feature_rows(static_global.get("features"), 10_000),
        "graph_attention": {
            "top_nodes": prompt_feature_rows(graph_attention.get("top_nodes"), 10_000),
            "top_edges": prompt_edge_rows(graph_attention.get("top_edges"), 10_000),
        },
        "caveats": [
            "SHAP/attention are model evidence, not causal proof.",
            "Obfuscation or missing extraction can reduce confidence.",
        ],
    }
    payload = json.dumps(compact_payload, ensure_ascii=False, indent=2)
    return f"""你是安卓恶意软件分析专家，下面是使用机器学习模型对一个 APK 的分类结果（良性或恶意），包括 LightGBM 的局部SHAP贡献以及图模型的节点/边注意力这些归因证据。
请你基于这些证据，理解权限、API、方法名、类名等特征的用途，并解释它们可能代表的安卓行为。
核心要求：
1. 如果你有联网检索能力，请先主动检索最重要、最不确定的权限、Android API、组件和方法名。优先查 Android Developers 官方 API Reference 与 Android 安全/权限文档；第三方库请查其官方文档。核对名称、所属类、版本限制和实际用途，再解释证据。不要只凭名称猜测。
2. 对关键技术解释给出可核验的文档链接，并说明链接支持哪项用途。不得编造网址、引文或检索结果。如果当前无法联网，明确说明“未完成联网核验”，仅作有保留的解释，不得声称已查证。
3. 将 APK 中实际观测到的特征、模型归因、官方文档说明、由此推测的行为区分开。某个 API/权限的存在或高注意力不等于该行为实际发生，更不能单独证明恶意意图。不要把 SHAP 或 attention 当作因果证明。
4. 优先解释局部正向 SHAP 与图注意力 Top 节点/边，也说明局部负向 SHAP 是否削弱恶意判断。全局 SHAP 仅说明模型在样本集合中的平均关注程度，不能代替该 APK 的局部证据。
5. 对混淆名、非 Android 官方 API、无法核验或证据不足处，明确说不确定，不要补造代码调用链、网络连接、窃取行为或用户操作。
6. 若最终预测为恶意，输出：最终判断、关键证据解释、可能行为、防护建议（面向非专业人员）；若预测为良性，输出：最终判断、关键证据解释。语言简洁。

证据 JSON：
```json
{payload}
```
"""


def main() -> None:
    args = parse_args()
    prediction_paths = {
        "static": Path(args.static_predictions),
        "graph": Path(args.graph_predictions),
        "ensemble": Path(args.ensemble_predictions),
    }
    indexed_predictions = index_predictions(prediction_paths)
    sample_id = choose_sample_id(indexed_predictions, args.sample_id)
    modalities = indexed_predictions.get(sample_id, {})
    apk_name = (modalities.get("ensemble") or modalities.get("static") or modalities.get("graph") or {}).get("apk_name") or f"{sample_id}.apk"
    ml_explain_dir = Path(args.ml_explain_dir)
    sample_dir = ml_explain_dir / safe_filename(str(apk_name))

    static_local, static_warnings = load_static_local_evidence(
        sample_dir,
        sample_id,
        args.top_static_positive,
        args.top_static_negative,
        args.max_name_len,
    )
    static_global = load_static_global_summary(ml_explain_dir, args.split, args.top_global_static, args.max_name_len)
    graph_evidence, graph_warnings = load_graph_evidence(
        sample_dir,
        sample_id,
        args.top_graph_nodes,
        args.top_graph_edges,
        args.max_name_len,
    )

    evidence = {
        "prediction": build_prediction_summary(sample_id, modalities),
        "static_local_shap": static_local,
        "static_global_shap": static_global,
        "graph_attention": graph_evidence,
        "caveats": [
            "SHAP explains how static features changed the LightGBM malware score; it is not proof of malicious intent by itself.",
            "Graph attention indicates model focus in the graph classifier; high-attention nodes are candidates for review, not guaranteed malicious code.",
            "Obfuscated method/class names and incomplete failed extractions can reduce interpretability.",
            "API and permission semantics should be verified against authoritative documentation before behavioral inference.",
        ],
        "warnings": static_warnings + graph_warnings,
    }

    output_dir = Path(args.output) / safe_filename(str(apk_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "llm_evidence.json"
    prompt_path = output_dir / "llm_prompt.md"
    write_json(evidence_path, evidence)
    prompt_path.write_text(build_prompt(evidence), encoding="utf-8")

    print(f"sample_id: {sample_id}")
    print(f"evidence: {evidence_path}")
    print(f"prompt: {prompt_path}")
    if evidence["warnings"]:
        print(json.dumps({"warnings": evidence["warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
