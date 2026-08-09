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
    parser.add_argument("--output", default="runs/llm_explain")
    parser.add_argument("--static-predictions", default="runs/static_lgbm/predictions/test_predictions.csv")
    parser.add_argument("--graph-predictions", default="runs/hetero_gatv2/predictions/test_predictions.csv")
    parser.add_argument("--ensemble-predictions", default="runs/ensemble/ensemble_predictions.csv")
    parser.add_argument("--shap-reports", default="runs/static_lgbm/reports")
    parser.add_argument("--graph-explain-dir", default="runs/hetero_gatv2/explanations")
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


def find_waterfall_csv(shap_reports: Path, sample_id: str, split: str) -> Optional[Path]:
    preferred = shap_reports / f"{split}_{safe_filename(sample_id)}_waterfall_contributions.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(shap_reports.glob(f"*{safe_filename(sample_id)}*waterfall_contributions.csv"))
    if candidates:
        return candidates[0]
    for path in sorted(shap_reports.glob("*waterfall_contributions.csv")):
        rows = read_csv(path)
        if any(str(row.get("sample_id")) == sample_id for row in rows):
            return path
    return None


def load_static_local_evidence(
    shap_reports: Path,
    sample_id: str,
    split: str,
    top_positive: int,
    top_negative: int,
    max_name_len: int,
) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    path = find_waterfall_csv(shap_reports, sample_id, split)
    if path is None:
        warnings.append(
            "No local SHAP waterfall contribution CSV was found. Run phase2.lightgbm.explain with --waterfall-sample-id for stronger local evidence."
        )
        return {"available": False, "top_positive_features": [], "top_negative_features": []}, warnings

    rows = [row for row in read_csv(path) if not row.get("sample_id") or row.get("sample_id") == sample_id]
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


def load_static_global_summary(shap_reports: Path, split: str, top_k: int, max_name_len: int) -> Dict[str, object]:
    path = shap_reports / f"{split}_shap_summary.csv"
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
                "mean_abs_shap": to_float(row.get("mean_abs_shap")),
                "mean_shap": to_float(row.get("mean_shap")),
            }
            for index, row in enumerate(rows)
        ],
    }


def load_graph_evidence(
    graph_explain_dir: Path,
    sample_id: str,
    top_nodes: int,
    top_edges: int,
    max_name_len: int,
) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    node_path = graph_explain_dir / "node_attention" / f"{sample_id}.csv"
    edge_path = graph_explain_dir / "edge_attention" / f"{sample_id}.csv"
    node_rows = read_csv(node_path)
    edge_rows = read_csv(edge_path)
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
        # "static_shap_global_top": prompt_feature_rows(static_global.get("features"), 10_000),
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
    return f"""你是安卓恶意软件分析专家。下面是一个 APK 的模型预测结果与可解释性证据，包括 LightGBM 的局部/全局 SHAP 特征，以及图模型的 attention 节点和边。
请你基于这些证据自己理解权限、API、方法名、类名或其他特征的用途，不要依赖预设标签。你的目标是解释：这些证据可能代表什么安卓行为，为什么是恶意的软件
要求：
1. 不要把 SHAP 或 attention 写成绝对因果证明；使用“模型关注到”“可能说明”“需要结合上下文确认”等表述。
2. 优先解释正向 SHAP 特征、图 attention Top 节点/边；同时说明负向 SHAP 是否削弱恶意判断。
3. 对你看得懂的 API/权限/方法名，简要解释其通常用途；对混淆名、未知名或证据不足处，明确说不确定。
4. 如果判定为恶意软件则输出：最终判断、关键证据解释、可能行为、防护建议（给非专业人员)。
5. 如果判定为良性软件则输出：最终判断、关键证据解释。
6. 解释内容尽量简洁，但又能清楚表达。

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

    static_local, static_warnings = load_static_local_evidence(
        Path(args.shap_reports),
        sample_id,
        args.split,
        args.top_static_positive,
        args.top_static_negative,
        args.max_name_len,
    )
    static_global = load_static_global_summary(Path(args.shap_reports), args.split, args.top_global_static, args.max_name_len)
    graph_evidence, graph_warnings = load_graph_evidence(
        Path(args.graph_explain_dir),
        sample_id,
        args.top_graph_nodes,
        args.top_graph_edges,
        args.max_name_len,
    )

    evidence = {
        "prediction": build_prediction_summary(sample_id, modalities),
        "static_local_shap": static_local,
        # "static_global_shap": static_global,
        "graph_attention": graph_evidence,
        "caveats": [
            "SHAP explains how static features changed the LightGBM malware score; it is not proof of malicious intent by itself.",
            "Graph attention indicates model focus in the graph classifier; high-attention nodes are candidates for review, not guaranteed malicious code.",
            "Obfuscated method/class names and incomplete failed extractions can reduce interpretability.",
            "DEX image modality is intentionally excluded from this LLM evidence packet because it is weaker and not directly behavior-interpretable.",
            "No hard-coded API behavior labels are added; the LLM should infer API/permission/method semantics from the raw evidence.",
        ],
        "warnings": static_warnings + graph_warnings,
    }

    output_dir = Path(args.output) / safe_filename(sample_id)
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


"""
默认给 LLM 更多原始证据：局部 SHAP 正向贡献 Top-20
局部 SHAP 负向贡献 Top-10
全局 SHAP Top-20
图 attention Top-25 节点
图 attention Top-20 边
静态/图/集成模型预测概率

prompt 会明确要求 LLM 自己解释 API、权限、方法名用途，并总结为什么被判定为恶意/良性，以及给防护建议。

继续这样用：
python -m phase3.llm_evidence \
  --sample-id com.thecybernanny.adroapp \
  --static-predictions inference/lightgbm/predictions/test_predictions.csv \
  --graph-predictions inference/graph/predictions/test_predictions.csv \
  --ensemble-predictions ensemble/ensemble_predictions.csv \
  --shap-reports explain/lightgbm/reports \
  --graph-explain-dir explain/graph/explanations \
  --output ./llm_explain

如果你想喂更多信息：
--top-static-positive 30 \
--top-static-negative 15 \
--top-global-static 30 \
--top-graph-nodes 40 \
--top-graph-edges 30
"""