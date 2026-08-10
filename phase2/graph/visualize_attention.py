import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

"""
explain-dir目录下要包含egde_attention和node_attention目录(即对应样本其对应的节点/边注意力权重)
# 默认用这个,不支持多样本，要一个个生成图
python3 -m phase2.graph.visualize_attention \
  --explain-dir ml_explain/graph/explanations \
  --top-k-nodes 30 \
  --neighbor-hops 1 \
  --max-display-nodes 60 \
  --max-display-edges 180 \
  --circular-order spread \
  --title "TASGATv2-org.ooma.oomaapp" \
  --sample-id org.ooma.oomaapp \
  --output-dir ml_explain

如果你想更像论文里“自然分散”的网络结构，也可以直接用力导向布局：
python -m phase2.graph.visualize_attention \
  --run-dir runs/hetero_gatv2 \
  --sample-id sample001 \
  --top-k-nodes 30 \
  --neighbor-hops 1 \
  --max-display-nodes 60 \
  --layout spring

产出ml_explain/com.thecybernanny.adroapp/attention_paper_figures/com.thecybernanny.png
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build publication-style TASGATv2 attention figures from graph explanation CSV files."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Graph run directory, for example runs/hetero_gatv2. Used when --explain-dir is not provided.",
    )
    parser.add_argument(
        "--explain-dir",
        default=None,
        help="Directory containing node_attention/ and edge_attention/. Overrides --run-dir.",
    )
    parser.add_argument("--sample-id", default=None, help="Sample id to visualize. If omitted, an informative sample is chosen.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <explain-dir>/attention_paper_figures.")
    parser.add_argument("--top-k-nodes", type=int, default=30, help="Number of most important nodes to keep before expansion.")
    parser.add_argument(
        "--neighbor-hops",
        type=int,
        default=1,
        help="Number of hops around the Top-K nodes to include. Use 0 for Top-K nodes only.",
    )
    parser.add_argument(
        "--max-display-nodes",
        type=int,
        default=60,
        help="Maximum nodes in the final figure after neighbor expansion.",
    )
    parser.add_argument("--max-display-edges", type=int, default=180, help="Maximum edges in the final figure.")
    parser.add_argument(
        "--score-column",
        choices=["auto", "attention", "normalized_attention"],
        default="auto",
        help="Node score used for colors and ranking. Auto prefers raw attention for clearer single-sample figures.",
    )
    parser.add_argument(
        "--layout",
        choices=["circular", "spring"],
        default="circular",
        help="Figure layout. Circular is usually clearer for paper-style overview figures.",
    )
    parser.add_argument(
        "--circular-order",
        choices=["spread", "rank", "node_id", "type"],
        default="spread",
        help=(
            "Node order for circular layout. 'spread' keeps rank labels but distributes high-attention "
            "nodes around the circle instead of placing rank 1,2,3,... consecutively."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["rank", "node_id", "none"],
        default="rank",
        help="Text shown inside nodes. The mapping table is always saved.",
    )
    parser.add_argument(
        "--node-types",
        nargs="*",
        default=None,
        help="Optional node type filter, for example: --node-types method api class.",
    )
    parser.add_argument(
        "--min-edge-attention",
        type=float,
        default=0.0,
        help="Drop edges with attention below this value before drawing.",
    )
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Draw TASGATv2 self-loop attention edges. Disabled by default because they usually clutter paper figures.",
    )
    parser.add_argument("--title", default=None, help="Optional short title shown under the graph, for example '(b) TASGATv2'.")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Output formats, for example: png pdf svg.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42, help="Seed used by spring layout.")
    parser.add_argument(
        "--node-font-size",
        type=float,
        default=16.0,
        help="Font size of node rank/id labels. Values above 12 may overflow small nodes.",
    )
    parser.add_argument(
        "--info-font-size",
        type=float,
        default=16.0,
        help="Font size of the prediction summary shown above the graph.",
    )
    return parser.parse_args()


def resolve_explain_dir(args: argparse.Namespace) -> Path:
    if args.explain_dir:
        explain_dir = Path(args.explain_dir)
    elif args.run_dir:
        explain_dir = Path(args.run_dir) / "explanations"
    else:
        raise SystemExit("Please provide --run-dir runs/hetero_gatv2 or --explain-dir runs/hetero_gatv2/explanations.")

    node_dir = explain_dir / "node_attention"
    edge_dir = explain_dir / "edge_attention"
    if not node_dir.exists() or not edge_dir.exists():
        raise SystemExit(
            f"{explain_dir} must contain node_attention/ and edge_attention/. "
            "Run phase2.graph.explain first."
        )
    return explain_dir


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def choose_sample(explain_dir: Path, sample_id: Optional[str]) -> str:
    node_dir = explain_dir / "node_attention"
    edge_dir = explain_dir / "edge_attention"
    if sample_id:
        node_path = node_dir / f"{sample_id}.csv"
        edge_path = edge_dir / f"{sample_id}.csv"
        if not node_path.exists():
            raise SystemExit(f"Node attention file not found: {node_path}")
        if not edge_path.exists():
            raise SystemExit(f"Edge attention file not found: {edge_path}")
        return sample_id

    candidates = []
    for node_path in sorted(node_dir.glob("*.csv")):
        rows = read_csv_rows(node_path)
        if not rows:
            continue
        first = rows[0]
        prob = to_float(first.get("prob_malware"))
        y_true = to_int(first.get("y_true"), -1)
        pred = to_int(first.get("pred"), -1)
        is_true_positive = int(y_true == 1 and pred == 1)
        is_malware = int(y_true == 1)
        candidates.append((is_true_positive, is_malware, prob, node_path.stem))

    if not candidates:
        raise SystemExit(f"No node attention CSV files found under {node_dir}")

    candidates.sort(reverse=True)
    return candidates[0][3]


def pick_score_column(rows: Sequence[Dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    if rows and "attention" in rows[0]:
        values = [to_float(row.get("attention")) for row in rows[: min(len(rows), 200)]]
        if any(value > 0 for value in values):
            return "attention"
    if rows and "normalized_attention" in rows[0]:
        values = [to_float(row.get("normalized_attention")) for row in rows[: min(len(rows), 200)]]
        if any(value > 0 for value in values):
            return "normalized_attention"
    return "attention"


def normalize_node_rows(
    rows: Sequence[Dict[str, str]],
    score_column: str,
    node_types: Optional[Set[str]],
) -> Dict[int, Dict[str, object]]:
    nodes: Dict[int, Dict[str, object]] = {}
    for row in rows:
        node_id = to_int(row.get("node"), -1)
        if node_id < 0:
            continue
        node_type = str(row.get("node_type") or "unknown")
        if node_types and node_type not in node_types:
            continue
        nodes[node_id] = {
            "node": node_id,
            "node_type": node_type,
            "local_index": to_int(row.get("local_index"), -1),
            "node_name": row.get("node_name") or f"node:{node_id}",
            "attention": to_float(row.get("attention")),
            "normalized_attention": to_float(row.get("normalized_attention")),
            "score": to_float(row.get(score_column)),
            "sample_id": row.get("sample_id") or "",
            "apk_name": row.get("apk_name") or "",
            "y_true": to_int(row.get("y_true"), -1),
            "pred": to_int(row.get("pred"), -1),
            "prob_malware": to_float(row.get("prob_malware")),
        }
    return nodes


def normalize_edge_rows(
    rows: Sequence[Dict[str, str]],
    min_attention: float,
    include_self_loops: bool,
) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    for row in rows:
        attention = to_float(row.get("attention"))
        if attention < min_attention:
            continue
        src = to_int(row.get("source_node"), -1)
        dst = to_int(row.get("target_node"), -1)
        if src < 0 or dst < 0:
            continue
        if src == dst and not include_self_loops:
            continue
        edges.append(
            {
                "source_node": src,
                "target_node": dst,
                "source_type": row.get("source_type") or "unknown",
                "target_type": row.get("target_type") or "unknown",
                "source_name": row.get("source_name") or f"node:{src}",
                "target_name": row.get("target_name") or f"node:{dst}",
                "attention": attention,
            }
        )
    return edges


def expand_top_nodes(
    nodes: Dict[int, Dict[str, object]],
    edges: Sequence[Dict[str, object]],
    top_k: int,
    neighbor_hops: int,
    max_display_nodes: int,
) -> Tuple[Set[int], Set[int]]:
    ranked_nodes = sorted(nodes, key=lambda node_id: float(nodes[node_id]["score"]), reverse=True)
    top_nodes = set(ranked_nodes[: max(top_k, 1)])
    display_nodes = set(top_nodes)
    frontier = set(top_nodes)

    adjacency: Dict[int, Set[int]] = {node_id: set() for node_id in nodes}
    for edge in edges:
        src = int(edge["source_node"])
        dst = int(edge["target_node"])
        if src in nodes and dst in nodes:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)

    for _ in range(max(neighbor_hops, 0)):
        next_frontier: Set[int] = set()
        for node_id in frontier:
            next_frontier.update(adjacency.get(node_id, set()))
        next_frontier.difference_update(display_nodes)
        display_nodes.update(next_frontier)
        frontier = next_frontier
        if len(display_nodes) >= max_display_nodes:
            break

    if len(display_nodes) > max_display_nodes:
        top_keep = set(ranked_nodes[: min(len(ranked_nodes), top_k)])
        remaining = [node_id for node_id in ranked_nodes if node_id in display_nodes and node_id not in top_keep]
        display_nodes = top_keep | set(remaining[: max(0, max_display_nodes - len(top_keep))])

    return display_nodes, top_nodes


def aggregate_edges(
    edges: Sequence[Dict[str, object]],
    display_nodes: Set[int],
    max_display_edges: int,
) -> List[Dict[str, object]]:
    pair_to_edge: Dict[Tuple[int, int], Dict[str, object]] = {}
    for edge in edges:
        src = int(edge["source_node"])
        dst = int(edge["target_node"])
        if src not in display_nodes or dst not in display_nodes:
            continue
        key = (src, dst)
        current = pair_to_edge.get(key)
        if current is None:
            pair_to_edge[key] = dict(edge, edge_count=1)
        else:
            current["edge_count"] = int(current.get("edge_count", 1)) + 1
            current["attention"] = max(float(current["attention"]), float(edge["attention"]))

    selected = sorted(pair_to_edge.values(), key=lambda item: float(item["attention"]), reverse=True)
    return selected[: max_display_edges]


def display_rank_map(nodes: Dict[int, Dict[str, object]], display_nodes: Iterable[int]) -> Dict[int, int]:
    ranked = sorted(display_nodes, key=lambda node_id: float(nodes[node_id]["score"]), reverse=True)
    return {node_id: rank + 1 for rank, node_id in enumerate(ranked)}


def circular_node_order(
    nodes: Dict[int, Dict[str, object]],
    display_nodes: Iterable[int],
    rank_map: Dict[int, int],
    circular_order: str,
) -> List[int]:
    if circular_order == "node_id":
        return sorted(display_nodes)
    if circular_order == "type":
        return sorted(
            display_nodes,
            key=lambda node_id: (str(nodes[node_id]["node_type"]), rank_map[node_id], node_id),
        )

    ranked = sorted(display_nodes, key=lambda node_id: rank_map[node_id])
    if circular_order == "rank" or len(ranked) <= 2:
        return ranked

    count = len(ranked)
    step = max(1, round(count * 0.382))
    while math.gcd(step, count) != 1:
        step += 1
        if step >= count:
            step = 1
            break

    spread: List[Optional[int]] = [None] * count
    position = 0
    for node_id in ranked:
        spread[position] = node_id
        position = (position + step) % count
    return [node_id for node_id in spread if node_id is not None]


def safe_filename(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def build_table_rows(
    nodes: Dict[int, Dict[str, object]],
    display_nodes: Set[int],
    top_nodes: Set[int],
    rank_map: Dict[int, int],
    degree: Dict[int, int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for node_id in sorted(display_nodes, key=lambda item: rank_map[item]):
        node = nodes[node_id]
        rows.append(
            {
                "display_id": rank_map[node_id],
                "node": node_id,
                "node_type": node["node_type"],
                "local_index": node["local_index"],
                "node_name": node["node_name"],
                "attention": node["attention"],
                "normalized_attention": node["normalized_attention"],
                "score": node["score"],
                "is_top_k": int(node_id in top_nodes),
                "degree_in_figure": degree.get(node_id, 0),
                "prob_malware": node["prob_malware"],
                "y_true": node["y_true"],
                "pred": node["pred"],
            }
        )
    return rows


def build_edge_table_rows(
    edges: Sequence[Dict[str, object]],
    nodes: Dict[int, Dict[str, object]],
    rank_map: Dict[int, int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for edge in sorted(edges, key=lambda item: float(item["attention"]), reverse=True):
        src = int(edge["source_node"])
        dst = int(edge["target_node"])
        rows.append(
            {
                "source_display_id": rank_map[src],
                "target_display_id": rank_map[dst],
                "source_node": src,
                "target_node": dst,
                "source_type": nodes[src]["node_type"],
                "target_type": nodes[dst]["node_type"],
                "source_name": nodes[src]["node_name"],
                "target_name": nodes[dst]["node_name"],
                "attention": edge["attention"],
                "edge_count": edge.get("edge_count", 1),
            }
        )
    return rows


def class_label(value: int) -> str:
    if value == 1:
        return "Malware"
    if value == 0:
        return "Benign"
    return "Unknown"


def plot_attention_graph(
    output_base: Path,
    formats: Sequence[str],
    nodes: Dict[int, Dict[str, object]],
    display_nodes: Set[int],
    edges: Sequence[Dict[str, object]],
    rank_map: Dict[int, int],
    layout: str,
    circular_order: str,
    label_mode: str,
    title: Optional[str],
    score_label: str,
    prob_malware: float,
    y_true: int,
    pred: int,
    node_font_size: float,
    info_font_size: float,
    dpi: int,
    seed: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise SystemExit(
            "Drawing attention figures requires matplotlib and networkx. "
            "Install them with: pip install matplotlib networkx"
        ) from exc

    graph = nx.DiGraph()
    ordered_nodes = sorted(display_nodes, key=lambda node_id: rank_map[node_id])
    layout_nodes = circular_node_order(nodes, display_nodes, rank_map, circular_order)
    for node_id in layout_nodes:
        graph.add_node(node_id)
    for edge in edges:
        graph.add_edge(int(edge["source_node"]), int(edge["target_node"]), attention=float(edge["attention"]))

    if layout == "spring":
        pos = nx.spring_layout(graph, seed=seed, k=1.2 / math.sqrt(max(graph.number_of_nodes(), 1)), iterations=200)
    else:
        pos = nx.circular_layout(graph)

    scores = [float(nodes[node_id]["score"]) for node_id in ordered_nodes]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    if math.isclose(max_score, min_score):
        min_score = 0.0
    sizes = [360.0 + 980.0 * (score / max(max_score, 1e-12)) for score in scores]

    edge_weights = [float(data.get("attention", 0.0)) for _, _, data in graph.edges(data=True)]
    max_edge = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.35 + 2.2 * (value / max(max_edge, 1e-12)) for value in edge_weights]
    edge_alphas = [0.18 + 0.42 * (value / max(max_edge, 1e-12)) for value in edge_weights]

    fig_width = 8.8 if graph.number_of_nodes() <= 45 else 10.5
    fig_height = 7.6 if graph.number_of_nodes() <= 45 else 9.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.set_axis_off()

    true_name = class_label(y_true)
    pred_name = class_label(pred)
    if y_true in {0, 1} and pred in {0, 1}:
        correctness = "Correct" if y_true == pred else "Incorrect"
    else:
        correctness = "Label unavailable"
    summary = (
        f"P(malware) = {prob_malware:.2%}   |   "
        f"True: {true_name} ({y_true})   |   "
        f"Predicted: {pred_name} ({pred})   |   {correctness}"
    )
    summary_edge = "#3A7D44" if y_true == pred and y_true in {0, 1} else "#B03A2E"
    fig.suptitle(
        summary,
        fontsize=info_font_size,
        fontweight="semibold",
        y=0.985,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#F7F7F7",
            "edgecolor": summary_edge,
            "linewidth": 1.2,
        },
    )

    for (src, dst, data), width, alpha in zip(graph.edges(data=True), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=[(src, dst)],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=8,
            width=width,
            edge_color="#444444",
            alpha=alpha,
            connectionstyle="arc3,rad=0.08",
        )

    node_collection = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=ordered_nodes,
        node_color=scores,
        cmap=plt.cm.Reds,
        vmin=min_score,
        vmax=max_score,
        node_size=sizes,
        linewidths=0.8,
        edgecolors="#333333",
    )

    if label_mode != "none":
        midpoint = min_score + (max_score - min_score) * 0.62
        for node_id in ordered_nodes:
            label = str(node_id) if label_mode == "node_id" else str(rank_map[node_id])
            score = float(nodes[node_id]["score"])
            font_color = "white" if score >= midpoint else "#111111"
            x_coord, y_coord = pos[node_id]
            ax.text(
                x_coord,
                y_coord,
                label,
                ha="center",
                va="center",
                fontsize=node_font_size,
                fontweight="semibold",
                color=font_color,
                zorder=10,
            )

    colorbar = fig.colorbar(node_collection, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(score_label, fontsize=16)
    colorbar.ax.tick_params(labelsize=10)

    if title:
        fig.text(0.5, 0.02, title, ha="center", va="bottom", fontsize=18, family="serif")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        clean_fmt = fmt.lower().lstrip(".")
        fig.savefig(output_base.with_suffix(f".{clean_fmt}"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    explain_dir = resolve_explain_dir(args)
    sample_id = choose_sample(explain_dir, args.sample_id)
    node_path = explain_dir / "node_attention" / f"{sample_id}.csv"
    edge_path = explain_dir / "edge_attention" / f"{sample_id}.csv"
    # output_dir = Path(args.output_dir) if args.output_dir else explain_dir / "attention_paper_figures"
    if not args.output_dir:
        print("please provide output dir")
        exit(1)
    else:
        output_dir = Path(args.output_dir) / f"{sample_id}" / "attention_paper_figures"
    node_rows = read_csv_rows(node_path)
    edge_rows = read_csv_rows(edge_path)
    if not node_rows:
        raise SystemExit(f"No node rows in {node_path}")
    if not edge_rows:
        raise SystemExit(f"No edge rows in {edge_path}")

    score_column = pick_score_column(node_rows, args.score_column)
    node_types = set(args.node_types) if args.node_types else None
    nodes = normalize_node_rows(node_rows, score_column, node_types)
    edges = normalize_edge_rows(edge_rows, args.min_edge_attention, args.include_self_loops)
    if not nodes:
        raise SystemExit("No nodes left after filtering. Relax --node-types or check the explanation CSV.")
    if not edges:
        print(
            "warning: no drawable non-self-loop edges remain. "
            "Use --include-self-loops or lower --min-edge-attention if you want to show self-loop/low-score edges.",
            file=sys.stderr,
        )
    if args.neighbor_hops > 0 and args.max_display_nodes <= args.top_k_nodes:
        print(
            "warning: --max-display-nodes is not larger than --top-k-nodes, "
            "so neighbor expansion has no room after truncation.",
            file=sys.stderr,
        )

    display_nodes, top_nodes = expand_top_nodes(
        nodes,
        edges,
        top_k=args.top_k_nodes,
        neighbor_hops=args.neighbor_hops,
        max_display_nodes=args.max_display_nodes,
    )
    edges = aggregate_edges(edges, display_nodes, args.max_display_edges)

    degree = {node_id: 0 for node_id in display_nodes}
    for edge in edges:
        degree[int(edge["source_node"])] += 1
        degree[int(edge["target_node"])] += 1

    rank_map = display_rank_map(nodes, display_nodes)
    table_rows = build_table_rows(nodes, display_nodes, top_nodes, rank_map, degree)
    edge_table_rows = build_edge_table_rows(edges, nodes, rank_map)

    title = args.title
    output_base = output_dir / safe_filename(sample_id)

    score_label = "TASGATv2 Node Attention"
    if score_column == "normalized_attention":
        score_label = "Normalized TASGATv2 Node Attention"

    sample_metadata = next(iter(nodes.values()))

    plot_attention_graph(
        output_base=output_base,
        formats=args.formats,
        nodes=nodes,
        display_nodes=display_nodes,
        edges=edges,
        rank_map=rank_map,
        layout=args.layout,
        label_mode=args.label_mode,
        circular_order=args.circular_order,
        title=title,
        score_label=score_label,
        prob_malware=float(sample_metadata["prob_malware"]),
        y_true=int(sample_metadata["y_true"]),
        pred=int(sample_metadata["pred"]),
        node_font_size=args.node_font_size,
        info_font_size=args.info_font_size,
        dpi=args.dpi,
        seed=args.seed,
    )

    # write_csv_rows(
    #     output_base.with_name(f"{output_base.name}_node_legend.csv"),
    #     table_rows,
    #     [
    #         "display_id",
    #         "node",
    #         "node_type",
    #         "local_index",
    #         "node_name",
    #         "attention",
    #         "normalized_attention",
    #         "score",
    #         "is_top_k",
    #         "degree_in_figure",
    #         "prob_malware",
    #         "y_true",
    #         "pred",
    #     ],
    # )
    # write_csv_rows(
    #     output_base.with_name(f"{output_base.name}_edge_legend.csv"),
    #     edge_table_rows,
    #     [
    #         "source_display_id",
    #         "target_display_id",
    #         "source_node",
    #         "target_node",
    #         "source_type",
    #         "target_type",
    #         "source_name",
    #         "target_name",
    #         "attention",
    #         "edge_count",
    #     ],
    # )
    # write_json(
    #     output_base.with_name(f"{output_base.name}_figure_config.json"),
    #     {
    #         "sample_id": sample_id,
    #         "node_attention_csv": str(node_path),
    #         "edge_attention_csv": str(edge_path),
    #         "score_column": score_column,
    #         "node_types": sorted(node_types) if node_types else "all",
    #         "top_k_nodes": args.top_k_nodes,
    #         "neighbor_hops": args.neighbor_hops,
    #         "max_display_nodes": args.max_display_nodes,
    #         "max_display_edges": args.max_display_edges,
    #         "min_edge_attention": args.min_edge_attention,
    #         "include_self_loops": args.include_self_loops,
    #         "layout": args.layout,
    #         "circular_order": args.circular_order,
    #         "label_mode": args.label_mode,
    #         "n_display_nodes": len(display_nodes),
    #         "n_display_edges": len(edges),
    #         "note": (
    #             "Node color is an attention-derived importance score. It explains model focus, "
    #             "not a formal causal proof that the node is malicious."
    #         ),
    #     },
    # )
    # 无用存储文件

    figure_paths = [str(output_base.with_suffix(f".{fmt.lower().lstrip('.')}")) for fmt in args.formats]
    print(f"sample_id: {sample_id}")
    print(f"figures: {', '.join(figure_paths)}")
    print(f"node legend: {output_base.with_name(f'{output_base.name}_node_legend.csv')}")
    print(f"edge legend: {output_base.with_name(f'{output_base.name}_edge_legend.csv')}")


if __name__ == "__main__":
    main()
