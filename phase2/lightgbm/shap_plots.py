"""Readable SHAP figures from LightGBM's native feature contributions."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import sparse


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    return plt, MaxNLocator


def _label(name: str, limit: int = 66) -> str:
    name = str(name).replace("\n", " ").strip()
    if len(name) <= limit:
        return name
    head = (limit - 3) // 2
    return f"{name[:head]}...{name[-(limit - 3 - head):]}"


def _save(figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    return path


def plot_local_waterfall(
    contributions: np.ndarray,
    feature_names: Sequence[str],
    base_value: float,
    probability: float,
    sample_id: str,
    path: Path,
    top_k: int = 15,
    dpi: int = 220,
    *,
    y_true: Optional[int] = None,
    y_pred: Optional[int] = None,
) -> Path:
    """Plot the largest local contributions and aggregate the remainder."""
    plt, MaxNLocator = _plotting()
    contributions = np.asarray(contributions, dtype=float).reshape(-1)
    selected = [int(i) for i in np.argsort(np.abs(contributions))[::-1] if contributions[i] != 0][:top_k]
    rows = [(str(feature_names[i]), float(contributions[i])) for i in selected]
    other = float(contributions.sum() - sum(value for _, value in rows))
    if not np.isclose(other, 0.0, atol=1e-12):
        rows.append((f"Other {len(contributions) - len(selected)} features", other))
    if not rows:
        rows = [("No nonzero contributions", 0.0)]

    height = max(6.5, 0.52 * len(rows) + 3.2)
    fig, (ax, values_ax) = plt.subplots(
        1, 2, figsize=(17, height), sharey=True,
        gridspec_kw={"width_ratios": [4.3, 1.0]},
    )
    fig.subplots_adjust(left=0.38, right=0.96, top=0.84, bottom=0.12, wspace=0.05)

    starts = float(base_value) + np.r_[0.0, np.cumsum([value for _, value in rows[:-1]])]
    ends = starts + np.asarray([value for _, value in rows])
    y = np.arange(len(rows))
    for position, ((_, contribution), start, end) in enumerate(zip(rows, starts, ends)):
        ax.barh(
            position, abs(contribution), left=min(start, end), height=0.62,
            color="#c53b3b" if contribution >= 0 else "#2864ad",
            edgecolor="none",
        )
        values_ax.text(
            0.10, position, f"{contribution:+.4g}",
            va="center", ha="left", fontsize=20,
            color="#a22f2f" if contribution >= 0 else "#20528f",
        )

    final_value = float(base_value + contributions.sum())
    ax.axvline(base_value, color="#777777", linewidth=1.2, linestyle="--")
    ax.axvline(final_value, color="#202020", linewidth=1.2)
    lower = min(float(np.min(starts)), float(np.min(ends)), float(base_value), final_value)
    upper = max(float(np.max(starts)), float(np.max(ends)), float(base_value), final_value)
    spread = max(upper - lower, 0.25)
    ax.set_xlim(lower - 0.08 * spread, upper + 0.08 * spread)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_yticks(y)
    ax.set_yticklabels([_label(name) for name, _ in rows], fontsize=20)
    ax.invert_yaxis()
    ax.set_xlabel("Model output for malware class (raw score)", fontsize=20)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    values_ax.set_xlim(0, 1)
    values_ax.set_xticks([])
    values_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in values_ax.spines.values():
        spine.set_visible(False)
    values_ax.text(0.10, 1.015, "SHAP value", transform=values_ax.transAxes, fontsize=20, fontweight="bold")

    true_label = "unknown" if y_true is None else str(y_true)
    pred_label = "unknown" if y_pred is None else str(y_pred)
    fig.suptitle(
        f"SHAP Waterfall: {_label(sample_id, 72)} | P(malware) = {probability:.5f}, "
        f"y_true = {true_label}, y_pred = {pred_label}",
        y=0.895, fontsize=20,
    )
    result = _save(fig, path, dpi)
    plt.close(fig)
    return result


def plot_global_bar(
    mean_abs: np.ndarray,
    feature_names: Sequence[str],
    n_samples: int,
    path: Path,
    top_k: int = 20,
    dpi: int = 220,
) -> Path:
    """Plot global mean absolute SHAP values with a separate label margin."""
    plt, MaxNLocator = _plotting()
    mean_abs = np.asarray(mean_abs, dtype=float).reshape(-1)
    order = np.argsort(mean_abs)[::-1][:top_k][::-1]
    y = np.arange(len(order))
    scores = mean_abs[order]
    fig, ax = plt.subplots(figsize=(16, max(6.5, 0.52 * len(order) + 2.8)))
    fig.subplots_adjust(left=0.40, right=0.94, top=0.88, bottom=0.13)
    ax.barh(y, scores, height=0.63, color="#c53b3b", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([_label(feature_names[int(i)]) for i in order], fontsize=20)
    maximum = float(np.max(scores)) if len(scores) else 0.0
    scale = maximum if maximum > 0 else 1.0
    ax.set_xlim(0, scale * 1.22)
    for position, score in zip(y, scores):
        ax.text(float(score) + scale * 0.015, position, f"{score:.3g}", va="center", fontsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Mean |SHAP value| (malware raw score)", fontsize=20)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.suptitle(f"Global SHAP feature importance (n = {n_samples})", y=0.97, fontsize=20)
    # fig.text(0.40, 0.045, "Full feature names and values are in the CSV.", fontsize=20, color="#555555")
    result = _save(fig, path, dpi)
    plt.close(fig)
    return result


def plot_global_summary(
    contributions: np.ndarray,
    features,
    feature_names: Sequence[str],
    mean_abs: np.ndarray,
    path: Path,
    top_k: int = 20,
    dpi: int = 220,
) -> Path:
    """Plot per-sample SHAP spread; color shows each feature's relative value."""
    plt, MaxNLocator = _plotting()
    import matplotlib as mpl

    contributions = np.asarray(contributions, dtype=float)
    order = np.argsort(mean_abs)[::-1][:top_k][::-1]
    feature_values = features[:, order]
    if sparse.issparse(feature_values):
        feature_values = feature_values.toarray()
    feature_values = np.asarray(feature_values, dtype=float)
    fig, ax = plt.subplots(figsize=(16, max(6.5, 0.56 * len(order) + 2.8)))
    fig.subplots_adjust(left=0.40, right=0.85, top=0.88, bottom=0.13)
    rng = np.random.default_rng(42)
    for position, index in enumerate(order):
        x = contributions[:, int(index)]
        values = feature_values[:, position]
        finite = values[np.isfinite(values)]
        if len(finite):
            low, high = np.percentile(finite, [5, 95])
            colors = np.clip((values - low) / (high - low), 0, 1) if high > low else np.full(len(values), 0.5)
        else:
            colors = np.full(len(values), 0.5)
        jitter = rng.uniform(-0.26, 0.26, size=len(x))
        ax.scatter(
            x, position + jitter, c=colors, cmap="coolwarm", vmin=0, vmax=1,
            s=10, alpha=0.55, linewidths=0, rasterized=True,
        )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_label(feature_names[int(i)]) for i in order], fontsize=20)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel("SHAP value (impact on malware raw score)", fontsize=20)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    colorbar_ax = fig.add_axes([0.89, 0.19, 0.018, 0.58])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="coolwarm"), cax=colorbar_ax)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Relative feature value", fontsize=20)
    fig.suptitle(f"Global SHAP summary (n = {len(contributions)})", y=0.97, fontsize=20)
    fig.text(0.40, 0.045, "Each dot is one APK. Full feature names and values are in the CSV.", fontsize=20, color="#555555")
    result = _save(fig, path, dpi)
    plt.close(fig)
    return result
