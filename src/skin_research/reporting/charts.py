"""Matplotlib figures for the research report (static PNGs).

Visual conventions (validated palette, see dataviz guidance):
* condition groups keep one fixed colour each; colour is never the only cue
  (every bar is labelled on the axis and the report carries the full tables);
* magnitude uses a single-hue blue ramp; severities use ordinal blue steps;
* thin marks, recessive grid, text in ink colours only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed (WSL / servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from skin_research.data.loader import load_image  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
GROUP_COLORS = {"original": NEUTRAL, "degradation": "#eb6834", "enhancement": "#2a78d6",
                "degrade_enhance": "#1baf7a"}
GROUP_NAMES = {"original": "Original", "degradation": "Degraded", "enhancement": "Enhanced original",
               "degrade_enhance": "Degraded → enhanced"}
SEVERITY_COLORS = {"original": NEUTRAL, "mild": "#86b6ef", "moderate": "#3987e5", "severe": "#184f95"}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "font.size": 9, "text.color": INK,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _group_legend(ax: plt.Axes, groups: list[str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in groups]
    ax.legend(handles, [GROUP_NAMES[g] for g in groups], loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=len(groups))


def condition_overview(summary: pd.DataFrame, metric: str, path: Path) -> Path:
    """Horizontal bars of ``metric`` with 95% CI for every condition, baseline marked."""
    _style()
    df = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(df) + 1.4))
    y = np.arange(len(df))
    colors = [GROUP_COLORS.get(g, NEUTRAL) for g in df["group"]]
    ax.barh(y, df[metric], color=colors, height=0.72, edgecolor=SURFACE, linewidth=1)
    lo, hi = df.get(f"{metric}_ci_low"), df.get(f"{metric}_ci_high")
    if lo is not None:
        ax.errorbar(df[metric], y, xerr=[df[metric] - lo, hi - df[metric]], fmt="none", ecolor=INK_2,
                    elinewidth=0.8, capsize=0)
    base = float(summary.loc[summary["condition"] == "original", metric].iloc[0])
    ax.axvline(base, color=INK, linewidth=1, linestyle="--")
    ax.text(base, len(df) - 0.3, f" original {base:.3f}", color=INK, fontsize=8, va="bottom")
    ax.set_yticks(y, df["condition"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{metric.replace('_', ' ')} (bars: point estimate, lines: 95% lesion-bootstrap CI)")
    ax.set_title(f"{metric.replace('_', ' ').capitalize()} by experimental condition")
    ax.grid(axis="y", visible=False)
    _group_legend(ax, [g for g in GROUP_COLORS if g in set(df["group"])])
    return _save(fig, path)


def severity_small_multiples(summary: pd.DataFrame, metric: str, path: Path) -> Path:
    """One panel per degradation: metric at original → mild → moderate → severe, with CI band."""
    _style()
    deg = summary[summary["group"] == "degradation"]
    names = list(dict.fromkeys(deg["degradation"]))
    base = summary[summary["condition"] == "original"].iloc[0]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        sub = deg[deg["degradation"] == name]
        order = [s for s in ("mild", "moderate", "severe") if s in set(sub["severity"])]
        sub = sub.set_index("severity").loc[order]
        xs = ["original", *order]
        vals = [base[metric], *sub[metric]]
        lo = [base[f"{metric}_ci_low"], *sub[f"{metric}_ci_low"]]
        hi = [base[f"{metric}_ci_high"], *sub[f"{metric}_ci_high"]]
        ax.fill_between(range(len(xs)), lo, hi, color="#cde2fb", linewidth=0)
        ax.plot(range(len(xs)), vals, color="#2a78d6", linewidth=2)
        ax.scatter(range(len(xs)), vals, s=36, zorder=3, color=[SEVERITY_COLORS[x] for x in xs],
                   edgecolors=SURFACE, linewidths=2)
        ax.annotate(f"{vals[-1]:.2f}", (len(xs) - 1, vals[-1]), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color=INK)
        ax.set_xticks(range(len(xs)), ["orig", *order])
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.supylabel(metric.replace("_", " "), color=INK_2, fontsize=9)
    fig.suptitle(f"Degradation severity vs {metric.replace('_', ' ')} (band: 95% CI)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def enhancement_comparison(summary: pd.DataFrame, restoration: pd.DataFrame, metric: str, path: Path) -> Path:
    """Left: enhancement of original images. Right: degraded vs degraded→enhanced."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 2]})
    base = float(summary.loc[summary["condition"] == "original", metric].iloc[0])

    enh = summary[summary["group"].isin(["original", "enhancement"])]
    labels = [c.replace("enhance__", "") for c in enh["condition"]]
    ax1.bar(range(len(enh)), enh[metric], color=[GROUP_COLORS[g] for g in enh["group"]], width=0.7,
            yerr=[enh[metric] - enh[f"{metric}_ci_low"], enh[f"{metric}_ci_high"] - enh[metric]],
            error_kw={"elinewidth": 0.8, "ecolor": INK_2}, edgecolor=SURFACE)
    ax1.set_xticks(range(len(enh)), labels, rotation=35, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(metric.replace("_", " "))
    ax1.set_title("Enhancing good-quality originals")

    if len(restoration):
        x = np.arange(len(restoration))
        w = 0.38
        ax2.bar(x - w / 2, restoration["balacc_degraded"], w, color=GROUP_COLORS["degradation"],
                label="Degraded", edgecolor=SURFACE)
        ax2.bar(x + w / 2, restoration["balacc_enhanced"], w, color=GROUP_COLORS["degrade_enhance"],
                label="Degraded → enhanced", edgecolor=SURFACE)
        ax2.axhline(base, color=INK, linestyle="--", linewidth=1)
        ax2.text(-0.45, base + 0.01, f"original {base:.3f}", ha="left", va="bottom", fontsize=8)
        names = [c.replace("__moderate", " (mod)").replace("+", "\n+ ") for c in restoration["condition"]]
        ax2.set_xticks(x, names, fontsize=7)
        ax2.grid(axis="x", visible=False)
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
        ax2.set_title("Can enhancement recover degraded images? (balanced accuracy)")
    fig.tight_layout()
    return _save(fig, path)


def classwise_heatmap(per_class: pd.DataFrame, value: str, conditions: list[str], labels: list[str],
                      support: dict[str, int], path: Path) -> Path:
    """Condition × class heatmap (e.g. recall/sensitivity) with cell values."""
    _style()
    table = per_class.pivot(index="condition", columns="label", values=value).reindex(conditions)[labels]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(conditions) + 1.6))
    im = ax.imshow(table.to_numpy(dtype=float), cmap=BLUE_RAMP, vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if v > 0.6 else INK)
    ax.set_xticks(range(len(labels)), [f"{lbl}\n(n={support.get(lbl, 0)})" for lbl in labels])
    ax.set_yticks(range(len(conditions)), conditions, fontsize=7)
    ax.grid(False)
    ax.xaxis.tick_top()
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(value)
    ax.set_title(f"Per-class {value} by condition", pad=28)
    return _save(fig, path)


def confidence_comparison(predictions: pd.DataFrame, conditions: list[str], path: Path) -> Path:
    """Distribution of top-1 confidence per condition, split by correct vs wrong."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    pos = np.arange(len(conditions))
    for offset, correct, color, name in ((-0.18, True, "#2a78d6", "Correct"), (0.18, False, "#eb6834", "Wrong")):
        data = [predictions[(predictions["condition"] == c) & (predictions["correct"] == correct)]["confidence"]
                .to_numpy() for c in conditions]
        keep = [i for i, d in enumerate(data) if len(d)]
        bp = ax.boxplot([data[i] for i in keep], positions=pos[keep] + offset, widths=0.32, patch_artist=True,
                        showfliers=False, medianprops={"color": INK, "linewidth": 1.2})
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.85)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color=color, linewidth=1)
        ax.plot([], [], color=color, linewidth=6, label=name)
    ax.set_xticks(pos, conditions, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("top-1 predicted probability")
    ax.set_title("Prediction confidence: correct vs wrong predictions")
    ax.legend(loc="lower left")
    return _save(fig, path)


def reliability_diagram(details: dict, conditions: list[str], path: Path) -> Path:
    """Calibration: accuracy vs confidence per bin for selected conditions."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1, label="perfect calibration")
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]
    for color, cond in zip(palette, conditions[:3]):
        bins = details[cond]["reliability_bins"]
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=color, linewidth=2,
                marker="o", markersize=5, label=f"{cond} (ECE {details[cond]['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("observed accuracy in bin")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left", fontsize=7)
    return _save(fig, path)


def confusion_matrix_plot(cm: list[list[int]], labels: list[str], title: str, path: Path) -> Path:
    _style()
    arr = np.asarray(cm, dtype=float)
    norm = arr / np.clip(arr.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", fontsize=8,
                    color=SURFACE if norm[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(f"{title}\n(counts; shading = row-normalised recall)", fontsize=10)
    return _save(fig, path)


def example_grid(predictions: pd.DataFrame, image_id: str, conditions: list[str], path: Path) -> Path | None:
    """One lesion shown under several conditions with the model's prediction."""
    rows = predictions[(predictions["image_id"] == image_id) & predictions["condition"].isin(conditions)]
    rows = rows.set_index("condition").reindex(conditions).dropna(subset=["output_path"])
    if rows.empty:
        return None
    _style()
    cols = 4
    n = len(rows)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(12, 2.9 * np.ceil(n / cols)), squeeze=False)
    for ax, (cond, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(load_image(Path(r["output_path"])))
        verdict = "✓" if r["correct"] else "✗"
        ax.set_title(f"{cond}\npred {r['predicted_label']} ({r['confidence']:.2f}) {verdict}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"{image_id} — ground truth: {rows.iloc[0]['label']}", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def quality_distribution(quality: pd.DataFrame, path: Path) -> Path:
    """Histograms of the main quality metrics for the original images."""
    _style()
    metrics = [("brightness", "brightness (L*, 0-1)"), ("rms_contrast", "RMS contrast"),
               ("laplacian_var", "Laplacian variance (log scale)"), ("noise_sigma", "noise σ estimate")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, (col, label) in zip(axes, metrics):
        data = quality[col].dropna()
        if col == "laplacian_var":
            data = np.log10(data.clip(lower=1e-3))
            label = "log10 Laplacian variance"
        ax.hist(data, bins=25, color="#3987e5", edgecolor=SURFACE, linewidth=0.8)
        ax.set_xlabel(label)
    axes[0].set_ylabel("images")
    fig.suptitle("Image quality of the original evaluation images", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
