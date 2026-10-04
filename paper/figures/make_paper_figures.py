from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = OUT / "source_data"

BLUE = "#1967B3"
ORANGE = "#D45A2A"
GREEN = "#16856B"
PURPLE = "#7355A6"
GRAY = "#656B73"
LIGHT = "#E8EDF2"
DARK = "#1E252B"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def mm(x: float) -> float:
    return x / 25.4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def polish(ax: mpl.axes.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.7, length=3)
    if grid:
        ax.grid(axis=grid, color="#D9DEE3", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.18,
        f"({label})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=DARK,
    )


def save(fig: mpl.figure.Figure, stem: str, axes: list[mpl.axes.Axes] | None = None) -> None:
    if axes:
        qa = OUT / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig,
            axes=axes,
            json_out=qa / f"{stem}.alignment.json",
            overlay_svg=qa / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, title, body, color, fontsize=8.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="none",
        facecolor=mpl.colors.to_rgba(color, 0.07),
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=fontsize + 0.4, fontweight="bold", color=DARK)
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center",
            fontsize=fontsize, color=GRAY, linespacing=1.15)
    return patch


def arrow(ax, start, end, color=GRAY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=color, shrinkA=5, shrinkB=10))


def figure1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(mm(180), mm(110)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.02, 0.38), (0.12, 0.23), "EEG window", "C × T\n250 Hz", BLUE, fontsize=8.0)
    add_box(ax, (0.18, 0.38), (0.15, 0.23), "Foldable local\nmixer", "parallel local paths\nfolded at inference\nkernels 15, 7", BLUE, fontsize=8.0)
    arrow(ax, (0.14, 0.495), (0.18, 0.495))

    add_box(ax, (0.37, 0.68), (0.18, 0.21), "Complex spectrum", "6–45 Hz; 0.25-Hz grid\nRe, Im, log |z|", GREEN, fontsize=8.0)
    add_box(ax, (0.37, 0.40), (0.18, 0.19), "Temporal\ncandidates", "4 time segments\nharmonic demodulation", PURPLE, fontsize=8.0)
    add_box(ax, (0.37, 0.10), (0.18, 0.19), "Spectral\ncandidates", "complex bins near\nharmonics (±2)", ORANGE, fontsize=8.0)
    arrow(ax, (0.33, 0.53), (0.38, 0.78))
    arrow(ax, (0.33, 0.49), (0.38, 0.485))
    arrow(ax, (0.33, 0.45), (0.38, 0.195))

    add_box(ax, (0.60, 0.67), (0.16, 0.22), "Reduced spectral\ntokens", "depthwise reduction\n+ position code", GREEN, fontsize=8.0)
    add_box(ax, (0.60, 0.27), (0.16, 0.21), "Fused candidate\ntokens", "temporal + spectral\nevidence", PURPLE, fontsize=8.0)
    arrow(ax, (0.55, 0.78), (0.61, 0.78))
    arrow(ax, (0.55, 0.49), (0.61, 0.40))
    arrow(ax, (0.55, 0.20), (0.61, 0.34))

    add_box(ax, (0.80, 0.52), (0.18, 0.25), "Late global\nfusion", "harmonic cross-attention\n→ candidate attention", BLUE, fontsize=8.0)
    arrow(ax, (0.76, 0.78), (0.81, 0.67))
    arrow(ax, (0.76, 0.37), (0.81, 0.59))
    add_box(ax, (0.82, 0.16), (0.14, 0.18), "Shared scorer", "one logit per\nstimulus candidate", BLUE, fontsize=8.0)
    arrow(ax, (0.89, 0.51), (0.89, 0.34))

    save(fig, "fig1_architecture")


def figure2_calibration_free() -> None:
    df = pd.read_csv(DATA / "fig2_calibration_free.csv")
    fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(72)), sharey=True)
    colors = {"HarmonicFoldNet": BLUE, "Spectral Transformer": GRAY}
    markers = {"HarmonicFoldNet": "o", "Spectral Transformer": "s"}
    for ax, dataset, label in zip(axes, ["Benchmark", "BETA"], ["a", "b"]):
        part = df[df.dataset == dataset]
        for method in colors:
            rows = part[part.method == method].sort_values("window_s")
            ax.plot(rows.window_s, rows.balanced_accuracy_pct, marker=markers[method],
                    markersize=4, color=colors[method], label=method)
            ax.fill_between(
                rows.window_s,
                rows.ci95_low_pct,
                rows.ci95_high_pct,
                color=colors[method],
                alpha=0.10,
                linewidth=0,
            )
        hfn = part[part.method == "HarmonicFoldNet"].sort_values("window_s")
        for _, row in hfn.iterrows():
            if pd.notna(row.holm_p) and row.holm_p < 0.05:
                direction = 1 if row.difference_hfn_minus_reference_pp > 0 else -1
                y = row.ci95_high_pct + 1.2 if direction > 0 else row.ci95_low_pct - 1.7
                ax.scatter(
                    [row.window_s], [y], marker="*", s=22, color=BLUE,
                    linewidths=0.5, zorder=6, clip_on=False,
                )
        ax.set_title(dataset)
        ax.set_xlabel("Observation window (s)")
        ax.set_xticks(sorted(part.window_s.unique()))
        ax.set_ylim(20, 86)
        polish(ax)
        panel(ax, label)
    axes[0].set_ylabel("Balanced accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.995, 0.01, "* Holm-adjusted p < 0.05", ha="right", va="bottom", fontsize=8, color=GRAY)
    fig.subplots_adjust(bottom=0.22, wspace=0.13)
    save(fig, "fig2_calibration_free", list(axes))


def figure3_harmonic_mechanism() -> None:
    df = pd.read_csv(DATA / "fig3_harmonic_mechanism.csv")
    fig, axes = plt.subplots(1, 3, figsize=(mm(180), mm(67)))
    for ax, record, title, label in [
        (axes[0], "harmonic_neighbourhood_mass", "Harmonic-neighbourhood mass", "a"),
        (axes[1], "peak_within_target", "Peak within target ±0.5 Hz", "b"),
    ]:
        part = df[df.record == record]
        windows = sorted(part.window_s.unique())
        x = np.arange(len(windows))
        width = 0.34
        for i, (condition, color) in enumerate([("bias", BLUE), ("masked", LIGHT if record else LIGHT)]):
            vals = [float(part[(part.window_s == w) & (part.condition == condition)].value_pct.iloc[0]) for w in windows]
            ax.bar(x + (i - 0.5) * width, vals, width, label=condition.capitalize(),
                   color=color, edgecolor=GRAY if condition == "masked" else color, linewidth=0.7)
        ax.set_xticks(x, [f"{w:.1f}" for w in windows])
        ax.set_xlabel("Window (s)")
        ax.set_ylabel("Attention mass (%)" if record == "harmonic_neighbourhood_mass" else "Trial–head observations (%)")
        ax.set_ylim(0, 108)
        ax.set_title(title)
        polish(ax)
        panel(ax, label)
    axes[0].legend(loc="upper right")

    ax = axes[2]
    part = df[df.record == "full_minus_no_bias_accuracy"].sort_values("window_s")
    colors = [GREEN if v > 0 else ORANGE for v in part.value_pct]
    yerr = np.vstack([
        part.value_pct.to_numpy() - part.ci95_low_pct.to_numpy(),
        part.ci95_high_pct.to_numpy() - part.value_pct.to_numpy(),
    ])
    ax.bar(
        part.window_s.astype(str),
        part.value_pct,
        yerr=yerr,
        error_kw={"elinewidth": 0.8, "ecolor": DARK, "capsize": 2.0},
        color=colors,
        width=0.68,
    )
    ax.axhline(0, color=DARK, linewidth=0.75)
    for i, row in enumerate(part.itertuples()):
        if pd.notna(row.holm_p) and row.holm_p < 0.05:
            ax.scatter(
                [i], [row.ci95_high_pct + 0.12], marker="*", s=22,
                color=DARK, linewidths=0.5, zorder=6,
            )
    ax.set_title("Accuracy contribution of harmonic bias", pad=12)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Full − no bias (percentage points)")
    ax.set_ylim(-1.9, 2.9)
    polish(ax)
    panel(ax, "c")
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig3_harmonic_mechanism", list(axes))


def figure4_adaptation() -> None:
    df = pd.read_csv(DATA / "fig4_adaptation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(72)))
    ax = axes[0]
    part = df[df.record == "calibration_curve"]
    colors = {0.4: ORANGE, 0.8: BLUE, 1.2: GREEN}
    for window in [0.4, 0.8, 1.2]:
        rows = part[part.window_s == window].copy()
        x = np.array([0, 1, 2])
        y = rows.balanced_accuracy_pct.to_numpy()
        yerr = np.vstack([
            y - rows.ci95_low_pct.to_numpy(),
            rows.ci95_high_pct.to_numpy() - y,
        ])
        ax.errorbar(
            x, y, yerr=yerr, marker="o", markersize=4, capsize=2,
            linewidth=1.4, elinewidth=0.8, color=colors[window], label=f"{window:.1f} s",
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Target-user calibration blocks")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("113-parameter adapter")
    ax.legend(title="Window")
    polish(ax)
    panel(ax, "a")

    ax = axes[1]
    part = df[df.record == "two_block_comparison"]
    methods = ["113-parameter adapter", "TRCA", "eTRCA", "TDCA"]
    windows = [0.4, 0.8, 1.2]
    x = np.arange(len(windows))
    width = 0.19
    palette = [BLUE, LIGHT, PURPLE, GREEN]
    for i, (method, color) in enumerate(zip(methods, palette)):
        rows = part[part.method_or_blocks == method].set_index("window_s")
        vals = [rows.loc[w, "balanced_accuracy_pct"] for w in windows]
        lo = [rows.loc[w, "ci95_low_pct"] for w in windows]
        hi = [rows.loc[w, "ci95_high_pct"] for w in windows]
        yerr = np.vstack([np.asarray(vals) - np.asarray(lo), np.asarray(hi) - np.asarray(vals)])
        ax.bar(
            x + (i - 1.5) * width, vals, width, yerr=yerr, label=method, color=color,
            edgecolor=GRAY if color == LIGHT else color, linewidth=0.6,
            error_kw={"elinewidth": 0.7, "ecolor": DARK, "capsize": 1.5},
        )
    ax.set_xticks(x, [f"{w:.1f}" for w in windows])
    ax.set_xlabel("Observation window (s)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Matched two-block calibration")
    ax.legend(loc="upper left", frameon=False)
    polish(ax)
    panel(ax, "b")
    fig.subplots_adjust(wspace=0.26)
    save(fig, "fig4_adaptation", list(axes))


def figure5_electrode_transfer() -> None:
    df = pd.read_csv(DATA / "fig5_electrode_transfer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(72)))
    colors = {"dry to dry": GRAY, "dry to wet": BLUE, "wet to dry": ORANGE, "wet to wet": GREEN}
    markers = {"dry to dry": "o", "dry to wet": "s", "wet to dry": "^", "wet to wet": "D"}
    ax = axes[0]
    for condition in colors:
        rows = df[df.train_test == condition].sort_values("window_s")
        y = rows.balanced_accuracy_pct.to_numpy()
        lo = rows.ci95_low_pct.to_numpy()
        hi = rows.ci95_high_pct.to_numpy()
        ax.fill_between(rows.window_s, lo, hi, color=colors[condition], alpha=0.10, linewidth=0)
        ax.plot(rows.window_s, y, marker=markers[condition], markersize=3.8,
                color=colors[condition], label=condition.replace(" to ", "→"))
    ax.set_xlabel("Observation window (s)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Directional electrode-condition transfer")
    ax.set_xticks(sorted(df.window_s.unique()))
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    polish(ax)
    panel(ax, "a")

    ax = axes[1]
    piv = df.pivot(index="window_s", columns="train_test", values="balanced_accuracy_pct")
    effects = pd.DataFrame(
        {
            "dry→wet minus wet→wet": piv["dry to wet"] - piv["wet to wet"],
            "wet→dry minus dry→dry": piv["wet to dry"] - piv["dry to dry"],
            "dry→wet minus wet→dry": piv["dry to wet"] - piv["wet to dry"],
        }
    )
    effect_colors = [BLUE, ORANGE, PURPLE]
    for col, color in zip(effects.columns, effect_colors):
        ax.plot(effects.index, effects[col], marker="o", markersize=3.5, color=color, label=col)
    ax.axhline(0, color=DARK, linewidth=0.75)
    ax.set_xlabel("Observation window (s)")
    ax.set_ylabel("Paired difference (percentage points)")
    ax.set_title("Asymmetric transfer penalties")
    ax.set_xticks(sorted(df.window_s.unique()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    polish(ax)
    panel(ax, "b")
    fig.subplots_adjust(wspace=0.27, bottom=0.28)
    save(fig, "fig5_electrode_transfer", list(axes))


def figure6_deployment_pareto() -> None:
    df = pd.read_csv(DATA / "fig6_deployment_pareto.csv")
    if (df["parameters"] <= 0).any():
        raise ValueError("Log-scaled parameter counts must be positive")
    fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(72)))
    palette = {
        "HarmonicFoldNet": BLUE,
        "Spectral Transformer": GRAY,
        "Filter-bank Transformer": ORANGE,
        "MTSNet reconstruction": PURPLE,
    }
    markers = {"HarmonicFoldNet": "o", "Spectral Transformer": "s", "Filter-bank Transformer": "^", "MTSNet reconstruction": "D"}

    ax = axes[0]
    for row in df.itertuples():
        ax.errorbar(
            row.parameters / 1e6,
            row.balanced_accuracy_pct_1_2s,
            yerr=[[row.balanced_accuracy_pct_1_2s - row.accuracy_ci95_low_pct],
                  [row.accuracy_ci95_high_pct - row.balanced_accuracy_pct_1_2s]],
            fmt=markers[row.method], markersize=6, capsize=2, elinewidth=0.8,
            color=palette[row.method], label=row.method, zorder=3,
        )
        label_offset = {
            "HarmonicFoldNet": (9, 7),
            "Spectral Transformer": (9, 7),
            "Filter-bank Transformer": (10, 8),
            "MTSNet reconstruction": (12, 10),
        }[row.method]
        ax.annotate(row.method.replace(" reconstruction", ""),
                    (row.parameters / 1e6, row.balanced_accuracy_pct_1_2s),
                    xytext=label_offset, textcoords="offset points", fontsize=8, color=DARK)
    ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("BETA balanced accuracy at 1.2 s (%)")
    ax.set_title("Accuracy–size trade-off")
    ax.set_ylim(58, 71)
    polish(ax)
    panel(ax, "a")

    ax = axes[1]
    for row in df.itertuples():
        size = 28 + 2.0 * row.cuda_peak_mib
        ax.errorbar(
            row.cpu_p50_ms,
            row.cuda_p50_ms,
            xerr=[[0.0], [row.cpu_p95_ms - row.cpu_p50_ms]],
            yerr=[[0.0], [row.cuda_p95_ms - row.cuda_p50_ms]],
            fmt="none", ecolor=mpl.colors.to_rgba(palette[row.method], 0.55),
            elinewidth=0.8, capsize=2, zorder=2,
        )
        ax.scatter(row.cpu_p50_ms, row.cuda_p50_ms, s=size,
                   marker=markers[row.method], color=palette[row.method], label=row.method,
                   edgecolor="white", linewidth=0.6, zorder=3)
        label_offset = {
            "HarmonicFoldNet": (10, 8),
            "Spectral Transformer": (9, 7),
            "Filter-bank Transformer": (12, -14),
            "MTSNet reconstruction": (13, 10),
        }[row.method]
        ax.annotate(row.method.replace(" reconstruction", ""),
                    (row.cpu_p50_ms, row.cuda_p50_ms), xytext=label_offset,
                    textcoords="offset points", fontsize=8, color=DARK)
    ax.set_xlabel("CPU P50 latency (ms, one thread)")
    ax.set_ylabel("CUDA P50 latency (ms)")
    ax.set_title("Measured latency; marker area scales with CUDA memory")
    polish(ax)
    panel(ax, "b")
    fig.subplots_adjust(wspace=0.29)
    save(fig, "fig6_deployment_pareto", list(axes))


def write_manifest() -> None:
    source_docs = [
        ROOT / "docs" / "FINAL_EVIDENCE_SUMMARY_v1.md",
        ROOT / "docs" / "DEPLOYMENT_PARETO_RESULTS_v1.md",
        ROOT / "docs" / "HARMONIC_ATTENTION_MECHANISM_RESULTS_v1.md",
        ROOT / "docs" / "SUBJECT_ADAPTATION_RESULTS_v2.md",
        ROOT / "docs" / "ELECTRODE_TRANSFER_RESULTS_v1.md",
    ]
    records = []
    for path in sorted(DATA.glob("*.csv")):
        records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)})
    manifest = {
        "generated_by": "paper/figures/make_paper_figures.py",
        "figure_outputs": [f"fig{i}_{name}" for i, name in [
            (1, "architecture"), (2, "calibration_free"), (3, "harmonic_mechanism"),
            (4, "adaptation"), (5, "electrode_transfer"), (6, "deployment_pareto")]],
        "source_tables": records,
        "frozen_result_documents": [
            {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p)} for p in source_docs
        ],
        "participant_source_data": {
            "path": "paper/source_data/submission_evidence.json",
            "sha256": sha256(ROOT / "paper" / "source_data" / "submission_evidence.json"),
        },
        "provenance": (
            "Figures 2 and 3 use source tables regenerated from preserved participant-level, "
            "three-seed run artifacts and validated against the frozen comparison records. "
            "Figures 4-6 retain the frozen result-document sources listed above."
        ),
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure1_architecture()
    figure2_calibration_free()
    figure3_harmonic_mechanism()
    figure4_adaptation()
    figure5_electrode_transfer()
    figure6_deployment_pareto()
    write_manifest()
    print("Generated Figures 1-6 as PDF, SVG, and 600-dpi PNG.")


if __name__ == "__main__":
    main()
