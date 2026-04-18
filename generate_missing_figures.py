"""Generate three missing figures: fig23, fig24, fig25."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUTDIR = "/home/bcheng/ATLAS/paper/figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "reddish": "#CC79A7",
    "black": "#000000",
}

SCORE_LABELS = {
    "sofa": "SOFA",
    "qsofa": "qSOFA",
    "apache2": "APACHE-II",
    "news2": "NEWS2",
}
AXIS_LABELS = {
    "race_cat": "Race",
    "sex": "Sex",
    "age_group": "Age Group",
    "diag_type": "Diagnosis Type",
}
METRIC_LABELS = {
    "eod": "EOD",
    "ppg": "PPG",
    "auroc_gap": "AUROC Gap",
    "cal_gap": "Calibration Gap",
    "ece_gap": "ECE Gap",
}


# ---------------------------------------------------------------------------
# FIG 24: Intersectional AUROC heatmap (race × age group, all 4 scores)
# ---------------------------------------------------------------------------
def fig24():
    df = pd.read_csv("/home/bcheng/ATLAS/experiments/exp_gossis/e2_intersectional.csv")
    sub = df[(df["axis1"] == "race_cat") & (df["axis2"] == "age_group")].copy()

    race_order = ["White", "Black", "Hispanic", "Asian", "Other"]
    age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    scores = ["sofa", "qsofa", "apache2", "news2"]

    # At 10" figure displayed at 6" (scale 0.60), desired PDF sizes:
    # cell text ~8pt → code 13; axis ticks ~9pt → 15; title ~10pt → 17
    CELL_FS   = 13
    TICK_FS   = 15
    TITLE_FS  = 17
    CBAR_FS   = 14
    SUPT_FS   = 17

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.42, wspace=0.38, left=0.10, right=0.95,
                        top=0.90, bottom=0.12)

    for idx, score in enumerate(scores):
        ax = axes[idx]
        pivot = (
            sub[sub["score"] == score]
            .pivot_table(index="group1", columns="group2", values="auroc", aggfunc="mean")
        )
        # Reorder
        pivot = pivot.reindex(index=[r for r in race_order if r in pivot.index],
                              columns=[a for a in age_order if a in pivot.columns])

        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.65, vmax=0.90, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=TICK_FS)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=TICK_FS)
        ax.set_title(SCORE_LABELS[score], fontweight="bold", fontsize=TITLE_FS)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Annotate cells
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    text_color = "white" if val < 0.72 or val > 0.87 else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=CELL_FS, color=text_color, fontweight="bold")

        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("AUROC", fontsize=CBAR_FS)
        cb.ax.tick_params(labelsize=CBAR_FS - 1)

    fig.suptitle("Intersectional AUROC: Race \u00d7 Age Group",
                 fontsize=SUPT_FS, fontweight="bold", y=0.98)

    out = os.path.join(OUTDIR, "fig24_intersectional_heatmap")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}.pdf")


# ---------------------------------------------------------------------------
# FIG 25: RSB gap by axis — showing age dominance across all scores & metrics
# ---------------------------------------------------------------------------
def fig25():
    df = pd.read_csv("/home/bcheng/ATLAS/experiments/exp_gossis/e4_rsb_full.csv")

    axes_order = ["age_group", "race_cat", "sex", "diag_type"]
    axis_colors = {
        "age_group": CB["vermilion"],
        "race_cat": CB["blue"],
        "sex": CB["green"],
        "diag_type": CB["orange"],
    }
    scores = ["sofa", "qsofa", "apache2", "news2"]
    metrics = ["eod", "ppg", "auroc_gap", "cal_gap"]

    # Compute mean RSB gap per axis × score, averaged over metrics
    summary = (
        df[df["metric"].isin(metrics)]
        .groupby(["score", "axis"])["rsb_gap"]
        .mean()
        .reset_index()
    )

    fig, axes_plots = plt.subplots(1, 4, figsize=(13, 4), sharey=False)

    for idx, score in enumerate(scores):
        ax = axes_plots[idx]
        sub = summary[summary["score"] == score]
        sub = sub.set_index("axis").reindex(axes_order).reset_index()

        colors = [axis_colors[a] for a in sub["axis"]]
        bars = ax.bar(
            [AXIS_LABELS[a] for a in sub["axis"]],
            sub["rsb_gap"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            width=0.6,
        )

        # Annotate age bar (always first)
        age_val = sub[sub["axis"] == "age_group"]["rsb_gap"].values
        if len(age_val):
            ax.text(0, age_val[0] + 0.002, f"{age_val[0]:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=CB["vermilion"])

        ax.set_title(SCORE_LABELS[score], fontweight="bold")
        ax.set_ylabel("Mean RSB Gap" if idx == 0 else "")
        ax.set_xticklabels([AXIS_LABELS[a] for a in sub["axis"]],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, sub["rsb_gap"].max() * 1.3)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=axis_colors[a], label=AXIS_LABELS[a])
        for a in axes_order
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)

    fig.suptitle("Relative Shortfall Burden (RSB) by Fairness Axis\n"
                 "(averaged over EOD, PPG, AUROC Gap, Calibration Gap)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTDIR, "fig25_rsb_by_axis")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}.pdf")


# ---------------------------------------------------------------------------
# FIG 23: ML shifts dumbbell — before/after fairness gaps with GRU-FAFT
# ---------------------------------------------------------------------------
def fig23():
    df = pd.read_csv("/home/bcheng/ATLAS/experiments/exp_gossis/e5_ml_improvement_full.csv")

    # Focus on SOFA (main score) and key axes / metrics
    focus_axes = ["age_group", "race_cat", "sex"]
    focus_metrics = ["eod", "ppg", "auroc_gap", "cal_gap"]
    sub = df[(df["score"] == "sofa") &
             (df["axis"].isin(focus_axes)) &
             (df["metric"].isin(focus_metrics))].copy()

    sub["label"] = sub["axis"].map(AXIS_LABELS) + " / " + sub["metric"].map(METRIC_LABELS)
    sub = sub.sort_values(["axis", "metric"])

    fig, ax = plt.subplots(figsize=(8, 6))

    y_positions = range(len(sub))
    y_labels = sub["label"].tolist()

    for i, (_, row) in enumerate(sub.iterrows()):
        before = row["score_fairness_gap"]
        after = row["ml_fairness_gap"]
        improved = after < before

        # Draw connecting line
        ax.plot([before, after], [i, i],
                color=CB["green"] if improved else CB["vermilion"],
                linewidth=1.5, zorder=1)

        # Draw dots
        ax.scatter(before, i, color=CB["blue"], s=55, zorder=2, label="SOFA" if i == 0 else "")
        ax.scatter(after, i, color=CB["orange"], s=55, zorder=2,
                   marker="D", label="GRU-FAFT" if i == 0 else "")

        # Arrow direction text
        delta = before - after
        if abs(delta) > 0.005:
            sign = "▼" if improved else "▲"
            color = CB["green"] if improved else CB["vermilion"]
            ax.text(max(before, after) + 0.003, i, f"{sign}{abs(delta):.3f}",
                    va="center", fontsize=6.5, color=color)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Fairness Gap (lower = fairer)", fontsize=10)
    ax.set_title("Fairness Shifts: SOFA → GRU-FAFT\n"
                 "(dumbbell = before/after per axis & metric)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.grid(False, axis="y")

    # Annotation box
    improved_count = (sub["ml_fairness_gap"] < sub["score_fairness_gap"]).sum()
    total = len(sub)
    ax.text(0.98, 0.02,
            f"{improved_count}/{total} metrics improved\nAge EOD worsens (−232%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CB["blue"],
               markersize=8, label="SOFA score"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=CB["orange"],
               markersize=8, label="GRU-FAFT"),
        Line2D([0], [0], color=CB["green"], linewidth=2, label="Improved"),
        Line2D([0], [0], color=CB["vermilion"], linewidth=2, label="Worsened"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    out = os.path.join(OUTDIR, "fig23_ml_shifts_dumbbell")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}.pdf")


if __name__ == "__main__":
    print("Generating fig24 (intersectional heatmap)...")
    fig24()
    print("Generating fig25 (RSB by axis)...")
    fig25()
    print("Generating fig23 (ML shifts dumbbell)...")
    fig23()
    print("All done.")
