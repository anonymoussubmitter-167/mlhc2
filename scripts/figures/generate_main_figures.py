"""Generate 2 new main-text figures for the ATLAS paper.

fig23: ML Shifts the Bias Frontier (dumbbell chart)
fig24: Intersectional Compounding Heatmap (race × age, 2×2 grid)

Style matched to improve_figures.py:
- Colorblind-safe Wong 2011 palette
- Sans-serif fonts (Helvetica/Arial)
- Clean annotations, consistent sizing
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
DATA = _ROOT / "experiments" / "exp_gossis"
OUT  = _ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colorblind-safe palette (Wong 2011) — matches improve_figures.py ──
CB = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermilion": "#D55E00",
    "sky":       "#56B4E9",
    "yellow":    "#F0E442",
    "purple":    "#CC79A7",
    "black":     "#000000",
}

SCORE_COLORS = {
    "sofa":    CB["blue"],
    "qsofa":   CB["orange"],
    "apache2": CB["green"],
    "news2":   CB["vermilion"],
}
SCORE_LABELS = {"sofa": "SOFA", "qsofa": "qSOFA",
                "apache2": "APACHE-II", "news2": "NEWS2"}
AXIS_LABELS = {"race_cat": "Race", "sex": "Sex",
               "age_group": "Age", "diag_type": "Diagnosis"}
METRIC_LABELS = {"eod": "Equalized Odds", "ppg": "Predictive Parity",
                 "cal_gap": "Calibration Gap", "auroc_gap": "AUROC Gap"}

AGE_GROUPS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

# ── Global style — matches improve_figures.py exactly ──
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size":            11,
    "axes.labelsize":       12,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.titlepad":        9,
    "axes.labelweight":     "regular",
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "legend.title_fontsize": 10,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#CCCCCC",
    "legend.borderpad":     0.5,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.1,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "axes.edgecolor":       "#444444",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.3,
    "grid.color":           "#AAAAAA",
    "lines.linewidth":      2.0,
    "lines.markersize":     6,
    "patch.linewidth":      0.5,
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
})


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  Saved: {name}")


# ════════════════════════════════════════════════════════════════════════
# Figure 23: ML Shifts the Bias Frontier (Dumbbell Chart)
# ════════════════════════════════════════════════════════════════════════

def plot_dumbbell():
    """Dumbbell chart: score gap vs ML gap per axis×metric."""
    df = pd.read_csv(DATA / "e5_ml_improvement_full.csv")

    metrics = ["eod", "cal_gap", "auroc_gap", "ppg"]
    axes_order = ["age_group", "race_cat", "sex", "diag_type"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axs_flat = axs.flatten()

    for col_idx, metric in enumerate(metrics):
        ax = axs_flat[col_idx]
        mdf = df[df["metric"] == metric].copy()

        y_labels = []
        y_pos = []
        pos = 0

        for axis in axes_order:
            adf = mdf[mdf["axis"] == axis].sort_values("score")
            if adf.empty:
                continue

            for _, row in adf.iterrows():
                sg = row["score_fairness_gap"]
                mg = row["ml_fairness_gap"]
                improved = mg < sg
                color = CB["green"] if improved else CB["vermilion"]

                # Connector line
                ax.plot([sg, mg], [pos, pos], color=color, linewidth=2.0,
                        alpha=0.8, zorder=1, solid_capstyle="round")
                # Score gap (gray circle)
                ax.scatter(sg, pos, color="#666666", s=45, zorder=2,
                           marker="o", edgecolors="white", linewidths=0.6)
                # ML gap (colored diamond)
                ax.scatter(mg, pos, color=color, s=55, zorder=3,
                           marker="D", edgecolors="white", linewidths=0.6)

                score_label = SCORE_LABELS.get(row["score"], row["score"])
                y_labels.append(f"{AXIS_LABELS[axis]}: {score_label}")
                y_pos.append(pos)
                pos += 1

            # Axis separator
            if axis != axes_order[-1]:
                ax.axhline(pos - 0.5, color="#DDDDDD", linewidth=0.5,
                           linestyle="-", zorder=0)
            pos += 0.7

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8.5)
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xlabel("Fairness Gap", fontsize=10)
        ax.axvline(0, color="#AAAAAA", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)

    # Shared legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
                   markersize=7, label="Classical score gap"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CB["green"],
                   markersize=7, label="GRU gap (improved)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CB["vermilion"],
                   markersize=7, label="GRU gap (worsened)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.01),
               frameon=True, edgecolor="#CCCCCC")

    fig.suptitle("ML Shifts Rather Than Eliminates Fairness Disparities",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    _save(fig, "fig23_ml_shifts_dumbbell")


# ════════════════════════════════════════════════════════════════════════
# Figure 24: Intersectional Compounding Heatmap
# ════════════════════════════════════════════════════════════════════════

def plot_intersectional_heatmap():
    """Race × Age heatmap of AUROC for each score."""
    df = pd.read_csv(DATA / "e2_intersectional.csv")

    # Filter to race × age intersections
    race_age = df[
        ((df["axis1"] == "race_cat") & (df["axis2"] == "age_group")) |
        ((df["axis1"] == "age_group") & (df["axis2"] == "race_cat"))
    ].copy()

    race_age["race"] = np.where(
        race_age["axis1"] == "race_cat", race_age["group1"], race_age["group2"]
    )
    race_age["age"] = np.where(
        race_age["axis1"] == "age_group", race_age["group1"], race_age["group2"]
    )

    scores = ["sofa", "qsofa", "apache2", "news2"]
    race_order = ["White", "Black", "Hispanic", "Asian", "Other"]
    age_order = AGE_GROUPS

    # Use a diverging colormap centered near the median AUROC
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    axs_flat = axs.flatten()

    for idx, score in enumerate(scores):
        ax = axs_flat[idx]
        sdf = race_age[race_age["score"] == score]

        pivot = sdf.pivot_table(index="race", columns="age",
                                values="auroc", aggfunc="first")
        pivot_n = sdf.pivot_table(index="race", columns="age",
                                  values="n", aggfunc="first")

        pivot = pivot.reindex(index=[r for r in race_order if r in pivot.index],
                              columns=[a for a in age_order if a in pivot.columns])
        pivot_n = pivot_n.reindex(index=pivot.index, columns=pivot.columns)

        # Heatmap via seaborn for better aesthetics
        sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=0.60, vmax=0.92,
                    annot=False, linewidths=0.8, linecolor="white",
                    cbar=idx == 1,  # Only show cbar on top-right
                    cbar_kws={"label": "AUROC", "shrink": 0.8} if idx == 1 else {})

        # Annotate with AUROC value and n
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                n = pivot_n.values[i, j]
                if np.isnan(val):
                    ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                            fontsize=7.5, color="#999999")
                else:
                    color = "white" if val < 0.70 else "#222222"
                    n_str = f"{int(n):,}" if not np.isnan(n) else "?"
                    ax.text(j + 0.5, i + 0.35, f"{val:.3f}",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color=color)
                    ax.text(j + 0.5, i + 0.65, f"n={n_str}",
                            ha="center", va="center", fontsize=6.5,
                            color=color, alpha=0.7)

        ax.set_title(SCORE_LABELS[score], fontsize=13, fontweight="bold")
        ax.set_xlabel("Age Group" if idx >= 2 else "", fontsize=11)
        ax.set_ylabel("Race / Ethnicity" if idx % 2 == 0 else "", fontsize=11)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Intersectional AUROC: Race × Age Compounding Across Scores",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, "fig24_intersectional_heatmap")


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating main-text figures 23-24...")
    plot_dumbbell()
    plot_intersectional_heatmap()
    print("Done!")
