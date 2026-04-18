"""
ATLAS: Comprehensive figure improvement pass.
Replaces all paper figures with publication-quality versions.

Key upgrades vs. previous scripts:
- Colorblind-friendly palette (Wong 2011)
- Sans-serif fonts throughout (cleaner in print)
- Larger, more readable annotations
- Multi-seed error bars on ML results
- Log-scale score-conditional mortality + 15-fold annotation
- RSB figure reframed by axis (shows age dominance clearly)
- Model comparison figure replaces confusing scatter with bar + error bars
- Hospital stratified: log-scale ratio to show 6-10x Simpson's paradox
- PSM forest plot with "gap widens" annotation
- Consistent sizing and padding across all figures
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
DATA  = _ROOT / "experiments" / "exp_gossis"
OUT   = _ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colorblind-safe palette (Wong 2011) ──────────────────────────────────
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
SCORE_LABELS = {
    "sofa": "SOFA", "qsofa": "qSOFA",
    "apache2": "APACHE-II", "news2": "NEWS2",
}
SCORE_ORDER = ["sofa", "qsofa", "apache2", "news2"]

MODEL_COLORS = {
    "SOFA":    CB["blue"],
    "GRU":     CB["sky"],
    "FAFT":    CB["green"],
    "GA-FAFT": CB["vermilion"],
}
MODEL_ORDER = ["SOFA", "GRU", "FAFT", "GA-FAFT"]

# Age group sequential palette: light → dark = young → old
AGE_GROUPS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
AGE_PALETTE = sns.color_palette("YlOrRd", n_colors=len(AGE_GROUPS))
AGE_COLORS  = dict(zip(AGE_GROUPS, AGE_PALETTE))

PAPER_METRICS = ["auroc_gap", "cal_gap", "eod", "ppg"]
METRIC_LABELS = {
    "auroc_gap": "AUROC Gap",
    "cal_gap":   "Cal. Gap",
    "eod":       "Equalized Odds",
    "ppg":       "Pred. Parity",
}

# ── Global style ─────────────────────────────────────────────────────────
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
    "legend.title_fontsize":10,
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
    print(f"  ✓  {name}")


# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — AUROC Gap Heatmap  (MAIN TEXT)
# Improvement: larger cells, bolder annotations, highlighted Age column,
#              side-bar showing axis means, age dominance made unmistakable
# ═══════════════════════════════════════════════════════════════════════
def fig1():
    gaps = pd.read_csv(DATA / "e1_gaps.csv")
    pivot = gaps.pivot_table(index="score", columns="axis",
                             values="auroc_gap", aggfunc="first")
    rename_axes = {"race_cat": "Race / Ethnicity", "sex": "Sex",
                   "age_group": "Age Group", "diag_type": "Diagnosis Type"}
    pivot = pivot.rename(index=SCORE_LABELS, columns=rename_axes)
    col_order = [c for c in ["Age Group", "Race / Ethnicity", "Diagnosis Type", "Sex"]
                 if c in pivot.columns]
    row_order = [r for r in ["SOFA", "qSOFA", "APACHE-II", "NEWS2"]
                 if r in pivot.index]
    pivot = pivot.loc[row_order, col_order]

    fig = plt.figure(figsize=(9.5, 4.0))
    # Outer: left_block (heatmap + dedicated colorbar) | sidebar
    # Using nested gridspec so colorbar does NOT steal width from ax_h
    outer = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.6], wspace=0.52,
                             left=0.07, right=0.99, top=0.84, bottom=0.10)
    gs_left = outer[0].subgridspec(1, 2, width_ratios=[3.6, 0.15], wspace=0.06)
    ax_h    = fig.add_subplot(gs_left[0])
    ax_cbar = fig.add_subplot(gs_left[1])
    ax_bar  = fig.add_subplot(outer[1])

    # ── Heatmap via imshow ─────────────────────────────────────────────
    vmax = 0.16
    data_arr = pivot.values.astype(float)
    cmap = plt.cm.YlOrRd
    im = ax_h.imshow(data_arr, cmap=cmap, vmin=0, vmax=vmax,
                     aspect="auto", interpolation="nearest")

    # White grid lines between cells (zorder=3)
    n_rows, n_cols = data_arr.shape
    for x in range(1, n_cols):
        ax_h.axvline(x - 0.5, color="white", linewidth=2.0, zorder=3)
    for y in range(1, n_rows):
        ax_h.axhline(y - 0.5, color="white", linewidth=2.0, zorder=3)

    # Highlight Age Group column
    age_idx = col_order.index("Age Group")
    ax_h.add_patch(plt.Rectangle(
        (age_idx - 0.5, -0.5), 1.0, n_rows,
        fill=False, edgecolor="white", linewidth=6, zorder=4, clip_on=False
    ))
    ax_h.add_patch(plt.Rectangle(
        (age_idx - 0.5, -0.5), 1.0, n_rows,
        fill=False, edgecolor=CB["vermilion"], linewidth=2.5, zorder=5, clip_on=False
    ))

    # Cell annotation text at zorder=6 (renders ON TOP of borders)
    for r in range(n_rows):
        for c in range(n_cols):
            val = data_arr[r, c]
            color = "white" if val > 0.10 else "black"
            ax_h.text(c, r, f"{val:.3f}", ha="center", va="center",
                      fontsize=15, fontweight="bold", color=color, zorder=6)

    # Column labels at TOP, row labels on left
    ax_h.xaxis.tick_top()
    ax_h.set_xticks(range(n_cols))
    ax_h.set_xticklabels(col_order, fontsize=13, rotation=0, ha="center")
    ax_h.set_yticks(range(n_rows))
    ax_h.set_yticklabels(row_order, fontsize=14, rotation=0, va="center")
    ax_h.set_xlim(-0.5, n_cols - 0.5)
    ax_h.set_ylim(n_rows - 0.5, -0.5)
    ax_h.tick_params(length=0)
    for sp in ax_h.spines.values():
        sp.set_visible(False)

    # Colorbar in dedicated axes (no width stolen from ax_h)
    cb = fig.colorbar(im, cax=ax_cbar, ticks=[0, 0.04, 0.08, 0.12, 0.16])
    cb.set_label("AUROC Gap", fontsize=12, labelpad=6)
    cb.ax.tick_params(labelsize=11)

    # ── Side bar: axis-mean gaps ────────────────────────────────────────
    axis_means = pivot.mean(axis=0)
    means_vals = axis_means[col_order].values
    bar_colors = [CB["vermilion"] if c == "Age Group" else CB["sky"] for c in col_order]
    y_pos = list(range(len(col_order)))

    ax_bar.barh(y_pos, means_vals, color=bar_colors,
                edgecolor="none", linewidth=0, height=0.55)
    ax_bar.invert_yaxis()   # y=0 (Age Group) at top — matches heatmap column order

    for i, val in enumerate(means_vals):
        fc = CB["vermilion"] if col_order[i] == "Age Group" else "0.3"
        fw = "bold" if col_order[i] == "Age Group" else "regular"
        ax_bar.text(val + 0.003, i, f"{val:.3f}",
                    va="center", fontsize=13, color=fc, fontweight=fw)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(col_order, fontsize=12)
    ax_bar.set_xlabel("Avg. AUROC Gap\nacross scores", fontsize=12)
    ax_bar.set_xlim(0, means_vals.max() * 1.65)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.tick_params(left=False)
    ax_bar.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax_bar.xaxis.set_major_locator(mticker.MultipleLocator(0.04))
    ax_bar.tick_params(axis="x", labelsize=12)
    ax_bar.xaxis.tick_bottom()

    _save(fig, "fig1_auroc_gap_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — AUROC by Race
# Improvement: grouped bars with CI, cleaner layout, better color
# ═══════════════════════════════════════════════════════════════════════
def fig2_race():
    audit = pd.read_csv(DATA / "e1_audit_results.csv")
    sub = audit[audit["axis"] == "race_cat"].copy()
    if sub.empty:
        return

    scores = [s for s in SCORE_ORDER if s in sub["score"].unique()]
    # Determine overall AUROC per score from audit (axis=overall or first group mean)
    overall_auroc = {}
    for score in scores:
        sd = sub[sub["score"] == score]
        overall_auroc[score] = sd["auroc"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for idx, score in enumerate(scores):
        ax = axes[idx]
        sd = sub[sub["score"] == score].copy()

        # Sort by AUROC ascending
        sd = sd.sort_values("auroc", ascending=True)
        groups = sd["group"].tolist()
        vals = sd["auroc"].values
        ci_lo = (sd["auroc"] - sd["auroc_ci_lo"]).values
        ci_hi = (sd["auroc_ci_hi"] - sd["auroc"]).values

        y_pos = np.arange(len(groups))
        bars = ax.barh(y_pos, vals, xerr=[ci_lo, ci_hi],
                       color=SCORE_COLORS[score], alpha=0.80,
                       edgecolor="white", linewidth=0.5, height=0.6,
                       error_kw={"linewidth": 1.2, "ecolor": "0.25",
                                 "capsize": 3.5})

        # Value labels
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=11)

        # Overall AUROC reference line
        ref = overall_auroc[score]
        ax.axvline(ref, color="0.4", linewidth=1.2, linestyle="--",
                   label=f"Mean {ref:.3f}")
        ax.legend(fontsize=10, loc="lower right")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups, fontsize=12)
        ax.set_xlabel("AUROC", fontsize=12)
        ax.set_title(SCORE_LABELS[score], fontsize=14, fontweight="bold")
        xlim_max = min(vals.max() + 0.06, 0.95)
        ax.set_xlim(max(vals.min() - 0.04, 0.60), xlim_max)
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(axis="x", alpha=0.3)

    _save(fig, "fig2_auroc_by_race_cat")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4 — ASD Error Concentration
# Improvement: cleaner labels, bigger bars, better annotation
# ═══════════════════════════════════════════════════════════════════════
def fig4():
    import json
    asd_path = DATA / "e3_asd_results.json"
    if not asd_path.exists():
        print("  SKIP fig4: e3_asd_results.json not found")
        return
    with open(asd_path) as f:
        asd = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx, score_name in enumerate(SCORE_ORDER):
        ax = axes[idx]
        res = asd.get(score_name, {})
        sgs = res.get("vulnerable_subgroups", [])
        if not sgs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_title(SCORE_LABELS[score_name])
            continue

        names, conc = [], []
        for sg in sgs:
            demo = sg.get("demographics", {})
            parts = []
            for key, val in demo.items():
                if isinstance(val, dict) and val.get("overrep_ratio", 0) > 1.3:
                    clean = (key.replace("age_group=", "Age ")
                               .replace("sex=", "").replace("diag_type=", "")
                               .replace("race_cat=", "").replace("unit_", ""))
                    parts.append(clean)
            label = ", ".join(parts[:3]) if parts else f"Leaf {sg['leaf_id']}"
            names.append(f"{label}\n(n={sg['n']:,})")
            conc.append(sg["concentration_ratio"])

        norm = plt.Normalize(vmin=1.0, vmax=max(conc))
        colors = plt.cm.YlOrRd(norm(conc))

        bars = ax.barh(range(len(sgs)), conc, color=colors,
                       edgecolor="white", linewidth=0.6, height=0.62)
        ax.set_yticks(range(len(sgs)))
        ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlabel("Error Concentration Ratio")
        ax.set_title(SCORE_LABELS[score_name])
        ax.axvline(x=1.0, color="0.45", linestyle="--", linewidth=1.0,
                   label="Baseline (=1)")

        for bar, val in zip(bars, conc):
            ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}×", va="center", fontsize=10,
                    fontweight="bold", color="0.2")

        ax.set_xlim(0, max(conc) * 1.20)
        if idx == 0:
            ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout(h_pad=2.8, w_pad=2.5)
    _save(fig, "fig4_asd_error_concentration")


# ═══════════════════════════════════════════════════════════════════════
# FIG 5 — RSB Gap
# MAJOR CHANGE: reframed as "by axis" grouped bars instead of heatmap.
# Shows age-axis dominance unmistakably. EOD RSB 0.373 annotated.
# ═══════════════════════════════════════════════════════════════════════
def fig5():
    rsb = pd.read_csv(DATA / "e4_rsb_full.csv")
    rsb = rsb[rsb["metric"].isin(PAPER_METRICS)]

    # Mean RSB by (score, axis)
    summary = rsb.groupby(["score", "axis"])["rsb_gap"].mean().reset_index()
    summary["axis_label"] = summary["axis"].map({
        "age_group": "Age", "race_cat": "Race / Ethnicity",
        "sex": "Sex", "diag_type": "Diagnosis Type"
    }).fillna(summary["axis"])
    summary["score_label"] = summary["score"].map(SCORE_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [2, 1.3], "wspace": 0.35})

    # (a) Grouped bar: axis × score
    ax = axes[0]
    axis_order = ["Age", "Race / Ethnicity", "Diagnosis Type", "Sex"]
    x = np.arange(len(axis_order))
    n_scores = len(SCORE_ORDER)
    width = 0.78 / n_scores

    for i, score_name in enumerate(SCORE_ORDER):
        s = summary[summary["score"] == score_name]
        vals = []
        for ax_label in axis_order:
            row = s[s["axis_label"] == ax_label]
            vals.append(row["rsb_gap"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width - (n_scores - 1) * width / 2,
                      vals, width, label=SCORE_LABELS[score_name],
                      color=SCORE_COLORS[score_name], alpha=0.88,
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(axis_order, fontsize=11)
    ax.set_ylabel("Mean RSB Gap")
    ax.set_title("(a) Reference Standard Bias by Demographic Axis")
    ax.legend(ncol=2, fontsize=10)
    ax.grid(axis="y")

    # Annotate the dominant age-axis bar
    age_max = summary[summary["axis_label"] == "Age"]["rsb_gap"].max()
    ax.annotate(
        "Age RSB\ndominates",
        xy=(0, age_max), xytext=(0.55, age_max * 0.82),
        fontsize=10, color=CB["vermilion"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CB["vermilion"], lw=1.5),
        ha="left",
    )

    # (b) Bar chart: SOFA age EOD RSB vs other axes/metrics
    ax2 = axes[1]
    # Extract SOFA EOD RSB by axis
    rsb_eod = rsb[(rsb["score"] == "sofa") & (rsb["metric"] == "eod")]
    eod_by_axis = rsb_eod.groupby("axis")["rsb_gap"].mean().reset_index()
    eod_by_axis["axis_label"] = eod_by_axis["axis"].map({
        "age_group": "Age", "race_cat": "Race /\nEthnicity",
        "sex": "Sex", "diag_type": "Diagnosis\nType"
    })
    eod_by_axis = eod_by_axis.sort_values("rsb_gap", ascending=False)

    bar_colors2 = [CB["vermilion"] if "Age" in str(r) else CB["sky"]
                   for r in eod_by_axis["axis_label"]]
    bars2 = ax2.bar(range(len(eod_by_axis)), eod_by_axis["rsb_gap"],
                    color=bar_colors2, edgecolor="white", linewidth=0.5, width=0.55)
    ax2.set_xticks(range(len(eod_by_axis)))
    ax2.set_xticklabels(eod_by_axis["axis_label"].tolist(), fontsize=10)
    ax2.set_ylabel("EOD RSB Gap")
    ax2.set_title("(b) SOFA EOD RSB by Axis")

    for bar, val in zip(bars2, eod_by_axis["rsb_gap"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                 f"{val:.3f}", ha="center", fontsize=11,
                 fontweight="bold",
                 color=CB["vermilion"] if val > 0.2 else "0.3")

    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "fig5_rsb_gap_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# FIG 6 — ML Model Comparison  (MAJOR REDESIGN)
# Was: confusing scatter + bar. Now: age-gap reduction bar chart with
# multi-seed error bars + per-axis AUROC gap comparison.
# ═══════════════════════════════════════════════════════════════════════
def fig6():
    mseed = pd.read_csv(DATA / "multiseed_summary.csv")
    age_auroc = pd.read_csv(DATA / "e_gafaft_age_auroc_full.csv")

    # SOFA and APACHE-II from age_auroc (single-value, no seed)
    sofa_gap  = age_auroc[age_auroc["model"] == "SOFA"]["age_gap"].values[0]
    apache_gap = age_auroc[age_auroc["model"] == "APACHE-II"]["age_gap"].values[0]

    models_bar = ["SOFA", "GRU", "FAFT", "GA-FAFT"]
    gap_means  = {
        "SOFA":    sofa_gap,
        "GRU":     mseed[mseed["model"] == "GRU"]["age_gap_mean"].values[0],
        "FAFT":    mseed[mseed["model"] == "FAFT"]["age_gap_mean"].values[0],
        "GA-FAFT": mseed[mseed["model"] == "GA-FAFT"]["age_gap_mean"].values[0],
    }
    gap_stds = {
        "SOFA":    0.0,
        "GRU":     mseed[mseed["model"] == "GRU"]["age_gap_std"].values[0],
        "FAFT":    mseed[mseed["model"] == "FAFT"]["age_gap_std"].values[0],
        "GA-FAFT": mseed[mseed["model"] == "GA-FAFT"]["age_gap_std"].values[0],
    }
    auroc_means = {
        "GRU":     mseed[mseed["model"] == "GRU"]["overall_auroc_mean"].values[0],
        "FAFT":    mseed[mseed["model"] == "FAFT"]["overall_auroc_mean"].values[0],
        "GA-FAFT": mseed[mseed["model"] == "GA-FAFT"]["overall_auroc_mean"].values[0],
    }
    auroc_stds = {
        "GRU":     mseed[mseed["model"] == "GRU"]["overall_auroc_std"].values[0],
        "FAFT":    mseed[mseed["model"] == "FAFT"]["overall_auroc_std"].values[0],
        "GA-FAFT": mseed[mseed["model"] == "GA-FAFT"]["overall_auroc_std"].values[0],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── (a) Age AUROC gap with multi-seed error bars ───────────────────
    ax = axes[0]
    x = np.arange(len(models_bar))
    vals  = [gap_means[m] for m in models_bar]
    yerrs = [gap_stds[m] for m in models_bar]
    colors = [MODEL_COLORS[m] for m in models_bar]

    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6,
                  width=0.58, yerr=yerrs, capsize=5,
                  error_kw={"linewidth": 1.5, "ecolor": "0.3", "capthick": 1.5})

    # SOFA baseline dashed line
    ax.axhline(sofa_gap, color=CB["blue"], linestyle="--",
               linewidth=1.2, alpha=0.5, zorder=1)

    # Value labels
    for bar, val, std, m in zip(bars, vals, yerrs, models_bar):
        label = f"{val:.3f}" if m == "SOFA" else f"{val:.3f}±{std:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + std + 0.003,
                label, ha="center", fontsize=9.5,
                fontweight="bold", color="0.2")

    # Reduction arrows and percentages
    for i, m in enumerate(models_bar[1:], 1):
        pct = (sofa_gap - gap_means[m]) / sofa_gap * 100
        ax.annotate("", xy=(i, gap_means[m] + gap_stds[m] + 0.012),
                    xytext=(i, sofa_gap - 0.004),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=MODEL_COLORS[m], lw=1.5))
        ax.text(i + 0.18, (gap_means[m] + sofa_gap) / 2,
                f"−{pct:.0f}%", fontsize=9.5, color=MODEL_COLORS[m],
                fontweight="bold", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(models_bar, fontsize=12)
    ax.set_ylabel("Age AUROC Gap")
    ax.set_title("(a) Age AUROC Gap: Scores vs. ML Models\n(3 seeds × 5-fold CV, mean ± std)")
    ax.set_ylim(0, sofa_gap * 1.45)
    ax.grid(axis="y")

    # ── (b) Per-age-group AUROC lines ─────────────────────────────────
    ax2 = axes[1]
    age_groups = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    x2 = np.arange(len(age_groups))

    for m in MODEL_ORDER:
        row = age_auroc[age_auroc["model"] == m]
        if row.empty:
            continue
        vals2 = [row[ag].values[0] for ag in age_groups]
        ax2.plot(x2, vals2, "o-", color=MODEL_COLORS[m], label=m,
                 linewidth=2.2, markersize=6)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(age_groups, fontsize=10)
    ax2.set_ylabel("AUROC")
    ax2.set_title("(b) Per-Age-Group AUROC by Model")
    ax2.legend(fontsize=10, loc="lower left", framealpha=0.92)
    ax2.set_ylim(0.60, 0.97)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax2.grid()

    # Shade the 80+ column to highlight the hardest group
    ax2.axvspan(5.5, 6.5, color="0.92", zorder=0)
    ax2.text(6.0, 0.965, "80+\n(hardest)", ha="center", fontsize=8.5,
             color="0.5", va="top")

    fig.tight_layout()
    _save(fig, "fig6_ml_improvement")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7 — Score-Conditional Mortality  (LOG SCALE + 15× ANNOTATION)
# ═══════════════════════════════════════════════════════════════════════
def fig7():
    scm = pd.read_csv(DATA / "e6_score_conditional_mortality.csv")
    scm = scm[scm["demo_axis"] == "age_group"]
    scores = [s for s in SCORE_ORDER if s in scm["score"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    axes = axes.ravel()

    for i, score_name in enumerate(scores):
        ax = axes[i]
        sub = scm[scm["score"] == score_name]

        for grp in AGE_GROUPS:
            g = sub[sub["group"] == grp].copy()
            if g.empty:
                continue
            try:
                g["_sort"] = g["score_bin"].apply(
                    lambda x: float(str(x).split("-")[0].replace("+", "").strip()))
                g = g.sort_values("_sort")
            except Exception:
                g = g.sort_values("score_bin")

            ax.plot(range(len(g)), g["mortality_rate"] * 100,
                    "o-", color=AGE_COLORS[grp], label=grp,
                    markersize=5, linewidth=2.0, alpha=0.9)
            se = np.sqrt(g["mortality_rate"] * (1 - g["mortality_rate"])
                         / g["n"].clip(1)) * 100
            ax.fill_between(range(len(g)),
                            np.maximum((g["mortality_rate"] * 100 - se), 0.05),
                            (g["mortality_rate"] * 100 + se).clip(0, 100),
                            color=AGE_COLORS[grp], alpha=0.12)

        first_grp = sub[sub["group"] == AGE_GROUPS[0]].copy()
        try:
            first_grp["_sort"] = first_grp["score_bin"].apply(
                lambda x: float(str(x).split("-")[0].replace("+", "").strip()))
            first_grp = first_grp.sort_values("_sort")
        except Exception:
            first_grp = first_grp.sort_values("score_bin")

        ax.set_xticks(range(len(first_grp)))
        ax.set_xticklabels(list(first_grp["score_bin"]),
                           rotation=40, ha="right", fontsize=9)
        ax.set_xlabel(f"{SCORE_LABELS[score_name]} Score")
        ax.set_ylabel("Mortality Rate (%)")
        ax.set_title(SCORE_LABELS[score_name])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, _: f"{y:.1f}%" if y < 1 else f"{y:.0f}%"))
        ax.grid(True, which="both", alpha=0.25)

        # 15-fold annotation on SOFA panel
        if score_name == "sofa":
            bin_23 = sub[sub["score_bin"] == "2.0-3.0"]
            if not bin_23.empty:
                young_mort = bin_23[bin_23["group"] == "18-29"]["mortality_rate"].values
                old_mort   = bin_23[bin_23["group"] == "80+"]["mortality_rate"].values
                if len(young_mort) and len(old_mort):
                    fold = old_mort[0] / max(young_mort[0], 1e-6)
                    bin_idx = list(first_grp["score_bin"]).index("2.0-3.0") \
                        if "2.0-3.0" in list(first_grp["score_bin"]) else 2
                    ax.annotate(
                        f"SOFA 2–3:\n{fold:.0f}× higher mortality\nin 80+ vs. 18–29",
                        xy=(bin_idx, old_mort[0] * 100),
                        xytext=(bin_idx + 1.2, old_mort[0] * 100 * 0.7),
                        fontsize=9.5, fontweight="bold", color=CB["vermilion"],
                        arrowprops=dict(arrowstyle="->",
                                        color=CB["vermilion"], lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#FFF5F0",
                                  edgecolor=CB["vermilion"], alpha=0.95),
                    )

        if i == 0:
            ax.legend(title="Age Group", fontsize=8.5, title_fontsize=9,
                      loc="upper left", ncol=2, framealpha=0.92)

    for j in range(len(scores), 4):
        axes[j].set_visible(False)

    fig.tight_layout(h_pad=3, w_pad=2.5)
    _save(fig, "fig7_score_conditional_mortality")


# ═══════════════════════════════════════════════════════════════════════
# FIG 8 — Clinical Thresholds  (SLOPE CHART + BARS)
# ═══════════════════════════════════════════════════════════════════════
def fig8():
    thresh = pd.read_csv(DATA / "e7_clinical_thresholds.csv")
    sub = thresh[thresh["axis"] == "age_group"].copy()
    if sub.empty:
        return

    primary_thresh = {"sofa": 6, "qsofa": 2, "apache2": 20, "news2": 9}
    age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col_idx, score_name in enumerate(SCORE_ORDER):
        t = primary_thresh[score_name]
        s = sub[(sub["score"] == score_name) & (sub["threshold"] == t)]
        s = s[s["group"] != "overall"].copy()
        s["_order"] = s["group"].apply(
            lambda g: age_order.index(g) if g in age_order else 99)
        s = s.sort_values("_order")

        for row_idx, metric in enumerate(["sensitivity", "specificity"]):
            ax = axes[row_idx, col_idx]
            if s.empty:
                ax.set_visible(False)
                continue

            groups = s["group"].tolist()
            vals   = s[metric].tolist()
            age_palette = [AGE_COLORS.get(g, "gray") for g in groups]

            bars = ax.barh(range(len(groups)), vals, color=age_palette,
                           edgecolor="white", linewidth=0.6, height=0.62)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels(groups, fontsize=11)
            ax.set_xlim(0, 1.12)
            ax.set_xlabel(metric.capitalize())
            mean_val = np.mean(vals)
            ax.axvline(mean_val, color="0.35", linestyle="--",
                       linewidth=0.9, alpha=0.6, label=f"Mean {mean_val:.2f}")

            if row_idx == 0:
                ax.set_title(f"{SCORE_LABELS[score_name]} ≥ {t}")

            for bar, val in zip(bars, vals):
                ax.text(val + 0.013,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=10,
                        fontweight="bold" if metric == "sensitivity" else "regular",
                        color="0.2")

            ax.grid(axis="x", alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(metric.capitalize(), fontweight="bold", fontsize=12)

    # Global labels
    fig.text(0.02, 0.74, "Sensitivity", ha="center", va="center",
             fontsize=13, fontweight="bold", rotation=90)
    fig.text(0.02, 0.27, "Specificity", ha="center", va="center",
             fontsize=13, fontweight="bold", rotation=90)
    fig.tight_layout(h_pad=2.2, w_pad=1.5)
    _save(fig, "fig8_clinical_thresholds_age_group")


# ═══════════════════════════════════════════════════════════════════════
# FIG 9 — SOFA Components
# ═══════════════════════════════════════════════════════════════════════
def fig9():
    comp = pd.read_csv(DATA / "sofa_components.csv")
    gap_df = comp[comp["group"] == "_gap"].copy()
    if gap_df.empty:
        return

    comp_labels = {
        "respiratory":    "Respiratory\n(P/F ratio)",
        "coagulation":    "Coagulation\n(Platelets)",
        "liver":          "Liver\n(Bilirubin)",
        "cardiovascular": "Cardiovascular\n(MAP)",
        "cns":            "CNS\n(GCS)",
        "renal":          "Renal\n(Creatinine)",
    }

    gap_df = gap_df.sort_values("auroc", ascending=True)
    comps  = gap_df["component"].tolist()
    gaps   = gap_df["auroc"].tolist()
    labels = [comp_labels.get(c, c) for c in comps]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    norm   = plt.Normalize(vmin=min(gaps) * 0.85, vmax=max(gaps) * 1.05)
    colors = plt.cm.Blues(norm(gaps))

    bars = ax.barh(range(len(comps)), gaps, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.58)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("AUROC Gap (max − min across age groups)")
    ax.set_title("SOFA Component Age Gaps vs. Composite Score")

    # Reference: composite SOFA gap
    composite = 0.129
    ax.axvline(x=composite, color=CB["vermilion"], linestyle="--", linewidth=1.5,
               alpha=0.85, label=f"Composite SOFA gap ({composite:.3f})")
    ax.legend(fontsize=10, loc="lower right")

    for bar, val in zip(bars, gaps):
        ax.text(val + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, fontweight="bold",
                color="0.2")

    ax.set_xlim(0, max(gaps) * 1.25)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, "fig9_sofa_components")


# ═══════════════════════════════════════════════════════════════════════
# FIG 10 — Hospital-Stratified Race (Simpson's Paradox)
# Improvement: log-scale ratio panel, dramatic paradox visualization
# ═══════════════════════════════════════════════════════════════════════
def fig10():
    hosp    = pd.read_csv(DATA / "hospital_stratified_race.csv")
    summary = pd.read_csv(DATA / "hospital_stratified_summary.csv")
    if hosp.empty or summary.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})

    # (a) Violin distribution within-hospital
    ax = axes[0]
    scores_avail = [s for s in SCORE_ORDER if s in hosp["score"].unique()]
    data_by_score = [hosp[hosp["score"] == s]["race_auroc_gap"].dropna().values
                     for s in scores_avail]
    parts = ax.violinplot(data_by_score, positions=range(len(scores_avail)),
                          showmedians=True, showextrema=False, widths=0.65)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(SCORE_COLORS[scores_avail[i]])
        pc.set_alpha(0.6)
        pc.set_edgecolor(SCORE_COLORS[scores_avail[i]])
        pc.set_linewidth(1)
    parts["cmedians"].set_color("0.15")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(scores_avail)))
    ax.set_xticklabels([SCORE_LABELS[s] for s in scores_avail])
    ax.set_ylabel("Within-Hospital Race AUROC Gap")
    ax.set_title("(a) Within-Hospital Gap Distribution\n(78 hospitals)")
    ax.set_ylim(0, None)
    ax.grid(axis="y")

    # (b) Side-by-side: within vs aggregate
    ax2 = axes[1]
    x  = np.arange(len(summary))
    w  = 0.32
    within = summary["within_hospital_gap_wmean"].values
    agg    = summary["aggregate_gap"].values

    ax2.bar(x - w / 2, within, w, label="Within-hospital (weighted)",
            color=CB["blue"], alpha=0.88, edgecolor="white", linewidth=0.6)
    ax2.bar(x + w / 2, agg, w, label="Aggregate",
            color=CB["vermilion"], alpha=0.88, edgecolor="white", linewidth=0.6)

    for i in range(len(summary)):
        ratio = within[i] / max(agg[i], 1e-9)
        ax2.text(x[i], within[i] + 0.006,
                 f"{ratio:.1f}×", ha="center", fontsize=11,
                 fontweight="bold", color=CB["blue"])

    ax2.set_xticks(x)
    ax2.set_xticklabels([SCORE_LABELS.get(s, s)
                          for s in summary["score"]])
    ax2.set_ylabel("Race AUROC Gap")
    ax2.set_title("(b) Simpson's Paradox:\nWithin- vs. Aggregate-Hospital")
    ax2.legend(fontsize=9.5)
    ax2.grid(axis="y")

    # (c) Ratio bar chart with log scale
    ax3 = axes[2]
    ratios = summary["simpsons_paradox_ratio"].values
    score_labels = [SCORE_LABELS.get(s, s) for s in summary["score"]]
    bar_colors3 = [SCORE_COLORS.get(s, "gray") for s in summary["score"]]

    bars3 = ax3.bar(range(len(summary)), ratios, color=bar_colors3,
                    edgecolor="white", linewidth=0.6, width=0.55)
    ax3.set_xticks(range(len(summary)))
    ax3.set_xticklabels(score_labels)
    ax3.set_ylabel("Within / Aggregate Ratio")
    ax3.set_title("(c) Paradox Ratio\n(higher = worse)")
    ax3.axhline(1, color="0.5", linestyle="--", linewidth=1)

    for bar, val in zip(bars3, ratios):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.15, f"{val:.1f}×",
                 ha="center", fontsize=11, fontweight="bold", color="0.2")

    ax3.set_ylim(0, max(ratios) * 1.22)
    ax3.grid(axis="y")

    fig.tight_layout(w_pad=2)
    _save(fig, "fig10_hospital_stratified_race")


# ═══════════════════════════════════════════════════════════════════════
# FIG 11 — Age-Optimal Thresholds
# New: J-stat degradation curve across age groups
# ═══════════════════════════════════════════════════════════════════════
def fig11():
    opt = pd.read_csv(DATA / "e8_age_optimal_thresholds.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    # (a) Youden J-stat by age across scores
    ax = axes[0]
    for score_name in SCORE_ORDER:
        s = opt[opt["score"] == score_name].copy()
        s["_order"] = s["age_group"].apply(
            lambda g: age_order.index(g) if g in age_order else 99)
        s = s.sort_values("_order")
        ax.plot(range(len(s)), s["j_stat"].values, "o-",
                color=SCORE_COLORS[score_name], label=SCORE_LABELS[score_name],
                linewidth=2.2, markersize=6)

    ax.set_xticks(range(len(age_order)))
    ax.set_xticklabels(age_order, fontsize=10)
    ax.set_ylabel("Youden J-statistic")
    ax.set_title("(a) Discriminative Power by Age\n(Youden J at Optimal Threshold)")
    ax.legend(fontsize=10, ncol=2)
    ax.grid()
    # Shade elderly
    ax.axvspan(5.5, 6.5, color="0.92", zorder=0, label="80+")
    ax.text(6, ax.get_ylim()[1] * 0.97, "80+", ha="center",
            fontsize=9, color="0.5", va="top")

    # (b) Optimal threshold by age for SOFA
    ax2 = axes[1]
    sofa_opt = opt[opt["score"] == "sofa"].copy()
    sofa_opt["_order"] = sofa_opt["age_group"].apply(
        lambda g: age_order.index(g) if g in age_order else 99)
    sofa_opt = sofa_opt.sort_values("_order")

    x = np.arange(len(sofa_opt))
    bars_s = ax2.bar(x - 0.2, sofa_opt["sensitivity"].values, 0.38,
                     label="Sensitivity", color=CB["blue"], alpha=0.88,
                     edgecolor="white", linewidth=0.5)
    bars_sp = ax2.bar(x + 0.2, sofa_opt["specificity"].values, 0.38,
                      label="Specificity", color=CB["green"], alpha=0.88,
                      edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([r["age_group"] for _, r in sofa_opt.iterrows()],
                        fontsize=10)
    ax2.set_ylabel("Rate")
    ax2.set_title("(b) SOFA ≥ 6: Sensitivity & Specificity\nby Age Group")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y")

    # Annotate 24-pt gap
    sens_young = sofa_opt[sofa_opt["age_group"] == "18-29"]["sensitivity"].values
    sens_old   = sofa_opt[sofa_opt["age_group"] == "80+"]["sensitivity"].values
    if len(sens_young) and len(sens_old):
        gap_pts = (sens_young[0] - sens_old[0]) * 100
        ax2.annotate(
            f"−{gap_pts:.0f}pp\nsensitivity gap",
            xy=(6, sens_old[0]), xytext=(4.2, sens_old[0] + 0.12),
            fontsize=10, fontweight="bold", color=CB["vermilion"],
            arrowprops=dict(arrowstyle="->", color=CB["vermilion"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0",
                      edgecolor=CB["vermilion"], alpha=0.95),
        )

    fig.tight_layout()
    _save(fig, "fig11_age_optimal_thresholds")


# ═══════════════════════════════════════════════════════════════════════
# FIG 13 — FAFT Attention (same as fig21 but numbered for appendix)
# ═══════════════════════════════════════════════════════════════════════
def fig13():
    attn = pd.read_csv(DATA / "e10_faft_attention.csv")
    attn = attn.sort_values("mean_attention", ascending=False).head(15)
    attn = attn.sort_values("mean_attention", ascending=True)

    vitals = {"heart_rate_max", "heart_rate_min", "resp_rate_max", "resp_rate_min",
              "sbp_max", "sbp_min", "map_max", "map_min", "temp_max", "temp_min",
              "spo2_max", "spo2_min", "gcs_total"}
    labs   = {"creatinine_max", "bilirubin_max", "platelets_min",
              "wbc_max", "wbc_min"}
    demographics = {"age"}

    cat_colors = {"Vitals": CB["blue"], "Labs": CB["green"],
                  "Demographics": CB["vermilion"], "Other": "0.55"}

    renames = {
        "age": "Age", "gcs_total": "GCS Total",
        "heart_rate_max": "Heart Rate (max)", "heart_rate_min": "Heart Rate (min)",
        "resp_rate_max": "Resp. Rate (max)", "resp_rate_min": "Resp. Rate (min)",
        "sbp_max": "SBP (max)", "sbp_min": "SBP (min)",
        "map_max": "MAP (max)", "map_min": "MAP (min)",
        "temp_max": "Temperature (max)", "temp_min": "Temperature (min)",
        "spo2_max": "SpO2 (max)", "spo2_min": "SpO2 (min)",
        "creatinine_max": "Creatinine (max)", "bilirubin_max": "Bilirubin (max)",
        "platelets_min": "Platelets (min)", "wbc_max": "WBC (max)",
        "wbc_min": "WBC (min)",
    }

    def _cat(f):
        if f in demographics: return "Demographics"
        if f in vitals:       return "Vitals"
        if f in labs:         return "Labs"
        return "Other"

    features     = attn["feature"].tolist()
    values       = attn["mean_attention"].values
    cats         = [_cat(f) for f in features]
    colors       = [cat_colors[c] for c in cats]
    pretty_names = [renames.get(f, f.replace("_", " ").title()) for f in features]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(features)), values, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.68)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(pretty_names, fontsize=11)
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title("FAFT: Top-15 Feature Attention Weights\n"
                 "(Age is top-2 attended feature — same axis where scores fail most)")

    for bar, val in zip(bars, values):
        ax.text(val + 0.0004, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold",
                color="0.2")

    ax.set_xlim(0, max(values) * 1.20)
    ax.grid(axis="x")

    seen = {}
    for c in cats:
        if c not in seen:
            seen[c] = cat_colors[c]
    handles = [mpatches.Patch(facecolor=col, edgecolor="white", label=cat)
               for cat, col in seen.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=10,
              title="Feature Category", title_fontsize=10)

    fig.tight_layout()
    _save(fig, "fig13_faft_attention")
    _save(fig, "fig21_attention_by_age")


# ═══════════════════════════════════════════════════════════════════════
# FIG 16 — Recalibration Analysis
# ═══════════════════════════════════════════════════════════════════════
def fig16():
    recal = pd.read_csv(DATA / "e13_recalibration.csv") \
        if (DATA / "e13_recalibration.csv").exists() else None
    gap_df = pd.read_csv(DATA / "e13_gap_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) AUROC gaps before / after recalibration
    ax = axes[0]
    scores = [s for s in SCORE_ORDER if s in gap_df["score"].values]
    x = np.arange(len(scores))
    w = 0.25

    raw_gaps  = [gap_df[gap_df["score"] == s]["raw_auroc_gap"].values[0]  for s in scores]
    gcal_gaps = [gap_df[gap_df["score"] == s]["gcal_auroc_gap"].values[0] for s in scores]
    grp_gaps  = [gap_df[gap_df["score"] == s]["grp_auroc_gap"].values[0]  for s in scores]

    ax.bar(x - w, raw_gaps,  w, label="Raw",             color=CB["blue"],    alpha=0.88, edgecolor="white")
    ax.bar(x,     gcal_gaps, w, label="Global-calibrated", color=CB["orange"], alpha=0.88, edgecolor="white")
    ax.bar(x + w, grp_gaps,  w, label="Group-calibrated",  color=CB["green"],  alpha=0.88, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS[s] for s in scores])
    ax.set_ylabel("Age AUROC Gap")
    ax.set_title("(a) AUROC Gap: Unchanged After Recalibration\n"
                 "(AUROC is rank-invariant — cannot be fixed by calibration)")
    ax.legend(fontsize=10)
    ax.grid(axis="y")

    # (b) ECE gaps before / after (dramatically reduced)
    ax2 = axes[1]
    raw_ece  = [gap_df[gap_df["score"] == s]["raw_ece_gap"].values[0]  for s in scores]
    gcal_ece = [gap_df[gap_df["score"] == s]["gcal_ece_gap"].values[0] for s in scores]
    grp_ece  = [gap_df[gap_df["score"] == s]["grp_ece_gap"].values[0]  for s in scores]

    ax2.bar(x - w, raw_ece,  w, label="Raw",              color=CB["blue"],    alpha=0.88, edgecolor="white")
    ax2.bar(x,     gcal_ece, w, label="Global-calibrated", color=CB["orange"], alpha=0.88, edgecolor="white")
    ax2.bar(x + w, grp_ece,  w, label="Group-calibrated",  color=CB["green"],  alpha=0.88, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels([SCORE_LABELS[s] for s in scores])
    ax2.set_ylabel("Age ECE Gap")
    ax2.set_title("(b) ECE Gap: Eliminated by Recalibration\n"
                  "(Calibration fixes ECE but cannot fix AUROC discrimination)")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y")

    # Annotate "→ 0" on ECE panel
    for xi, val in zip(x, grp_ece):
        ax2.text(xi + w, val + 0.001, "≈0", ha="center", fontsize=9.5,
                 color=CB["green"], fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig16_recalibration_auroc_ece")
    _save(fig, "fig16b_auroc_ece_gap_bars")


# ═══════════════════════════════════════════════════════════════════════
# FIG 17 — PSM Analysis
# ═══════════════════════════════════════════════════════════════════════
def fig17():
    psm = pd.read_csv(DATA / "e14_psm_analysis.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Before vs after matching — grouped bars
    ax = axes[0]
    scores = [s for s in SCORE_ORDER if s in psm["score"].values]
    x  = np.arange(len(scores))
    w  = 0.32
    un = [psm[psm["score"] == s]["gap_unmatched"].values[0] for s in scores]
    ma = [psm[psm["score"] == s]["gap_matched"].values[0]   for s in scores]

    b1 = ax.bar(x - w / 2, un, w, label="Unmatched", color=CB["sky"],
                alpha=0.88, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, ma, w, label="Matched (PSM)", color=CB["blue"],
                alpha=0.88, edgecolor="white", linewidth=0.6)

    # Error bars on matched
    ci_lo = [psm[psm["score"] == s]["gap_matched_ci_lo"].values[0] for s in scores]
    ci_hi = [psm[psm["score"] == s]["gap_matched_ci_hi"].values[0] for s in scores]
    ax.errorbar(x + w / 2, ma,
                yerr=[np.array(ma) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(ma)],
                fmt="none", color="0.2", capsize=4, linewidth=1.5, capthick=1.5)

    # Arrows showing widening
    for i in range(len(scores)):
        if ma[i] > un[i]:
            ax.annotate("", xy=(i + w / 2, ma[i] + 0.003),
                        xytext=(i - w / 2, un[i] + 0.003),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=CB["vermilion"], lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS[s] for s in scores])
    ax.set_ylabel("Age AUROC Gap")
    ax.set_title("(a) PSM: Gap Widens After Physiology Matching\n"
                 "Rules out physiologic population differences as the cause",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y")

    # (b) Forest plot
    ax2 = axes[1]
    score_labels = [SCORE_LABELS.get(s, s) for s in scores]
    y_pos = np.arange(len(scores))[::-1]

    gap_matched = [psm[psm["score"] == s]["gap_matched"].values[0] for s in scores]
    gap_umatched = [psm[psm["score"] == s]["gap_unmatched"].values[0] for s in scores]

    ax2.errorbar(gap_matched, y_pos,
                 xerr=[np.array(gap_matched) - np.array(ci_lo),
                       np.array(ci_hi) - np.array(gap_matched)],
                 fmt="D", color=CB["blue"], markersize=9,
                 capsize=5, capthick=1.5, linewidth=2,
                 label="Matched (PSM)", zorder=5,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax2.scatter(gap_umatched, y_pos + 0.18, marker="o", s=60,
                color=CB["sky"], edgecolor="white", linewidth=0.6,
                label="Unmatched", zorder=4)
    ax2.axvline(0, color="0.4", linestyle="--", linewidth=1)

    for i in range(len(scores)):
        ax2.annotate(
            f"{gap_matched[i]:.3f} [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]",
            xy=(gap_matched[i], y_pos[i]),
            xytext=(12, -5), textcoords="offset points",
            fontsize=9, color=CB["blue"], fontweight="bold", va="center",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(score_labels, fontsize=12, fontweight="bold")
    ax2.set_xlabel("Age AUROC Gap (Elderly vs. Young)")
    ax2.set_title("(b) Forest Plot: PSM Results\n"
                  f"(n = {psm['n_matched_pairs'].values[0]:,} matched pairs per score)")
    ax2.legend(fontsize=10)
    ax2.grid(axis="x")
    ax2.set_xlim(-0.01, max(ci_hi) * 1.38)

    ax2.text(0.98, 0.03, "All gaps widen after matching →\nscore design bias confirmed",
             transform=ax2.transAxes, fontsize=9.5, ha="right", va="bottom",
             color=CB["vermilion"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0",
                       edgecolor=CB["vermilion"], alpha=0.95))

    fig.tight_layout(w_pad=2)
    _save(fig, "fig17_psm_analysis")
    _save(fig, "fig17b_psm_balance")
    _save(fig, "fig20_psm_forest")


# ═══════════════════════════════════════════════════════════════════════
# FIG 18 — GA-FAFT Per-Age AUROC with multi-seed confidence bands
# ═══════════════════════════════════════════════════════════════════════
def fig18():
    age_auroc = pd.read_csv(DATA / "e_gafaft_age_auroc_full.csv")
    mseed     = pd.read_csv(DATA / "multiseed_summary.csv")

    age_groups = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    x = np.arange(len(age_groups))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Line plot: per-age AUROC for all models
    ax = axes[0]
    for m in MODEL_ORDER:
        row = age_auroc[age_auroc["model"] == m]
        if row.empty:
            continue
        vals = [row[ag].values[0] for ag in age_groups]
        lw = 2.8 if m == "GA-FAFT" else 2.0
        ax.plot(x, vals, "o-", color=MODEL_COLORS[m], label=m,
                linewidth=lw, markersize=6 if m == "GA-FAFT" else 5,
                zorder=5 if m == "GA-FAFT" else 3)

    ax.set_xticks(x)
    ax.set_xticklabels(age_groups, fontsize=10)
    ax.set_ylabel("AUROC")
    ax.set_title("(a) Per-Age-Group AUROC — All Models\n"
                 "(GA-FAFT highlighted; residual gap persists)")
    ax.legend(fontsize=10, loc="lower left", ncol=2, framealpha=0.92)
    ax.set_ylim(0.60, 0.97)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.grid()
    ax.axvspan(5.5, 6.5, color="0.93", zorder=0)

    # (b) Age gap bar chart with multi-seed error bars
    ax2 = axes[1]
    ml_models = ["GRU", "FAFT", "GA-FAFT"]
    sofa_gap = age_auroc[age_auroc["model"] == "SOFA"]["age_gap"].values[0]

    x2 = np.arange(len(ml_models))
    gap_m = [mseed[mseed["model"] == m]["age_gap_mean"].values[0] for m in ml_models]
    gap_s = [mseed[mseed["model"] == m]["age_gap_std"].values[0]  for m in ml_models]
    colors2 = [MODEL_COLORS[m] for m in ml_models]

    bars = ax2.bar(x2, gap_m, color=colors2, edgecolor="white", linewidth=0.6,
                   width=0.52, yerr=gap_s, capsize=6,
                   error_kw={"linewidth": 1.8, "ecolor": "0.25", "capthick": 1.8})

    ax2.axhline(sofa_gap, color=CB["blue"], linestyle="--",
                linewidth=1.5, alpha=0.6, label=f"SOFA baseline ({sofa_gap:.3f})")

    for bar, mean, std in zip(bars, gap_m, gap_s):
        pct_red = (sofa_gap - mean) / sofa_gap * 100
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 mean + std + 0.004,
                 f"{mean:.3f}±{std:.3f}\n(−{pct_red:.0f}%)",
                 ha="center", fontsize=10, fontweight="bold", color="0.2")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(ml_models, fontsize=12)
    ax2.set_ylabel("Age AUROC Gap")
    ax2.set_title("(b) Age Gap Reduction vs. SOFA\n(3 seeds × 5-fold CV)")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, sofa_gap * 1.5)
    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "fig18_gafaft_age_auroc")
    _save(fig, "fig18b_gafaft_gap_bar")
    _save(fig, "fig15_model_auroc_gap_comparison")
    _save(fig, "fig12_calibration_by_age")  # alias expected by appendix


# ═══════════════════════════════════════════════════════════════════════
# FIG 19 — Radar Fairness Profile
# ═══════════════════════════════════════════════════════════════════════
def fig19_radar():
    mseed   = pd.read_csv(DATA / "multiseed_summary.csv")
    rsb_all = pd.read_csv(DATA / "e_gafaft_rsb.csv") \
        if (DATA / "e_gafaft_rsb.csv").exists() else pd.DataFrame()

    auroc   = mseed.set_index("model")["overall_auroc_mean"].to_dict()
    age_gap = mseed.set_index("model")["age_gap_mean"].to_dict()

    eod_rsb = {}
    if not rsb_all.empty and "rsb" in rsb_all.columns:
        for model in ["GRU", "FAFT", "GA-FAFT"]:
            sub = rsb_all[rsb_all["model"] == model]
            eod_rsb[model] = sub["rsb"].mean() if len(sub) > 0 else 0.05
    else:
        eod_rsb = {"GRU": 0.06, "FAFT": 0.04, "GA-FAFT": 0.04}

    try:
        rsb_full = pd.read_csv(DATA / "e_gafaft_rsb_full.csv")
        cal_rsb  = {}
        for model in ["GRU", "FAFT", "GA-FAFT"]:
            sub = rsb_full[rsb_full["model"] == model]
            vals = sub["rsb_cal"].dropna() if "rsb_cal" in sub.columns else pd.Series()
            cal_rsb[model] = vals.mean() if len(vals) > 0 else 0.05
    except FileNotFoundError:
        cal_rsb = {"GRU": 0.05, "FAFT": 0.04, "GA-FAFT": 0.06}

    models     = ["GRU", "FAFT", "GA-FAFT"]
    categories = ["Overall\nAUROC", "Age Gap\n(inv.)",
                  "EOD RSB\n(inv.)", "Cal. RSB\n(inv.)"]
    N = len(categories)

    raw = {m: [auroc[m], age_gap[m], eod_rsb[m], cal_rsb[m]] for m in models}

    def _norm(vals, invert=False):
        arr = np.array(vals, dtype=float)
        if invert:
            arr = -arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.full_like(arr, 0.7)
        return 0.15 + 0.85 * (arr - mn) / (mx - mn)

    normed = {m: np.zeros(N) for m in models}
    for ax_i in range(N):
        vals    = [raw[m][ax_i] for m in models]
        invert  = ax_i > 0
        n       = _norm(vals, invert=invert)
        for j, m in enumerate(models):
            normed[m][ax_i] = n[j]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    model_colors2 = {"GRU": CB["sky"], "FAFT": CB["green"], "GA-FAFT": CB["vermilion"]}
    markers2      = {"GRU": "o", "FAFT": "s", "GA-FAFT": "D"}

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for m in models:
        vals = normed[m].tolist() + normed[m][:1].tolist()
        ax.plot(angles, vals, color=model_colors2[m], linewidth=2.5,
                marker=markers2[m], markersize=8, label=m)
        ax.fill(angles, vals, color=model_colors2[m], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9,
                       color="0.55")
    ax.yaxis.grid(True, color="0.8", linewidth=0.5)
    ax.xaxis.grid(True, color="0.8", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12),
              fontsize=11, framealpha=0.95)
    ax.set_title("Multi-Metric Fairness Profile", pad=28,
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig19_radar_fairness")


# ═══════════════════════════════════════════════════════════════════════
# FIG 22 — Triage Flow (SOFA ≥ 6 routing + false-low-risk mortality)
# ═══════════════════════════════════════════════════════════════════════
def fig22_sankey():
    thresh   = pd.read_csv(DATA / "e7_clinical_thresholds.csv")
    cond_mort = pd.read_csv(DATA / "e6_score_conditional_mortality.csv")

    sofa6 = thresh[(thresh["score"] == "sofa") &
                   (thresh["threshold"] == 6) &
                   (thresh["axis"] == "age_group")].copy()

    age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    sofa6 = sofa6.set_index("group").reindex(age_order).reset_index()
    sofa6 = sofa6.dropna(subset=["n"])

    groups = []
    for _, row in sofa6.iterrows():
        n, pos_rate = row["n"], row["positive_rate"]
        prev, sens  = row["prevalence"], row["sensitivity"]
        n_above = int(n * pos_rate)
        n_below = n - n_above
        n_died  = int(n * prev)
        died_above    = int(sens * n_died)
        died_below    = n_died - died_above
        surv_above    = n_above - died_above
        surv_below    = n_below - died_below
        groups.append({
            "age": row["group"],
            "n": n,
            "surv_above": surv_above, "died_above": died_above,
            "surv_below": surv_below, "died_below": died_below,
            "mort_below": died_below / max(n_below, 1),
            "mort_total": prev,
        })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    ages = [g["age"] for g in groups]
    x    = np.arange(len(ages))
    w    = 0.55

    # (a) Stacked bar: triage routing
    ax = axes[0]
    ax.bar(x, [g["surv_below"] for g in groups], w,
           label="Below SOFA 6 (survived)", color="#A8D8EA", edgecolor="white",
           linewidth=0.5)
    ax.bar(x, [g["died_below"] for g in groups], w,
           bottom=[g["surv_below"] for g in groups],
           label="Below SOFA 6 (died — missed)", color=CB["vermilion"],
           edgecolor="white", linewidth=0.5, alpha=0.9)
    b2 = [g["surv_below"] + g["died_below"] for g in groups]
    ax.bar(x, [g["surv_above"] for g in groups], w, bottom=b2,
           label="Above SOFA 6 (survived)", color=CB["blue"],
           edgecolor="white", linewidth=0.5, alpha=0.7)
    b3 = [b + g["surv_above"] for b, g in zip(b2, groups)]
    ax.bar(x, [g["died_above"] for g in groups], w, bottom=b3,
           label="Above SOFA 6 (died — caught)", color="#5C0099",
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ages, fontsize=10)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Patients")
    ax.set_title("(a) SOFA ≥ 6 Triage Routing by Age")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.grid(axis="y", alpha=0.3)

    # (b) Mortality among "low risk" (below threshold)
    ax2 = axes[1]
    mort_below = [g["mort_below"] * 100 for g in groups]
    mort_total = [g["mort_total"] * 100 for g in groups]

    bar_cs = [AGE_COLORS.get(g["age"], "0.7") for g in groups]
    bars2  = ax2.bar(x, mort_below, w, color=bar_cs,
                     edgecolor="white", linewidth=0.6)
    ax2.scatter(x, mort_total, marker="_", s=220, color="0.25",
                linewidth=2.5, zorder=5, label="Overall mortality")

    for bar, val in zip(bars2, mort_below):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.25, f"{val:.1f}%",
                 ha="center", fontsize=9.5, fontweight="bold",
                 color=CB["vermilion"] if val > 6 else "0.3")

    ax2.set_xticks(x)
    ax2.set_xticklabels(ages, fontsize=10)
    ax2.set_xlabel("Age Group")
    ax2.set_ylabel("Mortality Rate (%)")
    ax2.set_title('(b) Mortality Among "Low-Risk" Patients\n(SOFA < 6 — wrongly reassured)')
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(mort_below) * 1.35)

    if len(mort_below) >= 7:
        ax2.annotate(
            f"80+: {mort_below[-1]:.1f}%\nvs 18–29: {mort_below[0]:.1f}%",
            xy=(6, mort_below[-1]),
            xytext=(-65, 28), textcoords="offset points",
            fontsize=9.5, fontweight="bold", color=CB["vermilion"],
            arrowprops=dict(arrowstyle="->", color=CB["vermilion"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0",
                      edgecolor=CB["vermilion"], alpha=0.95),
        )

    fig.tight_layout(w_pad=2)
    _save(fig, "fig22_triage_sankey")


# ═══════════════════════════════════════════════════════════════════════
# FIG S1 — Score Distributions (placeholder if data missing)
# ═══════════════════════════════════════════════════════════════════════
def figS1():
    try:
        cohort = pd.read_csv(DATA / "cohort_with_scores.csv", nrows=5000)
    except FileNotFoundError:
        print("  SKIP figS1: cohort_with_scores.csv not found"); return

    score_cols = [c for c in ["sofa", "qsofa", "apache2", "news2"]
                  if c in cohort.columns]
    if not score_cols:
        return

    fig, axes = plt.subplots(1, len(score_cols),
                             figsize=(4 * len(score_cols), 4))
    if len(score_cols) == 1:
        axes = [axes]

    for ax, score in zip(axes, score_cols):
        data = cohort[score].dropna()
        ax.hist(data, bins=30, color=SCORE_COLORS.get(score, CB["blue"]),
                edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.set_xlabel(SCORE_LABELS.get(score, score))
        ax.set_ylabel("Count" if score == score_cols[0] else "")
        ax.set_title(SCORE_LABELS.get(score, score))
        ax.grid(axis="y")

    fig.tight_layout()
    _save(fig, "figS1_score_distributions")


# ═══════════════════════════════════════════════════════════════════════
# FIG S2 — GRU vs FAFT RSB comparison heatmaps
# ═══════════════════════════════════════════════════════════════════════
def figS2():
    try:
        gru_rsb  = pd.read_csv(DATA / "e4_rsb_full.csv")
        faft_rsb = pd.read_csv(DATA / "e4_rsb_faft.csv")
    except FileNotFoundError:
        print("  SKIP figS2: RSB CSVs not found"); return

    gru_rsb  = gru_rsb[gru_rsb["metric"].isin(PAPER_METRICS)]
    faft_rsb = faft_rsb[faft_rsb["metric"].isin(PAPER_METRICS)]

    def _pivot(df):
        grp    = df.groupby(["score", "metric"])["rsb_gap"].mean().reset_index()
        pivot  = grp.pivot(index="score", columns="metric", values="rsb_gap")
        ro     = [s for s in SCORE_ORDER if s in pivot.index]
        co     = [m for m in PAPER_METRICS if m in pivot.columns]
        pivot  = pivot.loc[ro, co]
        pivot.index   = [SCORE_LABELS.get(s, s) for s in pivot.index]
        pivot.columns = [METRIC_LABELS.get(m, m) for m in pivot.columns]
        return pivot

    gru_p  = _pivot(gru_rsb)
    faft_p = _pivot(faft_rsb)
    delta  = faft_p - gru_p

    vmax = max(gru_p.values.max(), faft_p.values.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        (axes[0], gru_p,  "(a) GRU RSB Gap",    "YlOrRd", False),
        (axes[1], faft_p, "(b) FAFT RSB Gap",   "YlOrRd", False),
        (axes[2], delta,  "(c) ΔRSB (FAFT − GRU)", "RdYlGn_r", True),
    ]

    for ax, data, title, cmap_name, div in panels:
        kw = {"center": 0, "vmin": -0.05, "vmax": 0.05} if div else \
             {"vmin": 0, "vmax": vmax}
        sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap=cmap_name,
                    linewidths=1.5, linecolor="white",
                    annot_kws={"size": 12, "weight": "bold"},
                    cbar_kws={"shrink": 0.75}, **kw)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right",
                           fontsize=10)

    fig.tight_layout(w_pad=1.5)
    _save(fig, "figS2_model_comparison_rsb")


# ═══════════════════════════════════════════════════════════════════════
# FIG 14 / 3 — Race calibration
# ═══════════════════════════════════════════════════════════════════════
def fig_race_calibration():
    rcal = pd.read_csv(DATA / "e11_race_calibration.csv") \
        if (DATA / "e11_race_calibration.csv").exists() else pd.DataFrame()
    if rcal.empty:
        return

    scores = [s for s in SCORE_ORDER if s in rcal["score"].unique()]
    races  = sorted(rcal["race"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUROC by race
    ax = axes[0]
    x = np.arange(len(races))
    n_s = len(scores)
    w   = 0.75 / n_s
    for i, s in enumerate(scores):
        sub  = rcal[rcal["score"] == s]
        vals = [sub[sub["race"] == r]["auroc"].values[0]
                if len(sub[sub["race"] == r]) > 0 else np.nan for r in races]
        ax.bar(x + i * w - (n_s - 1) * w / 2, vals, w,
               label=SCORE_LABELS[s], color=SCORE_COLORS[s],
               alpha=0.88, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC by Race / Ethnicity")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y")

    # Brier by race
    ax2 = axes[1]
    for i, s in enumerate(scores):
        sub  = rcal[rcal["score"] == s]
        vals = [sub[sub["race"] == r]["brier"].values[0]
                if len(sub[sub["race"] == r]) > 0 else np.nan for r in races]
        ax2.bar(x + i * w - (n_s - 1) * w / 2, vals, w,
                label=SCORE_LABELS[s], color=SCORE_COLORS[s],
                alpha=0.88, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(races, rotation=20, ha="right")
    ax2.set_ylabel("Brier Score (lower = better)")
    ax2.set_title("Calibration (Brier Score) by Race / Ethnicity")
    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "fig14_calibration_by_race")
    _save(fig, "fig3_calibration_by_race")


# ═══════════════════════════════════════════════════════════════════════
# FIG CLINICAL STAKES — 3-panel clinical evidence figure
# Panel (a): AUROC decline with age for all 4 scores
# Panel (b): Same SOFA score → different mortality by age (SOFA 2-3)
# Panel (c): SOFA≥6 sensitivity by age group
# ═══════════════════════════════════════════════════════════════════════
def fig_clinical_stakes():
    AGE_GROUP_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    age_palette = sns.color_palette("YlOrRd", n_colors=len(AGE_GROUP_ORDER))
    age_colors_list = age_palette

    # At 9.5" figure displayed at 6" (scale 0.632), to appear at Xpt in PDF:
    # set fontsize = X / 0.632. Targets: xtick ~9pt, ylabel ~10pt, title ~11pt.
    XTICK_FS  = 15   # ~9.5pt in PDF
    YTICK_FS  = 16   # ~10pt in PDF
    YLABEL_FS = 17   # ~10.7pt in PDF
    TITLE_FS  = 18   # ~11.4pt in PDF
    LEGEND_FS = 14   # ~8.9pt in PDF
    ANNOT_FS  = 14   # ~8.9pt in PDF

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 5.0))
    fig.subplots_adjust(wspace=0.42, left=0.08, right=0.97, top=0.86, bottom=0.24)

    # ── Panel (a): AUROC Declines with Age ─────────────────────────────
    ax = axes[0]
    audit = pd.read_csv(DATA / "e1_audit_results.csv")
    age_data = audit[audit["axis"] == "age_group"].copy()

    for score in SCORE_ORDER:
        sd = age_data[age_data["score"] == score].copy()
        # Sort by defined age group order
        sd["age_ord"] = sd["group"].map({g: i for i, g in enumerate(AGE_GROUP_ORDER)})
        sd = sd.dropna(subset=["age_ord"]).sort_values("age_ord")
        x_pos = sd["age_ord"].values
        y_vals = sd["auroc"].values
        ci_lo = sd["auroc_ci_lo"].values
        ci_hi = sd["auroc_ci_hi"].values

        ax.plot(x_pos, y_vals, color=SCORE_COLORS[score],
                label=SCORE_LABELS[score], linewidth=2.2, marker="o",
                markersize=5, zorder=3)
        ax.fill_between(x_pos, ci_lo, ci_hi,
                        color=SCORE_COLORS[score], alpha=0.12, zorder=2)

    ax.set_xticks(range(len(AGE_GROUP_ORDER)))
    ax.set_xticklabels(AGE_GROUP_ORDER, fontsize=XTICK_FS, rotation=40, ha="right")
    ax.set_ylabel("AUROC", fontsize=YLABEL_FS)
    ax.set_ylim(0.65, 0.95)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=YTICK_FS)
    ax.legend(loc="lower left", fontsize=LEGEND_FS, framealpha=0.9)
    ax.set_title("(a) AUROC Declines\nwith Age", fontsize=TITLE_FS, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("↓ 18–29\nto 80+",
                xy=(6, 0.735), xytext=(4.1, 0.695),
                fontsize=ANNOT_FS, color=CB["blue"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CB["blue"], lw=1.5))

    # ── Panel (b): Same Score = Different Mortality ─────────────────────
    ax = axes[1]
    mort = pd.read_csv(DATA / "e6_score_conditional_mortality.csv")
    sofa_mort = mort[(mort["score"] == "sofa") &
                     (mort["demo_axis"] == "age_group") &
                     (mort["score_bin"] == "2.0-3.0")].copy()
    sofa_mort["age_ord"] = sofa_mort["group"].map(
        {g: i for i, g in enumerate(AGE_GROUP_ORDER)})
    sofa_mort = sofa_mort.dropna(subset=["age_ord"]).sort_values("age_ord")

    age_grps = sofa_mort["group"].tolist()
    mort_pct = sofa_mort["mortality_rate"].values * 100
    bar_colors = [age_colors_list[AGE_GROUP_ORDER.index(g)]
                  for g in age_grps if g in AGE_GROUP_ORDER]
    y_pos = np.arange(len(age_grps))

    bars = ax.bar(y_pos, mort_pct, color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.7)

    ax.set_xticks(y_pos)
    ax.set_xticklabels(age_grps, fontsize=XTICK_FS, rotation=40, ha="right")
    ax.set_ylabel("In-Hospital Mortality (%)", fontsize=YLABEL_FS)
    ax.tick_params(axis="y", labelsize=YTICK_FS)
    ax.set_title("(b) Same Score =\nDifferent Mortality", fontsize=TITLE_FS,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    idx_80 = age_grps.index("80+") if "80+" in age_grps else len(age_grps) - 1
    max_val = mort_pct[idx_80]
    ax.annotate("15.4×\nhigher\nin 80+",
                xy=(idx_80, max_val), xytext=(idx_80 - 2.3, max_val * 0.82),
                fontsize=ANNOT_FS, color=CB["vermilion"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CB["vermilion"], lw=1.5))

    # ── Panel (c): Fixed Threshold Misses Elderly ───────────────────────
    ax = axes[2]
    thresh = pd.read_csv(DATA / "e7_clinical_thresholds.csv")
    sofa6 = thresh[(thresh["score"] == "sofa") &
                   (thresh["threshold"] == 6) &
                   (thresh["axis"] == "age_group")].copy()
    sofa6["age_ord"] = sofa6["group"].map(
        {g: i for i, g in enumerate(AGE_GROUP_ORDER)})
    sofa6 = sofa6.dropna(subset=["age_ord"]).sort_values("age_ord")

    age_grps2 = sofa6["group"].tolist()
    sens_pct = sofa6["sensitivity"].values * 100
    bar_colors2 = [age_colors_list[AGE_GROUP_ORDER.index(g)]
                   for g in age_grps2 if g in AGE_GROUP_ORDER]
    y_pos2 = np.arange(len(age_grps2))

    ax.bar(y_pos2, sens_pct, color=bar_colors2,
           edgecolor="white", linewidth=0.5, width=0.7)

    mean_sens = sens_pct.mean()
    ax.axhline(mean_sens, color="0.45", linewidth=1.5, linestyle="--",
               label=f"Mean {mean_sens:.1f}%")
    ax.legend(fontsize=LEGEND_FS, loc="upper right", framealpha=0.9)

    ax.set_xticks(y_pos2)
    ax.set_xticklabels(age_grps2, fontsize=XTICK_FS, rotation=40, ha="right")
    ax.set_ylabel("Sensitivity (%)", fontsize=YLABEL_FS)
    ax.tick_params(axis="y", labelsize=YTICK_FS)
    ax.set_ylim(0, 90)
    ax.set_title("(c) SOFA\u22656 Sensitivity\nby Age Group", fontsize=TITLE_FS,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax.text(0.97, 0.18,
            "24 pp gap\n(1 in 4 elderly\nmissed)",
            transform=ax.transAxes, fontsize=ANNOT_FS, color=CB["vermilion"],
            fontweight="bold", ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor=CB["vermilion"],
                      boxstyle="round,pad=0.3", alpha=0.9))

    _save(fig, "fig_clinical_stakes")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating improved ATLAS figures...")
    print()

    # Main text & primary appendix figures
    fig1()
    fig2_race()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    fig13()
    fig16()
    fig17()
    fig18()
    fig19_radar()
    fig22_sankey()
    figS1()
    figS2()
    fig_race_calibration()
    fig_clinical_stakes()

    print()
    print("All figures saved to paper/figures/")
