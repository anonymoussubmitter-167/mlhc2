"""Regenerate all ATLAS paper figures — NeurIPS-quality publication standard."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
DATA = Path("experiments/exp_gossis")
OUT  = Path("paper/figures")

# ── Global NeurIPS-quality style ──────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "patch.linewidth": 0.4,
})

SCORE_LABELS = {"sofa": "SOFA", "qsofa": "qSOFA",
                "apache2": "APACHE-II", "news2": "NEWS2"}
SCORE_COLORS = {"sofa": "#2166ac", "qsofa": "#ef8a62",
                "apache2": "#4dac26", "news2": "#b2182b"}
SCORE_ORDER = ["sofa", "qsofa", "apache2", "news2"]

# 4 paper-reported fairness metrics (exclude ece_gap)
PAPER_METRICS = ["auroc_gap", "cal_gap", "eod", "ppg"]
METRIC_LABELS = {"auroc_gap": "AUROC Gap", "cal_gap": "Cal. Gap",
                 "eod": "Equalized Odds", "ppg": "Pred. Parity"}


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  OK {name}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 1: AUROC Gap Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig1():
    gaps = pd.read_csv(DATA / "e1_gaps.csv")
    pivot = gaps.pivot_table(index="score", columns="axis",
                             values="auroc_gap", aggfunc="first")
    rename_axes = {"race_cat": "Race/Ethnicity", "sex": "Sex",
                   "age_group": "Age", "diag_type": "Diagnosis Type"}
    pivot = pivot.rename(index=SCORE_LABELS, columns=rename_axes)
    col_order = [c for c in ["Age", "Race/Ethnicity", "Diagnosis Type", "Sex"]
                 if c in pivot.columns]
    row_order = [r for r in ["SOFA", "qSOFA", "APACHE-II", "NEWS2"]
                 if r in pivot.index]
    pivot = pivot.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.heatmap(pivot, annot=True, fmt=".3f",
                cmap="YlOrRd", vmin=0, vmax=0.16,
                linewidths=2, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"},
                cbar_kws={"label": "AUROC Gap", "shrink": 0.82,
                           "aspect": 15},
                ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")
    # Fix seaborn y-tick dash artifact: set labels explicitly
    ax.set_yticklabels(row_order, rotation=0, fontsize=12, va="center")
    ax.set_xticklabels(col_order, rotation=25, ha="right", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig1_auroc_gap_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# Fig 2: AUROC by Race
# ═══════════════════════════════════════════════════════════════════════
def fig2_race():
    audit = pd.read_csv(DATA / "e1_audit_results.csv")
    sub = audit[audit["axis"] == "race_cat"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scores = [s for s in SCORE_ORDER if s in sub["score"].unique()]
    groups = sorted(sub["group"].unique())
    x = np.arange(len(groups))
    width = 0.8 / len(scores)

    for i, score in enumerate(scores):
        s = sub[sub["score"] == score]
        vals, yerr_lo, yerr_hi = [], [], []
        for g in groups:
            row = s[s["group"] == g]
            if len(row) > 0:
                v = row["auroc"].values[0]
                lo = row["auroc_ci_lo"].values[0]
                hi = row["auroc_ci_hi"].values[0]
                vals.append(v)
                yerr_lo.append(v - lo)
                yerr_hi.append(hi - v)
            else:
                vals.append(np.nan); yerr_lo.append(0); yerr_hi.append(0)

        ax.bar(x + i * width, vals, width, label=SCORE_LABELS[score],
               color=SCORE_COLORS[score],
               yerr=[yerr_lo, yerr_hi], capsize=2, alpha=0.88,
               edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})

    ax.set_xlabel("Race/Ethnicity")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x + width * (len(scores) - 1) / 2)
    ax.set_xticklabels(groups, rotation=0)
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(0.60, 0.88)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "fig2_auroc_by_race_cat")


# ═══════════════════════════════════════════════════════════════════════
# Fig 4: ASD Error Concentration — with real demographic labels
# ═══════════════════════════════════════════════════════════════════════
def fig4():
    with open(DATA / "e3_asd_results.json") as f:
        asd = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
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

        # Build readable labels from demographics
        names = []
        for sg in sgs:
            demo = sg.get("demographics", {})
            # Find overrepresented demographics (ratio > 1.5)
            parts = []
            for key, val in demo.items():
                if isinstance(val, dict) and val.get("overrep_ratio", 0) > 1.3:
                    # Clean raw field names for display
                    clean = key.replace("age_group=", "Age ")
                    clean = clean.replace("sex=", "")
                    clean = clean.replace("diag_type=", "")
                    clean = clean.replace("race_cat=", "")
                    clean = clean.replace("unit_", "")
                    parts.append(clean)
            label = ", ".join(parts[:3]) if parts else f"Leaf {sg['leaf_id']}"
            names.append(f"{label}\n(n={sg['n']:,})")

        conc = [sg["concentration_ratio"] for sg in sgs]
        colors = sns.color_palette("Reds_r", len(sgs))

        bars = ax.barh(range(len(sgs)), conc, color=colors,
                       edgecolor="white", linewidth=0.6, height=0.65)
        ax.set_yticks(range(len(sgs)))
        ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlabel("Error Concentration Ratio")
        ax.set_title(SCORE_LABELS[score_name])
        ax.axvline(x=1.0, color="0.5", linestyle="--", linewidth=0.9,
                   label="Baseline" if idx == 0 else None)

        for bar, val in zip(bars, conc):
            ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

        ax.set_xlim(0, max(conc) * 1.18)
        if idx == 0:
            ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig4_asd_error_concentration")


# ═══════════════════════════════════════════════════════════════════════
# Fig 5: RSB Gap Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig5():
    rsb = pd.read_csv(DATA / "e4_rsb_full.csv")
    rsb = rsb[rsb["metric"].isin(PAPER_METRICS)]
    summary = rsb.groupby(["score", "metric"])["rsb_gap"].mean().reset_index()
    pivot = summary.pivot(index="score", columns="metric", values="rsb_gap")

    col_labels = {m: METRIC_LABELS[m] for m in PAPER_METRICS}
    pivot = pivot.rename(index=SCORE_LABELS, columns=col_labels)
    row_order = [r for r in ["SOFA", "qSOFA", "APACHE-II", "NEWS2"] if r in pivot.index]
    col_order = [c for c in [METRIC_LABELS[m] for m in PAPER_METRICS] if c in pivot.columns]
    pivot = pivot.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="PuBu",
                vmin=0, linewidths=2, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"},
                cbar_kws={"label": "RSB Gap", "shrink": 0.82, "aspect": 15},
                ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticklabels(row_order, rotation=0, fontsize=12, va="center")
    ax.set_xticklabels(col_order, rotation=0, ha="center", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig5_rsb_gap_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# Fig 6: ML Improvement — per-metric grouped bars (matches paper Table 5)
# ═══════════════════════════════════════════════════════════════════════
def fig6():
    imp = pd.read_csv(DATA / "e5_ml_improvement_full.csv")
    imp = imp[imp["metric"].isin(PAPER_METRICS)]

    # Panel A: scatter of score gap vs ML gap per metric
    summary = imp.groupby(["score", "metric"]).agg(
        score_gap=("score_fairness_gap", "mean"),
        ml_gap=("ml_fairness_gap", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) Scatter
    ax = axes[0]
    marker_map = {"auroc_gap": "o", "cal_gap": "s", "eod": "D", "ppg": "^"}
    for score_name in SCORE_ORDER:
        s = summary[summary["score"] == score_name]
        for _, row in s.iterrows():
            m = row["metric"]
            ax.scatter(row["score_gap"], row["ml_gap"],
                       color=SCORE_COLORS[score_name],
                       marker=marker_map.get(m, "o"),
                       s=80, zorder=3, edgecolors="0.3", linewidth=0.4)
    # Score legend
    for sn in SCORE_ORDER:
        ax.scatter([], [], color=SCORE_COLORS[sn], marker="o", s=60,
                   label=SCORE_LABELS[sn], edgecolors="0.3", linewidth=0.4)
    # Metric legend
    for m, mk in marker_map.items():
        ax.scatter([], [], color="0.5", marker=mk, s=50,
                   label=METRIC_LABELS[m], edgecolors="0.3", linewidth=0.4)
    lim = max(summary["score_gap"].max(), summary["ml_gap"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", alpha=0.25, linewidth=1, zorder=1)
    ax.set_xlabel("Classical Score Fairness Gap")
    ax.set_ylabel("GRU Fairness Gap")
    ax.set_title("(a) Score vs. ML Fairness Gap")
    ax.legend(fontsize=8, ncol=2, loc="upper left",
              handletextpad=0.3, columnspacing=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid()

    # (b) Per-metric improvement by score — grouped bars
    ax = axes[1]
    pct = imp.groupby(["score", "metric"])["pct_improvement"].mean().reset_index()
    metrics_in_data = [m for m in PAPER_METRICS if m in pct["metric"].unique()]
    x = np.arange(len(SCORE_ORDER))
    n_metrics = len(metrics_in_data)
    width = 0.8 / n_metrics
    metric_colors = {"auroc_gap": "#4393c3", "cal_gap": "#92c5de",
                     "eod": "#d6604d", "ppg": "#fddbc7"}

    for i, m in enumerate(metrics_in_data):
        vals = []
        for sn in SCORE_ORDER:
            row = pct[(pct["score"] == sn) & (pct["metric"] == m)]
            vals.append(row["pct_improvement"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width - (n_metrics - 1) * width / 2, vals, width,
                      label=METRIC_LABELS[m], color=metric_colors.get(m, "gray"),
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS[s] for s in SCORE_ORDER])
    ax.set_ylabel("Fairness Improvement (%)")
    ax.set_title("(b) ML Improvement by Metric")
    ax.axhline(y=0, color="0.4", linestyle="-", linewidth=0.6)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.grid(axis="y")

    fig.tight_layout()
    _save(fig, "fig6_ml_improvement")


# ═══════════════════════════════════════════════════════════════════════
# Fig 7: Score-Conditional Mortality — 2×2 layout (was unreadable 1×4)
# ═══════════════════════════════════════════════════════════════════════
def fig7():
    scm = pd.read_csv(DATA / "e6_score_conditional_mortality.csv")
    scores = [s for s in SCORE_ORDER if s in scm["score"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    groups = sorted(scm["group"].unique())
    palette = sns.color_palette("husl", len(groups))
    group_colors = dict(zip(groups, palette))

    for i, score_name in enumerate(scores):
        ax = axes[i]
        sub = scm[scm["score"] == score_name]

        for grp in groups:
            g = sub[sub["group"] == grp].copy()
            if g.empty:
                continue
            try:
                g["_sort"] = g["score_bin"].apply(
                    lambda x: float(str(x).split("-")[0].replace("+", "").strip()))
                g = g.sort_values("_sort")
            except Exception:
                g = g.sort_values("score_bin")

            ax.plot(range(len(g)), g["mortality_rate"],
                    "o-", color=group_colors[grp], label=grp,
                    markersize=4, linewidth=1.6)
            se = np.sqrt(g["mortality_rate"] * (1 - g["mortality_rate"])
                         / g["n"].clip(1))
            ax.fill_between(range(len(g)),
                            (g["mortality_rate"] - se).clip(0),
                            (g["mortality_rate"] + se).clip(0, 1),
                            color=group_colors[grp], alpha=0.10)

        first_grp = sub[sub["group"] == groups[0]].copy()
        try:
            first_grp["_sort"] = first_grp["score_bin"].apply(
                lambda x: float(str(x).split("-")[0].replace("+", "").strip()))
            first_grp = first_grp.sort_values("_sort")
        except Exception:
            first_grp = first_grp.sort_values("score_bin")

        ax.set_xticks(range(len(first_grp)))
        ax.set_xticklabels(list(first_grp["score_bin"]), rotation=45,
                           ha="right", fontsize=9)
        ax.set_xlabel(f"{SCORE_LABELS[score_name]} Score")
        ax.set_ylabel("Observed Mortality Rate")
        ax.set_title(SCORE_LABELS[score_name])
        ax.set_ylim(0, min(1.0, sub["mortality_rate"].max() * 1.3))
        ax.grid()
        if i == 0:
            ax.legend(title="Age Group", fontsize=8, title_fontsize=9,
                      loc="upper left", ncol=2)

    for j in range(len(scores), 4):
        axes[j].set_visible(False)

    fig.tight_layout(h_pad=2.5, w_pad=2)
    _save(fig, "fig7_score_conditional_mortality")


# ═══════════════════════════════════════════════════════════════════════
# Fig 8: Clinical Thresholds — enlarged with better colors
# ═══════════════════════════════════════════════════════════════════════
def fig8():
    thresh = pd.read_csv(DATA / "e7_clinical_thresholds.csv")
    sub = thresh[thresh["axis"] == "age_group"].copy()
    if sub.empty:
        return

    scores = ["sofa", "qsofa", "apache2", "news2"]
    primary_thresh = {"sofa": 6, "qsofa": 2, "apache2": 20, "news2": 9}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col_idx, score_name in enumerate(scores):
        t = primary_thresh[score_name]
        s = sub[(sub["score"] == score_name) & (sub["threshold"] == t)]
        s = s[s["group"] != "overall"].copy()
        # Sort by age group
        age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        s["_order"] = s["group"].apply(
            lambda g: age_order.index(g) if g in age_order else 99)
        s = s.sort_values("_order")

        for row_idx, metric in enumerate(["sensitivity", "specificity"]):
            ax = axes[row_idx, col_idx]
            if s.empty:
                continue
            groups = s["group"].tolist()
            vals = s[metric].tolist()

            # Diverging color: blue (good) to red (bad)
            norm_vals = np.array(vals)
            cmap = plt.cm.RdYlBu
            colors = cmap((norm_vals - 0.3) / 0.7)  # normalize 0.3-1.0 range

            bars = ax.barh(range(len(groups)), vals, color=colors,
                           edgecolor="white", linewidth=0.6, height=0.65)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels(groups, fontsize=11)
            ax.set_xlim(0, 1.08)
            ax.set_xlabel(metric.capitalize())
            mean_val = np.mean(vals)
            ax.axvline(mean_val, color="0.3", linestyle="--",
                       linewidth=0.8, alpha=0.5)
            if row_idx == 0:
                ax.set_title(f"{SCORE_LABELS[score_name]} ≥ {t}")

            for bar, val in zip(bars, vals):
                ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=10)

    axes[0, 0].set_ylabel("Sensitivity", fontweight="bold", fontsize=13)
    axes[1, 0].set_ylabel("Specificity", fontweight="bold", fontsize=13)
    fig.tight_layout(h_pad=2, w_pad=1.5)
    _save(fig, "fig8_clinical_thresholds_age_group")


# ═══════════════════════════════════════════════════════════════════════
# Fig 9: SOFA Components — single-color gradient sorted by value
# ═══════════════════════════════════════════════════════════════════════
def fig9():
    comp = pd.read_csv(DATA / "sofa_components.csv")
    gap_df = comp[comp["group"] == "_gap"].copy()
    if gap_df.empty:
        return

    comp_labels = {
        "respiratory": "Respiratory (PF ratio)",
        "coagulation": "Coagulation (Platelets)",
        "liver": "Liver (Bilirubin)",
        "cardiovascular": "Cardiovascular (MAP)",
        "cns": "CNS (GCS)",
        "renal": "Renal (Creatinine)",
    }

    # Sort by gap value for visual clarity
    gap_df = gap_df.sort_values("auroc", ascending=True)
    comps = gap_df["component"].tolist()
    gaps = gap_df["auroc"].tolist()
    labels = [comp_labels.get(c, c) for c in comps]

    fig, ax = plt.subplots(figsize=(8, 4))
    # Single-hue gradient: darker = larger gap
    norm = plt.Normalize(vmin=min(gaps) * 0.8, vmax=max(gaps) * 1.1)
    colors = plt.cm.Blues(norm(gaps))

    bars = ax.barh(range(len(comps)), gaps, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.6)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("AUROC Gap (max − min across age groups)")
    ax.set_title("SOFA Component Attribution: Age-Related AUROC Gaps")

    for bar, val in zip(bars, gaps):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, fontweight="bold")

    # Reference line for composite SOFA gap
    ax.axvline(x=0.129, color="#b2182b", linestyle="--", linewidth=1,
               alpha=0.7, label="Composite SOFA gap (0.129)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, max(gaps) * 1.22)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, "fig9_sofa_components")


# ═══════════════════════════════════════════════════════════════════════
# Fig 10: Hospital-Stratified Race (Simpson's Paradox)
# ═══════════════════════════════════════════════════════════════════════
def fig10():
    hosp = pd.read_csv(DATA / "hospital_stratified_race.csv")
    summary = pd.read_csv(DATA / "hospital_stratified_summary.csv")
    if hosp.empty or summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) Violin
    ax = axes[0]
    scores = [s for s in SCORE_ORDER if s in hosp["score"].unique()]
    data_by_score = [hosp[hosp["score"] == s]["race_auroc_gap"].values
                     for s in scores]
    parts = ax.violinplot(data_by_score, positions=range(len(scores)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(SCORE_COLORS[scores[i]])
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("0.2")
    parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([SCORE_LABELS[s] for s in scores])
    ax.set_ylabel("Within-Hospital Race AUROC Gap")
    ax.set_title("(a) Within-Hospital Gap Distribution")
    ax.set_ylim(0, None)
    ax.grid(axis="y")

    # (b) Side-by-side comparison with ratio annotations
    ax = axes[1]
    x = np.arange(len(summary))
    width = 0.32
    within = summary["within_hospital_gap_wmean"].values
    agg = summary["aggregate_gap"].values
    b1 = ax.bar(x - width / 2, within, width,
                label="Within-hospital (weighted)",
                color="#2166ac", alpha=0.85, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + width / 2, agg, width,
                label="Aggregate",
                color="#b2182b", alpha=0.85, edgecolor="white", linewidth=0.6)

    # Add ratio annotations
    for i in range(len(summary)):
        if agg[i] > 0:
            ratio = within[i] / agg[i]
            ax.text(x[i], within[i] + 0.005, f"{ratio:.1f}×",
                    ha="center", fontsize=10, fontweight="bold", color="#2166ac")

    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS.get(s, s) for s in summary["score"]])
    ax.set_ylabel("Race AUROC Gap")
    ax.set_title("(b) Simpson's Paradox: Within vs. Aggregate")
    ax.legend(fontsize=10)
    ax.grid(axis="y")

    fig.tight_layout(w_pad=2)
    _save(fig, "fig10_hospital_stratified_race")


# ═══════════════════════════════════════════════════════════════════════
# FigS2: GRU vs FAFT RSB Comparison — enlarged 2-row
# ═══════════════════════════════════════════════════════════════════════
def figS2():
    try:
        gru_rsb = pd.read_csv(DATA / "e4_rsb_full.csv")
        faft_rsb = pd.read_csv(DATA / "e4_rsb_faft.csv")
    except FileNotFoundError:
        print("  Missing RSB data for figS2"); return

    gru_rsb = gru_rsb[gru_rsb["metric"].isin(PAPER_METRICS)]
    faft_rsb = faft_rsb[faft_rsb["metric"].isin(PAPER_METRICS)]

    def _pivot(df):
        grp = df.groupby(["score", "metric"])["rsb_gap"].mean().reset_index()
        pivot = grp.pivot(index="score", columns="metric", values="rsb_gap")
        row_order = [s for s in SCORE_ORDER if s in pivot.index]
        col_order = [m for m in PAPER_METRICS if m in pivot.columns]
        pivot = pivot.loc[row_order, col_order]
        pivot.index = [SCORE_LABELS.get(s, s) for s in pivot.index]
        pivot.columns = [METRIC_LABELS.get(m, m) for m in pivot.columns]
        return pivot

    gru_p = _pivot(gru_rsb)
    faft_p = _pivot(faft_rsb)
    delta = faft_p - gru_p

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    panels = [
        (gs[0, 0], gru_p,  "(a) GRU RSB Gap",          "YlOrRd"),
        (gs[0, 1], faft_p, "(b) FAFT RSB Gap",          "YlOrRd"),
        (gs[1, :], delta,  "(c) ΔRSB (FAFT − GRU): negative = FAFT reduces bias", "div"),
    ]

    vmax = max(gru_p.values.max(), faft_p.values.max())

    for spec, data, title, cmap_name in panels:
        ax = fig.add_subplot(spec)
        if cmap_name == "div":
            sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn_r",
                        center=0, vmin=-0.05, vmax=0.05,
                        linewidths=1.5, linecolor="white",
                        annot_kws={"size": 13, "weight": "bold"},
                        cbar_kws={"shrink": 0.7})
        else:
            sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap=cmap_name,
                        vmin=0, vmax=vmax,
                        linewidths=1.5, linecolor="white",
                        annot_kws={"size": 13, "weight": "bold"},
                        cbar_kws={"shrink": 0.7})
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)

    _save(fig, "figS2_model_comparison_rsb")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Regenerating ATLAS figures (NeurIPS quality)...")
    fig1()
    fig2_race()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    figS2()
    print("\nDone.")
