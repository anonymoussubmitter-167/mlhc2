# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""Publication-quality figure generation for ATLAS paper."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
from ..data.config import FIGURES_DIR, PAPER_FIGURES_DIR

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (7, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SCORE_COLORS = {
    "sofa": "#1f77b4",
    "qsofa": "#ff7f0e",
    "apache2": "#2ca02c",
    "news2": "#d62728",
}

SCORE_LABELS = {
    "sofa": "SOFA",
    "qsofa": "qSOFA",
    "apache2": "APACHE-II",
    "news2": "NEWS2",
}


def _save(fig, name):
    """Save figure as PNG and PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(PAPER_FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: {name}.png/pdf")


def plot_auroc_gap_heatmap(gaps_df: pd.DataFrame):
    """Figure 1: Heatmap of AUROC gaps across scores x demographic axes."""
    pivot = gaps_df.pivot_table(index="score", columns="axis",
                                 values="auroc_gap", aggfunc="first")
    # Rename for display
    rename_axes = {
        "race_cat": "Race", "sex": "Sex", "age_group": "Age",
        "insurance_cat": "Insurance", "diag_type": "Diagnosis",
    }
    rename_scores = SCORE_LABELS
    pivot = pivot.rename(index=rename_scores, columns=rename_axes)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, ax=ax, linewidths=0.5,
                cbar_kws={"label": "AUROC Gap (max - min)"})
    ax.set_title("Performance Disparity: AUROC Gap by Score and Demographic Axis")
    ax.set_ylabel("")
    ax.set_xlabel("")
    _save(fig, "fig1_auroc_gap_heatmap")


def plot_subgroup_performance(audit_results: pd.DataFrame, axis: str = "race_cat"):
    """Figure 2: Bar chart of AUROC by subgroup for each score along one axis."""
    sub = audit_results[audit_results["axis"] == axis].copy()
    if sub.empty:
        print(f"  No data for axis={axis}")
        return

    axis_label = {"race_cat": "Race", "sex": "Sex", "age_group": "Age",
                  "insurance_cat": "Insurance", "diag_type": "Diagnosis"}.get(axis, axis)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scores = sub["score"].unique()
    groups = sub["group"].unique()
    x = np.arange(len(groups))
    width = 0.8 / len(scores)

    for i, score in enumerate(scores):
        s = sub[sub["score"] == score]
        vals = [s[s["group"] == g]["auroc"].values[0] if len(s[s["group"] == g]) > 0
                else np.nan for g in groups]
        # CIs
        lo = [s[s["group"] == g]["auroc_ci_lo"].values[0] if len(s[s["group"] == g]) > 0
              else np.nan for g in groups]
        hi = [s[s["group"] == g]["auroc_ci_hi"].values[0] if len(s[s["group"] == g]) > 0
              else np.nan for g in groups]
        yerr_lo = [v - l if not (np.isnan(v) or np.isnan(l)) else 0 for v, l in zip(vals, lo)]
        yerr_hi = [h - v if not (np.isnan(v) or np.isnan(h)) else 0 for v, h in zip(vals, hi)]

        ax.bar(x + i * width, vals, width, label=SCORE_LABELS.get(score, score),
               color=SCORE_COLORS.get(score, "gray"),
               yerr=[yerr_lo, yerr_hi], capsize=2, alpha=0.85)

    ax.set_xlabel(axis_label)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Mortality Prediction AUROC by {axis_label}")
    ax.set_xticks(x + width * (len(scores) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)
    _save(fig, f"fig2_auroc_by_{axis}")


def plot_calibration_curves(data: pd.DataFrame):
    """Figure 3: Calibration curves per score, split by race."""
    from .audit import _score_to_prob
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    scores = ["sofa", "qsofa", "apache2", "news2"]
    races = data["race_cat"].unique()
    race_colors = dict(zip(sorted(races), sns.color_palette("Set2", len(races))))

    for idx, score_name in enumerate(scores):
        ax = axes[idx]
        prob = _score_to_prob(data[score_name], data["mortality"])

        for race in sorted(races):
            mask = data["race_cat"] == race
            if mask.sum() < 20:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(
                    data.loc[mask, "mortality"], prob[mask],
                    n_bins=8, strategy="quantile")
                ax.plot(mean_pred, frac_pos, "o-",
                        color=race_colors[race], label=race, markersize=4)
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_title(SCORE_LABELS[score_name])
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Calibration by Race for Each Acuity Score", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_calibration_by_race")


def plot_asd_results(asd_results: dict):
    """Figure 4: Adversarial subgroup discovery — error concentration."""
    fig, axes = plt.subplots(1, len(asd_results), figsize=(4 * len(asd_results), 4))
    if len(asd_results) == 1:
        axes = [axes]

    for ax, (score_name, res) in zip(axes, asd_results.items()):
        sgs = res["vulnerable_subgroups"]
        if not sgs:
            ax.text(0.5, 0.5, "No subgroups found", ha="center", va="center")
            ax.set_title(SCORE_LABELS.get(score_name, score_name))
            continue

        names = [f"SG{i+1}\n(n={sg['n']})" for i, sg in enumerate(sgs)]
        concentrations = [sg["concentration_ratio"] for sg in sgs]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sgs)))

        ax.barh(range(len(sgs)), concentrations, color=colors)
        ax.set_yticks(range(len(sgs)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Error Concentration Ratio")
        ax.set_title(SCORE_LABELS.get(score_name, score_name))
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Adversarial Subgroup Discovery: Error Concentration", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_asd_error_concentration")


def plot_rsb_gaps(rsb_df: pd.DataFrame):
    """Figure 5: RSB gap visualization — how much benchmarking is distorted."""
    # Pivot to score x metric, averaged over demographic axes
    summary = rsb_df.groupby(["score", "metric"])["rsb_gap"].mean().reset_index()
    pivot = summary.pivot(index="score", columns="metric", values="rsb_gap")

    rename_scores = SCORE_LABELS
    rename_metrics = {"eod": "Equalized\nOdds", "ppg": "Predictive\nParity",
                      "cal_gap": "Calibration\nGap", "auroc_gap": "AUROC\nGap"}
    pivot = pivot.rename(index=rename_scores, columns=rename_metrics)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="PuBu",
                vmin=0, ax=ax, linewidths=0.5,
                cbar_kws={"label": "RSB Gap"})
    ax.set_title("Reference Standard Bias: Fairness Evaluation Distortion")
    ax.set_ylabel("")
    _save(fig, "fig5_rsb_gap_heatmap")


def plot_ml_improvement(improvement_df: pd.DataFrame):
    """Figure 6: ML fairness improvement over classical scores."""
    summary = improvement_df.groupby(["score", "metric"]).agg(
        score_gap=("score_fairness_gap", "mean"),
        ml_gap=("ml_fairness_gap", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Score gap vs ML gap
    ax = axes[0]
    for score_name in summary["score"].unique():
        s = summary[summary["score"] == score_name]
        ax.scatter(s["score_gap"], s["ml_gap"],
                   color=SCORE_COLORS.get(score_name, "gray"),
                   label=SCORE_LABELS.get(score_name, score_name),
                   s=80, zorder=3)
    lim = max(summary["score_gap"].max(), summary["ml_gap"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="No improvement")
    ax.set_xlabel("Classical Score Fairness Gap")
    ax.set_ylabel("ML Model Fairness Gap")
    ax.set_title("(a) ML vs. Classical Score Equity")
    ax.legend(fontsize=8)

    # Panel B: Percent improvement by score
    ax = axes[1]
    pct = improvement_df.groupby("score")["pct_improvement"].mean()
    bars = ax.bar(range(len(pct)), pct.values,
                  color=[SCORE_COLORS.get(s, "gray") for s in pct.index])
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([SCORE_LABELS.get(s, s) for s in pct.index])
    ax.set_ylabel("Mean Fairness Improvement (%)")
    ax.set_title("(b) ML Fairness Improvement over Classical Scores")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig6_ml_improvement")


def plot_score_distributions(data: pd.DataFrame):
    """Supplementary: Score distributions by race."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    scores = ["sofa", "qsofa", "apache2", "news2"]
    for ax, score_name in zip(axes.ravel(), scores):
        for race in sorted(data["race_cat"].unique()):
            subset = data[data["race_cat"] == race][score_name]
            ax.hist(subset, bins=20, alpha=0.5, label=race, density=True)
        ax.set_title(SCORE_LABELS[score_name])
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        if score_name == "sofa":
            ax.legend(fontsize=8)
    fig.suptitle("Score Distributions by Race", y=1.01)
    fig.tight_layout()
    _save(fig, "figS1_score_distributions")


def plot_score_conditional_mortality(scm_df: pd.DataFrame,
                                     demo_axis: str = "age_group"):
    """Figure 7: Same score value → different mortality rates across groups.

    For each score, show mortality rate by score bin x demographic group.
    This is the mechanistic evidence for score bias.
    """
    scores = scm_df["score"].unique()
    n_scores = len(scores)
    fig, axes = plt.subplots(1, n_scores, figsize=(4 * n_scores, 4.5))
    if n_scores == 1:
        axes = [axes]

    axis_label = {"age_group": "Age Group", "race_cat": "Race",
                  "sex": "Sex", "diag_type": "Diagnosis"}.get(demo_axis, demo_axis)

    groups = sorted(scm_df["group"].unique())
    palette = sns.color_palette("tab10", len(groups))
    group_colors = dict(zip(groups, palette))

    for ax, score_name in zip(axes, scores):
        sub = scm_df[scm_df["score"] == score_name]
        bins = sub["score_bin"].unique()

        for grp in groups:
            g = sub[sub["group"] == grp]
            if g.empty:
                continue
            # Sort bins by their first numeric value
            g = g.copy()
            try:
                g["_sort"] = g["score_bin"].apply(
                    lambda x: float(str(x).split("-")[0].replace("+", "").strip())
                )
                g = g.sort_values("_sort")
            except Exception:
                g = g.sort_values("score_bin")

            ax.plot(range(len(g)), g["mortality_rate"],
                    "o-", color=group_colors[grp], label=grp,
                    markersize=5, linewidth=1.8)
            # Add band for sample size uncertainty (simple ±1 SE)
            se = np.sqrt(g["mortality_rate"] * (1 - g["mortality_rate"]) / g["n"].clip(1))
            ax.fill_between(range(len(g)),
                            (g["mortality_rate"] - se).clip(0),
                            (g["mortality_rate"] + se).clip(0, 1),
                            color=group_colors[grp], alpha=0.12)

        # x-axis ticks from bins of the first group
        first_grp = sub[sub["group"] == groups[0]].copy()
        try:
            first_grp["_sort"] = first_grp["score_bin"].apply(
                lambda x: float(str(x).split("-")[0].replace("+", "").strip())
            )
            first_grp = first_grp.sort_values("_sort")
        except Exception:
            first_grp = first_grp.sort_values("score_bin")
        n_bins = len(first_grp)
        ax.set_xticks(range(n_bins))
        tick_labels = list(first_grp["score_bin"])
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel(f"{SCORE_LABELS.get(score_name, score_name)} Score")
        ax.set_ylabel("Observed Mortality Rate")
        ax.set_title(SCORE_LABELS.get(score_name, score_name))
        ax.set_ylim(0, min(1.0, sub["mortality_rate"].max() * 1.3))
        if score_name == scores[0]:
            ax.legend(title=axis_label, fontsize=8, title_fontsize=8)

    fig.suptitle(f"Score-Conditional Mortality by {axis_label}", y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_score_conditional_mortality")


def plot_clinical_thresholds(threshold_df: pd.DataFrame,
                              axis: str = "age_group"):
    """Figure 8: Sensitivity and specificity at clinical thresholds by group."""
    sub = threshold_df[threshold_df["axis"] == axis].copy()
    if sub.empty:
        print(f"  No threshold data for axis={axis}")
        return

    axis_label = {"age_group": "Age Group", "race_cat": "Race",
                  "sex": "Sex", "diag_type": "Diagnosis"}.get(axis, axis)

    scores = ["sofa", "qsofa", "apache2", "news2"]
    # Use primary threshold per score
    primary_thresh = {"sofa": 6, "qsofa": 2, "apache2": 20, "news2": 9}

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col_idx, score_name in enumerate(scores):
        thresh = primary_thresh.get(score_name, None)
        if thresh is None:
            continue
        s = sub[(sub["score"] == score_name) & (sub["threshold"] == thresh)]
        s = s[s["group"] != "overall"]

        # Row 0: sensitivity
        ax = axes[0, col_idx]
        if not s.empty:
            groups = s["group"].tolist()
            sens = s["sensitivity"].tolist()
            colors = plt.cm.RdYlGn([v for v in sens])
            ax.barh(range(len(groups)), sens, color=colors)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels(groups, fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Sensitivity")
            ax.set_title(f"{SCORE_LABELS.get(score_name, score_name)}\n≥{thresh}")
            ax.axvline(s["sensitivity"].mean(), color="k", linestyle="--",
                       alpha=0.4, linewidth=1)

        # Row 1: specificity
        ax = axes[1, col_idx]
        if not s.empty:
            spec = s["specificity"].tolist()
            colors = plt.cm.RdYlGn([v for v in spec])
            ax.barh(range(len(groups)), spec, color=colors)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels(groups, fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Specificity")
            ax.axvline(s["specificity"].mean(), color="k", linestyle="--",
                       alpha=0.4, linewidth=1)

    axes[0, 0].set_ylabel("Sensitivity")
    axes[1, 0].set_ylabel("Specificity")
    fig.suptitle(f"Clinical Threshold Performance by {axis_label}", y=1.02)
    fig.tight_layout()
    _save(fig, f"fig8_clinical_thresholds_{axis}")


def plot_sofa_components(comp_df: pd.DataFrame):
    """Figure 9: SOFA component AUROC gaps by age group (mechanistic)."""
    gap_df = comp_df[comp_df["group"] == "_gap"].copy()
    if gap_df.empty:
        print("  No SOFA component gap data.")
        return

    comp_labels = {
        "respiratory": "Respiratory\n(PF ratio)",
        "coagulation": "Coagulation\n(Platelets)",
        "liver": "Liver\n(Bilirubin)",
        "cardiovascular": "Cardiovascular\n(MAP)",
        "cns": "CNS\n(GCS)",
        "renal": "Renal\n(Creatinine)",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    comps = gap_df["component"].tolist()
    gaps = gap_df["auroc"].tolist()
    labels = [comp_labels.get(c, c) for c in comps]

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(comps)))
    bars = ax.barh(range(len(comps)), gaps, color=colors)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUROC Gap (max - min across age groups)")
    ax.set_title("SOFA Component Attribution: Age-Related AUROC Gaps")

    for i, (bar, val) in enumerate(zip(bars, gaps)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, max(gaps) * 1.25)
    fig.tight_layout()
    _save(fig, "fig9_sofa_components")


def plot_hospital_stratified_race(hosp_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Figure 10: Within-hospital vs. aggregate race AUROC gaps (Simpson's paradox)."""
    if hosp_df.empty or summary_df.empty:
        print("  No hospital-stratified data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: violin plot of within-hospital gaps by score
    ax = axes[0]
    scores = hosp_df["score"].unique()
    data_by_score = [hosp_df[hosp_df["score"] == s]["race_auroc_gap"].values
                     for s in scores]
    parts = ax.violinplot(data_by_score, positions=range(len(scores)),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([SCORE_LABELS.get(s, s) for s in scores])
    ax.set_ylabel("Within-Hospital Race AUROC Gap")
    ax.set_title("(a) Distribution of Within-Hospital Race Gaps")
    ax.set_ylim(0, None)

    # Panel B: within-hospital vs aggregate gap comparison
    ax = axes[1]
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["within_hospital_gap_wmean"],
           width, label="Within-hospital (weighted)", color="#2166ac", alpha=0.8)
    ax.bar(x + width / 2, summary_df["aggregate_gap"],
           width, label="Aggregate gap", color="#d62728", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS.get(s, s) for s in summary_df["score"]])
    ax.set_ylabel("Race AUROC Gap")
    ax.set_title("(b) Within-Hospital vs. Aggregate Race Gap\n(Simpson's Paradox)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "fig10_hospital_stratified_race")


def plot_model_comparison(gru_rsb: pd.DataFrame, faft_rsb: pd.DataFrame,
                           gru_improvement: pd.DataFrame, faft_improvement: pd.DataFrame):
    """figS2: GRU vs FAFT RSB gap comparison and ML improvement.

    Panel (a): RSB gap heatmap for GRU (full cohort)
    Panel (b): RSB gap heatmap for FAFT
    Panel (c): Delta RSB (FAFT - GRU), negative = FAFT reduces RSB
    """
    SCORE_LABELS = {"sofa": "SOFA", "qsofa": "qSOFA",
                    "apache2": "APACHE-II", "news2": "NEWS2"}
    METRIC_LABELS = {"eod": "EOD", "ppg": "PPG",
                     "cal_gap": "Cal. gap", "auroc_gap": "AUROC gap", "ece_gap": "ECE gap"}

    def _pivot(df):
        grp = df.groupby(["score", "metric"])["rsb_gap"].mean().reset_index()
        pivot = grp.pivot(index="score", columns="metric", values="rsb_gap")
        pivot.index = [SCORE_LABELS.get(s, s) for s in pivot.index]
        pivot.columns = [METRIC_LABELS.get(m, m) for m in pivot.columns]
        return pivot

    gru_p  = _pivot(gru_rsb)
    faft_p = _pivot(faft_rsb)
    delta  = faft_p - gru_p  # negative = FAFT lower RSB

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax = max(gru_p.values.max(), faft_p.values.max())

    for ax, data, title in zip(axes,
                                [gru_p, faft_p, delta],
                                ["(a) GRU RSB Gap", "(b) FAFT RSB Gap",
                                 "(c) ΔRSB (FAFT − GRU)\nnegative = FAFT better"]):
        if "ΔRSB" in title:
            sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn_r",
                        center=0, vmin=-0.05, vmax=0.05,
                        linewidths=0.5, cbar_kws={"shrink": 0.8})
        else:
            sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                        vmin=0, vmax=vmax,
                        linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Reference Standard Bias: GRU vs. FAFT", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "figS2_model_comparison_rsb")
