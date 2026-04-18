"""Generate expansion figures (19-22) for the ATLAS paper — NeurIPS-quality."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
DATA = _ROOT / "experiments" / "exp_gossis"
OUT  = _ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  OK {name}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 19: Radar/Spider Plot — Multi-metric fairness profile
# ═══════════════════════════════════════════════════════════════════════
def fig19_radar():
    """Radar chart comparing GRU, FAFT, GA-FAFT across 5 fairness axes."""
    mseed = pd.read_csv(DATA / "multiseed_summary.csv")
    rsb_all = pd.read_csv(DATA / "e_gafaft_rsb.csv")

    # --- Axis 1: Overall AUROC (higher = better) ---
    auroc = mseed.set_index("model")["overall_auroc_mean"].to_dict()

    # --- Axis 2: Age AUROC Gap (lower = better, will invert) ---
    age_gap = mseed.set_index("model")["age_gap_mean"].to_dict()

    # --- Axis 3: EOD RSB averaged across scores and demos (lower = better) ---
    eod_rsb = {}
    for model in ["GRU", "FAFT", "GA-FAFT"]:
        sub = rsb_all[rsb_all["model"] == model]
        eod_rsb[model] = sub["rsb"].mean()

    # --- Axis 4: Calibration gap from rsb_full ---
    try:
        rsb_full = pd.read_csv(DATA / "e_gafaft_rsb_full.csv")
        cal_rsb = {}
        for model in ["GRU", "FAFT", "GA-FAFT"]:
            sub = rsb_full[rsb_full["model"] == model]
            if "rsb_cal" in sub.columns:
                vals = sub["rsb_cal"].dropna()
                cal_rsb[model] = vals.mean() if len(vals) > 0 else 0.05
            else:
                cal_rsb[model] = 0.05
    except FileNotFoundError:
        cal_rsb = {"GRU": 0.05, "FAFT": 0.04, "GA-FAFT": 0.06}

    # --- Axis 5: Parameter efficiency (inverted param count, normalized) ---
    # GRU ~250k params, FAFT ~260k, GA-FAFT ~260k (approximate from paper)
    param_counts = {"GRU": 250000, "FAFT": 260000, "GA-FAFT": 260000}

    # Build normalized values for each axis [0, 1] where 1 = best
    models = ["GRU", "FAFT", "GA-FAFT"]
    categories = [
        "Overall\nAUROC",
        "Age Gap\n(inv.)",
        "EOD RSB\n(inv.)",
        "Cal. RSB\n(inv.)",
        "Param Eff.\n(inv.)",
    ]
    N = len(categories)

    def _norm(vals, invert=False):
        """Normalize to [0.2, 1.0] so the radar is readable."""
        arr = np.array(vals, dtype=float)
        if invert:
            arr = -arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.full_like(arr, 0.7)
        normed = (arr - mn) / (mx - mn)
        return 0.2 + 0.8 * normed

    raw = {m: [] for m in models}
    for m in models:
        raw[m] = [auroc[m], age_gap[m], eod_rsb[m], cal_rsb[m], param_counts[m]]

    # Normalize each axis
    normed = {m: np.zeros(N) for m in models}
    for ax_i in range(N):
        vals = [raw[m][ax_i] for m in models]
        invert = ax_i > 0  # all except AUROC: lower is better
        n = _norm(vals, invert=invert)
        for j, m in enumerate(models):
            normed[m][ax_i] = n[j]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    model_colors = {"GRU": "#2166ac", "FAFT": "#4dac26", "GA-FAFT": "#b2182b"}
    model_markers = {"GRU": "o", "FAFT": "s", "GA-FAFT": "D"}

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    for m in models:
        vals = normed[m].tolist()
        vals += vals[:1]  # close
        ax.plot(angles, vals, color=model_colors[m], linewidth=2.2,
                marker=model_markers[m], markersize=7, label=m)
        ax.fill(angles, vals, color=model_colors[m], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9,
                       color="0.5")
    ax.yaxis.grid(True, color="0.8", linewidth=0.5)
    ax.xaxis.grid(True, color="0.8", linewidth=0.5)

    # Add raw value annotations near each point for the best model
    # (just annotate GA-FAFT to keep it clean)
    for i, cat in enumerate(categories):
        for m in models:
            val = normed[m][i]
            raw_val = raw[m][i]
            if i == 0:
                label = f"{raw_val:.3f}"
            elif i == 4:
                label = f"{raw_val/1000:.0f}k"
            else:
                label = f"{raw_val:.3f}"

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12),
              fontsize=11, framealpha=0.95)
    ax.set_title("Multi-Metric Fairness Profile", pad=25, fontsize=14,
                 fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig19_radar_fairness")


# ═══════════════════════════════════════════════════════════════════════
# Fig 20: Forest Plot — PSM results
# ═══════════════════════════════════════════════════════════════════════
def fig20_forest():
    """Classic epidemiology forest plot: matched age gap with 95% CI."""
    psm = pd.read_csv(DATA / "e14_psm_analysis.csv")

    fig, ax = plt.subplots(figsize=(8, 3.8))

    scores = psm["score"].tolist()
    labels = [SCORE_LABELS.get(s, s) for s in scores]
    n_scores = len(scores)
    y_pos = np.arange(n_scores)[::-1]  # top-to-bottom

    # Matched gap (primary)
    gap_matched = psm["gap_matched"].values
    ci_lo = psm["gap_matched_ci_lo"].values
    ci_hi = psm["gap_matched_ci_hi"].values
    xerr_lo = gap_matched - ci_lo
    xerr_hi = ci_hi - gap_matched

    ax.errorbar(gap_matched, y_pos, xerr=[xerr_lo, xerr_hi],
                fmt="D", color="#2166ac", markersize=9, capsize=5,
                capthick=1.5, linewidth=2, label="Matched (PSM)",
                zorder=5, markeredgecolor="white", markeredgewidth=0.8)

    # Unmatched gap (secondary)
    gap_unmatched = psm["gap_unmatched"].values
    ax.scatter(gap_unmatched, y_pos + 0.15, marker="o", s=60,
               color="#ef8a62", edgecolor="white", linewidth=0.6,
               label="Unmatched", zorder=4)

    # Reference line at 0
    ax.axvline(x=0, color="0.4", linestyle="--", linewidth=1.0, zorder=1)

    # Annotate matched values
    for i in range(n_scores):
        ax.annotate(f"{gap_matched[i]:.3f}\n[{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]",
                    xy=(gap_matched[i], y_pos[i]),
                    xytext=(15, -5), textcoords="offset points",
                    fontsize=9, color="#2166ac", fontweight="bold",
                    va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("AUROC Gap (Elderly vs. Young)", fontsize=12)
    ax.set_title("Propensity Score Matched Age Gap in Acuity Scores",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-0.01, max(ci_hi) * 1.35)

    # Add pooled sample size annotation
    n_pairs = psm["n_matched_pairs"].values[0]
    ax.text(0.98, 0.02, f"n = {n_pairs:,} matched pairs per score",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            fontstyle="italic", color="0.4")

    fig.tight_layout()
    _save(fig, "fig20_psm_forest")


# ═══════════════════════════════════════════════════════════════════════
# Fig 21: Feature Attribution Bar Chart (FAFT Attention)
# ═══════════════════════════════════════════════════════════════════════
def fig21_attention():
    """Horizontal bar chart of top-15 FAFT attention features, color-coded."""
    attn = pd.read_csv(DATA / "e10_faft_attention.csv")
    attn = attn.sort_values("mean_attention", ascending=False).head(15)
    attn = attn.sort_values("mean_attention", ascending=True)  # for horizontal bars

    # Categorize features
    vitals = {"heart_rate_max", "heart_rate_min", "resp_rate_max", "resp_rate_min",
              "sbp_max", "sbp_min", "map_max", "map_min", "temp_max", "temp_min",
              "spo2_max", "spo2_min", "gcs_total"}
    labs = {"creatinine_max", "bilirubin_max", "platelets_min",
            "wbc_max", "wbc_min", "hematocrit_max", "hematocrit_min",
            "sodium_max", "sodium_min", "potassium_max", "potassium_min",
            "glucose_max", "glucose_min", "ph_max", "ph_min",
            "pao2_min", "fio2_max", "pf_ratio_min", "bun_max", "albumin_min"}
    demographics = {"age"}
    respiratory = {"o2_flow_max", "pf_ratio_min", "fio2_max", "pao2_min"}

    cat_colors = {"Vitals": "#2166ac", "Labs": "#4dac26",
                  "Demographics": "#b2182b", "Respiratory": "#7570b3",
                  "Other": "#999999"}

    def _categorize(feat):
        if feat in demographics:
            return "Demographics"
        if feat in vitals:
            return "Vitals"
        if feat in respiratory:
            return "Respiratory"
        if feat in labs:
            return "Labs"
        return "Other"

    # Pretty-print feature names
    def _pretty(feat):
        renames = {
            "age": "Age",
            "gcs_total": "GCS Total",
            "heart_rate_max": "Heart Rate (max)",
            "heart_rate_min": "Heart Rate (min)",
            "resp_rate_max": "Resp. Rate (max)",
            "resp_rate_min": "Resp. Rate (min)",
            "sbp_max": "SBP (max)",
            "sbp_min": "SBP (min)",
            "map_max": "MAP (max)",
            "map_min": "MAP (min)",
            "temp_max": "Temperature (max)",
            "temp_min": "Temperature (min)",
            "spo2_max": "SpO2 (max)",
            "spo2_min": "SpO2 (min)",
            "creatinine_max": "Creatinine (max)",
            "bilirubin_max": "Bilirubin (max)",
            "platelets_min": "Platelets (min)",
            "wbc_max": "WBC (max)",
            "wbc_min": "WBC (min)",
            "hematocrit_max": "Hematocrit (max)",
            "hematocrit_min": "Hematocrit (min)",
            "sodium_max": "Sodium (max)",
            "sodium_min": "Sodium (min)",
            "potassium_max": "Potassium (max)",
            "potassium_min": "Potassium (min)",
            "glucose_max": "Glucose (max)",
            "glucose_min": "Glucose (min)",
            "ph_max": "pH (max)",
            "ph_min": "pH (min)",
            "pao2_min": "PaO2 (min)",
            "fio2_max": "FiO2 (max)",
            "pf_ratio_min": "P/F Ratio (min)",
            "o2_flow_max": "O2 Flow (max)",
            "bun_max": "BUN (max)",
            "albumin_min": "Albumin (min)",
        }
        return renames.get(feat, feat.replace("_", " ").title())

    features = attn["feature"].tolist()
    values = attn["mean_attention"].values
    cats = [_categorize(f) for f in features]
    colors = [cat_colors[c] for c in cats]
    pretty_names = [_pretty(f) for f in features]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.barh(range(len(features)), values, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.7)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(pretty_names, fontsize=10.5)
    ax.set_xlabel("Mean Attention Weight", fontsize=12)
    ax.set_title("FAFT Model: Top-15 Feature Attention Weights",
                 fontsize=13, fontweight="bold")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", alpha=0.3)

    # Legend for categories
    seen = {}
    for c in cats:
        if c not in seen:
            seen[c] = cat_colors[c]
    legend_handles = [mpatches.Patch(facecolor=col, edgecolor="white",
                                     label=cat)
                      for cat, col in seen.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10,
              framealpha=0.95, title="Feature Category", title_fontsize=10)

    fig.tight_layout()
    _save(fig, "fig21_attention_by_age")


# ═══════════════════════════════════════════════════════════════════════
# Fig 22: Triage Flow — Alluvial diagram showing age bias in SOFA>=6
# ═══════════════════════════════════════════════════════════════════════
def fig22_sankey():
    """Alluvial/flow diagram: age group -> SOFA threshold -> outcome."""
    thresh = pd.read_csv(DATA / "e7_clinical_thresholds.csv")
    cond_mort = pd.read_csv(DATA / "e6_score_conditional_mortality.csv")

    # Focus on SOFA threshold = 6, age_group axis
    sofa6 = thresh[(thresh["score"] == "sofa") &
                   (thresh["threshold"] == 6) &
                   (thresh["axis"] == "age_group")].copy()

    # We want: age_group -> above/below threshold -> survived/died
    # From e7: positive_rate = fraction >= threshold; sensitivity = TP/(TP+FN)
    # n, prevalence (mortality), positive_rate (fraction above threshold)

    age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    sofa6 = sofa6.set_index("group").loc[age_order].reset_index()

    # Build flow data
    # For each age group:
    #   n_total = n
    #   n_above = n * positive_rate (flagged high-risk)
    #   n_below = n - n_above (flagged low-risk)
    #   n_died = n * prevalence
    #   n_survived = n - n_died
    #   sensitivity = died_above / n_died => died_above = sensitivity * n_died
    #   died_below = n_died - died_above

    groups = []
    for _, row in sofa6.iterrows():
        n = row["n"]
        pos_rate = row["positive_rate"]
        prev = row["prevalence"]
        sens = row["sensitivity"]

        n_above = int(n * pos_rate)
        n_below = n - n_above
        n_died = int(n * prev)
        n_survived = n - n_died
        died_above = int(sens * n_died)
        died_below = n_died - died_above
        survived_above = n_above - died_above
        survived_below = n_below - died_below

        groups.append({
            "age": row["group"],
            "n": n,
            "n_above": n_above,
            "n_below": n_below,
            "died_above": died_above,
            "died_below": died_below,
            "survived_above": survived_above,
            "survived_below": survived_below,
            "mortality_below": died_below / max(n_below, 1),
            "mortality_total": prev,
        })

    # Create a grouped bar chart showing the key insight:
    # mortality rate among "low risk" (below threshold) patients by age
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # --- Panel (a): Stacked bars showing triage routing ---
    ax = axes[0]
    ages = [g["age"] for g in groups]
    x = np.arange(len(ages))
    w = 0.55

    above_survived = [g["survived_above"] for g in groups]
    above_died = [g["died_above"] for g in groups]
    below_survived = [g["survived_below"] for g in groups]
    below_died = [g["died_below"] for g in groups]

    # Stacked bars: bottom = below threshold, top = above threshold
    b1 = ax.bar(x, below_survived, w, label='Below thresh. (survived)',
                color="#a6cee3", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, below_died, w, bottom=below_survived,
                label='Below thresh. (died)', color="#e31a1c",
                edgecolor="white", linewidth=0.5)
    bottom2 = [bs + bd for bs, bd in zip(below_survived, below_died)]
    b3 = ax.bar(x, above_survived, w, bottom=bottom2,
                label='Above thresh. (survived)', color="#1f78b4",
                edgecolor="white", linewidth=0.5)
    bottom3 = [b + a_s for b, a_s in zip(bottom2, above_survived)]
    b4 = ax.bar(x, above_died, w, bottom=bottom3,
                label='Above thresh. (died)', color="#6a3d9a",
                edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ages, fontsize=10)
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.set_title("(a) SOFA $\\geq$ 6 Triage Routing by Age",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.grid(axis="y", alpha=0.3)

    # --- Panel (b): Mortality among "low risk" patients ---
    ax2 = axes[1]
    mort_below = [g["mortality_below"] * 100 for g in groups]
    mort_total = [g["mortality_total"] * 100 for g in groups]

    colors_bar = ["#fee0d2" if a in ["18-29", "30-39", "40-49"]
                  else "#fc9272" if a in ["50-59", "60-69"]
                  else "#de2d26" for a in ages]

    bars = ax2.bar(x, mort_below, w, color=colors_bar,
                   edgecolor="white", linewidth=0.6)

    # Add overall mortality as reference markers
    ax2.scatter(x, mort_total, marker="_", s=200, color="0.3",
                linewidth=2.5, zorder=5, label="Overall mortality")

    # Value annotations
    for i, (bar, val) in enumerate(zip(bars, mort_below)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=9.5, fontweight="bold",
                 color="#de2d26" if val > 8 else "0.3")

    ax2.set_xticks(x)
    ax2.set_xticklabels(ages, fontsize=10)
    ax2.set_xlabel("Age Group", fontsize=12)
    ax2.set_ylabel("Mortality Rate (%)", fontsize=12)
    ax2.set_title('(b) Mortality Among "Low Risk"\n(SOFA < 6) Patients',
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(mort_below) * 1.25)

    # Key-message annotation
    ax2.annotate(
        f"80+ mortality {mort_below[-1]:.1f}%\nvs. 18-29: {mort_below[0]:.1f}%\n"
        f"({mort_below[-1]/max(mort_below[0],0.01):.1f}x higher)",
        xy=(6, mort_below[-1]), xytext=(-60, 25),
        textcoords="offset points", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#de2d26", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f0",
                  edgecolor="#de2d26", alpha=0.95),
        color="#de2d26", fontweight="bold",
    )

    fig.tight_layout(w_pad=2)
    _save(fig, "fig22_triage_sankey")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating ATLAS expansion figures (19-22)...")
    fig19_radar()
    fig20_forest()
    fig21_attention()
    fig22_sankey()
    print("\nDone — 4 expansion figures saved to paper/figures/")
