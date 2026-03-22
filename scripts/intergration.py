import glob
import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


RESULTS_BASE_DIR = "./results/"
GROUP_OUT_DIR = os.path.join(RESULTS_BASE_DIR, "group_summary")
os.makedirs(GROUP_OUT_DIR, exist_ok=True)

CONDITIONS = ["Divergent", "Convergent", "Random"]

# 1. Elegant, muted pastel palette for publication
COLORS = {
    "Divergent": "#CB7A5C",  # Muted Terracotta / Brick Red
    "Convergent": "#5C7CA3", # Muted Steel Blue
    "Random": "#7DA889"      # Muted Sage Green
}

NETWORK_TYPE_COLORS = {
    "strong": "#4A6478",
    "weak": "#A8B6C1",
    "strong_threshold": "#8C5669",
    "weak_threshold": "#BCA9AE",
}
NETWORK_TYPE_LABELS = {
    "strong": "Strong (rank)",
    "weak": "Weak (rank)",
    "strong_threshold": "Strong (threshold)",
    "weak_threshold": "Weak (threshold)",
}

ID_TO_COND = {"1": "Divergent", "2": "Convergent", "3": "Random"}
COND_ALIASES = {
    "divergent": "Divergent",
    "convergent": "Convergent",
    "random": "Random",
    "1": "Divergent",
    "2": "Convergent",
    "3": "Random",
}

GRAPH_METRICS = ["efficiency", "modularity", "local_efficiency", "avg_clustering"]

# ==========================================
# Global Publication Aesthetics (Minimalist)
# ==========================================
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

def style_axis(ax):
    """Minimalist axis styling."""
    sns.despine(ax=ax, trim=False)
    ax.tick_params(axis='both', which='major', length=5, pad=6)

# ==========================================
# Data Loading & Parsing (Logic Unchanged)
# ==========================================
def normalize_condition(value):
    if value is None: return None
    s = str(value).strip()
    if s in ID_TO_COND: return ID_TO_COND[s]
    return COND_ALIASES.get(s.lower())

def safe_float(value, default=np.nan):
    try:
        if value is None: return default
        return float(value)
    except Exception: return default

def load_optional_csv(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path): return None
    try: return pd.read_csv(path)
    except Exception: return None

def load_optional_csv_by_pattern(data_dir, pattern):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths: return None
    try: return pd.read_csv(paths[0])
    except Exception: return None

def load_all_mice_bundles(base_dir):
    pattern = os.path.join(base_dir, "*", "data", "*_statistics.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files: return []

    bundles = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mouse_id = payload.get("mouse_id", os.path.basename(fp).replace("_statistics.json", ""))
        data_dir = os.path.dirname(fp)
        bundles.append({
            "mouse_id": mouse_id, "json_path": fp, "data_dir": data_dir, "payload": payload,
            "trial_shape_summary": load_optional_csv(data_dir, "trial_response_shape_summary.csv"),
            "effective_dim": load_optional_csv(data_dir, "effective_dimensionality_by_class.csv"),
            "graph_sw": load_optional_csv(data_dir, "network_metrics_strong_vs_weak.csv"),
            "graph_thr": load_optional_csv_by_pattern(data_dir, "network_metrics_threshold_*.csv"),
            "sig_noise_summary": load_optional_csv(data_dir, "sig_noise_strength_summary_by_condition.csv"),
            "noise_decile_coupling": load_optional_csv(data_dir, "noise_corr_decile_coupling.csv"),
            "rr_overlap": load_optional_csv(data_dir, "rr_overlap_summary.csv"),
            "corr_deciles_csv": load_optional_csv(data_dir, f"{mouse_id}_correlation_deciles.csv"),
        })
    print(f"[*] Loaded {len(bundles)} mice from {base_dir}")
    return bundles

def _build_dict_by_condition(records, name_key):
    out = {}
    for item in records or []:
        cond = normalize_condition(item.get(name_key))
        if cond is not None: out[cond] = item
    return out

def _participants_by_condition(raw_dict):
    out = {}
    for k, v in (raw_dict or {}).items():
        cond = normalize_condition(k)
        if cond is not None: out[cond] = safe_float(v)
    return out

def _graph_condition_metrics(graph_df, condition):
    cols = {}
    if graph_df is None or graph_df.empty: return cols
    g = graph_df.copy()
    g["Class_Name"] = g["Class_Name"].map(normalize_condition)
    g["Network_Type"] = g["Network_Type"].astype(str).str.lower()
    sub = g[g["Class_Name"] == condition]
    if sub.empty: return cols
    for metric in GRAPH_METRICS:
        strong = safe_float(sub.loc[sub["Network_Type"] == "strong", metric].iloc[0]) if (sub["Network_Type"] == "strong").any() else np.nan
        weak = safe_float(sub.loc[sub["Network_Type"] == "weak", metric].iloc[0]) if (sub["Network_Type"] == "weak").any() else np.nan
        cols[f"GraphStrong_{metric}"] = strong
        cols[f"GraphWeak_{metric}"] = weak
        cols[f"GraphGap_{metric}"] = strong - weak if pd.notna(strong) and pd.notna(weak) else np.nan
    return cols

def _graph_threshold_condition_metrics(graph_df, condition):
    cols = {}
    if graph_df is None or graph_df.empty: return cols
    g = graph_df.copy()
    g["Class_Name"] = g["Class_Name"].map(normalize_condition)
    g["Network_Type"] = g["Network_Type"].astype(str).str.lower()
    sub = g[g["Class_Name"] == condition]
    if sub.empty: return cols
    for metric in GRAPH_METRICS:
        strong = safe_float(sub.loc[sub["Network_Type"] == "strong_threshold", metric].iloc[0]) if (sub["Network_Type"] == "strong_threshold").any() else np.nan
        weak = safe_float(sub.loc[sub["Network_Type"] == "weak_threshold", metric].iloc[0]) if (sub["Network_Type"] == "weak_threshold").any() else np.nan
        cols[f"GraphThrStrong_{metric}"] = strong
        cols[f"GraphThrWeak_{metric}"] = weak
        cols[f"GraphThrGap_{metric}"] = strong - weak if pd.notna(strong) and pd.notna(weak) else np.nan
    return cols

def _trial_shape_by_condition(trial_shape_summary_df, condition):
    cols = {}
    if trial_shape_summary_df is None or trial_shape_summary_df.empty: return cols
    t = trial_shape_summary_df.copy()
    t["Class_Name"] = t["Class_Name"].map(normalize_condition)
    row = t[t["Class_Name"] == condition]
    if row.empty: return cols
    row = row.iloc[0]
    for key in ["Gini_Mean", "Gini_STD", "PR_Mean", "PR_STD", "PR_Norm_Mean", "PR_Norm_STD"]:
        if key in row: cols[key] = safe_float(row[key])
    return cols

def _effective_dim_by_condition(effective_dim_df, condition):
    cols = {}
    if effective_dim_df is None or effective_dim_df.empty: return cols
    d = effective_dim_df.copy()
    d["Class_Name"] = d["Class_Name"].map(normalize_condition)
    row = d[d["Class_Name"] == condition]
    if row.empty: return cols
    row = row.iloc[0]
    for key in ["Effective_Dim_PR", "Effective_Dim_eRank", "Effective_Dim_90Var"]:
        if key in row: cols[key] = safe_float(row[key])
    return cols

def _sig_noise_by_condition(sig_noise_df, condition):
    cols = {}
    if sig_noise_df is None or sig_noise_df.empty: return cols
    s = sig_noise_df.copy()
    s["Class_Name"] = s["Class_Name"].map(normalize_condition)
    row = s[s["Class_Name"] == condition]
    if row.empty: return cols
    row = row.iloc[0]
    field_map = {
        "Mean_Signal_Corr": "Sig_Mean_Corr", "Mean_Noise_Corr": "Noise_Mean_Corr",
        "Mean_Abs_Signal_Corr": "SigAbs_Mean_Corr", "Mean_Abs_Noise_Corr": "NoiseAbs_Mean_Corr",
        "Signal_Noise_Coupling_r": "SigNoise_Coupling_r",
    }
    for src, dst in field_map.items():
        if src in row: cols[dst] = safe_float(row[src])
    return cols

def build_master_dataframe(bundles):
    rows = []
    for b in bundles:
        mouse_id = b["mouse_id"]
        payload = b["payload"]
        entropy_by_cond = _build_dict_by_condition(payload.get("entropy_metrics", []), "Stimulus")
        corr_by_cond = _build_dict_by_condition(payload.get("network_correlation", []), "Class_Name")
        part_by_cond = _participants_by_condition(payload.get("rr_participants_ratio", {}))
        for cond in CONDITIONS:
            ent = entropy_by_cond.get(cond, {})
            corr = corr_by_cond.get(cond, {})
            row = {
                "mouse_id": mouse_id, "Condition": cond,
                "Entropy": safe_float(ent.get("Entropy")),
                "Mean_RSM_Sim": safe_float(ent.get("Mean_Sim")),
                "Mean_Correlation": safe_float(corr.get("Mean_Correlation", corr.get("Mean_Abs_Correlation"))),
                "Strong_Correlation": safe_float(corr.get("Strong_Correlation_Mean", corr.get("Strong_Abs_Correlation_Mean"))),
                "Weak_Correlation": safe_float(corr.get("Weak_Correlation_Mean", corr.get("Weak_Abs_Correlation_Mean"))),
                "Strong_Weak_Gap": safe_float(corr.get("Strong_Weak_Gap")),
                "Participants_Ratio": safe_float(part_by_cond.get(cond)),
            }
            row.update(_trial_shape_by_condition(b["trial_shape_summary"], cond))
            row.update(_effective_dim_by_condition(b["effective_dim"], cond))
            row.update(_sig_noise_by_condition(b["sig_noise_summary"], cond))
            row.update(_graph_condition_metrics(b["graph_sw"], cond))
            row.update(_graph_threshold_condition_metrics(b["graph_thr"], cond))
            rows.append(row)
    return pd.DataFrame(rows)

def build_decile_dataframe(bundles):
    rows = []
    for b in bundles:
        mouse_id = b["mouse_id"]
        deciles = b["payload"].get("network_correlation_deciles")
        if deciles is None and b["corr_deciles_csv"] is not None:
            deciles = b["corr_deciles_csv"].to_dict(orient="records")
        if not deciles: continue
        for item in deciles:
            cond = normalize_condition(item.get("Class_Name") or item.get("Condition") or item.get("Class_ID"))
            if cond is None: continue
            rows.append({
                "mouse_id": mouse_id, "Condition": cond,
                "Decile_Index": int(item.get("Decile_Index")),
                "Mean_Correlation": safe_float(item.get("Mean_Correlation", item.get("Mean_Abs_Correlation"))),
            })
    return pd.DataFrame(rows)

def build_graph_sw_long_dataframe(bundles):
    rows = []
    for b in bundles:
        g = b["graph_sw"]
        if g is None or g.empty: continue
        tmp = g.copy()
        tmp["mouse_id"] = b["mouse_id"]
        tmp["Condition"] = tmp["Class_Name"].map(normalize_condition)
        tmp["Network_Type"] = tmp["Network_Type"].astype(str).str.lower()
        tmp = tmp[tmp["Condition"].isin(CONDITIONS)]
        for metric in GRAPH_METRICS:
            if metric in tmp.columns:
                keep = tmp[["mouse_id", "Condition", "Network_Type", metric]].copy().rename(columns={metric: "Value"})
                keep["Metric"] = metric
                rows.append(keep)
    if not rows: return pd.DataFrame(columns=["mouse_id", "Condition", "Network_Type", "Value", "Metric"])
    return pd.concat(rows, ignore_index=True)

def build_noise_decile_coupling_long_dataframe(bundles):
    rows = []
    for b in bundles:
        c = b["noise_decile_coupling"]
        if c is None or c.empty: continue
        tmp = c.copy()
        tmp["mouse_id"] = b["mouse_id"]
        tmp["Condition"] = tmp["Class_Name"].map(normalize_condition)
        tmp = tmp[tmp["Condition"].isin(CONDITIONS)]
        expected = ["mouse_id", "Condition", "Decile_Index", "Mean_Correlation", "Noise_Mean_Corr", "Corr_Delta_vs_D1", "Noise_Delta_vs_D1"]
        have = [x for x in expected if x in tmp.columns]
        rows.append(tmp[have])
    if not rows: return pd.DataFrame(columns=["mouse_id", "Condition", "Decile_Index", "Mean_Correlation", "Noise_Mean_Corr", "Corr_Delta_vs_D1", "Noise_Delta_vs_D1"])
    return pd.concat(rows, ignore_index=True)

def perform_statistical_tests(df, metric):
    pivot = df.pivot(index="mouse_id", columns="Condition", values=metric).reindex(columns=CONDITIONS).dropna()
    if len(pivot) < 3:
        return {"main_effect": "N too small", "p_main": np.nan, "post_hoc": {}}

    stat, p_val = stats.friedmanchisquare(pivot["Divergent"], pivot["Convergent"], pivot["Random"])
    out = {"main_effect": f"Friedman $\chi^2$={stat:.2f}, $p$={p_val:.3e}", "p_main": p_val, "post_hoc": {}}

    for c1, c2 in combinations(CONDITIONS, 2):
        try:
            _, p_pair = stats.wilcoxon(pivot[c1], pivot[c2])
            out["post_hoc"][f"{c1} vs {c2}"] = p_pair
        except Exception:
            out["post_hoc"][f"{c1} vs {c2}"] = np.nan
    return out

def p_to_star(p_val):
    if pd.isna(p_val): return "ns"
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"


# ==========================================
# Elegant Publication Plotting Functions
# ==========================================
def save_figure_variants(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)
    # Produce no-title variant for manuscript
    suptitle = fig._suptitle
    if suptitle is not None: suptitle.set_visible(False)
    for ax in fig.axes: ax.set_title("")
    stem, ext = os.path.splitext(save_path)
    fig.savefig(f"{stem}_notitle{ext}", dpi=300, bbox_inches="tight", transparent=False)
    return save_path

def plot_group_metric(df, metric, ylabel, title, stat_res, save_name):
    """
    UPGRADE: Clean, minimalist Barplot with error bars.
    Removes visual clutter (spaghetti lines) and focuses heavily on the group mean differences.
    """
    if metric not in df.columns or df[metric].isna().all(): return None

    sub = df[["mouse_id", "Condition", metric]].dropna().copy()
    if sub.empty: return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(4.5, 5))
    
    # Elegant Barplot
    sns.barplot(
        data=sub, 
        x="Condition", 
        y=metric, 
        palette=COLORS,
        errorbar="se",          # standard error
        capsize=0.15,           # crisp error bar caps
        err_kws={'linewidth': 2, 'color': '#333333'},
        linewidth=1.5,
        edgecolor="#333333",
        alpha=0.85,             # slight transparency for modern look
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{stat_res.get('main_effect', '')}", pad=15)
    style_axis(ax)

    # Significance Brackets
    if pd.notna(stat_res.get("p_main")) and stat_res.get("p_main", 1.0) < 0.1:
        y_max = sub[metric].max() * 1.05
        y_range = max(sub[metric].max() - sub[metric].min(), 1e-6)
        step = y_range * 0.1
        base = y_max

        sig_pairs = []
        for c1, c2 in combinations(CONDITIONS, 2):
            p = stat_res.get("post_hoc", {}).get(f"{c1} vs {c2}", np.nan)
            if p_to_star(p) != "ns": sig_pairs.append((c1, c2, p_to_star(p)))
        
        for i, (c1, c2, star) in enumerate(sig_pairs):
            x1, x2 = CONDITIONS.index(c1), CONDITIONS.index(c2)
            y = base + i * step
            ax.plot([x1, x1, x2, x2], [y, y + step*0.2, y + step*0.2, y], lw=1.5, c="#333333")
            ax.text((x1 + x2) * 0.5, y + step*0.25, star, ha="center", va="bottom", color="#111111", fontsize=12, fontweight='bold')
        if sig_pairs: ax.set_ylim(top=base + len(sig_pairs)*step + step*1.5)

    out = os.path.join(GROUP_OUT_DIR, save_name)
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_combined_strong_weak(df):
    """
    NEW: Combines Strong and Weak correlations into a single, grouped comparison plot.
    Highlights that Strong has no effect, while Weak does.
    """
    required = ["Strong_Correlation", "Weak_Correlation"]
    if not all(c in df.columns for c in required): return None

    sub = df[["mouse_id", "Condition", "Strong_Correlation", "Weak_Correlation"]].dropna().copy()
    if sub.empty: return None

    # Melt dataframe for grouped plotting
    melted = sub.melt(
        id_vars=["mouse_id", "Condition"], 
        value_vars=["Strong_Correlation", "Weak_Correlation"],
        var_name="Connection_Type", 
        value_name="Correlation"
    )
    
    # Rename for cleaner labels
    melted["Connection_Type"] = melted["Connection_Type"].replace({
        "Strong_Correlation": "Strong (Top 10%)",
        "Weak_Correlation": "Weak (Bottom 10%)"
    })
    melted["Condition"] = pd.Categorical(melted["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Grouped Pointplot highlights the interaction perfectly
    sns.pointplot(
        data=melted, 
        x="Connection_Type", 
        y="Correlation", 
        hue="Condition",
        palette=COLORS,
        dodge=0.25,      # Spread them out slightly
        errorbar="se", 
        capsize=0.08, 
        markers=["o", "s", "D"], 
        linestyles="-", 
        linewidth=2.5,
        err_kws={'linewidth': 2},
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_title("Divergent Modulation: Strong vs. Weak Connections", pad=15)
    style_axis(ax)
    
    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="", frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

    out = os.path.join(GROUP_OUT_DIR, "group_combined_strong_weak.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_decile_curve(decile_df):
    """UPGRADE: Smooth shaded error bands instead of vertical error bars."""
    if decile_df.empty: return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for cond in CONDITIONS:
        sub = decile_df[decile_df["Condition"] == cond]
        if sub.empty: continue
        agg = sub.groupby("Decile_Index")["Mean_Correlation"].agg(['mean', 'sem'])
        
        # Main Line
        ax.plot(agg.index, agg['mean'], label=cond, color=COLORS[cond], lw=2.5, marker='o', markersize=6, markeredgecolor='white')
        # Smooth Shaded Error Band
        ax.fill_between(agg.index, agg['mean'] - agg['sem'], agg['mean'] + agg['sem'], 
                        color=COLORS[cond], alpha=0.2, edgecolor="none")

    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Correlation Decile (1 = Weakest, 10 = Strongest)")
    ax.set_ylabel("Mean Correlation")
    ax.set_title("Hierarchical Correlation Structure", pad=15)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    out = os.path.join(GROUP_OUT_DIR, "group_corr_decile_curve.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_noise_decile_curve(noise_decile_df):
    """UPGRADE: Smooth shaded error bands for Noise Correlation."""
    if noise_decile_df.empty or "Noise_Mean_Corr" not in noise_decile_df.columns: return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for cond in CONDITIONS:
        sub = noise_decile_df[noise_decile_df["Condition"] == cond]
        if sub.empty: continue
        agg = sub.groupby("Decile_Index")["Noise_Mean_Corr"].agg(['mean', 'sem'])
        
        ax.plot(agg.index, agg['mean'], label=cond, color=COLORS[cond], lw=2.5, marker='o', markersize=6, markeredgecolor='white')
        ax.fill_between(agg.index, agg['mean'] - agg['sem'], agg['mean'] + agg['sem'], 
                        color=COLORS[cond], alpha=0.2, edgecolor="none")

    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Correlation Decile (Total Rank)")
    ax.set_ylabel("Mean Noise Correlation")
    ax.set_title("Noise Correlation across Hierarchy", pad=15)
    style_axis(ax)
    ax.legend(frameon=False, loc="best")

    out = os.path.join(GROUP_OUT_DIR, "group_noise_corr_decile_curve.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    bundles = load_all_mice_bundles(RESULTS_BASE_DIR)

    master_df = build_master_dataframe(bundles)
    decile_df = build_decile_dataframe(bundles)
    graph_long_df = build_graph_sw_long_dataframe(bundles)
    noise_decile_long_df = build_noise_decile_coupling_long_dataframe(bundles)

    # ---------------- Metrics Testing ----------------
    metrics_to_test = [m for m in [
        "Entropy", "Mean_RSM_Sim", "Mean_Correlation", "Strong_Correlation", "Weak_Correlation",
        "Strong_Weak_Gap", "Participants_Ratio", "Gini_Mean", "PR_Mean", "Effective_Dim_PR",
        "Sig_Mean_Corr", "Noise_Mean_Corr"
    ] if m in master_df.columns and not master_df[m].isna().all()]

    stat_results = {m: perform_statistical_tests(master_df, m) for m in metrics_to_test}

    # ---------------- Visualization Gen ----------------
    image_paths = {}
    
    # 1. New Combined Plot (Strong vs Weak)
    image_paths["Combined Strong vs Weak"] = plot_combined_strong_weak(master_df)

    # 2. Clean Core Metrics (Barplots)
    core_metrics = [
        ("Mean_RSM_Sim", "Cosine similarity", "RSM Mean Similarity"),
        ("Strong_Correlation", "Correlation", "Strong Connections (Top 10%)"),
        ("Weak_Correlation", "Correlation", "Weak Connections (Bottom 10%)"),
        ("Strong_Weak_Gap", "Correlation gap", "Strong-Weak Correlation Gap"),
        ("Participants_Ratio", "Ratio", "RR Participants Ratio"),
        ("Gini_Mean", "Gini Coefficient", "Response Gini (Mean)"),
    ]
    for metric, ylabel, title in core_metrics:
        image_paths[title] = plot_group_metric(master_df, metric, ylabel, title, stat_res=stat_results.get(metric, {}), save_name=f"group_{metric.lower()}.png")

    # 3. Smooth Decile Curves
    image_paths["Decile Correlation Curve"] = plot_decile_curve(decile_df)
    image_paths["Noise Decile Curve"] = plot_noise_decile_curve(noise_decile_long_df)

    print("====== Group integration visualization completed ======")