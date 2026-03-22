import glob
import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

RESULTS_BASE_DIR = "./results/"
GROUP_OUT_DIR = os.path.join(RESULTS_BASE_DIR, "group_summary")
os.makedirs(GROUP_OUT_DIR, exist_ok=True)

CONDITIONS = ["Divergent", "Convergent", "Random"]

# 1. 莫兰迪高级配色方案 (Elegant, muted pastel palette)
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
# 2. 全局极简学术排版设置 (Global Publication Aesthetics)
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
    """极简坐标轴样式"""
    sns.despine(ax=ax, trim=False)
    ax.tick_params(axis='both', which='major', length=5, pad=6)

# ==========================================
# 3. 数据加载与解析核心逻辑 (Data Loading & Parsing)
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

def build_rr_overlap_dataframe(bundles):
    rows = []
    for b in bundles:
        rr = b["rr_overlap"]
        if rr is None or rr.empty: continue
        tmp = rr.copy()
        tmp["mouse_id"] = b["mouse_id"]
        rows.append(tmp)
    if not rows: return pd.DataFrame(columns=["mouse_id", "Subset", "Subset_Size"])
    return pd.concat(rows, ignore_index=True)

def perform_statistical_tests(df, metric):
    pivot = df.pivot(index="mouse_id", columns="Condition", values=metric).reindex(columns=CONDITIONS).dropna()
    if len(pivot) < 3:
        return {"main_effect": "N too small", "p_main": np.nan, "post_hoc": {}}

    stat, p_val = stats.friedmanchisquare(pivot["Divergent"], pivot["Convergent"], pivot["Random"])
    # 修复：加上 rf 前缀防止 \c 被警告，输出标准的卡方符号
    out = {"main_effect": rf"Friedman $\chi^2$={stat:.2f}, $p$={p_val:.3e}", "p_main": p_val, "post_hoc": {}}

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
# 4. 优雅的可视化函数 (Elegant Publication Plotting Functions)
# ==========================================
def save_figure_variants(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)
    # 生成供论文排版的无标题版本
    suptitle = fig._suptitle
    if suptitle is not None: suptitle.set_visible(False)
    for ax in fig.axes: ax.set_title("")
    stem, ext = os.path.splitext(save_path)
    fig.savefig(f"{stem}_notitle{ext}", dpi=300, bbox_inches="tight", transparent=False)
    return save_path

def plot_group_metric(df, metric, ylabel, title, stat_res, save_name):
    """极简带误差棒的柱状图 (Minimalist Barplot) - 避免点线过杂"""
    if metric not in df.columns or df[metric].isna().all(): return None

    sub = df[["mouse_id", "Condition", metric]].dropna().copy()
    if sub.empty: return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(4.2, 5))
    
    # 修复了 Seaborn 未指定 hue 的告警
    sns.barplot(
        data=sub, x="Condition", y=metric, 
        hue="Condition", palette=COLORS, legend=False,
        errorbar="se", capsize=0.15, 
        err_kws={'linewidth': 2, 'color': '#333333'},
        linewidth=1.5, edgecolor="#333333", alpha=0.85, ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{stat_res.get('main_effect', '')}", pad=15)
    style_axis(ax)

    # 绘制显著性星号支架
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
    UPGRADE: 使用高级双面板箱型图 (Dual-panel Boxplots) 解决尺度压缩和杂乱问题。
    左图强连接，右图弱连接，独立Y轴，避免点线互相干扰。
    """
    required = ["Strong_Correlation", "Weak_Correlation"]
    if not all(c in df.columns for c in required): return None

    sub = df[["mouse_id", "Condition", "Strong_Correlation", "Weak_Correlation"]].dropna().copy()
    if sub.empty: return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    # 创建双面板图
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5), gridspec_kw={'wspace': 0.35})

    metrics = [
        ("Strong_Correlation", "Strong Connections\n(Top 10%)"),
        ("Weak_Correlation", "Weak Connections\n(Bottom 10%)")
    ]

    for ax, (col, title) in zip(axes, metrics):
        # 画高级箱型图：宽度收窄，去除离群点标志，透明度调整
        sns.boxplot(
            data=sub, x="Condition", y=col,
            hue="Condition", palette=COLORS, legend=False,
            ax=ax, width=0.5, linewidth=1.5, fliersize=0,
            boxprops=dict(alpha=0.75, edgecolor='#333')
        )
        
        # 叠加非常低调的半透明散点以展示 N=8，避免过于空白
        sns.stripplot(
            data=sub, x="Condition", y=col,
            color="#333333", alpha=0.5, size=4, jitter=0.15, ax=ax
        )
        
        ax.set_title(title, pad=15, fontsize=13, fontweight='bold')
        ax.set_ylabel("Mean Correlation" if ax == axes[0] else "")
        ax.set_xlabel("")
        style_axis(ax)

    # 统一的大标题，稍微调高 y 避免与子图标题重叠
    fig.suptitle("Divergent Modulation: Strong vs. Weak Network Couplings", 
                 fontsize=15, fontweight='bold', y=1.06)

    out = os.path.join(GROUP_OUT_DIR, "group_combined_strong_weak.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_decile_curve(decile_df):
    """带有平滑阴影误差带的分层曲线 (Shaded Error Bands)"""
    if decile_df.empty: return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for cond in CONDITIONS:
        sub = decile_df[decile_df["Condition"] == cond]
        if sub.empty: continue
        agg = sub.groupby("Decile_Index")["Mean_Correlation"].agg(['mean', 'sem'])
        
        # 主线
        ax.plot(agg.index, agg['mean'], label=cond, color=COLORS[cond], lw=2.5, marker='o', markersize=6, markeredgecolor='white')
        # 阴影带
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
# 5. Markdown 报告自动生成模块
# ==========================================
def generate_group_markdown(master_df, stat_results, image_paths, rr_overlap_df):
    md_path = os.path.join(GROUP_OUT_DIR, "Group_Analysis_Report.md")
    
    # 忽略 FutureWarning (observed=False)
    numeric_cols = [c for c in master_df.columns if c not in ["mouse_id", "Condition"] and pd.api.types.is_numeric_dtype(master_df[c])]
    summary_df = master_df.groupby("Condition", observed=False)[numeric_cols].agg(["mean", "sem"]).round(4)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group-level Multi-mouse Analysis Report\n\n")
        f.write(f"**Number of mice**: {master_df['mouse_id'].nunique()}\n\n")
        f.write(f"**Mouse IDs**: {', '.join(sorted(master_df['mouse_id'].unique()))}\n\n")

        f.write("## 1. Descriptive Statistics (Mean ± SEM)\n\n")
        desc = pd.DataFrame(index=summary_df.index)
        for col in summary_df.columns.levels[0]:
            desc[col] = summary_df[col]["mean"].astype(str) + " ± " + summary_df[col]["sem"].astype(str)
        f.write(desc.reset_index().to_markdown(index=False) + "\n\n")

        f.write("## 2. Friedman + Wilcoxon Tests\n\n")
        f.write("| Metric | Main Effect | Div vs Con | Div vs Rand | Con vs Rand |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for metric, res in stat_results.items():
            ph = res.get("post_hoc", {})
            f.write(
                f"| **{metric}** | {res.get('main_effect', 'N/A')} | "
                f"p={ph.get('Divergent vs Convergent', np.nan):.4f} ({p_to_star(ph.get('Divergent vs Convergent', np.nan))}) | "
                f"p={ph.get('Divergent vs Random', np.nan):.4f} ({p_to_star(ph.get('Divergent vs Random', np.nan))}) | "
                f"p={ph.get('Convergent vs Random', np.nan):.4f} ({p_to_star(ph.get('Convergent vs Random', np.nan))}) |\n"
            )
        f.write("\n")

        if not rr_overlap_df.empty:
            f.write("## 3. RR Overlap Summary Across Mice\n\n")
            rr_summary = rr_overlap_df.groupby("Subset", as_index=False).agg(Mean_Size=("Subset_Size", "mean"), SEM_Size=("Subset_Size", "sem"))
            f.write(rr_summary.to_markdown(index=False) + "\n\n")

        f.write("## 4. Figures\n\n")
        for name, path in image_paths.items():
            if path is None: continue
            rel = os.path.basename(path)
            f.write(f"### {name}\n![{name}](./{rel})\n\n")
    print(f"[*] Group markdown report written to: {md_path}")

def plot_cross_animal_binding(df):
    # 确保需要的列存在
    required_cols = ["Gini_Mean", "Mean_RSM_Sim"]
    if not all(c in df.columns for c in required_cols):
        print("Missing columns for cross-animal binding.")
        return None

    # 提取所需数据
    sub = df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    
    # 将数据透视，每只小鼠一行
    pivot_gini = sub.pivot(index="mouse_id", columns="Condition", values="Participants_Ratio")
    pivot_rsm = sub.pivot(index="mouse_id", columns="Condition", values="Mean_RSM_Sim")
    
    # 确保三个条件都存在
    for cond in ["Divergent", "Convergent", "Random"]:
        if cond not in pivot_gini.columns or cond not in pivot_rsm.columns:
            return None

    # 计算 Coherent Motion 的均值
    pivot_gini["Coherent"] = pivot_gini[["Divergent", "Convergent"]].mean(axis=1)
    pivot_rsm["Coherent"] = pivot_rsm[["Divergent", "Convergent"]].mean(axis=1)

    # 计算调制量 (Delta = Coherent - Random)
    delta_df = pd.DataFrame(index=pivot_gini.index)
    delta_df["Delta_Participants_Ratio"] = pivot_gini["Coherent"] - pivot_gini["Random"]
    delta_df["Delta_RSM"] = pivot_rsm["Coherent"] - pivot_rsm["Random"]
    delta_df = delta_df.dropna()

    if len(delta_df) < 3:
        return None

    # 统计检验 (Spearman for robustness, Pearson for reference)
    spearman_r, spearman_p = stats.spearmanr(delta_df["Delta_Participants_Ratio"], delta_df["Delta_RSM"])
    pearson_r, pearson_p = stats.pearsonr(delta_df["Delta_Participants_Ratio"], delta_df["Delta_RSM"])

    # 开始绘图
    fig, ax = plt.subplots(figsize=(5.5, 5))
    
    # 修复了 seaborn 和 matplotlib 之间的 linewidth 别名冲突报错
    sns.regplot(
        data=delta_df, x="Delta_Participants_Ratio", y="Delta_RSM",
        ax=ax, color="#404040", 
        scatter_kws={"s": 60, "alpha": 0.8, "edgecolors": "white", "linewidths": 1},
        line_kws={"linewidth": 2, "color": "#202020", "alpha": 0.8}
    )

    # 在图中标注每只鼠的 ID（帮助检查 outlier）
    for mouse_id, row in delta_df.iterrows():
        ax.annotate(mouse_id, (row["Delta_Participants_Ratio"], row["Delta_RSM"]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color="#606060", alpha=0.7)

    # 添加统计结果文本框
    stat_text = (f"Spearman $r_s$ = {spearman_r:.2f}, $p$ = {spearman_p:.3f}\n"
                 f"Pearson $r$ = {pearson_r:.2f}, $p$ = {pearson_p:.3f}")
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    # 增加过零点的辅助线
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)

    # 美化坐标轴，加上 r 前缀修复 \Delta 警告
    ax.set_xlabel(r"$\Delta$ Response Inequality (Participants$_{Coherent}$ - Participants$_{Random}$)")
    ax.set_ylabel(r"$\Delta$ Representational Stability (RSM$_{Coherent}$ - RSM$_{Random}$)")
    ax.set_title("Cross-Animal Binding", pad=15)
    style_axis(ax)

    out = os.path.join(GROUP_OUT_DIR, "group_cross_animal_binding.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_absolute_state_binding(df):
    """
    计算并绘制绝对状态的跨条件定量耦合分析 (Absolute State Space Binding)
    使用: Absolute Participants_Ratio vs Absolute Mean_RSM_Sim (N=24 data points)
    """
    required_cols = ["Participants_Ratio", "Mean_RSM_Sim"]
    if not all(c in df.columns for c in required_cols):
        return None

    # 提取所有小鼠在所有条件下的绝对值 (8 * 3 = 24 data points)
    plot_df = df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    plot_df["Condition"] = pd.Categorical(plot_df["Condition"], categories=CONDITIONS, ordered=True)

    if len(plot_df) < 5:
        return None

    # 整体相关性检验 (N=24)
    spearman_r, spearman_p = stats.spearmanr(plot_df["Participants_Ratio"], plot_df["Mean_RSM_Sim"])
    pearson_r, pearson_p = stats.pearsonr(plot_df["Participants_Ratio"], plot_df["Mean_RSM_Sim"])

    fig, ax = plt.subplots(figsize=(6, 5))

    # 1. 绘制底层的整体线性回归线 (不区分条件)
    sns.regplot(
        data=plot_df, x="Participants_Ratio", y="Mean_RSM_Sim",
        scatter=False, ax=ax, color="#404040", 
        line_kws={"linewidth": 2, "linestyle": "--", "alpha": 0.6}
    )

    # 2. 绘制散点，按照条件上色 (区分三种网络状态)
    sns.scatterplot(
        data=plot_df, x="Participants_Ratio", y="Mean_RSM_Sim",
        hue="Condition", palette=COLORS, s=70, alpha=0.85, 
        edgecolor="white", linewidth=1, ax=ax, zorder=3
    )

    # 添加统计结果文本框
    stat_text = (f"Overall Correlation (N={len(plot_df)} states)\n"
                 f"Spearman $r_s$ = {spearman_r:.2f}, $p$ = {spearman_p:.3e}\n"
                 f"Pearson $r$ = {pearson_r:.2f}, $p$ = {pearson_p:.3e}")
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    # 美化坐标轴
    ax.set_xlabel("Response Concentration (Participants Ratio)")
    ax.set_ylabel("Trial-to-Trial Population Stability (Mean RSM)")
    ax.set_title("Global State-Space Binding", pad=15)
    style_axis(ax)
    
    # 调整图例
    ax.legend(title="", frameon=False, loc="lower right")

    out = os.path.join(GROUP_OUT_DIR, "group_absolute_state_binding.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_lmm_state_binding(df):
    """
    终极版：使用线性混合效应模型 (LMM) 控制鼠内重复测量。
    模型: RSM ~ Participants_Ratio + (1 | mouse_id)
    """
    required_cols = ["Participants_Ratio", "Mean_RSM_Sim"]
    if not all(c in df.columns for c in required_cols):
        return None

    plot_df = df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    plot_df["Condition"] = pd.Categorical(plot_df["Condition"], categories=CONDITIONS, ordered=True)

    if len(plot_df) < 5:
        return None

    # ==========================================
    # 1. 拟合线性混合效应模型 (LMM)
    # 随机截距模型: 控制每只鼠自身的 baseline
    # ==========================================
    md = smf.mixedlm("Mean_RSM_Sim ~ Participants_Ratio", plot_df, groups=plot_df["mouse_id"])
    mdf = md.fit()

    coef = mdf.params["Participants_Ratio"]
    p_val = mdf.pvalues["Participants_Ratio"]
    global_intercept = mdf.params["Intercept"]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # ==========================================
    # 2. 绘制可视化：个体拟合线 (Random Intercepts)
    # 这将向审稿人完美展示“鼠内协变 (within-mouse covariance)”
    # ==========================================
    x_vals = np.array([plot_df["Participants_Ratio"].min() * 0.9, plot_df["Participants_Ratio"].max() * 1.05])
    
    for mouse_id, group_data in plot_df.groupby("mouse_id"):
        if mouse_id in mdf.random_effects:
            # 获取这只老鼠的随机截距 (Random effect)
            rand_int = mdf.random_effects[mouse_id].iloc[0] 
            intercept = global_intercept + rand_int
            y_vals = intercept + coef * x_vals
            # 画出属于这只老鼠自己的平行回归线
            ax.plot(x_vals, y_vals, color="#A0A0A0", alpha=0.35, linewidth=1.2, zorder=1)

    # ==========================================
    # 3. 绘制总体固定效应线 (Fixed Effect)
    # ==========================================
    global_y_vals = global_intercept + coef * x_vals
    ax.plot(x_vals, global_y_vals, color="#202020", linewidth=3, linestyle="--", zorder=2, label="Fixed effect (LMM)")

    # ==========================================
    # 4. 绘制原始散点 (区分条件)
    # ==========================================
    sns.scatterplot(
        data=plot_df, x="Participants_Ratio", y="Mean_RSM_Sim",
        hue="Condition", palette=COLORS, s=75, alpha=0.9, 
        edgecolor="white", linewidth=1, ax=ax, zorder=3
    )

    # 添加 LMM 统计结果文本框
    stat_text = (f"Linear Mixed-Effects Model (LMM)\n"
                 f"RSM $\\sim$ PR + $(1 | Mouse)$\n\n"
                 f"Fixed Effect $\\beta$ = {coef:.4f}\n"
                 f"$p$-value = {p_val:.3e}")
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))

    # 美化坐标轴
    ax.set_xlabel("Response Concentration (Participants Ratio)")
    ax.set_ylabel("Trial-to-Trial Population Stability (Mean RSM)")
    ax.set_title("Within-Animal State-Space Binding", pad=15)
    style_axis(ax)
    
    # 清理图例，去除重复项
    handles, labels = ax.get_legend_handles_labels()
    # 保留 Fixed effect 和 三个条件
    keep_indices = [labels.index("Fixed effect (LMM)")] + [labels.index(c) for c in CONDITIONS if c in labels]
    ax.legend([handles[i] for i in keep_indices], [labels[i] for i in keep_indices], 
              title="", frameon=False, loc="lower right")

    out = os.path.join(GROUP_OUT_DIR, "group_lmm_state_binding.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out
# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    bundles = load_all_mice_bundles(RESULTS_BASE_DIR)

    master_df = build_master_dataframe(bundles)
    decile_df = build_decile_dataframe(bundles)
    noise_decile_long_df = build_noise_decile_coupling_long_dataframe(bundles)
    rr_overlap_df = build_rr_overlap_dataframe(bundles) # 恢复 RR Overlap 数据

    # ---------------- 测试指标选择 ----------------
    metrics_to_test = [m for m in [
        "Entropy", "Mean_RSM_Sim", "Mean_Correlation", "Strong_Correlation", "Weak_Correlation",
        "Strong_Weak_Gap", "Participants_Ratio", "Gini_Mean", "PR_Mean", "Effective_Dim_PR",
        "Sig_Mean_Corr", "Noise_Mean_Corr"
    ] if m in master_df.columns and not master_df[m].isna().all()]

    stat_results = {m: perform_statistical_tests(master_df, m) for m in metrics_to_test}

    # ---------------- 可视化生成 ----------------
    image_paths = {}
    
    # 1. 强弱对比双面板箱型图 (Boxplot Dual-panel)
    image_paths["Combined Strong vs Weak"] = plot_combined_strong_weak(master_df)

    # 2. 核心指标的极简柱状图
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

    # 3. 分层曲线 (阴影误差带)
    image_paths["Decile Correlation Curve"] = plot_decile_curve(decile_df)
    image_paths["Noise Decile Curve"] = plot_noise_decile_curve(noise_decile_long_df)
    image_paths["Cross-animal Binding"] = plot_cross_animal_binding(master_df)
    image_paths["Absolute State Binding"] = plot_absolute_state_binding(master_df)
    image_paths["LMM State Binding"] = plot_lmm_state_binding(master_df)
    # 4. 生成Markdown报告
    generate_group_markdown(master_df, stat_results, image_paths, rr_overlap_df)

    print("====== Group integration visualization & markdown completed ======")