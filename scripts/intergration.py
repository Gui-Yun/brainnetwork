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

def load_optional_csv_any(data_dir, filenames):
    for fn in filenames:
        df = load_optional_csv(data_dir, fn)
        if df is not None:
            return df
    return None

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
            # Task1-2 decoder chain
            "decoder_summary": load_optional_csv(data_dir, "decoder_summary.csv"),
            "decoder_ablation_summary": load_optional_csv(data_dir, "decoder_ablation_summary.csv"),
            # Task3 FC decoder
            "fc_decoder_summary": load_optional_csv(data_dir, "fc_decoder_summary.csv"),
            # Task4 robust importance (edge-level preferred, component-level fallback)
            "fc_edge_stability": load_optional_csv_any(
                data_dir,
                ["fc_edge_importance_stability.csv", "fc_component_stability_selection.csv"],
            ),
            "fc_edge_ablation": load_optional_csv_any(
                data_dir,
                ["fc_edge_ablation_delta_acc.csv", "fc_component_ablation_delta_acc.csv"],
            ),
            "fc_projection_decile": load_optional_csv(data_dir, "fc_projection_by_strength_decile_task4.csv"),
            "fc_projection_layer_pair": load_optional_csv(data_dir, "fc_projection_by_layer_pair_task4.csv"),
            "fc_projection_strong_weak": load_optional_csv(data_dir, "fc_projection_strong_weak_match_task4.csv"),
            # Task5
            "fc_edge_decile_enrichment": load_optional_csv(data_dir, "fc_edge_decile_enrichment.csv"),
            # Task6
            "neuron_overlap_enrichment": load_optional_csv(data_dir, "neuron_overlap_enrichment.csv"),
            "neuron_selectivity_overlap": load_optional_csv(data_dir, "neuron_selectivity_by_overlap.csv"),
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

def _concat_bundle_table(bundles, key, default_columns=None):
    rows = []
    for b in bundles:
        df = b.get(key)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["mouse_id"] = b["mouse_id"]
        rows.append(tmp)
    if not rows:
        cols = default_columns or []
        return pd.DataFrame(columns=(["mouse_id"] + cols if "mouse_id" not in cols else cols))
    return pd.concat(rows, ignore_index=True)

def _first_row_bundle_table(bundles, key, default_columns=None):
    rows = []
    for b in bundles:
        df = b.get(key)
        if df is None or df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["mouse_id"] = b["mouse_id"]
        rows.append(row)
    cols = (["mouse_id"] + (default_columns or []))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

def build_decoder_summary_dataframe(bundles):
    return _first_row_bundle_table(
        bundles,
        "decoder_summary",
        default_columns=[
            "accuracy_mean", "shuffle_accuracy_mean", "accuracy_minus_shuffle",
            "recall_Divergent", "recall_Convergent", "recall_Random",
        ],
    )

def build_decoder_ablation_summary_dataframe(bundles):
    return _first_row_bundle_table(
        bundles,
        "decoder_ablation_summary",
        default_columns=[
            "full_accuracy_mean", "top10_ablation_accuracy_mean", "random_drop_mean_accuracy",
            "delta_full_minus_top10", "delta_top10_minus_random_mean", "ablation_rank_in_random",
        ],
    )

def build_fc_decoder_summary_dataframe(bundles):
    return _first_row_bundle_table(
        bundles,
        "fc_decoder_summary",
        default_columns=[
            "accuracy_mean", "shuffle_accuracy_mean", "accuracy_minus_shuffle",
            "fc_minus_activity_ref", "recall_Divergent", "recall_Convergent", "recall_Random",
        ],
    )

def build_fc_edge_stability_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "fc_edge_stability",
        default_columns=[
            "rank", "edge_idx", "strength_decile", "corr_mean",
            "edge_importance", "edge_importance_raw",
            "selection_frequency", "mean_abs_coef",
        ],
    )

def build_fc_edge_ablation_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "fc_edge_ablation",
        default_columns=[
            "ablation_type", "drop_fraction", "n_edges_dropped", "repeat_idx",
            "base_accuracy_mean", "accuracy_mean", "delta_vs_base",
        ],
    )

def build_fc_projection_decile_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "fc_projection_decile",
        default_columns=["strength_decile", "n_edges", "importance_sum", "importance_mean", "corr_mean"],
    )

def build_fc_projection_layer_pair_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "fc_projection_layer_pair",
        default_columns=["layer_pair", "n_edges", "importance_sum", "importance_mean", "corr_mean", "layer_source"],
    )

def build_fc_projection_strong_weak_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "fc_projection_strong_weak",
        default_columns=[
            "importance_strong_tail_decile10", "importance_weak_tail_decile1",
            "importance_gap_d10_minus_d1", "corr_mean_decile10", "corr_mean_decile1",
        ],
    )

def build_fc_edge_decile_enrichment_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "fc_edge_decile_enrichment",
        default_columns=[
            "level_type", "level", "observed_prop", "expected_prop",
            "enrichment_ratio", "log2_enrichment", "p_two_sided", "p_fdr_bh",
        ],
    )

def build_neuron_overlap_enrichment_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "neuron_overlap_enrichment",
        default_columns=[
            "overlap_category", "observed_important_fraction", "expected_important_fraction",
            "enrichment_ratio", "log2_enrichment", "p_two_sided", "p_fdr_bh",
        ],
    )

def build_neuron_selectivity_overlap_long_dataframe(bundles):
    return _concat_bundle_table(
        bundles,
        "neuron_selectivity_overlap",
        default_columns=[
            "level_type", "overlap_category", "important_fraction",
            "mean_selectivity_index", "mean_decoder_importance",
            "mean_ablation_effect_proxy", "mean_ablation_drop_actual",
        ],
    )

def build_fc_edge_stability_mouse_summary(stability_long_df):
    cols = [
        "mouse_id", "n_items", "importance_mean", "importance_sem",
        "weak_tail_importance_sum", "strong_tail_importance_sum", "strong_minus_weak",
        "top10_mean_importance",
    ]
    if stability_long_df is None or stability_long_df.empty:
        return pd.DataFrame(columns=cols)

    score_col = None
    for candidate in ["edge_importance", "selection_frequency", "mean_abs_coef"]:
        if candidate in stability_long_df.columns:
            score_col = candidate
            break
    if score_col is None:
        return pd.DataFrame(columns=cols)

    rows = []
    for mouse_id, sub in stability_long_df.groupby("mouse_id"):
        vals = pd.to_numeric(sub[score_col], errors="coerce").dropna()
        if vals.empty:
            continue

        topk = max(1, int(np.ceil(len(vals) * 0.10)))
        top10_mean = float(vals.nlargest(topk).mean())
        weak_sum = np.nan
        strong_sum = np.nan
        if "strength_decile" in sub.columns:
            dec = pd.to_numeric(sub["strength_decile"], errors="coerce")
            weak_sum = float(pd.to_numeric(sub.loc[dec.isin([1, 2]), score_col], errors="coerce").sum())
            strong_sum = float(pd.to_numeric(sub.loc[dec.isin([9, 10]), score_col], errors="coerce").sum())

        rows.append(
            {
                "mouse_id": mouse_id,
                "n_items": int(vals.size),
                "importance_mean": float(vals.mean()),
                "importance_sem": float(vals.sem()) if vals.size > 1 else np.nan,
                "weak_tail_importance_sum": weak_sum,
                "strong_tail_importance_sum": strong_sum,
                "strong_minus_weak": strong_sum - weak_sum if pd.notna(weak_sum) and pd.notna(strong_sum) else np.nan,
                "top10_mean_importance": top10_mean,
            }
        )
    return pd.DataFrame(rows, columns=cols)

def _paired_wilcoxon_summary(df, left_col, right_col, label):
    if df is None or df.empty or left_col not in df.columns or right_col not in df.columns:
        return None
    sub = df[["mouse_id", left_col, right_col]].dropna()
    if sub.empty:
        return None
    delta = sub[left_col] - sub[right_col]
    p_val = np.nan
    if len(sub) >= 3:
        try:
            _, p_val = stats.wilcoxon(sub[left_col], sub[right_col])
        except Exception:
            p_val = np.nan
    return {
        "Analysis": label,
        "N_mice": int(len(sub)),
        "Mean_Delta": float(delta.mean()),
        "SEM_Delta": float(delta.sem()) if len(sub) > 1 else np.nan,
        "p_value": p_val,
        "Significance": p_to_star(p_val),
    }

def _onesample_wilcoxon_summary(series, label):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    p_val = np.nan
    if len(vals) >= 3:
        try:
            _, p_val = stats.wilcoxon(vals)
        except Exception:
            p_val = np.nan
    return {
        "Analysis": label,
        "N_mice": int(len(vals)),
        "Mean_Delta": float(vals.mean()),
        "SEM_Delta": float(vals.sem()) if len(vals) > 1 else np.nan,
        "p_value": p_val,
        "Significance": p_to_star(p_val),
    }

def build_decoder_chain_stats_table(
    task1_df,
    task2_df,
    task3_df,
    task4_ablation_long_df,
    task5_enrichment_long_df,
    task6_overlap_enrichment_long_df,
):
    rows = []

    r = _paired_wilcoxon_summary(task1_df, "accuracy_mean", "shuffle_accuracy_mean", "Task1: activity decoder vs shuffle")
    if r is not None: rows.append(r)
    r = _paired_wilcoxon_summary(task2_df, "full_accuracy_mean", "top10_ablation_accuracy_mean", "Task2: full vs top10 neuron ablation")
    if r is not None: rows.append(r)
    r = _paired_wilcoxon_summary(task2_df, "top10_ablation_accuracy_mean", "random_drop_mean_accuracy", "Task2: top10 ablation vs random drop")
    if r is not None: rows.append(r)
    r = _paired_wilcoxon_summary(task3_df, "accuracy_mean", "shuffle_accuracy_mean", "Task3: FC decoder vs shuffle")
    if r is not None: rows.append(r)

    if task1_df is not None and not task1_df.empty and task3_df is not None and not task3_df.empty:
        merged = pd.merge(
            task1_df[["mouse_id", "accuracy_mean"]].rename(columns={"accuracy_mean": "task1_acc"}),
            task3_df[["mouse_id", "accuracy_mean"]].rename(columns={"accuracy_mean": "task3_acc"}),
            on="mouse_id",
            how="inner",
        )
        r = _paired_wilcoxon_summary(merged, "task3_acc", "task1_acc", "Task3 vs Task1: FC decoder vs activity decoder")
        if r is not None: rows.append(r)

    if task4_ablation_long_df is not None and not task4_ablation_long_df.empty:
        if {"ablation_type", "drop_fraction", "mouse_id", "delta_vs_base"}.issubset(task4_ablation_long_df.columns):
            top_df = task4_ablation_long_df[task4_ablation_long_df["ablation_type"] == "top"].copy()
            rand_df = task4_ablation_long_df[task4_ablation_long_df["ablation_type"] == "random"].copy()
            rand_mean = (
                rand_df.groupby(["mouse_id", "drop_fraction"], as_index=False)["delta_vs_base"]
                .mean()
                .rename(columns={"delta_vs_base": "rand_delta_vs_base"})
            )
            top_keep = top_df[["mouse_id", "drop_fraction", "delta_vs_base"]].rename(columns={"delta_vs_base": "top_delta_vs_base"})
            merged = pd.merge(top_keep, rand_mean, on=["mouse_id", "drop_fraction"], how="inner")
            for frac in sorted(merged["drop_fraction"].dropna().unique().tolist()):
                sub = merged[merged["drop_fraction"] == frac]
                r = _paired_wilcoxon_summary(
                    sub,
                    "top_delta_vs_base",
                    "rand_delta_vs_base",
                    f"Task4: top-edge vs random-edge ablation (drop={int(round(frac * 100))}%)",
                )
                if r is not None: rows.append(r)

    if task5_enrichment_long_df is not None and not task5_enrichment_long_df.empty:
        if {"level_type", "level", "mouse_id", "log2_enrichment"}.issubset(task5_enrichment_long_df.columns):
            regime = task5_enrichment_long_df[task5_enrichment_long_df["level_type"] == "regime"].copy()
            pivot = regime.pivot_table(index="mouse_id", columns="level", values="log2_enrichment", aggfunc="first").reset_index()
            if {"WeakTail_D1D2", "StrongTail_D9D10"}.issubset(pivot.columns):
                r = _paired_wilcoxon_summary(
                    pivot,
                    "WeakTail_D1D2",
                    "StrongTail_D9D10",
                    "Task5: weak-tail vs strong-tail log2 enrichment",
                )
                if r is not None: rows.append(r)
            if "WeakTail_D1D2" in pivot.columns:
                r = _onesample_wilcoxon_summary(pivot["WeakTail_D1D2"], "Task5: weak-tail enrichment vs 0")
                if r is not None: rows.append(r)

    if task6_overlap_enrichment_long_df is not None and not task6_overlap_enrichment_long_df.empty:
        if {"mouse_id", "overlap_category", "log2_enrichment"}.issubset(task6_overlap_enrichment_long_df.columns):
            pivot = task6_overlap_enrichment_long_df.pivot_table(
                index="mouse_id", columns="overlap_category", values="log2_enrichment", aggfunc="first"
            ).reset_index()
            if {"Shared_Core", "Condition_Biased"}.issubset(pivot.columns):
                r = _paired_wilcoxon_summary(
                    pivot,
                    "Shared_Core",
                    "Condition_Biased",
                    "Task6: Shared_Core vs Condition_Biased enrichment",
                )
                if r is not None: rows.append(r)
            if "Shared_Core" in pivot.columns:
                r = _onesample_wilcoxon_summary(pivot["Shared_Core"], "Task6: Shared_Core enrichment vs 0")
                if r is not None: rows.append(r)

    return pd.DataFrame(rows)

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

def df_to_markdown(df, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:
        data = df.copy()
        if not index and data.index.name is not None:
            data = data.reset_index(drop=True)
        if index:
            data = data.reset_index()
        cols = list(data.columns)
        lines = []
        lines.append("| " + " | ".join([str(c) for c in cols]) + " |")
        lines.append("| " + " | ".join([":---"] * len(cols)) + " |")
        for _, row in data.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                vals.append("" if pd.isna(v) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

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
def generate_group_markdown(
    master_df,
    stat_results,
    image_paths,
    rr_overlap_df,
    table_paths=None,
    decoder_chain_summary_df=None,
    decoder_chain_stats_df=None,
):
    md_path = os.path.join(GROUP_OUT_DIR, "Group_Analysis_Report.md")
    table_paths = table_paths or {}

    def _fmt(v, digits=4):
        return "NA" if pd.isna(v) else f"{v:.{digits}f}"

    numeric_cols = [c for c in master_df.columns if c not in ["mouse_id", "Condition"] and pd.api.types.is_numeric_dtype(master_df[c])]
    summary_df = master_df.groupby("Condition", observed=False)[numeric_cols].agg(["mean", "sem"]).round(4)

    desc = pd.DataFrame(index=summary_df.index)
    for col in summary_df.columns.levels[0]:
        desc[col] = [f"{_fmt(m)} ± {_fmt(s)}" for m, s in zip(summary_df[col]["mean"], summary_df[col]["sem"])]
    desc = desc.reset_index()

    stat_rows = []
    for metric, res in stat_results.items():
        ph = res.get("post_hoc", {})
        stat_rows.append(
            {
                "Metric": metric,
                "Main_Effect": res.get("main_effect", "N/A"),
                "p_main": res.get("p_main", np.nan),
                "Main_Star": p_to_star(res.get("p_main", np.nan)),
                "Div_vs_Con": ph.get("Divergent vs Convergent", np.nan),
                "Div_vs_Rand": ph.get("Divergent vs Random", np.nan),
                "Con_vs_Rand": ph.get("Convergent vs Random", np.nan),
            }
        )
    stat_df = pd.DataFrame(stat_rows).sort_values(by="p_main", na_position="last") if stat_rows else pd.DataFrame()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group-level Multi-mouse Analysis Report\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- Number of mice: {master_df['mouse_id'].nunique()}\n")
        f.write(f"- Mouse IDs: {', '.join(sorted(master_df['mouse_id'].unique()))}\n")
        f.write(f"- Conditions: {', '.join(CONDITIONS)}\n\n")

        f.write("## 2. Exported Data Tables\n\n")
        if table_paths:
            for name, path in table_paths.items():
                f.write(f"- {name}: `{os.path.basename(path)}`\n")
        else:
            f.write("- No external tables exported in this run.\n")
        f.write("\n")

        f.write("## 3. Descriptive Statistics (Mean ± SEM)\n\n")
        f.write(df_to_markdown(desc, index=False) + "\n\n")

        f.write("## 4. Friedman + Wilcoxon Tests\n\n")
        if stat_df.empty:
            f.write("No valid condition-level tests were computed.\n\n")
        else:
            stat_disp = stat_df.copy()
            for c in ["p_main", "Div_vs_Con", "Div_vs_Rand", "Con_vs_Rand"]:
                stat_disp[c] = stat_disp[c].map(lambda x: _fmt(x, digits=4))
            f.write(df_to_markdown(stat_disp, index=False) + "\n\n")

        section_idx = 5
        if not rr_overlap_df.empty:
            f.write(f"## {section_idx}. RR Overlap Summary Across Mice\n\n")
            rr_summary = rr_overlap_df.groupby("Subset", as_index=False).agg(
                Mean_Size=("Subset_Size", "mean"),
                SEM_Size=("Subset_Size", "sem"),
            )
            rr_summary["Mean_Size"] = rr_summary["Mean_Size"].round(4)
            rr_summary["SEM_Size"] = rr_summary["SEM_Size"].round(4)
            f.write(df_to_markdown(rr_summary, index=False) + "\n\n")
            section_idx += 1

        if decoder_chain_summary_df is not None and not decoder_chain_summary_df.empty:
            f.write(f"## {section_idx}. Decoder Chain Summary (Tasks 1-6)\n\n")
            f.write(df_to_markdown(decoder_chain_summary_df, index=False) + "\n\n")
            section_idx += 1

        if decoder_chain_stats_df is not None and not decoder_chain_stats_df.empty:
            f.write(f"## {section_idx}. Decoder Chain Statistical Tests (Tasks 1-6)\n\n")
            stat_local = decoder_chain_stats_df.copy()
            for c in ["Mean_Delta", "SEM_Delta", "p_value"]:
                if c in stat_local.columns:
                    stat_local[c] = stat_local[c].map(lambda x: _fmt(x, digits=4))
            f.write(df_to_markdown(stat_local, index=False) + "\n\n")
            section_idx += 1

        f.write(f"## {section_idx}. Figures\n\n")
        figure_groups = [
            ("Core and Correlation Metrics", [
                "Combined Strong vs Weak",
                "RSM Mean Similarity",
                "Strong Connections (Top 10%)",
                "Weak Connections (Bottom 10%)",
                "Strong-Weak Correlation Gap",
                "RR Participants Ratio",
                "Response Gini (Mean)",
                "Decile Correlation Curve",
                "Noise Decile Curve",
            ]),
            ("Binding Analyses", ["Cross-animal Binding", "Absolute State Binding", "LMM State Binding"]),
            ("Decoder Chain (Tasks 1-6)", [
                "Decoder Accuracy (Task1+Task3)",
                "Decoder Ablation (Task2)",
                "Edge Ablation Robustness (Task4)",
                "Edge Decile Enrichment (Task5)",
                "Neuron Linking (Task6)",
            ]),
        ]
        for group_name, names in figure_groups:
            available = [name for name in names if image_paths.get(name)]
            if not available:
                continue
            f.write(f"### {group_name}\n\n")
            for name in available:
                rel = os.path.basename(image_paths[name])
                f.write(f"#### {name}\n![{name}](./{rel})\n\n")

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

def _mean_sem(values):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if vals.empty:
        return np.nan, np.nan
    m = float(vals.mean())
    sem = float(vals.sem()) if len(vals) > 1 else np.nan
    return m, sem

def _plot_paired_metric(ax, df, left_col, right_col, left_label, right_label, colors, ylabel, title):
    if df is None or df.empty or left_col not in df.columns or right_col not in df.columns:
        return False
    sub = df[["mouse_id", left_col, right_col]].dropna()
    if sub.empty:
        return False

    x = np.array([0, 1], dtype=float)
    for _, row in sub.iterrows():
        y = [float(row[left_col]), float(row[right_col])]
        ax.plot(x, y, color="#999999", alpha=0.45, linewidth=1.0, zorder=1)
        ax.scatter(x, y, color=[colors[0], colors[1]], s=26, zorder=2)

    mean_left, sem_left = _mean_sem(sub[left_col])
    mean_right, sem_right = _mean_sem(sub[right_col])
    ax.errorbar(
        x,
        [mean_left, mean_right],
        yerr=[sem_left, sem_right],
        fmt="o-",
        color="#222222",
        linewidth=2.0,
        capsize=4,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([left_label, right_label], rotation=18)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nN={sub['mouse_id'].nunique()} mice")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    style_axis(ax)
    return True

def plot_decoder_chain_accuracy(task1_df, task3_df):
    has_task1 = task1_df is not None and not task1_df.empty
    has_task3 = task3_df is not None and not task3_df.empty
    if not has_task1 and not has_task3:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180)
    used_left = _plot_paired_metric(
        axes[0],
        task1_df,
        "shuffle_accuracy_mean",
        "accuracy_mean",
        "Shuffle",
        "Activity",
        colors=["#B9CFE7", "#4C78A8"],
        ylabel="Accuracy",
        title="Task1 Decoder",
    ) if has_task1 else False
    if not used_left:
        axes[0].axis("off")
        axes[0].set_title("Task1 Decoder (not available)")

    used_right = _plot_paired_metric(
        axes[1],
        task3_df,
        "shuffle_accuracy_mean",
        "accuracy_mean",
        "Shuffle",
        "FC",
        colors=["#C7D5B8", "#54A24B"],
        ylabel="Accuracy",
        title="Task3 FC Decoder",
    ) if has_task3 else False
    if not used_right:
        axes[1].axis("off")
        axes[1].set_title("Task3 FC Decoder (not available)")

    fig.tight_layout()
    out = os.path.join(GROUP_OUT_DIR, "group_decoder_chain_accuracy.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_task2_ablation_summary(task2_df):
    required = {"full_accuracy_mean", "top10_ablation_accuracy_mean", "random_drop_mean_accuracy"}
    if task2_df is None or task2_df.empty or not required.issubset(task2_df.columns):
        return None
    sub = task2_df[["mouse_id", "full_accuracy_mean", "top10_ablation_accuracy_mean", "random_drop_mean_accuracy"]].dropna()
    if sub.empty:
        return None

    labels = ["Full", "Top10 ablation", "Random drop"]
    cols = ["full_accuracy_mean", "top10_ablation_accuracy_mean", "random_drop_mean_accuracy"]
    x = np.arange(3)
    colors = ["#4C78A8", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=180)
    for _, row in sub.iterrows():
        y = [float(row[c]) for c in cols]
        ax.plot(x, y, color="#999999", alpha=0.45, linewidth=1.0, zorder=1)
        ax.scatter(x, y, color=colors, s=28, zorder=2)

    means = [sub[c].mean() for c in cols]
    sems = [sub[c].sem() if len(sub) > 1 else np.nan for c in cols]
    ax.errorbar(x, means, yerr=sems, fmt="o-", color="#222222", linewidth=2.0, capsize=4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Task2 Top10% Neuron Ablation Across Mice\nN={sub['mouse_id'].nunique()} mice")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    style_axis(ax)

    out = os.path.join(GROUP_OUT_DIR, "group_decoder_ablation_task2.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_task4_edge_ablation(task4_edge_ablation_long_df):
    required = {"mouse_id", "ablation_type", "drop_fraction", "delta_vs_base"}
    if task4_edge_ablation_long_df is None or task4_edge_ablation_long_df.empty or not required.issubset(task4_edge_ablation_long_df.columns):
        return None

    sub = task4_edge_ablation_long_df.copy()
    top_df = sub[sub["ablation_type"] == "top"][["mouse_id", "drop_fraction", "delta_vs_base"]].rename(
        columns={"delta_vs_base": "top_delta"}
    )
    rand_df = (
        sub[sub["ablation_type"] == "random"]
        .groupby(["mouse_id", "drop_fraction"], as_index=False)["delta_vs_base"]
        .mean()
        .rename(columns={"delta_vs_base": "random_delta"})
    )
    merged = pd.merge(top_df, rand_df, on=["mouse_id", "drop_fraction"], how="outer")
    if merged.empty:
        return None

    plot_rows = []
    for frac, g in merged.groupby("drop_fraction"):
        mt, st = _mean_sem(g["top_delta"])
        mr, sr = _mean_sem(g["random_delta"])
        plot_rows.append(
            {
                "drop_fraction": float(frac),
                "top_mean": mt, "top_sem": st,
                "random_mean": mr, "random_sem": sr,
                "n_mice": int(g["mouse_id"].nunique()),
            }
        )
    plot_df = pd.DataFrame(plot_rows).sort_values("drop_fraction")
    if plot_df.empty:
        return None

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=180)
    ax.errorbar(x, plot_df["top_mean"], yerr=plot_df["top_sem"], fmt="o-", color="#E45756", capsize=4, label="Top-edge drop")
    ax.errorbar(x, plot_df["random_mean"], yerr=plot_df["random_sem"], fmt="s--", color="#54A24B", capsize=4, label="Random-edge drop")
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(v * 100))}%" for v in plot_df["drop_fraction"]])
    ax.set_xlabel("Dropped edge fraction")
    ax.set_ylabel("Accuracy drop vs baseline")
    ax.set_title("Task4 Edge Ablation Robustness Across Mice")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    style_axis(ax)

    out = os.path.join(GROUP_OUT_DIR, "group_fc_edge_ablation_task4.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_task5_decile_enrichment(task5_enrichment_long_df):
    required = {"mouse_id", "level_type", "level", "log2_enrichment"}
    if task5_enrichment_long_df is None or task5_enrichment_long_df.empty or not required.issubset(task5_enrichment_long_df.columns):
        return None

    dec = task5_enrichment_long_df[task5_enrichment_long_df["level_type"] == "decile"].copy()
    if dec.empty:
        return None
    dec["decile_idx"] = dec["level"].astype(str).str.replace("D", "", regex=False)
    dec["decile_idx"] = pd.to_numeric(dec["decile_idx"], errors="coerce")
    dec = dec.dropna(subset=["decile_idx"])
    if dec.empty:
        return None

    dec_stat = dec.groupby("decile_idx", as_index=False)["log2_enrichment"].agg(
        log2_mean="mean",
        log2_sem="sem",
    )

    regime = task5_enrichment_long_df[
        (task5_enrichment_long_df["level_type"] == "regime")
        & (task5_enrichment_long_df["level"].isin(["WeakTail_D1D2", "StrongTail_D9D10"]))
    ].copy()
    regime_stat = (
        regime.groupby("level", as_index=False)["log2_enrichment"].agg(
            log2_mean="mean",
            log2_sem="sem",
        )
        if not regime.empty
        else pd.DataFrame()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)

    x = dec_stat["decile_idx"].to_numpy(dtype=int)
    axes[0].errorbar(x, dec_stat["log2_mean"], yerr=dec_stat["log2_sem"], fmt="o-", color="#4C78A8", capsize=4)
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_xticks(np.arange(1, 11, 1))
    axes[0].set_xlabel("Decile (1=weak, 10=strong)")
    axes[0].set_ylabel("Mean log2 enrichment")
    axes[0].set_title("Task5 Decile Enrichment")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    style_axis(axes[0])

    if regime_stat.empty:
        axes[1].axis("off")
        axes[1].set_title("Task5 Regime Enrichment (not available)")
    else:
        regime_stat["level"] = pd.Categorical(
            regime_stat["level"],
            categories=["WeakTail_D1D2", "StrongTail_D9D10"],
            ordered=True,
        )
        regime_stat = regime_stat.sort_values("level")
        x2 = np.arange(len(regime_stat))
        axes[1].bar(x2, regime_stat["log2_mean"], yerr=regime_stat["log2_sem"], color=["#72B7B2", "#F58518"], alpha=0.9, capsize=4)
        axes[1].axhline(0.0, color="#666666", linewidth=1.0)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(regime_stat["level"].astype(str), rotation=15)
        axes[1].set_ylabel("Mean log2 enrichment")
        axes[1].set_title("Task5 Regime Enrichment")
        axes[1].grid(axis="y", linestyle="--", alpha=0.25)
        style_axis(axes[1])

    fig.tight_layout()
    out = os.path.join(GROUP_OUT_DIR, "group_fc_edge_decile_enrichment_task5.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_task6_linking(task6_overlap_enrichment_long_df, task6_selectivity_long_df):
    has_overlap = (
        task6_overlap_enrichment_long_df is not None
        and not task6_overlap_enrichment_long_df.empty
        and {"overlap_category", "log2_enrichment"}.issubset(task6_overlap_enrichment_long_df.columns)
    )
    has_selectivity = (
        task6_selectivity_long_df is not None
        and not task6_selectivity_long_df.empty
        and {"level_type", "overlap_category", "mean_selectivity_index"}.issubset(task6_selectivity_long_df.columns)
    )
    if not has_overlap and not has_selectivity:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)

    if has_overlap:
        enr = task6_overlap_enrichment_long_df.groupby("overlap_category", as_index=False)["log2_enrichment"].agg(
            log2_mean="mean",
            log2_sem="sem",
        )
        x = np.arange(len(enr))
        axes[0].bar(x, enr["log2_mean"], yerr=enr["log2_sem"], color="#4C78A8", alpha=0.9, capsize=4)
        axes[0].axhline(0.0, color="#666666", linewidth=1.0)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(enr["overlap_category"].astype(str), rotation=18)
        axes[0].set_ylabel("Mean log2 enrichment")
        axes[0].set_title("Task6 Overlap Enrichment")
        axes[0].grid(axis="y", linestyle="--", alpha=0.25)
        style_axis(axes[0])
    else:
        axes[0].axis("off")
        axes[0].set_title("Task6 Overlap Enrichment (not available)")

    if has_selectivity:
        sel = task6_selectivity_long_df[task6_selectivity_long_df["level_type"] == "coarse"].copy()
        if sel.empty:
            axes[1].axis("off")
            axes[1].set_title("Task6 Selectivity by Overlap (not available)")
        else:
            sel_stat = sel.groupby("overlap_category", as_index=False)["mean_selectivity_index"].agg(
                si_mean="mean",
                si_sem="sem",
            )
            x2 = np.arange(len(sel_stat))
            axes[1].bar(x2, sel_stat["si_mean"], yerr=sel_stat["si_sem"], color="#72B7B2", alpha=0.9, capsize=4)
            axes[1].set_xticks(x2)
            axes[1].set_xticklabels(sel_stat["overlap_category"].astype(str), rotation=18)
            axes[1].set_ylabel("Mean selectivity index")
            axes[1].set_title("Task6 Selectivity by Overlap")
            axes[1].grid(axis="y", linestyle="--", alpha=0.25)
            style_axis(axes[1])
    else:
        axes[1].axis("off")
        axes[1].set_title("Task6 Selectivity by Overlap (not available)")

    fig.tight_layout()
    out = os.path.join(GROUP_OUT_DIR, "group_neuron_linking_task6.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def build_decoder_chain_summary_table(
    task1_df,
    task2_df,
    task3_df,
    task4_edge_ablation_long_df,
    task5_enrichment_long_df,
    task6_overlap_enrichment_long_df,
):
    rows = []

    def _append_metric(df, col, label):
        if df is None or df.empty or col not in df.columns:
            return
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return
        rows.append(
            {
                "Metric": label,
                "N_mice": int(vals.size),
                "Mean": float(vals.mean()),
                "SEM": float(vals.sem()) if vals.size > 1 else np.nan,
            }
        )

    _append_metric(task1_df, "accuracy_mean", "Task1 activity decoder accuracy")
    _append_metric(task1_df, "accuracy_minus_shuffle", "Task1 activity decoder minus shuffle")
    _append_metric(task2_df, "delta_full_minus_top10", "Task2 full minus top10 ablation")
    _append_metric(task3_df, "accuracy_mean", "Task3 FC decoder accuracy")
    _append_metric(task3_df, "accuracy_minus_shuffle", "Task3 FC decoder minus shuffle")

    if task4_edge_ablation_long_df is not None and not task4_edge_ablation_long_df.empty:
        top = task4_edge_ablation_long_df[task4_edge_ablation_long_df["ablation_type"] == "top"].copy()
        for frac in sorted(top["drop_fraction"].dropna().unique().tolist()):
            vals = pd.to_numeric(top.loc[top["drop_fraction"] == frac, "delta_vs_base"], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "Metric": f"Task4 top-edge ablation delta (drop={int(round(frac * 100))}%)",
                    "N_mice": int(vals.size),
                    "Mean": float(vals.mean()),
                    "SEM": float(vals.sem()) if vals.size > 1 else np.nan,
                }
            )

    if task5_enrichment_long_df is not None and not task5_enrichment_long_df.empty:
        weak = task5_enrichment_long_df[
            (task5_enrichment_long_df["level_type"] == "regime")
            & (task5_enrichment_long_df["level"] == "WeakTail_D1D2")
        ]
        _append_metric(weak, "log2_enrichment", "Task5 weak-tail log2 enrichment")

    if task6_overlap_enrichment_long_df is not None and not task6_overlap_enrichment_long_df.empty:
        shared = task6_overlap_enrichment_long_df[task6_overlap_enrichment_long_df["overlap_category"] == "Shared_Core"]
        _append_metric(shared, "log2_enrichment", "Task6 Shared_Core log2 enrichment")

    return pd.DataFrame(rows)

def run_group_integration():
    bundles = load_all_mice_bundles(RESULTS_BASE_DIR)
    if not bundles:
        print("[!] No mice loaded. Please check the results directory structure.")
        return

    table_paths = {}

    # ===== Original multi-mouse integration outputs =====
    master_df = build_master_dataframe(bundles)
    master_path = os.path.join(GROUP_OUT_DIR, "group_master_metrics.csv")
    master_df.to_csv(master_path, index=False)
    table_paths["Master metrics"] = master_path

    decile_df = build_decile_dataframe(bundles)
    decile_path = os.path.join(GROUP_OUT_DIR, "group_corr_deciles_long.csv")
    decile_df.to_csv(decile_path, index=False)
    table_paths["Correlation deciles long"] = decile_path

    noise_decile_long_df = build_noise_decile_coupling_long_dataframe(bundles)
    noise_decile_path = os.path.join(GROUP_OUT_DIR, "group_noise_corr_decile_coupling_long.csv")
    noise_decile_long_df.to_csv(noise_decile_path, index=False)
    table_paths["Noise decile coupling long"] = noise_decile_path

    rr_overlap_df = build_rr_overlap_dataframe(bundles)
    rr_overlap_path = os.path.join(GROUP_OUT_DIR, "group_rr_overlap_long.csv")
    rr_overlap_df.to_csv(rr_overlap_path, index=False)
    table_paths["RR overlap long"] = rr_overlap_path

    # ===== New Task1-6 multi-mouse integration =====
    task1_df = build_decoder_summary_dataframe(bundles)
    task1_path = os.path.join(GROUP_OUT_DIR, "group_decoder_summary_long.csv")
    task1_df.to_csv(task1_path, index=False)
    table_paths["Task1 decoder summary long"] = task1_path

    task2_df = build_decoder_ablation_summary_dataframe(bundles)
    task2_path = os.path.join(GROUP_OUT_DIR, "group_decoder_ablation_summary_long.csv")
    task2_df.to_csv(task2_path, index=False)
    table_paths["Task2 ablation summary long"] = task2_path

    task3_df = build_fc_decoder_summary_dataframe(bundles)
    task3_path = os.path.join(GROUP_OUT_DIR, "group_fc_decoder_summary_long.csv")
    task3_df.to_csv(task3_path, index=False)
    table_paths["Task3 FC decoder summary long"] = task3_path

    task4_stability_long_df = build_fc_edge_stability_long_dataframe(bundles)
    task4_stability_path = os.path.join(GROUP_OUT_DIR, "group_fc_edge_importance_stability_long.csv")
    task4_stability_long_df.to_csv(task4_stability_path, index=False)
    table_paths["Task4 edge stability long"] = task4_stability_path

    task4_stability_summary_df = build_fc_edge_stability_mouse_summary(task4_stability_long_df)
    task4_stability_summary_path = os.path.join(GROUP_OUT_DIR, "group_fc_edge_importance_mouse_summary.csv")
    task4_stability_summary_df.to_csv(task4_stability_summary_path, index=False)
    table_paths["Task4 edge stability mouse summary"] = task4_stability_summary_path

    task4_ablation_long_df = build_fc_edge_ablation_long_dataframe(bundles)
    task4_ablation_path = os.path.join(GROUP_OUT_DIR, "group_fc_edge_ablation_long.csv")
    task4_ablation_long_df.to_csv(task4_ablation_path, index=False)
    table_paths["Task4 edge ablation long"] = task4_ablation_path

    task4_proj_decile_long_df = build_fc_projection_decile_long_dataframe(bundles)
    task4_proj_decile_path = os.path.join(GROUP_OUT_DIR, "group_fc_projection_by_strength_decile_long.csv")
    task4_proj_decile_long_df.to_csv(task4_proj_decile_path, index=False)
    table_paths["Task4 projection decile long"] = task4_proj_decile_path

    task4_proj_layer_long_df = build_fc_projection_layer_pair_long_dataframe(bundles)
    task4_proj_layer_path = os.path.join(GROUP_OUT_DIR, "group_fc_projection_by_layer_pair_long.csv")
    task4_proj_layer_long_df.to_csv(task4_proj_layer_path, index=False)
    table_paths["Task4 projection layer pair long"] = task4_proj_layer_path

    task4_proj_sw_long_df = build_fc_projection_strong_weak_long_dataframe(bundles)
    task4_proj_sw_path = os.path.join(GROUP_OUT_DIR, "group_fc_projection_strong_weak_match_long.csv")
    task4_proj_sw_long_df.to_csv(task4_proj_sw_path, index=False)
    table_paths["Task4 projection strong-weak match long"] = task4_proj_sw_path

    task5_enrichment_long_df = build_fc_edge_decile_enrichment_long_dataframe(bundles)
    task5_path = os.path.join(GROUP_OUT_DIR, "group_fc_edge_decile_enrichment_long.csv")
    task5_enrichment_long_df.to_csv(task5_path, index=False)
    table_paths["Task5 edge decile enrichment long"] = task5_path

    task6_overlap_long_df = build_neuron_overlap_enrichment_long_dataframe(bundles)
    task6_overlap_path = os.path.join(GROUP_OUT_DIR, "group_neuron_overlap_enrichment_long.csv")
    task6_overlap_long_df.to_csv(task6_overlap_path, index=False)
    table_paths["Task6 neuron overlap enrichment long"] = task6_overlap_path

    task6_selectivity_long_df = build_neuron_selectivity_overlap_long_dataframe(bundles)
    task6_selectivity_path = os.path.join(GROUP_OUT_DIR, "group_neuron_selectivity_by_overlap_long.csv")
    task6_selectivity_long_df.to_csv(task6_selectivity_path, index=False)
    table_paths["Task6 neuron selectivity by overlap long"] = task6_selectivity_path

    coverage_frames = [
        ("Task1 decoder summary", task1_df),
        ("Task2 ablation summary", task2_df),
        ("Task3 FC decoder summary", task3_df),
        ("Task4 edge stability", task4_stability_long_df),
        ("Task4 edge ablation", task4_ablation_long_df),
        ("Task5 edge decile enrichment", task5_enrichment_long_df),
        ("Task6 overlap enrichment", task6_overlap_long_df),
        ("Task6 selectivity by overlap", task6_selectivity_long_df),
    ]
    for name, df_now in coverage_frames:
        n_rows = 0 if df_now is None else len(df_now)
        if df_now is None or df_now.empty or "mouse_id" not in df_now.columns:
            n_mice = 0
        else:
            n_mice = int(df_now["mouse_id"].nunique())
        print(f"[*] Coverage - {name}: {n_mice} mice, {n_rows} rows")

    # ===== Condition-level statistical tests =====
    metrics_to_test = [
        m
        for m in [
            "Entropy",
            "Mean_RSM_Sim",
            "Mean_Correlation",
            "Strong_Correlation",
            "Weak_Correlation",
            "Strong_Weak_Gap",
            "Participants_Ratio",
            "Gini_Mean",
            "PR_Mean",
            "Effective_Dim_PR",
            "Sig_Mean_Corr",
            "Noise_Mean_Corr",
        ]
        if m in master_df.columns and not master_df[m].isna().all()
    ]
    stat_results = {m: perform_statistical_tests(master_df, m) for m in metrics_to_test}

    stat_rows = []
    for metric, res in stat_results.items():
        ph = res.get("post_hoc", {})
        stat_rows.append(
            {
                "Metric": metric,
                "Main_Effect": res.get("main_effect", "N/A"),
                "p_main": res.get("p_main", np.nan),
                "Div_vs_Con": ph.get("Divergent vs Convergent", np.nan),
                "Div_vs_Rand": ph.get("Divergent vs Random", np.nan),
                "Con_vs_Rand": ph.get("Convergent vs Random", np.nan),
            }
        )
    stat_df = pd.DataFrame(stat_rows)
    stat_path = os.path.join(GROUP_OUT_DIR, "group_statistical_tests_summary.csv")
    stat_df.to_csv(stat_path, index=False)
    table_paths["Condition-level statistical tests"] = stat_path

    decoder_chain_summary_df = build_decoder_chain_summary_table(
        task1_df,
        task2_df,
        task3_df,
        task4_ablation_long_df,
        task5_enrichment_long_df,
        task6_overlap_long_df,
    )
    decoder_chain_summary_path = os.path.join(GROUP_OUT_DIR, "group_decoder_chain_summary.csv")
    decoder_chain_summary_df.to_csv(decoder_chain_summary_path, index=False)
    table_paths["Task1-6 decoder chain summary"] = decoder_chain_summary_path

    decoder_chain_stats_df = build_decoder_chain_stats_table(
        task1_df,
        task2_df,
        task3_df,
        task4_ablation_long_df,
        task5_enrichment_long_df,
        task6_overlap_long_df,
    )
    decoder_chain_stats_path = os.path.join(GROUP_OUT_DIR, "group_decoder_chain_stat_tests.csv")
    decoder_chain_stats_df.to_csv(decoder_chain_stats_path, index=False)
    table_paths["Task1-6 decoder chain statistical tests"] = decoder_chain_stats_path

    # ===== Figures =====
    image_paths = {}
    image_paths["Combined Strong vs Weak"] = plot_combined_strong_weak(master_df)

    core_metrics = [
        ("Mean_RSM_Sim", "Cosine similarity", "RSM Mean Similarity"),
        ("Strong_Correlation", "Correlation", "Strong Connections (Top 10%)"),
        ("Weak_Correlation", "Correlation", "Weak Connections (Bottom 10%)"),
        ("Strong_Weak_Gap", "Correlation gap", "Strong-Weak Correlation Gap"),
        ("Participants_Ratio", "Ratio", "RR Participants Ratio"),
        ("Gini_Mean", "Gini Coefficient", "Response Gini (Mean)"),
    ]
    for metric, ylabel, title in core_metrics:
        image_paths[title] = plot_group_metric(
            master_df,
            metric,
            ylabel,
            title,
            stat_res=stat_results.get(metric, {}),
            save_name=f"group_{metric.lower()}.png",
        )

    image_paths["Decile Correlation Curve"] = plot_decile_curve(decile_df)
    image_paths["Noise Decile Curve"] = plot_noise_decile_curve(noise_decile_long_df)
    image_paths["Cross-animal Binding"] = plot_cross_animal_binding(master_df)
    image_paths["Absolute State Binding"] = plot_absolute_state_binding(master_df)
    image_paths["LMM State Binding"] = plot_lmm_state_binding(master_df)

    image_paths["Decoder Accuracy (Task1+Task3)"] = plot_decoder_chain_accuracy(task1_df, task3_df)
    image_paths["Decoder Ablation (Task2)"] = plot_task2_ablation_summary(task2_df)
    image_paths["Edge Ablation Robustness (Task4)"] = plot_task4_edge_ablation(task4_ablation_long_df)
    image_paths["Edge Decile Enrichment (Task5)"] = plot_task5_decile_enrichment(task5_enrichment_long_df)
    image_paths["Neuron Linking (Task6)"] = plot_task6_linking(task6_overlap_long_df, task6_selectivity_long_df)

    generate_group_markdown(
        master_df,
        stat_results,
        image_paths,
        rr_overlap_df,
        table_paths=table_paths,
        decoder_chain_summary_df=decoder_chain_summary_df,
        decoder_chain_stats_df=decoder_chain_stats_df,
    )
    print("====== Group integration (tables, stats, figures, markdown) completed ======")

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    run_group_integration()
