import glob
import json
import os
import warnings
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

# 1. 鑾叞杩珮绾ч厤鑹叉柟妗?(Elegant, muted pastel palette)
COLORS = {
    "Divergent": "#7F9C96",
    "Convergent": "#8B90A8",
    "Random": "#B98372",
}

COND_WEAK_COLORS = {
    "Divergent": "#B8CBC6",
    "Convergent": "#C3C6D5",
    "Random": "#D9B7AD",
}
COND_STRONG_COLORS = {
    "Divergent": "#5F7E77",
    "Convergent": "#666C86",
    "Random": "#8E5E50",
}

NETWORK_TYPE_COLORS = {
    "strong": "#5F7088",
    "weak": "#B8C1CE",
    "strong_threshold": "#666C86",
    "weak_threshold": "#C3C6D5",
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
GEOMETRY_METRICS = [
    "mean_norm",
    "angle_deg",
    "var_parallel",
    "var_orthogonal",
    "orth_parallel_ratio",
    "anisotropy_index",
    "lambda1",
    "lambda2",
]

# ==========================================
# 2. 鍏ㄥ眬鏋佺畝瀛︽湳鎺掔増璁剧疆 (Global Publication Aesthetics)
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

def style_axis(ax, light_grid=False):
    """Minimal publication axis style."""
    sns.despine(ax=ax, trim=False)
    ax.tick_params(axis='both', which='major', length=5, pad=6)
    if light_grid:
        ax.grid(axis="y", color="#E9E5DF", linewidth=0.8)
    else:
        ax.grid(False)

# ==========================================
# 3. 鏁版嵁鍔犺浇涓庤В鏋愭牳蹇冮€昏緫 (Data Loading & Parsing)
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
    data_dirs = sorted(glob.glob(os.path.join(base_dir, "*", "data")))
    if not data_dirs:
        return []

    bundles = []
    for data_dir in data_dirs:
        folder_mouse_id = os.path.basename(os.path.dirname(data_dir))
        all_stats = sorted(glob.glob(os.path.join(data_dir, "*_statistics.json")))
        if not all_stats:
            continue

        preferred = os.path.join(data_dir, f"{folder_mouse_id}_statistics.json")
        if os.path.exists(preferred):
            fp = preferred
        else:
            # Fallback: choose latest file, but warn because this directory is ambiguous.
            fp = max(all_stats, key=os.path.getmtime)
            if len(all_stats) > 1:
                print(
                    f"[!] Multiple statistics JSONs in {data_dir}; "
                    f"using latest: {os.path.basename(fp)}"
                )

        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload_mouse_id = payload.get("mouse_id", None)
        if payload_mouse_id is not None and str(payload_mouse_id) != str(folder_mouse_id):
            print(
                f"[!] mouse_id mismatch in {fp}: "
                f"folder={folder_mouse_id}, payload={payload_mouse_id}. "
                f"Use folder mouse_id as canonical."
            )

        mouse_id = str(folder_mouse_id)
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
            # Shuffle analysis outputs (per-mouse)
            "shuffle_manifest": load_optional_csv(data_dir, "population_pattern_shuffle_manifest.csv"),
            "shuffle_corr_long": load_optional_csv(data_dir, "group_corr_shuffle_long.csv"),
            "shuffle_corr_decile_long": load_optional_csv(data_dir, "group_corr_decile_shuffle_long.csv"),
            "shuffle_rsm_long": load_optional_csv(data_dir, "group_rsm_shuffle_long.csv"),
            "shuffle_delta_long": load_optional_csv(data_dir, "group_shuffle_delta_long.csv"),
            "shuffle_dose_long": load_optional_csv(data_dir, "group_shuffle_dose_response_long.csv"),
            "shuffle_alloc_long": load_optional_csv(data_dir, "group_allocation_shuffle_long.csv"),
            "shuffle_effect_stats": load_optional_csv(data_dir, "group_shuffle_effect_stats.csv"),
            "shuffle_condition_summary": load_optional_csv(data_dir, "group_shuffle_condition_summary.csv"),
            "shuffle_condition_stats": load_optional_csv(data_dir, "group_shuffle_condition_stats.csv"),
            "shuffle_sync_contribution": load_optional_csv(data_dir, "group_shuffle_sync_contribution.csv"),
            "shuffle_sync_contribution_repeats": load_optional_csv(data_dir, "group_shuffle_sync_contribution_repeats.csv"),
            # RSM geometry outputs (per-mouse)
            "geometry_condition_level": load_optional_csv(data_dir, "geometry_condition_level_long.csv"),
            "geometry_condition_pairwise": load_optional_csv(data_dir, "geometry_condition_pairwise.csv"),
            "geometry_model_compare": load_optional_csv(data_dir, "geometry_rsm_model_compare.csv"),
            "geometry_vs_dimensionality": load_optional_csv(data_dir, "geometry_vs_dimensionality_model_compare.csv"),
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


def _geometry_by_condition(geometry_df, condition):
    cols = {}
    if geometry_df is None or geometry_df.empty:
        return cols
    g = geometry_df.copy()
    if "Condition" in g.columns:
        g["Condition"] = g["Condition"].map(normalize_condition).fillna(g["Condition"])
    elif "Class_Name" in g.columns:
        g["Condition"] = g["Class_Name"].map(normalize_condition).fillna(g["Class_Name"])
    else:
        return cols
    row = g[g["Condition"] == condition]
    if row.empty:
        return cols
    row = row.iloc[0]
    field_map = {
        "mean_norm": "Geom_MeanNorm",
        "angle_deg": "Geom_AngleDeg",
        "var_parallel": "Geom_VarParallel",
        "var_orthogonal": "Geom_VarOrthogonal",
        "orth_parallel_ratio": "Geom_OrthParallelRatio",
        "anisotropy_index": "Geom_Anisotropy",
        "lambda1": "Geom_Lambda1",
        "lambda2": "Geom_Lambda2",
    }
    for src, dst in field_map.items():
        if src in row:
            cols[dst] = safe_float(row[src])
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
            row.update(_geometry_by_condition(b.get("geometry_condition_level"), cond))
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


def _concat_bundle_table(bundles, key):
    rows = []
    for b in bundles:
        df = b.get(key)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        if "mouse_id" not in tmp.columns:
            if "mouse" in tmp.columns:
                tmp["mouse_id"] = tmp["mouse"].astype(str)
            else:
                tmp["mouse_id"] = b["mouse_id"]
        if "Condition" in tmp.columns:
            tmp["Condition"] = tmp["Condition"].map(normalize_condition).fillna(tmp["Condition"])
        if "condition" in tmp.columns:
            tmp["condition"] = tmp["condition"].map(normalize_condition).fillna(tmp["condition"])
            if "Condition" not in tmp.columns:
                tmp["Condition"] = tmp["condition"]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_shuffle_manifest_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_manifest")


def build_shuffle_corr_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_corr_long")


def build_shuffle_corr_decile_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_corr_decile_long")


def build_shuffle_rsm_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_rsm_long")


def build_shuffle_delta_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_delta_long")


def build_shuffle_dose_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_dose_long")


def build_shuffle_alloc_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_alloc_long")


def build_shuffle_effect_stats_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_effect_stats")


def build_shuffle_condition_summary_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_condition_summary")


def build_shuffle_condition_stats_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_condition_stats")


def build_shuffle_sync_contribution_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_sync_contribution")


def build_shuffle_sync_contribution_repeats_dataframe(bundles):
    return _concat_bundle_table(bundles, "shuffle_sync_contribution_repeats")


def build_geometry_condition_long_dataframe(bundles):
    df = _concat_bundle_table(bundles, "geometry_condition_level")
    if df.empty:
        return df
    if "Condition" in df.columns:
        df["Condition"] = df["Condition"].map(normalize_condition).fillna(df["Condition"])
    return df


def build_geometry_pairwise_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "geometry_condition_pairwise")


def build_geometry_model_compare_long_dataframe(bundles):
    return _concat_bundle_table(bundles, "geometry_model_compare")


def build_shuffle_core_long_dataframe(shuffle_corr_df, shuffle_rsm_df, shuffle_alloc_df):
    if shuffle_corr_df is None or shuffle_corr_df.empty:
        return pd.DataFrame()

    out = shuffle_corr_df.copy()
    keys = ["mouse_id", "Condition", "data_type", "repeat_id"]

    if shuffle_rsm_df is not None and not shuffle_rsm_df.empty and "mean_rsm" in shuffle_rsm_df.columns:
        keep = [c for c in keys + ["mean_rsm", "rsm_std", "rsm_entropy"] if c in shuffle_rsm_df.columns]
        out = out.merge(shuffle_rsm_df[keep], on=[c for c in keys if c in keep], how="left")

    if shuffle_alloc_df is not None and not shuffle_alloc_df.empty:
        alloc_metrics = [c for c in ["participants_ratio", "gini_mean", "pr_mean", "pr_norm_mean"] if c in shuffle_alloc_df.columns]
        if alloc_metrics:
            keep = [c for c in keys + alloc_metrics if c in shuffle_alloc_df.columns]
            out = out.merge(shuffle_alloc_df[keep], on=[c for c in keys if c in keep], how="left")

    return out


def summarize_shuffle_mouse_condition(core_df):
    if core_df is None or core_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in [
        "mean_corr", "weak_corr", "strong_corr", "strong_weak_gap",
        "mean_rsm", "pr_mean", "participants_ratio", "gini_mean"
    ] if c in core_df.columns]
    if not metric_cols:
        return pd.DataFrame()
    grp = (
        core_df.groupby(["mouse_id", "Condition", "data_type"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    return grp


def test_shuffle_original_vs_shuffled(core_df):
    """
    Group-level test across mice:
    per mouse+condition compare original vs mean(shuffled) using Wilcoxon.
    """
    mc = summarize_shuffle_mouse_condition(core_df)
    if mc.empty:
        return pd.DataFrame()

    metric_cols = [c for c in [
        "mean_corr", "weak_corr", "strong_corr", "strong_weak_gap",
        "mean_rsm", "pr_mean", "participants_ratio", "gini_mean"
    ] if c in mc.columns]

    rows = []
    for metric in metric_cols:
        for cond in CONDITIONS:
            sub = mc[mc["Condition"] == cond]
            piv = sub.pivot(index="mouse_id", columns="data_type", values=metric)
            if "original" not in piv.columns or "shuffled" not in piv.columns:
                rows.append({
                    "metric": metric, "Condition": cond, "n_mice": 0,
                    "orig_mean": np.nan, "shuffled_mean": np.nan,
                    "delta_orig_minus_shuffled": np.nan, "wilcoxon_p": np.nan
                })
                continue
            piv = piv[["original", "shuffled"]].dropna()
            if piv.empty:
                rows.append({
                    "metric": metric, "Condition": cond, "n_mice": 0,
                    "orig_mean": np.nan, "shuffled_mean": np.nan,
                    "delta_orig_minus_shuffled": np.nan, "wilcoxon_p": np.nan
                })
                continue

            try:
                _, p_val = stats.wilcoxon(piv["original"], piv["shuffled"]) if len(piv) >= 3 else (np.nan, np.nan)
            except Exception:
                p_val = np.nan

            rows.append({
                "metric": metric,
                "Condition": cond,
                "n_mice": int(len(piv)),
                "orig_mean": float(piv["original"].mean()),
                "shuffled_mean": float(piv["shuffled"].mean()),
                "delta_orig_minus_shuffled": float((piv["original"] - piv["shuffled"]).mean()),
                "wilcoxon_p": float(p_val) if pd.notna(p_val) else np.nan,
            })
    return pd.DataFrame(rows)


def test_shuffle_condition_differences(core_df, data_type="shuffled"):
    """
    Group-level condition test across mice for shuffled-only (or original-only) values.
    """
    mc = summarize_shuffle_mouse_condition(core_df)
    if mc.empty:
        return pd.DataFrame()

    sub = mc[mc["data_type"] == data_type].copy()
    if sub.empty:
        return pd.DataFrame()

    metric_cols = [c for c in [
        "mean_corr", "weak_corr", "strong_corr", "strong_weak_gap",
        "mean_rsm", "pr_mean", "participants_ratio", "gini_mean"
    ] if c in sub.columns]

    rows = []
    for metric in metric_cols:
        dup_n = int(sub.duplicated(["mouse_id", "Condition"]).sum())
        if dup_n > 0:
            print(f"[!] Duplicates found for shuffle condition test ({metric}): {dup_n}. Use mean aggregation.")
        piv = (
            sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean")
            .reindex(columns=CONDITIONS)
            .dropna()
        )
        if len(piv) < 3:
            rows.append({
                "metric": metric, "data_type": data_type,
                "main_effect": "N too small", "p_main": np.nan,
                "Divergent_vs_Convergent": np.nan,
                "Divergent_vs_Random": np.nan,
                "Convergent_vs_Random": np.nan,
            })
            continue
        stat, p_val = stats.friedmanchisquare(piv["Divergent"], piv["Convergent"], piv["Random"])
        pair_p = {}
        for c1, c2 in combinations(CONDITIONS, 2):
            try:
                _, p_pair = stats.wilcoxon(piv[c1], piv[c2])
            except Exception:
                p_pair = np.nan
            pair_p[f"{c1}_vs_{c2}"] = p_pair
        rows.append({
            "metric": metric,
            "data_type": data_type,
            "main_effect": rf"Friedman $\chi^2$={stat:.2f}, $p$={p_val:.3e}",
            "p_main": float(p_val),
            "Divergent_vs_Convergent": pair_p.get("Divergent_vs_Convergent", np.nan),
            "Divergent_vs_Random": pair_p.get("Divergent_vs_Random", np.nan),
            "Convergent_vs_Random": pair_p.get("Convergent_vs_Random", np.nan),
        })
    return pd.DataFrame(rows)


def test_shuffle_sync_contribution(sync_df):
    """
    Group-level synchrony contribution test across mice.
    H0: median(sync_contribution_abs) = 0
    H1: median(sync_contribution_abs) > 0
    """
    if sync_df is None or sync_df.empty:
        return pd.DataFrame()

    work = sync_df.copy()
    if "mouse_id" not in work.columns and "mouse" in work.columns:
        work["mouse_id"] = work["mouse"].astype(str)

    if "sync_contribution_abs" not in work.columns:
        if {"abs_contrast_original", "abs_contrast_shuffled_mean"}.issubset(set(work.columns)):
            work["sync_contribution_abs"] = (
                pd.to_numeric(work["abs_contrast_original"], errors="coerce")
                - pd.to_numeric(work["abs_contrast_shuffled_mean"], errors="coerce")
            )
        else:
            return pd.DataFrame()

    per_mouse = (
        work.groupby(["mouse_id", "metric"], as_index=False)["sync_contribution_abs"]
        .mean(numeric_only=True)
    )
    if per_mouse.empty:
        return pd.DataFrame()

    rows = []
    for metric in sorted(per_mouse["metric"].dropna().unique()):
        vals = per_mouse.loc[per_mouse["metric"] == metric, "sync_contribution_abs"].astype(float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            continue

        try:
            sign_p = stats.binomtest(int(np.sum(vals > 0)), n=n, p=0.5, alternative="greater").pvalue
        except Exception:
            sign_p = np.nan

        if n >= 3:
            try:
                w_stat, w_p_two = stats.wilcoxon(vals)
            except Exception:
                w_stat, w_p_two = np.nan, np.nan
            try:
                _, w_p_greater = stats.wilcoxon(vals, alternative="greater")
            except Exception:
                w_p_greater = np.nan
        else:
            w_stat, w_p_two, w_p_greater = np.nan, np.nan, np.nan

        rows.append(
            {
                "metric": metric,
                "n_mice": n,
                "mean_sync_contribution_abs": float(np.mean(vals)),
                "sem_sync_contribution_abs": float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "median_sync_contribution_abs": float(np.median(vals)),
                "n_positive": int(np.sum(vals > 0)),
                "ratio_positive": float(np.mean(vals > 0)),
                "wilcoxon_stat": float(w_stat) if pd.notna(w_stat) else np.nan,
                "wilcoxon_p_two_sided": float(w_p_two) if pd.notna(w_p_two) else np.nan,
                "wilcoxon_p_one_sided_greater": float(w_p_greater) if pd.notna(w_p_greater) else np.nan,
                "binom_p_one_sided_greater": float(sign_p) if pd.notna(sign_p) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def run_geometry_lmm_models(master_df):
    need_cols = {
        "mouse_id",
        "Condition",
        "Mean_RSM_Sim",
        "Participants_Ratio",
        "Effective_Dim_PR",
        "Geom_AngleDeg",
        "Geom_OrthParallelRatio",
    }
    if master_df is None or master_df.empty or not need_cols.issubset(set(master_df.columns)):
        return pd.DataFrame()

    work = master_df[list(need_cols)].copy()
    work = work[work["Condition"].isin(CONDITIONS)].copy()

    model_specs = [
        ("M1", "Mean_RSM_Sim ~ Geom_AngleDeg", ["Mean_RSM_Sim", "Geom_AngleDeg"], ["Geom_AngleDeg"]),
        ("M2", "Mean_RSM_Sim ~ Geom_OrthParallelRatio", ["Mean_RSM_Sim", "Geom_OrthParallelRatio"], ["Geom_OrthParallelRatio"]),
        ("M3", "Mean_RSM_Sim ~ Participants_Ratio + Geom_AngleDeg", ["Mean_RSM_Sim", "Participants_Ratio", "Geom_AngleDeg"], ["Participants_Ratio", "Geom_AngleDeg"]),
        ("M4", "Mean_RSM_Sim ~ Participants_Ratio + Geom_OrthParallelRatio", ["Mean_RSM_Sim", "Participants_Ratio", "Geom_OrthParallelRatio"], ["Participants_Ratio", "Geom_OrthParallelRatio"]),
        ("A1", "Geom_AngleDeg ~ Participants_Ratio", ["Geom_AngleDeg", "Participants_Ratio"], ["Participants_Ratio"]),
        ("A2", "Geom_OrthParallelRatio ~ Participants_Ratio", ["Geom_OrthParallelRatio", "Participants_Ratio"], ["Participants_Ratio"]),
        ("D1", "Mean_RSM_Sim ~ Effective_Dim_PR", ["Mean_RSM_Sim", "Effective_Dim_PR"], ["Effective_Dim_PR"]),
        ("D2", "Mean_RSM_Sim ~ Geom_AngleDeg + Effective_Dim_PR", ["Mean_RSM_Sim", "Geom_AngleDeg", "Effective_Dim_PR"], ["Geom_AngleDeg", "Effective_Dim_PR"]),
        ("D3", "Mean_RSM_Sim ~ Geom_OrthParallelRatio + Effective_Dim_PR", ["Mean_RSM_Sim", "Geom_OrthParallelRatio", "Effective_Dim_PR"], ["Geom_OrthParallelRatio", "Effective_Dim_PR"]),
    ]

    rows = []
    for model_name, formula, cols, terms in model_specs:
        sub = work[["mouse_id"] + cols].dropna().copy()
        n_obs = int(len(sub))
        n_mice = int(sub["mouse_id"].nunique())
        if n_obs < 6 or n_mice < 3:
            for term in terms:
                rows.append(
                    {
                        "model_name": model_name,
                        "formula": formula,
                        "term": term,
                        "beta": np.nan,
                        "p_value": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "llf": np.nan,
                        "n_obs": n_obs,
                        "n_mice": n_mice,
                        "converged": False,
                        "note": "N too small",
                    }
                )
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, sub, groups=sub["mouse_id"])
                fit = model.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            for term in terms:
                rows.append(
                    {
                        "model_name": model_name,
                        "formula": formula,
                        "term": term,
                        "beta": safe_float(fit.params.get(term, np.nan)),
                        "p_value": safe_float(fit.pvalues.get(term, np.nan)),
                        "aic": safe_float(getattr(fit, "aic", np.nan)),
                        "bic": safe_float(getattr(fit, "bic", np.nan)),
                        "llf": safe_float(getattr(fit, "llf", np.nan)),
                        "n_obs": n_obs,
                        "n_mice": n_mice,
                        "converged": bool(getattr(fit, "converged", False)),
                        "note": "",
                    }
                )
        except Exception as exc:
            for term in terms:
                rows.append(
                    {
                        "model_name": model_name,
                        "formula": formula,
                        "term": term,
                        "beta": np.nan,
                        "p_value": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "llf": np.nan,
                        "n_obs": n_obs,
                        "n_mice": n_mice,
                        "converged": False,
                        "note": f"fit failed: {exc}",
                    }
                )

    return pd.DataFrame(rows)


def perform_statistical_tests(df, metric):
    dup_n = int(df[["mouse_id", "Condition", metric]].dropna().duplicated(["mouse_id", "Condition"]).sum())
    if dup_n > 0:
        print(f"[!] Duplicates found for {metric}: {dup_n}. Use mean aggregation by mouse+condition.")
    pivot = (
        df.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean")
        .reindex(columns=CONDITIONS)
        .dropna()
    )
    if len(pivot) < 3:
        return {"main_effect": "N too small", "p_main": np.nan, "post_hoc": {}}

    stat, p_val = stats.friedmanchisquare(pivot["Divergent"], pivot["Convergent"], pivot["Random"])
    # 淇锛氬姞涓?rf 鍓嶇紑闃叉 \c 琚鍛婏紝杈撳嚭鏍囧噯鐨勫崱鏂圭鍙?
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


def df_to_md(df):
    if df is None or len(df) == 0:
        return "_No data._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"

# ==========================================
# 4. 浼橀泤鐨勫彲瑙嗗寲鍑芥暟 (Elegant Publication Plotting Functions)
# ==========================================
def save_figure_variants(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)
    # 鐢熸垚渚涜鏂囨帓鐗堢殑鏃犳爣棰樼増鏈?
    suptitle = fig._suptitle
    if suptitle is not None: suptitle.set_visible(False)
    for ax in fig.axes: ax.set_title("")
    stem, ext = os.path.splitext(save_path)
    fig.savefig(f"{stem}_notitle{ext}", dpi=300, bbox_inches="tight", transparent=False)
    return save_path

def plot_group_metric(df, metric, ylabel, title, stat_res, save_name):
    """Paired estimation-style: violin + raw points + mouse lines + mean卤SEM."""
    if metric not in df.columns or df[metric].isna().all(): return None

    sub = df[["mouse_id", "Condition", metric]].dropna().copy()
    if sub.empty: return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    order = CONDITIONS
    x_pos = np.arange(len(order))

    sns.violinplot(
        data=sub,
        x="Condition",
        y=metric,
        order=order,
        hue="Condition",
        hue_order=order,
        palette=COLORS,
        inner="quartile",
        cut=0,
        linewidth=1.0,
        alpha=0.26,
        legend=False,
        ax=ax,
    )

    pivot = (
        sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean")
        .reindex(columns=order)
    )
    for _, row in pivot.iterrows():
        y = row.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            ax.plot(x_pos[mask], y[mask], color="#A9A39A", lw=0.9, alpha=0.55, zorder=2)

    sns.stripplot(
        data=sub,
        x="Condition",
        y=metric,
        order=order,
        hue="Condition",
        hue_order=order,
        palette=COLORS,
        dodge=False,
        size=4.2,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
        legend=False,
        ax=ax,
    )

    means = sub.groupby("Condition")[metric].mean().reindex(order)
    sems = sub.groupby("Condition")[metric].sem().reindex(order)
    for i, cond in enumerate(order):
        if pd.isna(means[cond]):
            continue
        ax.errorbar(
            i + 0.18,
            means[cond],
            yerr=sems[cond],
            fmt="o",
            color="#2F2F2F",
            ecolor="#2F2F2F",
            markersize=5.5,
            lw=2.0,
            capsize=0,
            zorder=5,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{stat_res.get('main_effect', '')}", pad=15)
    style_axis(ax)

    # 缁樺埗鏄捐憲鎬ф槦鍙锋敮鏋?
    if pd.notna(stat_res.get("p_main")) and stat_res.get("p_main", 1.0) < 0.1:
        y_max = sub[metric].max() * 1.02
        y_range = max(sub[metric].max() - sub[metric].min(), 1e-6)
        step = y_range * 0.08
        base = y_max

        sig_pairs = []
        for c1, c2 in combinations(CONDITIONS, 2):
            p = stat_res.get("post_hoc", {}).get(f"{c1} vs {c2}", np.nan)
            if p_to_star(p) != "ns": sig_pairs.append((c1, c2, p_to_star(p)))
        
        for i, (c1, c2, star) in enumerate(sig_pairs):
            x1, x2 = CONDITIONS.index(c1), CONDITIONS.index(c2)
            y = base + i * step
            ax.plot([x1, x1, x2, x2], [y, y + step*0.2, y + step*0.2, y], lw=1.3, c="#333333")
            ax.text((x1 + x2) * 0.5, y + step*0.25, star, ha="center", va="bottom", color="#111111", fontsize=12, fontweight='bold')
        if sig_pairs: ax.set_ylim(top=base + len(sig_pairs)*step + step*1.5)

    out = os.path.join(GROUP_OUT_DIR, save_name)
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_combined_strong_weak(df):
    """Condition-wise strong-vs-weak endpoint dumbbell panel (group mean 卤95% CI)."""
    required = ["Strong_Correlation", "Weak_Correlation"]
    if not all(c in df.columns for c in required): return None

    sub = df[["mouse_id", "Condition", "Strong_Correlation", "Weak_Correlation"]].dropna().copy()
    if sub.empty: return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    order = CONDITIONS
    x = np.arange(len(order))

    for i, cond in enumerate(order):
        vals_w = sub.loc[sub["Condition"] == cond, "Weak_Correlation"].astype(float).dropna()
        vals_s = sub.loc[sub["Condition"] == cond, "Strong_Correlation"].astype(float).dropna()
        if len(vals_w) == 0 or len(vals_s) == 0:
            continue

        m_w, m_s = float(vals_w.mean()), float(vals_s.mean())
        ci_w = 1.96 * float(vals_w.sem()) if len(vals_w) > 1 else 0.0
        ci_s = 1.96 * float(vals_s.sem()) if len(vals_s) > 1 else 0.0

        ax.plot([i, i], [m_w, m_s], color="#8D8A84", lw=2.0, alpha=0.9, zorder=2)
        ax.errorbar(
            i - 0.06, m_w, yerr=ci_w, fmt="o",
            color=COND_WEAK_COLORS[cond], ecolor=COND_WEAK_COLORS[cond],
            lw=1.8, capsize=0, zorder=4
        )
        ax.errorbar(
            i + 0.06, m_s, yerr=ci_s, fmt="o",
            color=COND_STRONG_COLORS[cond], ecolor=COND_STRONG_COLORS[cond],
            lw=1.8, capsize=0, zorder=4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel("Mean correlation")
    ax.set_title("Strong vs Weak Endpoints")
    style_axis(ax)

    out = os.path.join(GROUP_OUT_DIR, "group_combined_strong_weak.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_decile_curve(decile_df):
    """甯︽湁骞虫粦闃村奖璇樊甯︾殑鍒嗗眰鏇茬嚎 (Shaded Error Bands)"""
    if decile_df.empty: return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for cond in CONDITIONS:
        sub = decile_df[decile_df["Condition"] == cond]
        if sub.empty: continue
        agg = sub.groupby("Decile_Index")["Mean_Correlation"].agg(['mean', 'sem'])
        
        # 涓荤嚎
        ax.plot(agg.index, agg['mean'], label=cond, color=COLORS[cond], lw=2.5, marker='o', markersize=6, markeredgecolor='white')
        # 闃村奖甯?
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


def plot_geometry_condition_metric(geometry_df, metric, ylabel, save_name):
    if geometry_df is None or geometry_df.empty or metric not in geometry_df.columns:
        return None
    sub = geometry_df[["mouse_id", "Condition", metric]].dropna().copy()
    if sub.empty:
        return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
    order = CONDITIONS
    x_pos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    sns.violinplot(
        data=sub,
        x="Condition",
        y=metric,
        order=order,
        hue="Condition",
        hue_order=order,
        palette=COLORS,
        inner="quartile",
        cut=0,
        linewidth=1.0,
        alpha=0.25,
        legend=False,
        ax=ax,
    )

    piv = (
        sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean")
        .reindex(columns=order)
    )
    for _, row in piv.iterrows():
        y = row.values.astype(float)
        m = ~np.isnan(y)
        if m.sum() >= 2:
            ax.plot(x_pos[m], y[m], color="#AAA49A", lw=0.8, alpha=0.55, zorder=2)

    sns.stripplot(
        data=sub,
        x="Condition",
        y=metric,
        order=order,
        hue="Condition",
        hue_order=order,
        palette=COLORS,
        dodge=False,
        size=4.0,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        legend=False,
        ax=ax,
    )

    means = sub.groupby("Condition")[metric].mean().reindex(order)
    sems = sub.groupby("Condition")[metric].sem().reindex(order)
    for i, cond in enumerate(order):
        if pd.isna(means[cond]):
            continue
        ax.errorbar(
            i + 0.16,
            means[cond],
            yerr=sems[cond],
            fmt="o",
            color="#2F2F2F",
            ecolor="#2F2F2F",
            markersize=5.5,
            lw=2.0,
            capsize=0,
            zorder=5,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    style_axis(ax)
    out = os.path.join(GROUP_OUT_DIR, save_name)
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_geometry_vs_rsm(master_df, x_col, xlabel, save_name):
    need_cols = {"mouse_id", "Condition", "Mean_RSM_Sim", x_col}
    if master_df is None or master_df.empty or not need_cols.issubset(set(master_df.columns)):
        return None
    sub = master_df[["mouse_id", "Condition", "Mean_RSM_Sim", x_col]].dropna().copy()
    if sub.empty:
        return None
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sns.regplot(
        data=sub,
        x=x_col,
        y="Mean_RSM_Sim",
        scatter=False,
        color="#444444",
        line_kws={"linewidth": 2.0, "linestyle": "--", "alpha": 0.8},
        ax=ax,
    )
    sns.scatterplot(
        data=sub,
        x=x_col,
        y="Mean_RSM_Sim",
        hue="Condition",
        palette=COLORS,
        s=68,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    try:
        lr = stats.linregress(sub[x_col].values, sub["Mean_RSM_Sim"].values)
        text = f"slope={lr.slope:.4f}\np={lr.pvalue:.3e}"
    except Exception:
        text = "slope=NA\np=NA"
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#D0D0D0"),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean RSM similarity")
    ax.set_title(f"Mean RSM vs {xlabel}")
    style_axis(ax)
    ax.legend(frameon=False, title="")

    out = os.path.join(GROUP_OUT_DIR, save_name)
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_shuffle_orig_vs_shuffled(core_df, metric, ylabel, stat_df=None):
    mc = summarize_shuffle_mouse_condition(core_df)
    if mc.empty or metric not in mc.columns:
        return None
    sub = mc[["mouse_id", "Condition", "data_type", metric]].dropna().copy()
    if sub.empty:
        return None

    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
    order = CONDITIONS
    hue_order = ["original", "shuffled"]
    pal = {"original": "#2F4858", "shuffled": "#9AA5B1"}

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    sns.boxplot(
        data=sub, x="Condition", y=metric, hue="data_type",
        order=order, hue_order=hue_order, palette=pal,
        width=0.62, showfliers=False, linewidth=1.2, ax=ax
    )
    sns.stripplot(
        data=sub, x="Condition", y=metric, hue="data_type",
        order=order, hue_order=hue_order, palette=pal,
        dodge=True, size=3, alpha=0.45, linewidth=0, ax=ax
    )
    # remove duplicated legend from box+strip
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = []
        used = set()
        for h, l in zip(handles, labels):
            if l in used:
                continue
            used.add(l)
            uniq.append((h, l))
        ax.legend([h for h, _ in uniq], [l for _, l in uniq], frameon=False, title="", loc="best")

    if stat_df is not None and not stat_df.empty:
        s = stat_df[stat_df["metric"] == metric].copy()
        p_map = {r["Condition"]: r.get("wilcoxon_p", np.nan) for _, r in s.iterrows() if "Condition" in s.columns}
        y_top = sub[metric].max()
        y_step = max((sub[metric].max() - sub[metric].min()) * 0.08, 1e-4)
        for i, cond in enumerate(order):
            p = p_map.get(cond, np.nan)
            star = p_to_star(p)
            if star == "ns":
                continue
            ax.text(i, y_top + y_step, star, ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylim(top=y_top + y_step * 2.2)

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric}: Original vs Shuffled")
    style_axis(ax)

    out = os.path.join(GROUP_OUT_DIR, f"group_shuffle_orig_vs_shuffled_{metric}.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_shuffle_condition_only(core_df, metric, ylabel, data_type="shuffled", stat_df=None):
    mc = summarize_shuffle_mouse_condition(core_df)
    if mc.empty or metric not in mc.columns:
        return None
    sub = mc[mc["data_type"] == data_type][["mouse_id", "Condition", metric]].dropna().copy()
    if sub.empty:
        return None

    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.boxplot(
        data=sub, x="Condition", y=metric, hue="Condition",
        palette=COLORS, legend=False, width=0.62, showfliers=False, linewidth=1.2, ax=ax
    )
    sns.stripplot(
        data=sub, x="Condition", y=metric, color="#333333",
        alpha=0.55, size=3.5, jitter=0.14, ax=ax
    )

    title = f"{metric}: {data_type.capitalize()} condition differences"
    if stat_df is not None and not stat_df.empty:
        row = stat_df[(stat_df["metric"] == metric) & (stat_df["data_type"] == data_type)]
        if not row.empty and pd.notna(row.iloc[0].get("p_main", np.nan)):
            title += f"\nMain p={row.iloc[0]['p_main']:.3e}"
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    style_axis(ax)

    out = os.path.join(GROUP_OUT_DIR, f"group_shuffle_condition_{data_type}_{metric}.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_shuffle_dose_response_group(dose_df, metric, ylabel):
    if dose_df is None or dose_df.empty or metric not in dose_df.columns:
        return None
    if "Condition" not in dose_df.columns or "shuffle_fraction" not in dose_df.columns:
        return None

    # mean over repeats within each mouse first (avoid pseudo-replication)
    per_mouse = (
        dose_df.groupby(["mouse_id", "Condition", "shuffle_fraction"], as_index=False)[metric]
        .mean(numeric_only=True)
    )
    if per_mouse.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for cond in CONDITIONS:
        sub = per_mouse[per_mouse["Condition"] == cond]
        if sub.empty:
            continue
        agg = sub.groupby("shuffle_fraction")[metric].agg(["mean", "sem"]).reset_index()
        ax.plot(agg["shuffle_fraction"], agg["mean"], marker="o", lw=2.2, color=COLORS[cond], label=cond)
        ax.fill_between(
            agg["shuffle_fraction"],
            agg["mean"] - agg["sem"],
            agg["mean"] + agg["sem"],
            color=COLORS[cond], alpha=0.18, linewidth=0
        )
    ax.set_xlabel("Shuffle Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Dose-response: {metric}")
    ax.set_xticks(sorted(per_mouse["shuffle_fraction"].dropna().unique()))
    style_axis(ax)
    ax.legend(frameon=False, title="")

    out = os.path.join(GROUP_OUT_DIR, f"group_shuffle_dose_{metric}.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_shuffle_delta_group(delta_df):
    if delta_df is None or delta_df.empty:
        return None
    need = {"Condition", "metric", "delta_shuffle"}
    if not need.issubset(set(delta_df.columns)):
        return None

    per_mouse = (
        delta_df.groupby(["mouse_id", "Condition", "metric"], as_index=False)["delta_shuffle"]
        .mean(numeric_only=True)
    )
    if per_mouse.empty:
        return None

    metrics = list(per_mouse["metric"].dropna().unique())
    if not metrics:
        return None

    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), dpi=180)
    axes = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = per_mouse[per_mouse["metric"] == metric].copy()
        sns.boxplot(
            data=sub, x="Condition", y="delta_shuffle", hue="Condition",
            palette=COLORS, legend=False, showfliers=False, linewidth=1.2, ax=ax
        )
        sns.stripplot(
            data=sub, x="Condition", y="delta_shuffle", color="#303030",
            size=3.5, jitter=0.14, alpha=0.55, ax=ax
        )
        ax.axhline(0, color="#777777", lw=1, ls="--")
        ax.set_xlabel("")
        ax.set_ylabel("Delta")
        ax.set_title(metric)
        style_axis(ax)

    for j in range(len(metrics), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Shuffle Delta by Condition", y=1.02)
    out = os.path.join(GROUP_OUT_DIR, "group_shuffle_delta_by_condition.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out


def plot_shuffle_sync_contribution_group(sync_df, stat_df=None):
    if sync_df is None or sync_df.empty:
        return None

    work = sync_df.copy()
    if "mouse_id" not in work.columns and "mouse" in work.columns:
        work["mouse_id"] = work["mouse"].astype(str)

    if "sync_contribution_abs" not in work.columns:
        if {"abs_contrast_original", "abs_contrast_shuffled_mean"}.issubset(set(work.columns)):
            work["sync_contribution_abs"] = (
                pd.to_numeric(work["abs_contrast_original"], errors="coerce")
                - pd.to_numeric(work["abs_contrast_shuffled_mean"], errors="coerce")
            )
        else:
            return None

    plot_df = (
        work.groupby(["mouse_id", "metric"], as_index=False)["sync_contribution_abs"]
        .mean(numeric_only=True)
        .dropna()
    )
    if plot_df.empty:
        return None

    metric_order = [
        m for m in ["weak_corr", "strong_weak_gap", "mean_rsm", "strong_corr", "mean_corr", "pr_mean", "gini_mean", "participants_ratio"]
        if m in set(plot_df["metric"].astype(str))
    ]
    if not metric_order:
        metric_order = sorted(plot_df["metric"].astype(str).unique().tolist())

    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.72 * len(metric_order) + 1.8)), dpi=180)
    sns.boxplot(
        data=plot_df,
        y="metric",
        x="sync_contribution_abs",
        order=metric_order,
        color="#D6DDE4",
        linewidth=1.2,
        fliersize=0,
        width=0.58,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        y="metric",
        x="sync_contribution_abs",
        order=metric_order,
        color="#2E3A46",
        size=4.6,
        alpha=0.75,
        jitter=0.16,
        ax=ax,
    )

    means = plot_df.groupby("metric")["sync_contribution_abs"].mean().reindex(metric_order)
    sems = plot_df.groupby("metric")["sync_contribution_abs"].sem().reindex(metric_order)
    y_idx = np.arange(len(metric_order))
    ax.errorbar(
        means.values,
        y_idx,
        xerr=sems.values,
        fmt="o",
        color="#8C4A3E",
        ecolor="#8C4A3E",
        elinewidth=2.0,
        capsize=0,
        markersize=6.2,
        zorder=5,
    )

    ax.axvline(0, color="#666666", lw=1.0, ls="--")
    ax.set_xlabel("Synchrony Contribution (|Random - Coherent|: original - shuffled mean)")
    ax.set_ylabel("Metric")
    ax.set_title("Shuffle Synchrony Contribution Across Mice")
    style_axis(ax)

    if stat_df is not None and not stat_df.empty and "metric" in stat_df.columns:
        stat_map = stat_df.set_index("metric")
        x_max = np.nanmax(plot_df["sync_contribution_abs"].values)
        x_min = np.nanmin(plot_df["sync_contribution_abs"].values)
        x_span = max(1e-9, x_max - x_min)
        x_text = x_max + 0.05 * x_span
        for yi, metric in enumerate(metric_order):
            if metric not in stat_map.index:
                continue
            p_val = stat_map.loc[metric, "wilcoxon_p_one_sided_greater"] if "wilcoxon_p_one_sided_greater" in stat_map.columns else np.nan
            if isinstance(p_val, pd.Series):
                p_val = p_val.iloc[0]
            ax.text(x_text, yi, f"p={p_val:.3g}" if pd.notna(p_val) else "p=NA", va="center", ha="left", fontsize=9, color="#303030")
        ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.23 * x_span)

    out = os.path.join(GROUP_OUT_DIR, "group_shuffle_sync_contribution_abs.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

# ==========================================
# 5. Markdown 鎶ュ憡鑷姩鐢熸垚妯″潡
# ==========================================
def generate_group_markdown(
    master_df,
    stat_results,
    image_paths,
    rr_overlap_df,
    shuffle_payload=None,
    geometry_payload=None,
):
    md_path = os.path.join(GROUP_OUT_DIR, "Group_Analysis_Report.md")
    
    # 蹇界暐 FutureWarning (observed=False)
    numeric_cols = [c for c in master_df.columns if c not in ["mouse_id", "Condition"] and pd.api.types.is_numeric_dtype(master_df[c])]
    summary_df = master_df.groupby("Condition", observed=False)[numeric_cols].agg(["mean", "sem"]).round(4)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group-level Multi-mouse Analysis Report\n\n")
        f.write(f"**Number of mice**: {master_df['mouse_id'].nunique()}\n\n")
        f.write(f"**Mouse IDs**: {', '.join(sorted(master_df['mouse_id'].unique()))}\n\n")

        f.write("## 1. Descriptive Statistics (Mean 卤 SEM)\n\n")
        desc = pd.DataFrame(index=summary_df.index)
        for col in summary_df.columns.levels[0]:
            desc[col] = summary_df[col]["mean"].astype(str) + " 卤 " + summary_df[col]["sem"].astype(str)
        f.write(df_to_md(desc.reset_index()) + "\n\n")

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
            f.write(df_to_md(rr_summary) + "\n\n")

        if shuffle_payload is not None:
            ov = shuffle_payload.get("orig_vs_shuffled_stats", pd.DataFrame())
            cs = shuffle_payload.get("condition_stats", pd.DataFrame())
            sc = shuffle_payload.get("sync_contribution_stats", pd.DataFrame())
            csv_paths = shuffle_payload.get("csv_paths", {})
            if ov is not None and not ov.empty:
                f.write("## 4. Shuffle: Original vs Shuffled (Across Mice)\n\n")
                f.write(df_to_md(ov) + "\n\n")
            if cs is not None and not cs.empty:
                f.write("## 5. Shuffle: Condition Differences in Shuffled Surrogates\n\n")
                f.write(df_to_md(cs) + "\n\n")
            if sc is not None and not sc.empty:
                f.write("## 6. Shuffle: Synchrony Contribution (Random - Coherent)\n\n")
                f.write(df_to_md(sc) + "\n\n")
            if csv_paths:
                f.write("## 7. Group Shuffle Output Files\n\n")
                for k, p in csv_paths.items():
                    f.write(f"- {k}: `{p}`\n")
                f.write("\n")

        if geometry_payload is not None:
            g_cond = geometry_payload.get("condition_long", pd.DataFrame())
            g_pair = geometry_payload.get("pairwise_long", pd.DataFrame())
            g_model = geometry_payload.get("model_compare", pd.DataFrame())
            g_paths = geometry_payload.get("csv_paths", {})
            if g_cond is not None and not g_cond.empty:
                f.write("## Geometry Condition-level Table\n\n")
                f.write(df_to_md(g_cond) + "\n\n")
            if g_pair is not None and not g_pair.empty:
                f.write("## Geometry Pairwise Bootstrap Table\n\n")
                f.write(df_to_md(g_pair) + "\n\n")
            if g_model is not None and not g_model.empty:
                f.write("## Geometry Mixed-model Summary Table\n\n")
                f.write(df_to_md(g_model) + "\n\n")
            if g_paths:
                f.write("## Geometry Output Files\n\n")
                for k, p in g_paths.items():
                    f.write(f"- {k}: `{p}`\n")
                f.write("\n")

        f.write("## 8. Figures\n\n")
        for name, path in image_paths.items():
            if path is None: continue
            rel = os.path.basename(path)
            f.write(f"### {name}\n![{name}](./{rel})\n\n")
    print(f"[*] Group markdown report written to: {md_path}")

def plot_cross_animal_binding(df):
    # 纭繚闇€瑕佺殑鍒楀瓨鍦?
    required_cols = ["Gini_Mean", "Mean_RSM_Sim"]
    if not all(c in df.columns for c in required_cols):
        print("Missing columns for cross-animal binding.")
        return None

    # 鎻愬彇鎵€闇€鏁版嵁
    sub = df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    
    # 灏嗘暟鎹€忚锛屾瘡鍙皬榧犱竴琛?
    pivot_gini = sub.pivot_table(index="mouse_id", columns="Condition", values="Participants_Ratio", aggfunc="mean")
    pivot_rsm = sub.pivot_table(index="mouse_id", columns="Condition", values="Mean_RSM_Sim", aggfunc="mean")
    
    # 纭繚涓変釜鏉′欢閮藉瓨鍦?
    for cond in ["Divergent", "Convergent", "Random"]:
        if cond not in pivot_gini.columns or cond not in pivot_rsm.columns:
            return None

    # 璁＄畻 Coherent Motion 鐨勫潎鍊?
    pivot_gini["Coherent"] = pivot_gini[["Divergent", "Convergent"]].mean(axis=1)
    pivot_rsm["Coherent"] = pivot_rsm[["Divergent", "Convergent"]].mean(axis=1)

    # 璁＄畻璋冨埗閲?(Delta = Coherent - Random)
    delta_df = pd.DataFrame(index=pivot_gini.index)
    delta_df["Delta_Participants_Ratio"] = pivot_gini["Coherent"] - pivot_gini["Random"]
    delta_df["Delta_RSM"] = pivot_rsm["Coherent"] - pivot_rsm["Random"]
    delta_df = delta_df.dropna()

    if len(delta_df) < 3:
        return None

    # 缁熻妫€楠?(Spearman for robustness, Pearson for reference)
    spearman_r, spearman_p = stats.spearmanr(delta_df["Delta_Participants_Ratio"], delta_df["Delta_RSM"])
    pearson_r, pearson_p = stats.pearsonr(delta_df["Delta_Participants_Ratio"], delta_df["Delta_RSM"])

    # 寮€濮嬬粯鍥?
    fig, ax = plt.subplots(figsize=(5.5, 5))
    
    # 淇浜?seaborn 鍜?matplotlib 涔嬮棿鐨?linewidth 鍒悕鍐茬獊鎶ラ敊
    sns.regplot(
        data=delta_df, x="Delta_Participants_Ratio", y="Delta_RSM",
        ax=ax, color="#404040", 
        scatter_kws={"s": 60, "alpha": 0.8, "edgecolors": "white", "linewidths": 1},
        line_kws={"linewidth": 2, "color": "#202020", "alpha": 0.8}
    )

    # 鍦ㄥ浘涓爣娉ㄦ瘡鍙紶鐨?ID锛堝府鍔╂鏌?outlier锛?
    for mouse_id, row in delta_df.iterrows():
        ax.annotate(mouse_id, (row["Delta_Participants_Ratio"], row["Delta_RSM"]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color="#606060", alpha=0.7)

    # 娣诲姞缁熻缁撴灉鏂囨湰妗?
    stat_text = (f"Spearman $r_s$ = {spearman_r:.2f}, $p$ = {spearman_p:.3f}\n"
                 f"Pearson $r$ = {pearson_r:.2f}, $p$ = {pearson_p:.3f}")
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    # 澧炲姞杩囬浂鐐圭殑杈呭姪绾?
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)

    # 缇庡寲鍧愭爣杞达紝鍔犱笂 r 鍓嶇紑淇 \Delta 璀﹀憡
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
    璁＄畻骞剁粯鍒剁粷瀵圭姸鎬佺殑璺ㄦ潯浠跺畾閲忚€﹀悎鍒嗘瀽 (Absolute State Space Binding)
    浣跨敤: Absolute Participants_Ratio vs Absolute Mean_RSM_Sim (N=24 data points)
    """
    required_cols = ["Participants_Ratio", "Mean_RSM_Sim"]
    if not all(c in df.columns for c in required_cols):
        return None

    # 鎻愬彇鎵€鏈夊皬榧犲湪鎵€鏈夋潯浠朵笅鐨勭粷瀵瑰€?(8 * 3 = 24 data points)
    plot_df = df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    plot_df["Condition"] = pd.Categorical(plot_df["Condition"], categories=CONDITIONS, ordered=True)

    if len(plot_df) < 5:
        return None

    # 鏁翠綋鐩稿叧鎬ф楠?(N=24)
    spearman_r, spearman_p = stats.spearmanr(plot_df["Participants_Ratio"], plot_df["Mean_RSM_Sim"])
    pearson_r, pearson_p = stats.pearsonr(plot_df["Participants_Ratio"], plot_df["Mean_RSM_Sim"])

    fig, ax = plt.subplots(figsize=(6, 5))

    # 1. 缁樺埗搴曞眰鐨勬暣浣撶嚎鎬у洖褰掔嚎 (涓嶅尯鍒嗘潯浠?
    sns.regplot(
        data=plot_df, x="Participants_Ratio", y="Mean_RSM_Sim",
        scatter=False, ax=ax, color="#404040", 
        line_kws={"linewidth": 2, "linestyle": "--", "alpha": 0.6}
    )

    # 2. 缁樺埗鏁ｇ偣锛屾寜鐓ф潯浠朵笂鑹?(鍖哄垎涓夌缃戠粶鐘舵€?
    sns.scatterplot(
        data=plot_df, x="Participants_Ratio", y="Mean_RSM_Sim",
        hue="Condition", palette=COLORS, s=70, alpha=0.85, 
        edgecolor="white", linewidth=1, ax=ax, zorder=3
    )

    # 娣诲姞缁熻缁撴灉鏂囨湰妗?
    stat_text = (f"Overall Correlation (N={len(plot_df)} states)\n"
                 f"Spearman $r_s$ = {spearman_r:.2f}, $p$ = {spearman_p:.3e}\n"
                 f"Pearson $r$ = {pearson_r:.2f}, $p$ = {pearson_p:.3e}")
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    # 缇庡寲鍧愭爣杞?
    ax.set_xlabel("Response Concentration (Participants Ratio)")
    ax.set_ylabel("Trial-to-Trial Population Stability (Mean RSM)")
    ax.set_title("Global State-Space Binding", pad=15)
    style_axis(ax)
    
    # 璋冩暣鍥句緥
    ax.legend(title="", frameon=False, loc="lower right")

    out = os.path.join(GROUP_OUT_DIR, "group_absolute_state_binding.png")
    save_figure_variants(fig, out)
    plt.close(fig)
    return out

def plot_lmm_state_binding(df):
    """
    缁堟瀬鐗堬細浣跨敤绾挎€ф贩鍚堟晥搴旀ā鍨?(LMM) 鎺у埗榧犲唴閲嶅娴嬮噺銆?
    妯″瀷: RSM ~ Participants_Ratio + (1 | mouse_id)
    """
    required_cols = ["Participants_Ratio", "Mean_RSM_Sim"]
    if not all(c in df.columns for c in required_cols):
        return None

    plot_df = df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    plot_df["Condition"] = pd.Categorical(plot_df["Condition"], categories=CONDITIONS, ordered=True)

    if len(plot_df) < 5:
        return None

    # ==========================================
    # 1. 鎷熷悎绾挎€ф贩鍚堟晥搴旀ā鍨?(LMM)
    # 闅忔満鎴窛妯″瀷: 鎺у埗姣忓彧榧犺嚜韬殑 baseline
    # ==========================================
    md = smf.mixedlm("Mean_RSM_Sim ~ Participants_Ratio", plot_df, groups=plot_df["mouse_id"])
    mdf = md.fit()

    coef = mdf.params["Participants_Ratio"]
    p_val = mdf.pvalues["Participants_Ratio"]
    global_intercept = mdf.params["Intercept"]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # ==========================================
    # 2. 缁樺埗鍙鍖栵細涓綋鎷熷悎绾?(Random Intercepts)
    # 杩欏皢鍚戝绋夸汉瀹岀編灞曠ず鈥滈紶鍐呭崗鍙?(within-mouse covariance)鈥?
    # ==========================================
    x_vals = np.array([plot_df["Participants_Ratio"].min() * 0.9, plot_df["Participants_Ratio"].max() * 1.05])
    
    for mouse_id, group_data in plot_df.groupby("mouse_id"):
        if mouse_id in mdf.random_effects:
            # 鑾峰彇杩欏彧鑰侀紶鐨勯殢鏈烘埅璺?(Random effect)
            rand_int = mdf.random_effects[mouse_id].iloc[0] 
            intercept = global_intercept + rand_int
            y_vals = intercept + coef * x_vals
            # 鐢诲嚭灞炰簬杩欏彧鑰侀紶鑷繁鐨勫钩琛屽洖褰掔嚎
            ax.plot(x_vals, y_vals, color="#A0A0A0", alpha=0.35, linewidth=1.2, zorder=1)

    # ==========================================
    # 3. 缁樺埗鎬讳綋鍥哄畾鏁堝簲绾?(Fixed Effect)
    # ==========================================
    global_y_vals = global_intercept + coef * x_vals
    ax.plot(x_vals, global_y_vals, color="#202020", linewidth=3, linestyle="--", zorder=2, label="Fixed effect (LMM)")

    # ==========================================
    # 4. 缁樺埗鍘熷鏁ｇ偣 (鍖哄垎鏉′欢)
    # ==========================================
    sns.scatterplot(
        data=plot_df, x="Participants_Ratio", y="Mean_RSM_Sim",
        hue="Condition", palette=COLORS, s=75, alpha=0.9, 
        edgecolor="white", linewidth=1, ax=ax, zorder=3
    )

    # 娣诲姞 LMM 缁熻缁撴灉鏂囨湰妗?
    stat_text = (f"Linear Mixed-Effects Model (LMM)\n"
                 f"RSM $\\sim$ PR + $(1 | Mouse)$\n\n"
                 f"Fixed Effect $\\beta$ = {coef:.4f}\n"
                 f"$p$-value = {p_val:.3e}")
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))

    # 缇庡寲鍧愭爣杞?
    ax.set_xlabel("Response Concentration (Participants Ratio)")
    ax.set_ylabel("Trial-to-Trial Population Stability (Mean RSM)")
    ax.set_title("Within-Animal State-Space Binding", pad=15)
    style_axis(ax)
    
    # 娓呯悊鍥句緥锛屽幓闄ら噸澶嶉」
    handles, labels = ax.get_legend_handles_labels()
    # 淇濈暀 Fixed effect 鍜?涓変釜鏉′欢
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
    if not bundles:
        print("[!] No mouse bundles found. Stop.")
        raise SystemExit(0)

    # ---------------- Core tables ----------------
    master_df = build_master_dataframe(bundles)
    dup_mc = int(master_df.duplicated(["mouse_id", "Condition"]).sum())
    if dup_mc > 0:
        print(f"[!] Found duplicated mouse_id+Condition rows in master_df: {dup_mc}. Collapse by mean.")
        num_cols = [c for c in master_df.columns if c not in ["mouse_id", "Condition"]]
        master_df = (
            master_df.groupby(["mouse_id", "Condition"], as_index=False)[num_cols]
            .mean(numeric_only=True)
        )
    decile_df = build_decile_dataframe(bundles)
    noise_decile_long_df = build_noise_decile_coupling_long_dataframe(bundles)
    rr_overlap_df = build_rr_overlap_dataframe(bundles)
    geometry_long_df = build_geometry_condition_long_dataframe(bundles)
    geometry_pairwise_long_df = build_geometry_pairwise_long_dataframe(bundles)
    geometry_model_compare_long_df = build_geometry_model_compare_long_dataframe(bundles)

    # ---------------- Shuffle tables ----------------
    shuffle_manifest_df = build_shuffle_manifest_dataframe(bundles)
    shuffle_corr_long_df = build_shuffle_corr_long_dataframe(bundles)
    shuffle_corr_decile_long_df = build_shuffle_corr_decile_long_dataframe(bundles)
    shuffle_rsm_long_df = build_shuffle_rsm_long_dataframe(bundles)
    shuffle_delta_long_df = build_shuffle_delta_long_dataframe(bundles)
    shuffle_dose_long_df = build_shuffle_dose_long_dataframe(bundles)
    shuffle_alloc_long_df = build_shuffle_alloc_long_dataframe(bundles)
    shuffle_effect_stats_df = build_shuffle_effect_stats_dataframe(bundles)
    shuffle_cond_summary_df = build_shuffle_condition_summary_dataframe(bundles)
    shuffle_cond_stats_df = build_shuffle_condition_stats_dataframe(bundles)
    shuffle_sync_contrib_df = build_shuffle_sync_contribution_dataframe(bundles)
    shuffle_sync_contrib_repeats_df = build_shuffle_sync_contribution_repeats_dataframe(bundles)
    shuffle_core_long_df = build_shuffle_core_long_dataframe(
        shuffle_corr_long_df, shuffle_rsm_long_df, shuffle_alloc_long_df
    )

    # ---------------- Save core tables ----------------
    save_paths = {}
    save_paths["group_master_metrics"] = os.path.join(GROUP_OUT_DIR, "group_master_metrics.csv")
    master_df.to_csv(save_paths["group_master_metrics"], index=False)

    save_paths["group_corr_deciles_long"] = os.path.join(GROUP_OUT_DIR, "group_corr_deciles_long.csv")
    if not decile_df.empty:
        decile_df.to_csv(save_paths["group_corr_deciles_long"], index=False)

    save_paths["group_noise_corr_decile_coupling_long"] = os.path.join(
        GROUP_OUT_DIR, "group_noise_corr_decile_coupling_long.csv"
    )
    if not noise_decile_long_df.empty:
        noise_decile_long_df.to_csv(save_paths["group_noise_corr_decile_coupling_long"], index=False)

    save_paths["group_rr_overlap_long"] = os.path.join(GROUP_OUT_DIR, "group_rr_overlap_long.csv")
    if not rr_overlap_df.empty:
        rr_overlap_df.to_csv(save_paths["group_rr_overlap_long"], index=False)

    save_paths["group_geometry_condition_level_long"] = os.path.join(
        GROUP_OUT_DIR, "group_geometry_condition_level_long.csv"
    )
    if not geometry_long_df.empty:
        geometry_long_df.to_csv(save_paths["group_geometry_condition_level_long"], index=False)

    save_paths["group_geometry_condition_pairwise_long"] = os.path.join(
        GROUP_OUT_DIR, "group_geometry_condition_pairwise_long.csv"
    )
    if not geometry_pairwise_long_df.empty:
        geometry_pairwise_long_df.to_csv(save_paths["group_geometry_condition_pairwise_long"], index=False)

    save_paths["group_geometry_model_compare_long"] = os.path.join(
        GROUP_OUT_DIR, "group_geometry_model_compare_long.csv"
    )
    if not geometry_model_compare_long_df.empty:
        geometry_model_compare_long_df.to_csv(save_paths["group_geometry_model_compare_long"], index=False)

    # ---------------- Save shuffle tables ----------------
    shuffle_save_paths = {}
    shuffle_table_map = {
        "group_shuffle_manifest": (shuffle_manifest_df, "group_shuffle_manifest.csv"),
        "group_shuffle_corr_long": (shuffle_corr_long_df, "group_shuffle_corr_long.csv"),
        "group_shuffle_corr_decile_long": (shuffle_corr_decile_long_df, "group_shuffle_corr_decile_long.csv"),
        "group_shuffle_rsm_long": (shuffle_rsm_long_df, "group_shuffle_rsm_long.csv"),
        "group_shuffle_delta_long": (shuffle_delta_long_df, "group_shuffle_delta_long.csv"),
        "group_shuffle_dose_long": (shuffle_dose_long_df, "group_shuffle_dose_long.csv"),
        "group_shuffle_alloc_long": (shuffle_alloc_long_df, "group_shuffle_alloc_long.csv"),
        "group_shuffle_effect_stats_raw": (shuffle_effect_stats_df, "group_shuffle_effect_stats_raw.csv"),
        "group_shuffle_condition_summary_raw": (shuffle_cond_summary_df, "group_shuffle_condition_summary_raw.csv"),
        "group_shuffle_condition_stats_raw": (shuffle_cond_stats_df, "group_shuffle_condition_stats_raw.csv"),
        "group_shuffle_sync_contribution_raw": (shuffle_sync_contrib_df, "group_shuffle_sync_contribution_raw.csv"),
        "group_shuffle_sync_contribution_repeats_raw": (shuffle_sync_contrib_repeats_df, "group_shuffle_sync_contribution_repeats_raw.csv"),
        "group_shuffle_core_long": (shuffle_core_long_df, "group_shuffle_core_long.csv"),
    }
    for key, (df, fn) in shuffle_table_map.items():
        p = os.path.join(GROUP_OUT_DIR, fn)
        if df is not None and not df.empty:
            df.to_csv(p, index=False)
            shuffle_save_paths[key] = p

    # ---------------- 娴嬭瘯鎸囨爣閫夋嫨 ----------------
    metrics_to_test = [m for m in [
        "Entropy", "Mean_RSM_Sim", "Mean_Correlation", "Strong_Correlation", "Weak_Correlation",
        "Strong_Weak_Gap", "Participants_Ratio", "Gini_Mean", "PR_Mean", "Effective_Dim_PR",
        "Sig_Mean_Corr", "Noise_Mean_Corr",
        "Geom_AngleDeg", "Geom_OrthParallelRatio", "Geom_VarParallel", "Geom_VarOrthogonal", "Geom_Anisotropy"
    ] if m in master_df.columns and not master_df[m].isna().all()]

    stat_results = {m: perform_statistical_tests(master_df, m) for m in metrics_to_test}

    # Save core stat summary table
    stat_rows = []
    for metric, res in stat_results.items():
        ph = res.get("post_hoc", {})
        stat_rows.append({
            "metric": metric,
            "main_effect": res.get("main_effect", "N/A"),
            "p_main": res.get("p_main", np.nan),
            "Divergent_vs_Convergent": ph.get("Divergent vs Convergent", np.nan),
            "Divergent_vs_Random": ph.get("Divergent vs Random", np.nan),
            "Convergent_vs_Random": ph.get("Convergent vs Random", np.nan),
        })
    stat_summary_df = pd.DataFrame(stat_rows)
    save_paths["group_statistical_tests_summary"] = os.path.join(GROUP_OUT_DIR, "group_statistical_tests_summary.csv")
    stat_summary_df.to_csv(save_paths["group_statistical_tests_summary"], index=False)

    # ---------------- Geometry mixed-model summaries ----------------
    geometry_lmm_df = run_geometry_lmm_models(master_df)
    geometry_save_paths = {}
    for k in [
        "group_geometry_condition_level_long",
        "group_geometry_condition_pairwise_long",
        "group_geometry_model_compare_long",
    ]:
        p = save_paths.get(k, None)
        if isinstance(p, str) and os.path.exists(p) and os.path.getsize(p) > 0:
            geometry_save_paths[k] = p
    if not geometry_lmm_df.empty:
        geometry_save_paths["group_geometry_rsm_model_compare"] = os.path.join(
            GROUP_OUT_DIR, "group_geometry_rsm_model_compare.csv"
        )
        geometry_lmm_df.to_csv(geometry_save_paths["group_geometry_rsm_model_compare"], index=False)

        rsm_df = geometry_lmm_df[geometry_lmm_df["model_name"].astype(str).str.startswith("M")].copy()
        alloc_df = geometry_lmm_df[geometry_lmm_df["model_name"].astype(str).str.startswith("A")].copy()
        vs_dim_df = geometry_lmm_df[
            geometry_lmm_df["model_name"].astype(str).str.startswith(("M", "D"))
        ].copy()

        geometry_save_paths["group_geometry_rsm_lmm_summary"] = os.path.join(
            GROUP_OUT_DIR, "group_geometry_rsm_lmm_summary.md"
        )
        with open(geometry_save_paths["group_geometry_rsm_lmm_summary"], "w", encoding="utf-8") as f:
            f.write("# Group Geometry vs Mean RSM (LMM)\n\n")
            f.write(df_to_md(rsm_df) + "\n")

        geometry_save_paths["group_geometry_allocation_lmm_summary"] = os.path.join(
            GROUP_OUT_DIR, "group_geometry_allocation_lmm_summary.md"
        )
        with open(geometry_save_paths["group_geometry_allocation_lmm_summary"], "w", encoding="utf-8") as f:
            f.write("# Group Geometry vs Participants Ratio (LMM)\n\n")
            f.write(df_to_md(alloc_df) + "\n")

        geometry_save_paths["group_geometry_vs_dimensionality_model_compare"] = os.path.join(
            GROUP_OUT_DIR, "group_geometry_vs_dimensionality_model_compare.csv"
        )
        vs_dim_df.to_csv(geometry_save_paths["group_geometry_vs_dimensionality_model_compare"], index=False)

    # ---------------- Shuffle stats across mice ----------------
    shuffle_ov_stats_df = test_shuffle_original_vs_shuffled(shuffle_core_long_df)
    shuffle_cond_group_stats_df = test_shuffle_condition_differences(shuffle_core_long_df, data_type="shuffled")
    shuffle_sync_group_stats_df = test_shuffle_sync_contribution(shuffle_sync_contrib_df)
    shuffle_orig_group_stats_path = os.path.join(GROUP_OUT_DIR, "group_shuffle_original_vs_shuffled_stats.csv")
    shuffle_cond_group_stats_path = os.path.join(GROUP_OUT_DIR, "group_shuffle_condition_stats.csv")
    shuffle_sync_group_stats_path = os.path.join(GROUP_OUT_DIR, "group_shuffle_sync_contribution_stats.csv")
    if not shuffle_ov_stats_df.empty:
        shuffle_ov_stats_df.to_csv(shuffle_orig_group_stats_path, index=False)
        shuffle_save_paths["group_shuffle_original_vs_shuffled_stats"] = shuffle_orig_group_stats_path
    if not shuffle_cond_group_stats_df.empty:
        shuffle_cond_group_stats_df.to_csv(shuffle_cond_group_stats_path, index=False)
        shuffle_save_paths["group_shuffle_condition_stats"] = shuffle_cond_group_stats_path
    if not shuffle_sync_group_stats_df.empty:
        shuffle_sync_group_stats_df.to_csv(shuffle_sync_group_stats_path, index=False)
        shuffle_save_paths["group_shuffle_sync_contribution_stats"] = shuffle_sync_group_stats_path

    # ---------------- 鍙鍖栫敓鎴?----------------
    image_paths = {}
    
    # 1. 寮哄急瀵规瘮鍙岄潰鏉跨鍨嬪浘 (Boxplot Dual-panel)
    image_paths["Combined Strong vs Weak"] = plot_combined_strong_weak(master_df)

    # 2. 鏍稿績鎸囨爣鐨勬瀬绠€鏌辩姸鍥?
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

    # 3. 鍒嗗眰鏇茬嚎 (闃村奖璇樊甯?
    image_paths["Decile Correlation Curve"] = plot_decile_curve(decile_df)
    image_paths["Noise Decile Curve"] = plot_noise_decile_curve(noise_decile_long_df)
    image_paths["Cross-animal Binding"] = plot_cross_animal_binding(master_df)
    image_paths["Absolute State Binding"] = plot_absolute_state_binding(master_df)
    image_paths["LMM State Binding"] = plot_lmm_state_binding(master_df)
    image_paths["Geometry Angle Condition"] = plot_geometry_condition_metric(
        geometry_long_df, "angle_deg", "Angle between mean axis and PC1 (deg)", "group_geometry_angle_condition.png"
    )
    image_paths["Geometry Orth/Parallel Ratio Condition"] = plot_geometry_condition_metric(
        geometry_long_df, "orth_parallel_ratio", "Orthogonal / Parallel variance ratio", "group_geometry_orth_parallel_condition.png"
    )
    image_paths["Geometry Angle vs Mean RSM"] = plot_geometry_vs_rsm(
        master_df, "Geom_AngleDeg", "Geometry angle (deg)", "group_geometry_angle_vs_rsm.png"
    )
    image_paths["Geometry Ratio vs Mean RSM"] = plot_geometry_vs_rsm(
        master_df, "Geom_OrthParallelRatio", "Orthogonal / Parallel variance ratio", "group_geometry_ratio_vs_rsm.png"
    )

    # 4. Shuffle group-level figures
    shuffle_metric_y = {
        "weak_corr": "Weak correlation",
        "strong_weak_gap": "Strong-Weak gap",
        "mean_rsm": "Mean RSM",
        "pr_mean": "Participation ratio (PR)",
    }
    for metric, ylabel in shuffle_metric_y.items():
        if metric in set(shuffle_core_long_df.columns):
            image_paths[f"Shuffle Orig-vs-Shuffled: {metric}"] = plot_shuffle_orig_vs_shuffled(
                shuffle_core_long_df, metric, ylabel, stat_df=shuffle_ov_stats_df
            )
            image_paths[f"Shuffle Condition Difference: {metric}"] = plot_shuffle_condition_only(
                shuffle_core_long_df, metric, ylabel, data_type="shuffled", stat_df=shuffle_cond_group_stats_df
            )
        if metric in set(shuffle_dose_long_df.columns):
            image_paths[f"Shuffle Dose-response: {metric}"] = plot_shuffle_dose_response_group(
                shuffle_dose_long_df, metric, ylabel
            )
    image_paths["Shuffle Delta by Condition"] = plot_shuffle_delta_group(shuffle_delta_long_df)
    image_paths["Shuffle Synchrony Contribution"] = plot_shuffle_sync_contribution_group(
        shuffle_sync_contrib_df, stat_df=shuffle_sync_group_stats_df
    )
    # 5. Generate markdown report (including shuffle outputs)
    shuffle_payload = {
        "orig_vs_shuffled_stats": shuffle_ov_stats_df,
        "condition_stats": shuffle_cond_group_stats_df,
        "sync_contribution_stats": shuffle_sync_group_stats_df,
        "csv_paths": shuffle_save_paths,
    }
    geometry_payload = {
        "condition_long": geometry_long_df,
        "pairwise_long": geometry_pairwise_long_df,
        "model_compare": geometry_lmm_df if 'geometry_lmm_df' in locals() else pd.DataFrame(),
        "csv_paths": geometry_save_paths if 'geometry_save_paths' in locals() else {},
    }
    generate_group_markdown(
        master_df,
        stat_results,
        image_paths,
        rr_overlap_df,
        shuffle_payload=shuffle_payload,
        geometry_payload=geometry_payload,
    )

    print("====== Group integration visualization & markdown completed ======")
