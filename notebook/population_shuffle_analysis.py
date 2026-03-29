import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, friedmanchisquare, wilcoxon
from sklearn.metrics.pairwise import cosine_similarity


def build_balanced_trial_matrix(segments, labels, window=slice(10, 13), random_state=0):
    x = np.nanmean(np.asarray(segments, dtype=float)[:, :, window], axis=2)
    y = np.asarray(labels).astype(int)
    classes = sorted(np.unique(y).astype(int).tolist())
    min_count = min(np.sum(y == c) for c in classes)

    rng = np.random.default_rng(random_state)
    xs = []
    ys = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        pick = rng.choice(idx, size=min_count, replace=False)
        xs.append(x[pick])
        ys.append(np.full(min_count, c, dtype=int))
    return np.vstack(xs), np.concatenate(ys)


def upper_triangle_values(matrix):
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    return matrix[mask]


def robust_corrcoef(x, rowvar=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")
    if (rowvar and x.shape[1] < 2) or ((not rowvar) and x.shape[0] < 2):
        n = x.shape[0] if rowvar else x.shape[1]
        out = np.full((n, n), np.nan)
        np.fill_diagonal(out, 1.0)
        return out
    c = np.corrcoef(x, rowvar=rowvar)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def gini_coefficient(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.nan
    m = np.min(x)
    if m < 0:
        x = x - m
    s = np.sum(x)
    if s <= 1e-12:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * xs) / (n * s)) - (n + 1) / n)


def compute_corr_metrics(x_cond, class_id, class_name):
    corr = robust_corrcoef(x_cond, rowvar=False)
    vals = upper_triangle_values(corr)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        summary = {
            "Class_ID": class_id,
            "Class_Name": class_name,
            "Mean_Correlation": np.nan,
            "Weak_Correlation": np.nan,
            "Strong_Correlation": np.nan,
            "Strong_Weak_Gap": np.nan,
            "Pair_Count": 0,
        }
        return summary, [], corr

    sorted_vals = np.sort(vals)
    bins = np.array_split(sorted_vals, 10)
    deciles = []
    dec_means = []
    for i, b in enumerate(bins, start=1):
        q_low = (i - 1) / 10
        q_high = i / 10
        m = float(np.mean(b)) if b.size else np.nan
        dec_means.append(m)
        deciles.append(
            {
                "Class_ID": class_id,
                "Class_Name": class_name,
                "Decile_Index": i,
                "Decile_Label": f"{int(q_low*100)}-{int(q_high*100)}%",
                "Lower_Quantile": q_low,
                "Upper_Quantile": q_high,
                "Pair_Count": int(b.size),
                "Corr_Value": m,
            }
        )

    summary = {
        "Class_ID": class_id,
        "Class_Name": class_name,
        "Mean_Correlation": float(np.mean(vals)),
        "Weak_Correlation": float(dec_means[0]),
        "Strong_Correlation": float(dec_means[-1]),
        "Strong_Weak_Gap": float(dec_means[-1] - dec_means[0]),
        "Pair_Count": int(vals.size),
    }
    return summary, deciles, corr


def compute_rsm_metrics(x_cond, class_id, class_name, n_bins=50):
    rsm = cosine_similarity(x_cond)
    vals = upper_triangle_values(rsm)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "Class_ID": class_id,
            "Class_Name": class_name,
            "Mean_RSM": np.nan,
            "RSM_STD": np.nan,
            "RSM_Entropy": np.nan,
            "Pair_Count": 0,
        }, vals, rsm

    counts, _ = np.histogram(vals, bins=n_bins, range=(-1, 1), density=False)
    p = counts.astype(float)
    if p.sum() > 0:
        p = p / p.sum()
    p = p[p > 0]
    ent = float(entropy(p, base=2)) if p.size else np.nan
    summary = {
        "Class_ID": class_id,
        "Class_Name": class_name,
        "Mean_RSM": float(np.mean(vals)),
        "RSM_STD": float(np.std(vals)),
        "RSM_Entropy": ent,
        "Pair_Count": int(vals.size),
    }
    return summary, vals, rsm


def neuronwise_trial_permutation(x_cond, rng, fraction=1.0):
    x_cond = np.asarray(x_cond, dtype=float)
    t, n = x_cond.shape
    out = x_cond.copy()
    if fraction <= 0:
        return out
    n_sel = int(np.round(fraction * n))
    n_sel = max(1, min(n, n_sel))
    sel = rng.choice(np.arange(n), size=n_sel, replace=False)
    for i in sel:
        out[:, i] = out[rng.permutation(t), i]
    return out


def compute_participants_ratio(x_cond, class_id, rr_neurons_spi):
    if rr_neurons_spi is None:
        return np.nan
    rr_map = {int(k): set(map(int, v)) for k, v in rr_neurons_spi.items()}
    if class_id not in rr_map:
        return np.nan
    rr_union = set().union(*rr_map.values()) if rr_map else set()
    rr_cls = rr_map[class_id]
    other_rr = rr_union - rr_cls
    rr_cls = [i for i in rr_cls if 0 <= i < x_cond.shape[1]]
    other_rr = [i for i in other_rr if 0 <= i < x_cond.shape[1]]
    if not rr_cls:
        return np.nan
    m_rr = float(np.nanmean(x_cond[:, rr_cls]))
    if not other_rr:
        return m_rr
    m_oth = float(np.nanmean(x_cond[:, other_rr]))
    if abs(m_oth) < 1e-12:
        return np.nan
    return m_rr / m_oth


def empirical_two_sided_pvalue(original_value, shuffled_values):
    shuffled_values = np.asarray(shuffled_values, dtype=float)
    shuffled_values = shuffled_values[np.isfinite(shuffled_values)]
    if shuffled_values.size == 0 or not np.isfinite(original_value):
        return np.nan
    p_low = (np.sum(shuffled_values <= original_value) + 1.0) / (shuffled_values.size + 1.0)
    p_high = (np.sum(shuffled_values >= original_value) + 1.0) / (shuffled_values.size + 1.0)
    return float(min(1.0, 2.0 * min(p_low, p_high)))


def run_population_pattern_shuffle_analysis(
    *,
    segments_spi,
    labels_spi,
    data_out_dir,
    fig_out_dir,
    mouse_id,
    label_names=None,
    class_colors=None,
    rr_neurons_spi=None,
    shuffle_repeats=200,
    shuffle_seed=20260328,
    shuffle_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
    response_window=slice(10, 13),
    rsm_bins=50,
):
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(fig_out_dir, exist_ok=True)

    if label_names is None:
        label_names = {1: "Divergent", 2: "Convergent", 3: "Random"}
    if class_colors is None:
        class_colors = {1: "#1F77B4", 2: "#D55E00", 3: "#009E73"}

    x_base, y_base = build_balanced_trial_matrix(
        segments_spi,
        labels_spi,
        window=response_window,
        random_state=shuffle_seed,
    )
    class_ids = sorted(np.unique(y_base).astype(int).tolist())
    condition_order = [label_names.get(c, str(c)) for c in class_ids]
    condition_to_id = {label_names.get(c, str(c)): c for c in class_ids}
    palette = {label_names.get(c, str(c)): class_colors.get(c, "#4C4C4C") for c in class_ids}
    x_by_condition = {label_names.get(c, str(c)): x_base[y_base == c] for c in class_ids}

    rng_master = np.random.default_rng(shuffle_seed)
    manifest_rows = []
    corr_rows = []
    decile_rows = []
    rsm_rows = []
    dose_rows = []
    allocation_rows = []
    shuffle_example = {"condition": None, "rsm_original": None, "rsm_shuffled": None, "sim_original": None, "sim_shuffled": None}
    preferred_example = "Convergent" if "Convergent" in condition_order else condition_order[0]

    for cond_name in condition_order:
        class_id = condition_to_id[cond_name]
        x_cond = np.asarray(x_by_condition[cond_name], dtype=float)

        corr_orig, dec_orig, _ = compute_corr_metrics(x_cond, class_id, cond_name)
        rsm_orig, sim_orig, rsm_mat_orig = compute_rsm_metrics(x_cond, class_id, cond_name, n_bins=rsm_bins)

        corr_rows.append(
            {
                "mouse": mouse_id,
                "condition": cond_name,
                "data_type": "original",
                "repeat_id": 0,
                "mean_corr": corr_orig["Mean_Correlation"],
                "weak_corr": corr_orig["Weak_Correlation"],
                "strong_corr": corr_orig["Strong_Correlation"],
                "strong_weak_gap": corr_orig["Strong_Weak_Gap"],
                "pair_count": corr_orig["Pair_Count"],
            }
        )
        for d in dec_orig:
            decile_rows.append(
                {
                    "mouse": mouse_id,
                    "condition": cond_name,
                    "data_type": "original",
                    "repeat_id": 0,
                    "decile": int(d["Decile_Index"]),
                    "corr_value": d["Corr_Value"],
                    "pair_count": int(d["Pair_Count"]),
                }
            )
        rsm_rows.append(
            {
                "mouse": mouse_id,
                "condition": cond_name,
                "data_type": "original",
                "repeat_id": 0,
                "mean_rsm": rsm_orig["Mean_RSM"],
                "rsm_std": rsm_orig["RSM_STD"],
                "rsm_entropy": rsm_orig["RSM_Entropy"],
                "pair_count": rsm_orig["Pair_Count"],
            }
        )
        allocation_rows.append(
            {
                "mouse": mouse_id,
                "condition": cond_name,
                "data_type": "original",
                "repeat_id": 0,
                "participants_ratio": compute_participants_ratio(x_cond, class_id, rr_neurons_spi),
                "gini_mean": float(np.nanmean([gini_coefficient(v) for v in x_cond])) if x_cond.shape[0] > 0 else np.nan,
            }
        )
        dose_rows.append(
            {
                "mouse": mouse_id,
                "condition": cond_name,
                "shuffle_fraction": 0.0,
                "repeat_id": 0,
                "weak_corr": corr_orig["Weak_Correlation"],
                "strong_weak_gap": corr_orig["Strong_Weak_Gap"],
                "mean_rsm": rsm_orig["Mean_RSM"],
            }
        )

        if cond_name == preferred_example:
            shuffle_example["condition"] = cond_name
            shuffle_example["rsm_original"] = rsm_mat_orig
            shuffle_example["sim_original"] = sim_orig

        for rep in range(1, int(shuffle_repeats) + 1):
            for frac in shuffle_fractions:
                if frac <= 0:
                    continue
                seed = int(rng_master.integers(0, 2**32 - 1))
                rng = np.random.default_rng(seed)
                x_sh = neuronwise_trial_permutation(x_cond, rng=rng, fraction=float(frac))
                corr_sh, dec_sh, _ = compute_corr_metrics(x_sh, class_id, cond_name)
                rsm_sh, sim_sh, rsm_mat_sh = compute_rsm_metrics(x_sh, class_id, cond_name, n_bins=rsm_bins)

                dose_rows.append(
                    {
                        "mouse": mouse_id,
                        "condition": cond_name,
                        "shuffle_fraction": float(frac),
                        "repeat_id": int(rep),
                        "weak_corr": corr_sh["Weak_Correlation"],
                        "strong_weak_gap": corr_sh["Strong_Weak_Gap"],
                        "mean_rsm": rsm_sh["Mean_RSM"],
                    }
                )

                if np.isclose(frac, 1.0):
                    manifest_rows.append(
                        {
                            "mouse": mouse_id,
                            "condition": cond_name,
                            "repeat_id": int(rep),
                            "n_trials": int(x_cond.shape[0]),
                            "n_neurons": int(x_cond.shape[1]),
                            "random_seed": int(seed),
                        }
                    )
                    corr_rows.append(
                        {
                            "mouse": mouse_id,
                            "condition": cond_name,
                            "data_type": "shuffled",
                            "repeat_id": int(rep),
                            "mean_corr": corr_sh["Mean_Correlation"],
                            "weak_corr": corr_sh["Weak_Correlation"],
                            "strong_corr": corr_sh["Strong_Correlation"],
                            "strong_weak_gap": corr_sh["Strong_Weak_Gap"],
                            "pair_count": corr_sh["Pair_Count"],
                        }
                    )
                    for d in dec_sh:
                        decile_rows.append(
                            {
                                "mouse": mouse_id,
                                "condition": cond_name,
                                "data_type": "shuffled",
                                "repeat_id": int(rep),
                                "decile": int(d["Decile_Index"]),
                                "corr_value": d["Corr_Value"],
                                "pair_count": int(d["Pair_Count"]),
                            }
                        )
                    rsm_rows.append(
                        {
                            "mouse": mouse_id,
                            "condition": cond_name,
                            "data_type": "shuffled",
                            "repeat_id": int(rep),
                            "mean_rsm": rsm_sh["Mean_RSM"],
                            "rsm_std": rsm_sh["RSM_STD"],
                            "rsm_entropy": rsm_sh["RSM_Entropy"],
                            "pair_count": rsm_sh["Pair_Count"],
                        }
                    )
                    allocation_rows.append(
                        {
                            "mouse": mouse_id,
                            "condition": cond_name,
                            "data_type": "shuffled",
                            "repeat_id": int(rep),
                            "participants_ratio": compute_participants_ratio(x_sh, class_id, rr_neurons_spi),
                            "gini_mean": float(np.nanmean([gini_coefficient(v) for v in x_sh])) if x_sh.shape[0] > 0 else np.nan,
                        }
                    )
                    if cond_name == preferred_example and rep == 1:
                        shuffle_example["rsm_shuffled"] = rsm_mat_sh
                        shuffle_example["sim_shuffled"] = sim_sh

    df_manifest = pd.DataFrame(manifest_rows)
    df_corr = pd.DataFrame(corr_rows)
    df_corr_decile = pd.DataFrame(decile_rows)
    df_rsm = pd.DataFrame(rsm_rows)
    df_dose = pd.DataFrame(dose_rows)
    df_alloc = pd.DataFrame(allocation_rows)

    keys = ["mouse", "condition", "data_type", "repeat_id"]
    df_full = df_corr.merge(df_rsm[["mouse", "condition", "data_type", "repeat_id", "mean_rsm"]], on=keys, how="left")
    orig = df_full[df_full["data_type"] == "original"].set_index("condition")
    shuf = df_full[df_full["data_type"] == "shuffled"].copy()

    metric_cols = {
        "weak_corr": "Delta_Weak_Correlation",
        "strong_weak_gap": "Delta_Strong_Weak_Gap",
        "mean_rsm": "Delta_Mean_RSM",
    }
    delta_rows = []
    for cond_name in condition_order:
        if cond_name not in orig.index:
            continue
        for _, r in shuf[shuf["condition"] == cond_name].iterrows():
            for col, mname in metric_cols.items():
                delta_rows.append(
                    {
                        "mouse": mouse_id,
                        "condition": cond_name,
                        "metric": mname,
                        "repeat_id": int(r["repeat_id"]),
                        "delta_shuffle": float(orig.loc[cond_name, col] - r[col]),
                    }
                )
    df_delta = pd.DataFrame(delta_rows)

    outputs = {
        "manifest": os.path.join(data_out_dir, "population_pattern_shuffle_manifest.csv"),
        "corr_long": os.path.join(data_out_dir, "group_corr_shuffle_long.csv"),
        "corr_decile_long": os.path.join(data_out_dir, "group_corr_decile_shuffle_long.csv"),
        "rsm_long": os.path.join(data_out_dir, "group_rsm_shuffle_long.csv"),
        "delta_long": os.path.join(data_out_dir, "group_shuffle_delta_long.csv"),
        "dose_long": os.path.join(data_out_dir, "group_shuffle_dose_response_long.csv"),
        "alloc_long": os.path.join(data_out_dir, "group_allocation_shuffle_long.csv"),
    }
    df_manifest.to_csv(outputs["manifest"], index=False)
    df_corr.to_csv(outputs["corr_long"], index=False)
    df_corr_decile.to_csv(outputs["corr_decile_long"], index=False)
    df_rsm.to_csv(outputs["rsm_long"], index=False)
    df_delta.to_csv(outputs["delta_long"], index=False)
    df_dose.to_csv(outputs["dose_long"], index=False)
    df_alloc.to_csv(outputs["alloc_long"], index=False)

    # Stats: original vs shuffled effect
    effect_rows = []
    for cond_name in condition_order:
        c_o = df_corr[(df_corr["condition"] == cond_name) & (df_corr["data_type"] == "original")]
        c_s = df_corr[(df_corr["condition"] == cond_name) & (df_corr["data_type"] == "shuffled")]
        r_o = df_rsm[(df_rsm["condition"] == cond_name) & (df_rsm["data_type"] == "original")]
        r_s = df_rsm[(df_rsm["condition"] == cond_name) & (df_rsm["data_type"] == "shuffled")]
        if len(c_o) == 0 or len(c_s) == 0 or len(r_o) == 0 or len(r_s) == 0:
            continue
        values_map = {
            "weak_corr": (float(c_o["weak_corr"].iloc[0]), c_s["weak_corr"].values),
            "strong_weak_gap": (float(c_o["strong_weak_gap"].iloc[0]), c_s["strong_weak_gap"].values),
            "mean_rsm": (float(r_o["mean_rsm"].iloc[0]), r_s["mean_rsm"].values),
        }
        for metric_name, (orig_val, sh_vals) in values_map.items():
            effect_rows.append(
                {
                    "mouse": mouse_id,
                    "condition": cond_name,
                    "metric": metric_name,
                    "original_value": orig_val,
                    "shuffled_mean": float(np.nanmean(sh_vals)),
                    "shuffled_std": float(np.nanstd(sh_vals, ddof=1)),
                    "delta_original_minus_shuffled_mean": float(orig_val - np.nanmean(sh_vals)),
                    "empirical_p_two_sided": empirical_two_sided_pvalue(orig_val, sh_vals),
                }
            )
    df_effect_stats = pd.DataFrame(effect_rows)
    outputs["effect_stats"] = os.path.join(data_out_dir, "group_shuffle_effect_stats.csv")
    df_effect_stats.to_csv(outputs["effect_stats"], index=False)

    # Stats: delta comparison across conditions
    delta_stat_rows = []
    for metric_name in sorted(df_delta["metric"].unique()):
        sub = df_delta[df_delta["metric"] == metric_name].copy()
        piv = sub.pivot(index="repeat_id", columns="condition", values="delta_shuffle")
        piv = piv.reindex(columns=condition_order).dropna()
        if piv.shape[0] < 3:
            delta_stat_rows.append(
                {
                    "metric": metric_name,
                    "test": "friedman",
                    "comparison": "all_conditions",
                    "stat": np.nan,
                    "p_value": np.nan,
                    "note": "N too small",
                }
            )
            continue
        stat, p = friedmanchisquare(*[piv[c].values for c in condition_order])
        delta_stat_rows.append(
            {
                "metric": metric_name,
                "test": "friedman",
                "comparison": "all_conditions",
                "stat": float(stat),
                "p_value": float(p),
                "note": "",
            }
        )
        for c1, c2 in combinations(condition_order, 2):
            try:
                w_stat, w_p = wilcoxon(piv[c1].values, piv[c2].values)
            except Exception:
                w_stat, w_p = np.nan, np.nan
            delta_stat_rows.append(
                {
                    "metric": metric_name,
                    "test": "wilcoxon",
                    "comparison": f"{c1} vs {c2}",
                    "stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                    "p_value": float(w_p) if np.isfinite(w_p) else np.nan,
                    "note": "",
                }
            )
    df_delta_stats = pd.DataFrame(delta_stat_rows)
    outputs["delta_stats"] = os.path.join(data_out_dir, "group_shuffle_delta_stats.csv")
    df_delta_stats.to_csv(outputs["delta_stats"], index=False)

    figs = generate_shuffle_figures(
        df_corr=df_corr,
        df_corr_decile=df_corr_decile,
        df_rsm=df_rsm,
        df_delta=df_delta,
        df_dose=df_dose,
        condition_order=condition_order,
        palette=palette,
        shuffle_fractions=list(shuffle_fractions),
        shuffle_example=shuffle_example,
        fig_out_dir=fig_out_dir,
    )

    return {
        "mouse_id": mouse_id,
        "condition_order": condition_order,
        "outputs": outputs,
        "figures": figs,
        "tables": {
            "manifest": df_manifest,
            "corr": df_corr,
            "corr_decile": df_corr_decile,
            "rsm": df_rsm,
            "delta": df_delta,
            "dose": df_dose,
            "alloc": df_alloc,
            "effect_stats": df_effect_stats,
            "delta_stats": df_delta_stats,
        },
    }


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def _save_fig(fig, fig_out_dir, filename):
    out = os.path.join(fig_out_dir, filename)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_shuffle_figures(
    *,
    df_corr,
    df_corr_decile,
    df_rsm,
    df_delta,
    df_dose,
    condition_order,
    palette,
    shuffle_fractions,
    shuffle_example,
    fig_out_dir,
):
    figs = {}

    # Figure A: original vs shuffled core metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    plot_specs = [("weak_corr", "Weak correlation"), ("strong_weak_gap", "Strong-Weak gap"), ("mean_rsm", "Mean RSM similarity")]

    for ax, (metric, ylabel) in zip(axes, plot_specs):
        if metric == "mean_rsm":
            sh_df = df_rsm[df_rsm["data_type"] == "shuffled"].copy()
            orig_df = df_rsm[df_rsm["data_type"] == "original"].copy()
        else:
            sh_df = df_corr[df_corr["data_type"] == "shuffled"].copy()
            orig_df = df_corr[df_corr["data_type"] == "original"].copy()

        if sh_df.empty or orig_df.empty:
            ax.set_axis_off()
            continue
        sns.violinplot(
            data=sh_df,
            x="condition",
            y=metric,
            hue="condition",
            order=condition_order,
            hue_order=condition_order,
            palette=palette,
            cut=0,
            inner="quartile",
            linewidth=1,
            legend=False,
            ax=ax,
        )
        sns.stripplot(data=orig_df, x="condition", y=metric, order=condition_order, color="#111111", size=8, marker="D", jitter=False, ax=ax)
        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        _style_axis(ax)

    fig.suptitle("Original vs Full-Shuffled Metrics", y=1.02)
    figs["orig_vs_shuffled_core"] = _save_fig(fig, fig_out_dir, "shuffle_original_vs_shuffled_core_metrics.png")

    # Figure B: decile curves
    fig, axes = plt.subplots(1, len(condition_order), figsize=(5.2 * len(condition_order), 4.5), dpi=180, sharey=True)
    if len(condition_order) == 1:
        axes = [axes]
    for ax, cond_name in zip(axes, condition_order):
        sub = df_corr_decile[df_corr_decile["condition"] == cond_name].copy()
        sub_o = sub[sub["data_type"] == "original"].sort_values("decile")
        sub_s = sub[sub["data_type"] == "shuffled"].copy()
        if not sub_o.empty:
            ax.plot(sub_o["decile"], sub_o["corr_value"], marker="o", lw=2.5, color=palette.get(cond_name, "#444444"), label="Original")
        if not sub_s.empty:
            agg = sub_s.groupby("decile")["corr_value"].agg(["mean", "sem"]).reset_index()
            ax.plot(agg["decile"], agg["mean"], marker="o", lw=2.0, ls="--", color="#222222", label="Shuffled (mean)")
            ax.fill_between(agg["decile"], agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color="#777777", alpha=0.2, linewidth=0)
        ax.set_title(cond_name)
        ax.set_xticks(np.arange(1, 11))
        ax.set_xlabel("Decile (1=weakest, 10=strongest)")
        ax.set_ylabel("Mean correlation")
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Decile Curves: Original vs Full-Shuffled", y=1.03)
    figs["decile_orig_vs_shuffled"] = _save_fig(fig, fig_out_dir, "shuffle_decile_curve_original_vs_shuffled.png")

    # Figure C: example RSM
    if shuffle_example.get("rsm_original") is not None and shuffle_example.get("rsm_shuffled") is not None:
        cond = shuffle_example["condition"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
        sns.heatmap(shuffle_example["rsm_original"], cmap="viridis", vmin=-1, vmax=1, cbar=False, ax=axes[0])
        axes[0].set_title(f"{cond}: Original RSM")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Trial")
        sns.heatmap(shuffle_example["rsm_shuffled"], cmap="viridis", vmin=-1, vmax=1, cbar=False, ax=axes[1])
        axes[1].set_title(f"{cond}: Shuffled RSM (rep=1)")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Trial")
        figs["rsm_heatmap_example"] = _save_fig(fig, fig_out_dir, "shuffle_rsm_heatmap_example_original_vs_shuffled.png")

        fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=180)
        sns.kdeplot(shuffle_example["sim_original"], fill=True, alpha=0.25, lw=2, color=palette.get(cond, "#1F77B4"), label="Original", ax=ax)
        sns.kdeplot(shuffle_example["sim_shuffled"], fill=True, alpha=0.25, lw=2, color="#222222", label="Shuffled", ax=ax)
        ax.set_xlabel("RSM off-diagonal similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"{cond}: Similarity distribution")
        ax.legend(frameon=False)
        _style_axis(ax)
        figs["rsm_similarity_hist_example"] = _save_fig(fig, fig_out_dir, "shuffle_rsm_similarity_distribution_example.png")

    # Figure D: delta by condition
    metric_order = ["Delta_Weak_Correlation", "Delta_Strong_Weak_Gap", "Delta_Mean_RSM"]
    fig, axes = plt.subplots(1, len(metric_order), figsize=(5.2 * len(metric_order), 4.6), dpi=180, sharey=False)
    if len(metric_order) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_order):
        sub = df_delta[df_delta["metric"] == metric].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        sns.boxplot(
            data=sub,
            x="condition",
            y="delta_shuffle",
            hue="condition",
            order=condition_order,
            hue_order=condition_order,
            palette=palette,
            width=0.6,
            showfliers=False,
            legend=False,
            ax=ax,
        )
        sns.stripplot(data=sub, x="condition", y="delta_shuffle", order=condition_order, color="#222222", size=3, alpha=0.35, jitter=0.15, ax=ax)
        ax.axhline(0, color="#888888", lw=1, ls="--")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("delta_shuffle")
        _style_axis(ax)

    fig.suptitle("Delta-shuffle (original - shuffled) across conditions", y=1.03)
    figs["delta_shuffle"] = _save_fig(fig, fig_out_dir, "shuffle_delta_by_condition.png")

    # Figure E: dose-response curves
    dose_long = df_dose.melt(
        id_vars=["mouse", "condition", "shuffle_fraction", "repeat_id"],
        value_vars=["weak_corr", "strong_weak_gap", "mean_rsm"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180, sharex=True)
    for ax, metric in zip(axes, ["weak_corr", "strong_weak_gap", "mean_rsm"]):
        sub = dose_long[dose_long["metric"] == metric]
        agg = sub.groupby(["condition", "shuffle_fraction"])["value"].agg(["mean", "sem"]).reset_index()
        for cond_name in condition_order:
            s = agg[agg["condition"] == cond_name].sort_values("shuffle_fraction")
            if s.empty:
                continue
            ax.plot(s["shuffle_fraction"], s["mean"], marker="o", lw=2, color=palette.get(cond_name, "#444444"), label=cond_name)
            ax.fill_between(s["shuffle_fraction"], s["mean"] - s["sem"], s["mean"] + s["sem"], color=palette.get(cond_name, "#444444"), alpha=0.15, linewidth=0)
        ax.set_title(metric)
        ax.set_xlabel("Shuffle fraction")
        ax.set_ylabel(metric)
        ax.set_xticks(shuffle_fractions)
        _style_axis(ax)
    axes[0].legend(frameon=False, title="", loc="best")
    fig.suptitle("Shuffle dose-response", y=1.03)
    figs["dose_response"] = _save_fig(fig, fig_out_dir, "shuffle_dose_response_curves.png")

    return figs
