import argparse
import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

from brainnetwork import load_data, preprocess_spike_data, rr_selection_class


DEFAULT_BASE_DIR = "/beegfs_hdd/data/nfs_share/users/guiyun/nishome/Micedata/"
DEFAULT_MOUSE_IDS = ["M21_1107", "M71_1024", "M73_1128", "M77_1031", "M77_1107", "M78_1017", "M79_1128", "M91_1017"]
DEFAULT_RESULTS_DIR = "./results"
GROUP_DIR_NAME = "group_summary"
COND_MAP = {1: "Divergent", 2: "Convergent", 3: "Random"}
CONDITIONS = ["Divergent", "Convergent", "Random"]
COLORS = {"Divergent": "#7F9C96", "Convergent": "#8B90A8", "Random": "#B98372"}
EPS = 1e-12


sns.set_theme(style="ticks", context="paper")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)


def safe_unit(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= EPS:
        return np.zeros_like(v), 0.0
    return v / n, float(n)


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def upper_tri(m):
    m = np.asarray(m, dtype=float)
    return m[np.triu(np.ones_like(m, dtype=bool), k=1)]


def mean_rsm(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.nan
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n > EPS, n, 1.0)
    sim = (X / n) @ (X / n).T
    vals = upper_tri(sim)
    return float(np.mean(vals)) if vals.size > 0 else np.nan


def effective_dim_pr(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.nan
    Y = X - np.mean(X, axis=0, keepdims=True)
    cov = np.cov(Y, rowvar=False)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    s1 = np.sum(eig)
    s2 = np.sum(eig ** 2)
    return float((s1 ** 2) / s2) if s2 > EPS else np.nan


def geometry_metrics(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return {k: np.nan for k in ["mean_norm", "angle_deg", "var_parallel", "var_orthogonal", "orth_parallel_ratio", "anisotropy_index", "lambda1", "lambda2"]}
    mu = np.mean(X, axis=0)
    mu_hat, mu_norm = safe_unit(mu)
    Y = X - mu
    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    v1_hat, _ = safe_unit(vt[0])
    if np.linalg.norm(mu_hat) <= EPS or np.linalg.norm(v1_hat) <= EPS:
        angle_deg = np.nan
    else:
        angle_deg = float(np.degrees(np.arccos(np.clip(np.abs(np.dot(mu_hat, v1_hat)), 0.0, 1.0))))
    if np.linalg.norm(mu_hat) <= EPS:
        var_par = np.nan
        var_orth = np.nan
        ratio = np.nan
    else:
        a = Y @ mu_hat
        var_par = float(np.mean(a ** 2))
        r = Y - np.outer(a, mu_hat)
        var_orth = float(np.mean(np.sum(r ** 2, axis=1)))
        ratio = float(var_orth / (var_par + EPS))
    eig = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(Y, rowvar=False)), 0.0))[::-1]
    lam1 = float(eig[0]) if eig.size >= 1 else np.nan
    lam2 = float(eig[1]) if eig.size >= 2 else np.nan
    anis = float(lam1 / (np.sum(eig) + EPS)) if eig.size > 0 else np.nan
    return {
        "mean_norm": float(mu_norm),
        "angle_deg": angle_deg,
        "var_parallel": var_par,
        "var_orthogonal": var_orth,
        "orth_parallel_ratio": ratio,
        "anisotropy_index": anis,
        "lambda1": lam1,
        "lambda2": lam2,
    }


def bootstrap_dist(X, n_boot, seed):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    records = []
    for _ in range(n_boot):
        idx = rng.integers(0, T, size=T)
        records.append(geometry_metrics(X[idx]))
    d = pd.DataFrame(records)
    out = {}
    for c in ["angle_deg", "orth_parallel_ratio", "var_parallel", "var_orthogonal", "anisotropy_index"]:
        v = d[c].astype(float).values
        out[c] = v[np.isfinite(v)]
    return out


def pairwise_boot(dist_map, metric, cond_order):
    rows = []
    for c1, c2 in combinations(cond_order, 2):
        a = np.asarray(dist_map.get(c1, {}).get(metric, []), dtype=float)
        b = np.asarray(dist_map.get(c2, {}).get(metric, []), dtype=float)
        n = min(a.size, b.size)
        if n == 0:
            rows.append({"metric": metric, "condition_1": c1, "condition_2": c2, "n_boot": 0, "mean_diff_boot": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_boot_two_sided": np.nan})
            continue
        d = a[:n] - b[:n]
        p1 = (np.sum(d <= 0) + 1) / (n + 1)
        p2 = (np.sum(d >= 0) + 1) / (n + 1)
        rows.append({"metric": metric, "condition_1": c1, "condition_2": c2, "n_boot": int(n), "mean_diff_boot": float(np.mean(d)), "ci95_low": float(np.quantile(d, 0.025)), "ci95_high": float(np.quantile(d, 0.975)), "p_boot_two_sided": float(min(1.0, 2 * min(p1, p2)))})
    return rows


def plot_cov_ellipse(ax, xy, color):
    if xy.shape[0] < 3:
        return
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * np.sqrt(np.maximum(vals[:2], 1e-12))
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse((np.mean(xy[:, 0]), np.mean(xy[:, 1])), width, height, angle=ang, edgecolor=color, facecolor=color, alpha=0.12, lw=1.8))


def save_variants(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    for ax in fig.axes:
        ax.set_title("")
    fig.savefig(path.replace(".png", "_notitle.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_mouse(mouse_id, args, seed_i):
    save_root = os.path.join(args.results_dir, mouse_id)
    data_out = os.path.join(save_root, "data")
    fig_out = os.path.join(save_root, "figures")
    ensure_dir(data_out)
    ensure_dir(fig_out)
    print(f"[*] Running {mouse_id} in overwrite mode")

    data_path = os.path.join(args.base_dir, mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)
    segments_spi, labels_spi, _ = preprocess_spike_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    labels = np.asarray(labels_spi).astype(int)
    classes = [c for c in [1, 2, 3] if c in set(labels.tolist())]
    X_trial = np.mean(np.asarray(segments_spi, dtype=float)[:, :, args.response_start:args.response_end], axis=2)
    min_count = min(np.sum(labels == c) for c in classes)
    rng = np.random.default_rng(seed_i)
    X_list, y_list = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        pick = rng.choice(idx, size=min_count, replace=False)
        X_list.append(X_trial[pick])
        y_list.extend([c] * min_count)
    X_resp = np.vstack(X_list)
    y_resp = np.asarray(y_list, dtype=int)

    rr_raw = rr_selection_class(segments_spi, labels_spi)
    rr_sets = {int(k): set(map(int, v)) for k, v in rr_raw.items()}
    rr_union = set().union(*rr_sets.values()) if rr_sets else set()
    participants = {}
    rw = slice(args.response_start, args.response_end)
    for c in classes:
        c_trials = np.asarray(segments_spi, dtype=float)[labels == c]
        c_rr = sorted(rr_sets.get(c, set()))
        oth = sorted(rr_union - set(c_rr))
        if len(c_rr) == 0 or c_trials.size == 0:
            participants[c] = np.nan
            continue
        m_rr = np.nanmean(c_trials[:, c_rr, :], axis=(0, 1))
        if len(oth) == 0:
            participants[c] = float(np.nanmean(m_rr[rw]))
        else:
            m_oth = np.nanmean(c_trials[:, oth, :], axis=(0, 1))
            den = float(np.nanmean(m_oth[rw]))
            participants[c] = np.nan if abs(den) <= EPS else float(np.nanmean(m_rr[rw]) / den)

    rows, dist_map = [], {}
    for c in classes:
        name = COND_MAP[c]
        Xc = X_resp[y_resp == c]
        gm = geometry_metrics(Xc)
        rows.append({"mouse_id": mouse_id, "Class_ID": c, "Condition": name, **gm, "Mean_RSM_Sim": mean_rsm(Xc), "Participants_Ratio": participants.get(c, np.nan), "Effective_Dim_PR": effective_dim_pr(Xc)})
        dist_map[name] = bootstrap_dist(Xc, n_boot=args.bootstrap_n, seed=seed_i + c * 13)
    df_cond = pd.DataFrame(rows)
    df_cond.to_csv(os.path.join(data_out, "geometry_condition_level_long.csv"), index=False)

    pair_rows = []
    cond_order = [COND_MAP[c] for c in [1, 2, 3] if COND_MAP[c] in set(df_cond["Condition"])]
    for m in ["angle_deg", "orth_parallel_ratio", "var_parallel", "var_orthogonal", "anisotropy_index"]:
        pair_rows.extend(pairwise_boot(dist_map, m, cond_order))
    df_pair = pd.DataFrame(pair_rows)
    df_pair["mouse_id"] = mouse_id
    df_pair.to_csv(os.path.join(data_out, "geometry_condition_pairwise.csv"), index=False)

    # Figures
    fig, axes = plt.subplots(1, len(classes), figsize=(4.8 * len(classes), 4.2), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, classes):
        Xc = X_resp[y_resp == c]
        mu = np.mean(Xc, axis=0)
        Y = Xc - mu
        _, _, vt = np.linalg.svd(Y, full_matrices=False)
        basis = vt[:2].T
        z = Y @ basis
        mu_proj = mu @ basis
        color = COLORS[COND_MAP[c]]
        ax.scatter(z[:, 0], z[:, 1], s=20, alpha=0.45, color=color, edgecolor="none")
        plot_cov_ellipse(ax, z, color)
        scale = max(np.nanstd(z[:, 0]), np.nanstd(z[:, 1]), 1e-3)
        ax.arrow(0, 0, scale, 0, color="#111111", head_width=0.06 * scale, length_includes_head=True)
        ax.arrow(0, 0, mu_proj[0], mu_proj[1], color="#8C4A3E", head_width=0.06 * scale, length_includes_head=True)
        ang = float(df_cond.loc[df_cond["Class_ID"] == c, "angle_deg"].iloc[0])
        ax.set_title(f"{COND_MAP[c]}\nangle={ang:.2f} deg")
        style_axis(ax)
    save_variants(fig, os.path.join(fig_out, "geometry_example_mouse_pc_scatter.png"))

    for metric, ylabel, stem in [("angle_deg", "Angle (deg)", "geometry_angle_condition"), ("orth_parallel_ratio", "Orth/Parallel variance ratio", "geometry_orth_parallel_condition")]:
        f, a = plt.subplots(figsize=(5.0, 4.2), dpi=180)
        order = [x for x in CONDITIONS if x in set(df_cond["Condition"])]
        sns.barplot(data=df_cond, x="Condition", y=metric, order=order, hue="Condition", palette=COLORS, legend=False, alpha=0.85, ax=a)
        sns.stripplot(data=df_cond, x="Condition", y=metric, order=order, color="#222", size=5, jitter=False, ax=a)
        a.set_xlabel("")
        a.set_ylabel(ylabel)
        style_axis(a, grid=True)
        save_variants(f, os.path.join(fig_out, f"{stem}.png"))

    for metric, stem in [("angle_deg", "geometry_angle_vs_rsm"), ("orth_parallel_ratio", "geometry_ratio_vs_rsm")]:
        f, a = plt.subplots(figsize=(5.0, 4.2), dpi=180)
        sub = df_cond[[metric, "Mean_RSM_Sim", "Condition"]].dropna()
        if len(sub) > 1:
            sns.regplot(data=sub, x=metric, y="Mean_RSM_Sim", scatter=False, color="#404040", line_kws={"lw": 2}, ax=a)
        sns.scatterplot(data=sub, x=metric, y="Mean_RSM_Sim", hue="Condition", palette=COLORS, s=80, ax=a)
        a.set_xlabel(metric)
        a.set_ylabel("Mean RSM similarity")
        a.legend(frameon=False, title="")
        style_axis(a, grid=True)
        save_variants(f, os.path.join(fig_out, f"{stem}.png"))

    print(f"[*] Mouse done: {mouse_id}")
    return df_cond, df_pair


def compute_orth_parallel_expansion(df_group):
    if df_group is None or df_group.empty:
        return pd.DataFrame(), pd.DataFrame()
    need = {"mouse_id", "Condition", "var_parallel", "var_orthogonal"}
    if not need.issubset(set(df_group.columns)):
        return pd.DataFrame(), pd.DataFrame()

    work = df_group[list(need)].copy()
    work["Condition"] = pd.Categorical(work["Condition"], categories=CONDITIONS, ordered=True)

    piv_par = (
        work.pivot_table(index="mouse_id", columns="Condition", values="var_parallel", aggfunc="mean", observed=False)
        .reindex(columns=CONDITIONS)
    )
    piv_orth = (
        work.pivot_table(index="mouse_id", columns="Condition", values="var_orthogonal", aggfunc="mean", observed=False)
        .reindex(columns=CONDITIONS)
    )

    rows = []
    for mouse_id in sorted(set(work["mouse_id"].astype(str).tolist())):
        if mouse_id not in piv_par.index or mouse_id not in piv_orth.index:
            continue
        par_r = safe_float(piv_par.loc[mouse_id, "Random"])
        ort_r = safe_float(piv_orth.loc[mouse_id, "Random"])
        par_c = safe_float(np.nanmean([piv_par.loc[mouse_id, "Divergent"], piv_par.loc[mouse_id, "Convergent"]]))
        ort_c = safe_float(np.nanmean([piv_orth.loc[mouse_id, "Divergent"], piv_orth.loc[mouse_id, "Convergent"]]))
        if np.isnan(par_r) or np.isnan(ort_r) or np.isnan(par_c) or np.isnan(ort_c):
            continue
        d_par = par_c - par_r
        d_orth = ort_c - ort_r
        rows.append(
            {
                "mouse_id": mouse_id,
                "parallel_coherent_mean": par_c,
                "parallel_random": par_r,
                "delta_parallel_coherent_minus_random": d_par,
                "orthogonal_coherent_mean": ort_c,
                "orthogonal_random": ort_r,
                "delta_orthogonal_coherent_minus_random": d_orth,
                "delta_diff_orth_minus_parallel": d_orth - d_par,
                "delta_ratio_orth_over_parallel": np.nan if abs(d_par) <= EPS else d_orth / d_par,
            }
        )
    per_mouse = pd.DataFrame(rows)
    if per_mouse.empty:
        return per_mouse, pd.DataFrame()

    d_par = per_mouse["delta_parallel_coherent_minus_random"].astype(float).values
    d_orth = per_mouse["delta_orthogonal_coherent_minus_random"].astype(float).values
    d_diff = per_mouse["delta_diff_orth_minus_parallel"].astype(float).values

    def _wilcoxon_one(x, alt="two-sided"):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        try:
            return float(stats.wilcoxon(x, alternative=alt).pvalue)
        except Exception:
            return np.nan

    def _wilcoxon_pair(x, y, alt="two-sided"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 3:
            return np.nan
        try:
            return float(stats.wilcoxon(x, y, alternative=alt).pvalue)
        except Exception:
            return np.nan

    summary = pd.DataFrame(
        [
            {
                "n_mice": int(len(per_mouse)),
                "mean_delta_parallel": float(np.nanmean(d_par)),
                "mean_delta_orthogonal": float(np.nanmean(d_orth)),
                "mean_delta_diff_orth_minus_parallel": float(np.nanmean(d_diff)),
                "median_delta_diff_orth_minus_parallel": float(np.nanmedian(d_diff)),
                "paired_wilcoxon_p_two_sided": _wilcoxon_pair(d_orth, d_par, "two-sided"),
                "paired_wilcoxon_p_one_sided_orth_greater": _wilcoxon_pair(d_orth, d_par, "greater"),
                "onesample_diff_p_two_sided": _wilcoxon_one(d_diff, "two-sided"),
                "onesample_diff_p_one_sided_greater": _wilcoxon_one(d_diff, "greater"),
                "onesample_delta_orth_p_one_sided_greater": _wilcoxon_one(d_orth, "greater"),
                "onesample_delta_parallel_p_one_sided_greater": _wilcoxon_one(d_par, "greater"),
            }
        ]
    )
    return per_mouse, summary


def plot_orth_parallel_delta(per_mouse_df, out_path):
    if per_mouse_df is None or per_mouse_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), dpi=180)
    ax0, ax1 = axes

    # Panel A: paired deltas per mouse
    for _, r in per_mouse_df.iterrows():
        ax0.plot(
            [0, 1],
            [r["delta_parallel_coherent_minus_random"], r["delta_orthogonal_coherent_minus_random"]],
            color="#A5A09A",
            lw=0.9,
            alpha=0.75,
            zorder=1,
        )
    ax0.scatter(
        np.zeros(len(per_mouse_df)),
        per_mouse_df["delta_parallel_coherent_minus_random"].values,
        s=42,
        color="#5F7E77",
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
        label="Delta Parallel",
    )
    ax0.scatter(
        np.ones(len(per_mouse_df)),
        per_mouse_df["delta_orthogonal_coherent_minus_random"].values,
        s=42,
        color="#8E5E50",
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
        label="Delta Orthogonal",
    )
    ax0.axhline(0, color="#666666", lw=1.0, ls="--")
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Delta Parallel", "Delta Orthogonal"])
    ax0.set_ylabel("Coherent - Random")
    ax0.legend(frameon=False, loc="best")
    style_axis(ax0, grid=True)

    # Panel B: orth vs parallel delta with y=x
    x = per_mouse_df["delta_parallel_coherent_minus_random"].values
    y = per_mouse_df["delta_orthogonal_coherent_minus_random"].values
    lo = float(np.nanmin(np.r_[x, y]))
    hi = float(np.nanmax(np.r_[x, y]))
    pad = 0.08 * max(1e-6, hi - lo)
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#777777", lw=1.1, ls="--")
    ax1.scatter(x, y, s=58, color="#6B7C8F", edgecolor="white", linewidth=0.8, alpha=0.9)
    for _, r in per_mouse_df.iterrows():
        ax1.text(
            r["delta_parallel_coherent_minus_random"],
            r["delta_orthogonal_coherent_minus_random"],
            str(r["mouse_id"]),
            fontsize=7.5,
            alpha=0.75,
            ha="left",
            va="bottom",
        )
    ax1.set_xlabel("Delta Parallel (Coherent - Random)")
    ax1.set_ylabel("Delta Orthogonal (Coherent - Random)")
    style_axis(ax1, grid=True)
    fig.tight_layout()
    save_variants(fig, out_path)
    return out_path


def run_group(df_group, df_pair, args):
    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)
    df_group.to_csv(os.path.join(group_dir, "group_geometry_condition_level_long.csv"), index=False)
    if df_pair is not None and not df_pair.empty:
        df_pair.to_csv(os.path.join(group_dir, "group_geometry_condition_pairwise_long.csv"), index=False)

    # Direct test: is orthogonal expansion larger than parallel expansion?
    per_mouse_delta_df, delta_summary_df = compute_orth_parallel_expansion(df_group)
    if per_mouse_delta_df is not None and not per_mouse_delta_df.empty:
        per_mouse_delta_df.to_csv(os.path.join(group_dir, "group_geometry_orth_parallel_delta_per_mouse.csv"), index=False)
    if delta_summary_df is not None and not delta_summary_df.empty:
        delta_summary_df.to_csv(os.path.join(group_dir, "group_geometry_orth_parallel_delta_test.csv"), index=False)

    # Condition tests
    stat_rows = []
    for m in ["angle_deg", "orth_parallel_ratio", "var_parallel", "var_orthogonal", "anisotropy_index"]:
        piv = df_group[["mouse_id", "Condition", m]].dropna().pivot_table(index="mouse_id", columns="Condition", values=m, aggfunc="mean", observed=False).reindex(columns=CONDITIONS).dropna()
        if len(piv) < 3:
            stat_rows.append({"metric": m, "main_effect": "N too small", "p_main": np.nan, "Divergent_vs_Convergent": np.nan, "Divergent_vs_Random": np.nan, "Convergent_vs_Random": np.nan})
            continue
        s, p = stats.friedmanchisquare(piv["Divergent"], piv["Convergent"], piv["Random"])
        dvc = stats.wilcoxon(piv["Divergent"], piv["Convergent"])[1]
        dvr = stats.wilcoxon(piv["Divergent"], piv["Random"])[1]
        cvr = stats.wilcoxon(piv["Convergent"], piv["Random"])[1]
        stat_rows.append({"metric": m, "main_effect": rf"Friedman $\chi^2$={s:.2f}, $p$={p:.3e}", "p_main": p, "Divergent_vs_Convergent": dvc, "Divergent_vs_Random": dvr, "Convergent_vs_Random": cvr})
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(group_dir, "group_geometry_condition_stats.csv"), index=False)

    # LMM compare
    rows = []
    specs = [
        ("M1", "Mean_RSM_Sim ~ angle_deg", ["angle_deg"]),
        ("M2", "Mean_RSM_Sim ~ orth_parallel_ratio", ["orth_parallel_ratio"]),
        ("M3", "Mean_RSM_Sim ~ Participants_Ratio + angle_deg", ["Participants_Ratio", "angle_deg"]),
        ("M4", "Mean_RSM_Sim ~ Participants_Ratio + orth_parallel_ratio", ["Participants_Ratio", "orth_parallel_ratio"]),
        ("A1", "angle_deg ~ Participants_Ratio", ["Participants_Ratio"]),
        ("A2", "orth_parallel_ratio ~ Participants_Ratio", ["Participants_Ratio"]),
        ("D1", "Mean_RSM_Sim ~ Effective_Dim_PR", ["Effective_Dim_PR"]),
        ("D2", "Mean_RSM_Sim ~ angle_deg + Effective_Dim_PR", ["angle_deg", "Effective_Dim_PR"]),
        ("D3", "Mean_RSM_Sim ~ orth_parallel_ratio + Effective_Dim_PR", ["orth_parallel_ratio", "Effective_Dim_PR"]),
    ]
    for name, formula, terms in specs:
        lhs = formula.split("~")[0].strip()
        rhs = [x.strip() for x in formula.split("~")[1].split("+")]
        sub = df_group[["mouse_id", lhs] + rhs].dropna()
        if len(sub) < 6 or sub["mouse_id"].nunique() < 3:
            for t in terms:
                rows.append({"model_name": name, "formula": formula, "term": t, "beta": np.nan, "p_value": np.nan, "aic": np.nan, "bic": np.nan, "n_obs": len(sub), "n_mice": sub["mouse_id"].nunique(), "note": "N too small"})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(formula, sub, groups=sub["mouse_id"]).fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            for t in terms:
                rows.append({"model_name": name, "formula": formula, "term": t, "beta": fit.params.get(t, np.nan), "p_value": fit.pvalues.get(t, np.nan), "aic": fit.aic, "bic": fit.bic, "n_obs": len(sub), "n_mice": sub["mouse_id"].nunique(), "note": ""})
        except Exception as exc:
            for t in terms:
                rows.append({"model_name": name, "formula": formula, "term": t, "beta": np.nan, "p_value": np.nan, "aic": np.nan, "bic": np.nan, "n_obs": len(sub), "n_mice": sub["mouse_id"].nunique(), "note": f"fit failed: {exc}"})
    model_df = pd.DataFrame(rows)
    model_df.to_csv(os.path.join(group_dir, "group_geometry_rsm_model_compare.csv"), index=False)
    model_df[model_df["model_name"].str.startswith(("M", "D"))].to_csv(os.path.join(group_dir, "group_geometry_vs_dimensionality_model_compare.csv"), index=False)

    # Group figures
    fig_map = {}
    for metric, ylab, fn in [("angle_deg", "Angle between mean axis and PC1 (deg)", "group_geometry_angle_condition.png"), ("orth_parallel_ratio", "Orthogonal / Parallel variance ratio", "group_geometry_orth_parallel_condition.png")]:
        f, a = plt.subplots(figsize=(5.2, 4.5), dpi=180)
        sub = df_group[["mouse_id", "Condition", metric]].dropna()
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
        sns.violinplot(data=sub, x="Condition", y=metric, hue="Condition", palette=COLORS, cut=0, inner="quartile", legend=False, alpha=0.25, ax=a)
        piv = sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean", observed=False).reindex(columns=CONDITIONS)
        for _, row in piv.iterrows():
            y = row.values.astype(float)
            m = ~np.isnan(y)
            if m.sum() >= 2:
                a.plot(np.arange(len(CONDITIONS))[m], y[m], color="#AAA49A", lw=0.8, alpha=0.55)
        sns.stripplot(data=sub, x="Condition", y=metric, hue="Condition", palette=COLORS, dodge=False, legend=False, size=4, alpha=0.75, ax=a)
        a.set_xlabel("")
        a.set_ylabel(ylab)
        style_axis(a, grid=True)
        out = os.path.join(group_dir, fn)
        save_variants(f, out)
        fig_map[ylab] = out
    for metric, xlab, fn in [("angle_deg", "Geometry angle (deg)", "group_geometry_angle_vs_rsm.png"), ("orth_parallel_ratio", "Orthogonal / Parallel variance ratio", "group_geometry_ratio_vs_rsm.png")]:
        f, a = plt.subplots(figsize=(5.2, 4.5), dpi=180)
        sub = df_group[[metric, "Mean_RSM_Sim", "Condition"]].dropna()
        sns.regplot(data=sub, x=metric, y="Mean_RSM_Sim", scatter=False, color="#444", line_kws={"lw": 2, "ls": "--"}, ax=a)
        sns.scatterplot(data=sub, x=metric, y="Mean_RSM_Sim", hue="Condition", palette=COLORS, s=68, alpha=0.9, ax=a)
        a.set_xlabel(xlab)
        a.set_ylabel("Mean RSM similarity")
        a.legend(frameon=False, title="")
        style_axis(a, grid=True)
        out = os.path.join(group_dir, fn)
        save_variants(f, out)
        fig_map[xlab] = out

    fig_delta = plot_orth_parallel_delta(
        per_mouse_delta_df,
        os.path.join(group_dir, "group_geometry_orth_parallel_delta_comparison.png"),
    )
    if fig_delta is not None:
        fig_map["Orth-vs-Parallel expansion (Coherent - Random)"] = fig_delta

    # Group markdown
    md = os.path.join(group_dir, "Group_Geometry_Report.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Group Geometry Analysis Report\n\n")
        f.write(f"**Number of mice**: {df_group['mouse_id'].nunique()}\n\n")
        f.write("## Condition summary (mean +/- sem)\n\n")
        s = df_group.groupby("Condition", observed=False)[["angle_deg", "orth_parallel_ratio", "var_parallel", "var_orthogonal", "Mean_RSM_Sim"]].agg(["mean", "sem"]).round(4)
        f.write(s.to_markdown() + "\n\n")
        f.write("## Condition tests\n\n")
        f.write(stat_df.to_markdown(index=False) + "\n\n")
        f.write("## Mixed models\n\n")
        f.write(model_df.to_markdown(index=False) + "\n\n")
        if per_mouse_delta_df is not None and not per_mouse_delta_df.empty:
            f.write("## Orthogonal-vs-Parallel Expansion Test\n\n")
            f.write("Per-mouse deltas are computed as Coherent(mean of Divergent/Convergent) - Random.\n\n")
            f.write("### Per-mouse delta table\n\n")
            f.write(per_mouse_delta_df.to_markdown(index=False) + "\n\n")
        if delta_summary_df is not None and not delta_summary_df.empty:
            f.write("### Group-level paired test summary\n\n")
            f.write(delta_summary_df.to_markdown(index=False) + "\n\n")
        f.write("## Figures\n\n")
        for name, path in fig_map.items():
            f.write(f"### {name}\n![{name}](./{os.path.basename(path)})\n\n")
    print(f"[*] Group report saved: {md}")


def parse_args():
    p = argparse.ArgumentParser(description="Geometry-only pipeline: per-mouse + group.")
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)
    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260330)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    all_cond, all_pair = [], []
    base_seed = int(args.seed)
    for i, mouse in enumerate(args.mice):
        try:
            seed_i = int(base_seed + i * 101)
            c, p = run_mouse(mouse, args, seed_i=seed_i)
            if c is not None and not c.empty:
                all_cond.append(c)
            if p is not None and not p.empty:
                all_pair.append(p)
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")
    if not all_cond:
        print("[!] No valid mouse outputs. Stop.")
        return
    df_cond = pd.concat(all_cond, ignore_index=True)
    df_pair = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    run_group(df_cond, df_pair, args)
    print("====== Geometry-only pipeline completed ======")


if __name__ == "__main__":
    main()
