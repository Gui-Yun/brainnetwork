import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from brainnetwork import load_data, preprocess_spike_data


DEFAULT_BASE_DIR = "/beegfs_hdd/data/nfs_share/users/guiyun/nishome/Micedata/"
DEFAULT_MOUSE_IDS = [
    "M21_1107",
    "M71_1024",
    "M73_1128",
    "M77_1031",
    "M77_1107",
    "M78_1017",
    "M79_1128",
    "M91_1017",
]
DEFAULT_RESULTS_DIR = "./results"
GROUP_DIR_NAME = "group_summary"
EPS = 1e-12

COND_MAP = {1: "Divergent", 2: "Convergent", 3: "Random"}
COND_ORDER = ["Random", "Divergent", "Convergent", "Coherent"]
COND_COLORS = {
    "Random": "#B98372",
    "Divergent": "#7F9C96",
    "Convergent": "#8B90A8",
    "Coherent": "#4F6B8A",
}

STATE_ORDER = ["Neg", "WeakPos", "MidPos", "StrongPos"]


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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)


def save_variants(fig: plt.Figure, out_png: str):
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    base, _ = os.path.splitext(out_png)
    for ax in fig.axes:
        ax.set_title("")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    fig.savefig(base + "_notitle.png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_empty_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def upper_tri_values(matrix: np.ndarray) -> np.ndarray:
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    return matrix[mask]


def robust_corrcoef(X: np.ndarray, rowvar=False) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be 2D")
    if (rowvar and arr.shape[1] < 2) or ((not rowvar) and arr.shape[0] < 2):
        n = arr.shape[0] if rowvar else arr.shape[1]
        out = np.full((n, n), np.nan)
        np.fill_diagonal(out, 1.0)
        return out
    C = np.corrcoef(arr, rowvar=rowvar)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def normalize_condition_name(name: Any) -> str:
    if name is None:
        return ""
    t = str(name).strip().lower()
    if t in {"1", "divergent"}:
        return "Divergent"
    if t in {"2", "convergent"}:
        return "Convergent"
    if t in {"3", "random"}:
        return "Random"
    return str(name).strip()


def holm_adjust(p_values: list[float]) -> list[float]:
    arr = np.asarray(p_values, dtype=float)
    if arr.size == 0:
        return []
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return out.tolist()

    p = arr[finite_idx]
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.zeros_like(p_sorted)
    for i, pv in enumerate(p_sorted):
        adj_sorted[i] = (m - i) * pv
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.zeros_like(p_sorted)
    adj[order] = adj_sorted
    out[finite_idx] = adj
    return out.tolist()


def wilcoxon_zero(vals: np.ndarray) -> tuple[float, int]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, int(x.size)
    try:
        _, p = stats.wilcoxon(x)
        return float(p), int(x.size)
    except Exception:
        return np.nan, int(x.size)


def friedman_3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, int]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    cc = np.asarray(c, dtype=float)
    m = np.isfinite(aa) & np.isfinite(bb) & np.isfinite(cc)
    n = int(np.sum(m))
    if n < 3:
        return np.nan, n
    try:
        _, p = stats.friedmanchisquare(aa[m], bb[m], cc[m])
        return float(p), n
    except Exception:
        return np.nan, n


def compute_pair_tables_from_raw(mouse_id: str, args) -> pd.DataFrame:
    save_root = os.path.join(args.results_dir, mouse_id)
    data_out = os.path.join(save_root, "data")
    ensure_dir(data_out)

    data_path = os.path.join(args.base_dir, mouse_id)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    try:
        neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path, data_type="spikes")
    except TypeError:
        neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)

    segments_spi, labels_spi, _ = preprocess_spike_data(
        neuron_data,
        neuron_pos,
        start_edges,
        stimulus_data,
        extract_rr=True,
    )
    X_trials = np.asarray(segments_spi, dtype=float)
    y_trials = np.asarray(labels_spi, dtype=int)

    noise_window = slice(int(args.response_start), int(args.response_end))

    pair_rows = []
    summary_rows = []
    for cls in sorted(np.unique(y_trials).astype(int).tolist()):
        if cls not in [1, 2, 3]:
            continue
        trials_c = X_trials[y_trials == cls]
        if trials_c.shape[0] < 3:
            continue

        mean_time_profile = np.nanmean(trials_c, axis=0)  # (n_neurons, n_time)
        sig_corr = robust_corrcoef(mean_time_profile, rowvar=True)

        trial_resp = np.nanmean(trials_c[:, :, noise_window], axis=2)  # (n_trials, n_neurons)
        residual = trial_resp - np.nanmean(trial_resp, axis=0, keepdims=True)
        noi_corr = robust_corrcoef(residual, rowvar=False)

        sig_vals = upper_tri_values(sig_corr)
        noi_vals = upper_tri_values(noi_corr)
        n_pairs = min(sig_vals.size, noi_vals.size)
        cname = COND_MAP.get(int(cls), str(cls))

        for i in range(n_pairs):
            s = float(sig_vals[i])
            n = float(noi_vals[i])
            pair_rows.append(
                {
                    "Class_ID": int(cls),
                    "Class_Name": cname,
                    "Signal_Corr": s,
                    "Noise_Corr": n,
                    "Abs_Signal_Corr": abs(s),
                    "Abs_Noise_Corr": abs(n),
                }
            )

        summary_rows.append(
            {
                "Class_ID": int(cls),
                "Class_Name": cname,
                "Mean_Signal_Corr": float(np.mean(sig_vals)),
                "Mean_Noise_Corr": float(np.mean(noi_vals)),
                "Mean_Abs_Signal_Corr": float(np.mean(np.abs(sig_vals))),
                "Mean_Abs_Noise_Corr": float(np.mean(np.abs(noi_vals))),
                "Signal_Noise_Coupling_r": (
                    float(np.corrcoef(sig_vals, noi_vals)[0, 1]) if sig_vals.size > 1 else np.nan
                ),
            }
        )

    df_pair = pd.DataFrame(pair_rows)
    df_summary = pd.DataFrame(summary_rows)

    pair_path = os.path.join(data_out, args.pair_csv_name)
    summary_path = os.path.join(data_out, args.summary_csv_name)
    df_pair.to_csv(pair_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"[*] Saved fallback pair CSV: {pair_path}")
    print(f"[*] Saved fallback summary CSV: {summary_path}")
    return df_pair


def load_pair_csv_or_fallback(mouse_id: str, args) -> pd.DataFrame:
    data_out = os.path.join(args.results_dir, mouse_id, "data")
    pair_path = os.path.join(data_out, args.pair_csv_name)

    df_pair = None
    if os.path.isfile(pair_path):
        try:
            df_pair = pd.read_csv(pair_path, usecols=lambda c: c in {"Class_ID", "Class_Name", "Condition", "Noise_Corr"})
            print(f"[*] Loaded existing CSV: {pair_path}")
        except Exception as exc:
            print(f"[!] Failed reading existing pair CSV for {mouse_id}: {exc}")

    if df_pair is None and args.allow_raw_fallback:
        print(f"[*] Fallback to raw computation for mouse: {mouse_id}")
        try:
            df_pair = compute_pair_tables_from_raw(mouse_id, args)
        except Exception as exc:
            print(f"[!] Raw fallback failed for {mouse_id}: {exc}")
            return pd.DataFrame()

    if df_pair is None:
        return pd.DataFrame()

    if "Condition" not in df_pair.columns:
        if "Class_Name" in df_pair.columns:
            df_pair["Condition"] = df_pair["Class_Name"].map(normalize_condition_name)
        elif "Class_ID" in df_pair.columns:
            df_pair["Condition"] = df_pair["Class_ID"].map(lambda x: COND_MAP.get(int(x), str(x)))
        else:
            return pd.DataFrame()
    else:
        df_pair["Condition"] = df_pair["Condition"].map(normalize_condition_name)

    if "Noise_Corr" not in df_pair.columns:
        return pd.DataFrame()

    df_pair = df_pair[df_pair["Condition"].isin(["Divergent", "Convergent", "Random"])].copy()
    df_pair["Noise_Corr"] = pd.to_numeric(df_pair["Noise_Corr"], errors="coerce")
    df_pair = df_pair[np.isfinite(df_pair["Noise_Corr"])].copy()
    return df_pair


def regime_label(v: float, weak_thr: float, strong_thr: float) -> str:
    if v < 0:
        return "Neg"
    if v <= weak_thr:
        return "WeakPos"
    if v < strong_thr:
        return "MidPos"
    return "StrongPos"


def summarize_mouse_from_pairs(mouse_id: str, df_pair: pd.DataFrame, args) -> dict:
    out = {
        "condition_rows": [],
        "transition_matrix_rownorm": None,
        "transition_key_row": None,
    }
    if df_pair is None or df_pair.empty:
        return out

    arrays = {}
    for cond in ["Divergent", "Convergent", "Random"]:
        vals = df_pair.loc[df_pair["Condition"] == cond, "Noise_Corr"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        arrays[cond] = vals

    if any(arrays[c].size < 10 for c in ["Divergent", "Convergent", "Random"]):
        return out

    m = int(min(arrays["Divergent"].size, arrays["Convergent"].size, arrays["Random"].size))
    d = arrays["Divergent"][:m]
    c = arrays["Convergent"][:m]
    r = arrays["Random"][:m]
    coh = 0.5 * (d + c)

    weak_thr = float(args.weak_pos_max)
    strong_thr = float(np.quantile(r, float(args.strong_quantile)))

    def _metric_rows(arr: np.ndarray, cond: str) -> dict:
        strong_mask = arr >= strong_thr
        return {
            "mouse_id": mouse_id,
            "Condition": cond,
            "n_pairs": int(arr.size),
            "mean_noise_corr": float(np.mean(arr)),
            "median_noise_corr": float(np.median(arr)),
            "neg_frac": float(np.mean(arr < 0)),
            "weak_pos_frac": float(np.mean((arr > 0) & (arr <= weak_thr))),
            "strong_frac": float(np.mean(strong_mask)),
            "strong_mean": float(np.mean(arr[strong_mask])) if np.any(strong_mask) else np.nan,
            "weak_thr": weak_thr,
            "strong_thr_by_random_q": strong_thr,
        }

    out["condition_rows"].append(_metric_rows(r, "Random"))
    out["condition_rows"].append(_metric_rows(d, "Divergent"))
    out["condition_rows"].append(_metric_rows(c, "Convergent"))
    out["condition_rows"].append(_metric_rows(coh, "Coherent"))

    rand_states = np.asarray([regime_label(v, weak_thr, strong_thr) for v in r], dtype=object)
    coh_states = np.asarray([regime_label(v, weak_thr, strong_thr) for v in coh], dtype=object)
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    M = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=float)
    for s1, s2 in zip(rand_states, coh_states):
        M[idx[s1], idx[s2]] += 1.0
    row_sum = np.sum(M, axis=1, keepdims=True)
    M_row = np.where(row_sum > EPS, M / (row_sum + EPS), np.nan)
    out["transition_matrix_rownorm"] = M_row

    neg_mask = rand_states == "Neg"
    strong_mask = rand_states == "StrongPos"
    neg_to_nonneg = np.nan
    neg_to_weak = np.nan
    strong_stay = np.nan
    strong_to_notstrong = np.nan
    if np.any(neg_mask):
        neg_to_nonneg = float(np.mean(coh[neg_mask] >= 0))
        neg_to_weak = float(np.mean((coh[neg_mask] >= 0) & (coh[neg_mask] <= weak_thr)))
    if np.any(strong_mask):
        strong_stay = float(np.mean(coh[strong_mask] >= strong_thr))
        strong_to_notstrong = float(np.mean(coh[strong_mask] < strong_thr))

    row_r = _metric_rows(r, "Random")
    row_coh = _metric_rows(coh, "Coherent")
    out["transition_key_row"] = {
        "mouse_id": mouse_id,
        "n_pairs_aligned": int(m),
        "neg_to_nonneg_frac": neg_to_nonneg,
        "neg_to_weak_frac": neg_to_weak,
        "strong_stay_strong_frac": strong_stay,
        "strong_to_notstrong_frac": strong_to_notstrong,
        "weak_expand_delta": float(row_coh["weak_pos_frac"] - row_r["weak_pos_frac"]),
        "neg_shrink_delta": float(row_coh["neg_frac"] - row_r["neg_frac"]),
        "strong_mean_delta": float(row_coh["strong_mean"] - row_r["strong_mean"]),
        "mean_noise_delta": float(row_coh["mean_noise_corr"] - row_r["mean_noise_corr"]),
        "asymmetry_index_negrelief_minus_strongloss": (
            float(neg_to_nonneg - strong_to_notstrong)
            if np.isfinite(neg_to_nonneg) and np.isfinite(strong_to_notstrong)
            else np.nan
        ),
    }
    return out


def build_stats_tables(df_cond: pd.DataFrame, df_trans_key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["mean_noise_corr", "neg_frac", "weak_pos_frac", "strong_frac", "strong_mean"]
    cond_stats_rows = []
    for metric in metrics:
        piv = (
            df_cond[df_cond["Condition"].isin(["Random", "Divergent", "Convergent"])][["mouse_id", "Condition", metric]]
            .dropna()
            .pivot(index="mouse_id", columns="Condition", values=metric)
        )
        if piv.empty:
            continue
        need = ["Random", "Divergent", "Convergent"]
        ok = piv.dropna(subset=need)
        if ok.empty:
            continue
        a = ok["Divergent"].to_numpy(dtype=float)
        b = ok["Convergent"].to_numpy(dtype=float)
        c = ok["Random"].to_numpy(dtype=float)
        p_main, n_main = friedman_3(a, b, c)
        cond_stats_rows.append(
            {
                "scope": "three_condition",
                "metric": metric,
                "comparison": "Divergent vs Convergent vs Random",
                "n_mice": n_main,
                "test": "Friedman",
                "p_value": p_main,
                "p_holm": np.nan,
                "mean_delta": np.nan,
            }
        )
        pairs = [
            ("Divergent vs Convergent", ok["Divergent"].to_numpy(dtype=float) - ok["Convergent"].to_numpy(dtype=float)),
            ("Divergent vs Random", ok["Divergent"].to_numpy(dtype=float) - ok["Random"].to_numpy(dtype=float)),
            ("Convergent vs Random", ok["Convergent"].to_numpy(dtype=float) - ok["Random"].to_numpy(dtype=float)),
        ]
        pvals = []
        rows_tmp = []
        for comp, diff in pairs:
            p, n = wilcoxon_zero(diff)
            rows_tmp.append(
                {
                    "scope": "three_condition_pairwise",
                    "metric": metric,
                    "comparison": comp,
                    "n_mice": n,
                    "test": "Wilcoxon(paired)",
                    "p_value": p,
                    "p_holm": np.nan,
                    "mean_delta": float(np.nanmean(diff)),
                }
            )
            pvals.append(p)
        p_adj = holm_adjust(pvals)
        for r, p in zip(rows_tmp, p_adj):
            r["p_holm"] = p
        cond_stats_rows.extend(rows_tmp)

    coh_rand_rows = []
    for metric in metrics:
        piv = (
            df_cond[df_cond["Condition"].isin(["Random", "Coherent"])][["mouse_id", "Condition", metric]]
            .dropna()
            .pivot(index="mouse_id", columns="Condition", values=metric)
        )
        if piv.empty or "Coherent" not in piv.columns or "Random" not in piv.columns:
            continue
        d = (piv["Coherent"] - piv["Random"]).to_numpy(dtype=float)
        p, n = wilcoxon_zero(d)
        coh_rand_rows.append(
            {
                "scope": "coherent_vs_random",
                "metric": metric,
                "comparison": "Coherent - Random",
                "n_mice": n,
                "test": "Wilcoxon(delta vs 0)",
                "p_value": p,
                "p_holm": np.nan,
                "mean_delta": float(np.nanmean(d)),
            }
        )
    if coh_rand_rows:
        p_adj = holm_adjust([r["p_value"] for r in coh_rand_rows])
        for i, p in enumerate(p_adj):
            coh_rand_rows[i]["p_holm"] = p

    trans_rows = []
    trans_metrics = [
        "weak_expand_delta",
        "neg_shrink_delta",
        "strong_mean_delta",
        "mean_noise_delta",
        "asymmetry_index_negrelief_minus_strongloss",
    ]
    for metric in trans_metrics:
        if metric not in df_trans_key.columns:
            continue
        vals = df_trans_key[metric].to_numpy(dtype=float)
        p, n = wilcoxon_zero(vals)
        trans_rows.append(
            {
                "scope": "transition_key",
                "metric": metric,
                "comparison": f"{metric} vs 0",
                "n_mice": n,
                "test": "Wilcoxon(delta vs 0)",
                "p_value": p,
                "p_holm": np.nan,
                "mean_delta": float(np.nanmean(vals)),
            }
        )
    if trans_rows:
        p_adj = holm_adjust([r["p_value"] for r in trans_rows])
        for i, p in enumerate(p_adj):
            trans_rows[i]["p_holm"] = p

    return pd.DataFrame(cond_stats_rows), pd.DataFrame(coh_rand_rows), pd.DataFrame(trans_rows)


def plot_condition_metrics(df_cond: pd.DataFrame, out_path: str):
    metrics = [
        ("neg_frac", "Negative-corr fraction"),
        ("weak_pos_frac", "Weak positive fraction"),
        ("strong_mean", "Strong-tail mean corr"),
        ("mean_noise_corr", "Mean noise corr"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df_cond[df_cond["Condition"].isin(COND_ORDER)][["mouse_id", "Condition", metric]].dropna().copy()
        if sub.empty:
            continue
        piv = sub.pivot(index="mouse_id", columns="Condition", values=metric)
        order = [c for c in COND_ORDER if c in piv.columns]
        for mouse in piv.index:
            y = piv.loc[mouse, order].to_numpy(dtype=float)
            x = np.arange(len(order))
            ax.plot(x, y, color="#BFBFBF", lw=1.2, alpha=0.7)
            ax.scatter(x, y, color="#808080", s=16, alpha=0.8, zorder=3)
        means = np.asarray([np.nanmean(piv[c]) for c in order], dtype=float)
        sems = np.asarray([stats.sem(piv[c], nan_policy="omit") for c in order], dtype=float)
        x = np.arange(len(order))
        ax.plot(x, means, color="#1F2937", lw=2.5, marker="o", zorder=4)
        ax.fill_between(x, means - sems, means + sems, color="#1F2937", alpha=0.15, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    save_variants(fig, out_path)


def plot_coherent_random_delta(df_cond: pd.DataFrame, out_path: str):
    metrics = [
        ("neg_frac", "Neg frac (Coh-Rand)"),
        ("weak_pos_frac", "Weak+ frac (Coh-Rand)"),
        ("strong_mean", "Strong mean (Coh-Rand)"),
        ("mean_noise_corr", "Mean noise corr (Coh-Rand)"),
    ]
    rows = []
    for metric, _ in metrics:
        piv = (
            df_cond[df_cond["Condition"].isin(["Random", "Coherent"])][["mouse_id", "Condition", metric]]
            .dropna()
            .pivot(index="mouse_id", columns="Condition", values=metric)
        )
        if piv.empty or "Coherent" not in piv.columns or "Random" not in piv.columns:
            continue
        d = piv["Coherent"] - piv["Random"]
        for mouse, val in d.items():
            rows.append({"metric": metric, "mouse_id": mouse, "delta": float(val)})
    ddf = pd.DataFrame(rows)
    if ddf.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.6), dpi=180)
    order = [m for m, _ in metrics if m in set(ddf["metric"].tolist())]
    sns.boxplot(data=ddf, x="metric", y="delta", order=order, color="#E5E7EB", width=0.56, fliersize=0, ax=ax)
    sns.stripplot(data=ddf, x="metric", y="delta", order=order, color="#374151", size=5, jitter=0.15, ax=ax)
    ax.axhline(0.0, color="#9CA3AF", lw=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Delta (Coherent - Random)")
    ax.set_xticklabels([dict(metrics)[m] for m in order], rotation=18, ha="right")
    style_axis(ax, grid=True)
    save_variants(fig, out_path)


def plot_transition_heatmap(M_mean: np.ndarray, out_path: str):
    if M_mean is None or M_mean.size == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2), dpi=180)
    im = ax.imshow(M_mean, cmap="YlGnBu", vmin=0.0, vmax=np.nanmax(M_mean))
    ax.set_xticks(np.arange(len(STATE_ORDER)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels(STATE_ORDER, rotation=20, ha="right")
    ax.set_yticklabels(STATE_ORDER)
    ax.set_xlabel("Coherent state")
    ax.set_ylabel("Random state")
    for i in range(M_mean.shape[0]):
        for j in range(M_mean.shape[1]):
            val = M_mean[i, j]
            txt = "nan" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="#111827", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Transition probability")
    style_axis(ax, grid=False)
    save_variants(fig, out_path)


def plot_transition_keys(df_key: pd.DataFrame, out_path: str):
    if df_key.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), dpi=180)
    ax1, ax2 = axes

    x = np.arange(df_key.shape[0], dtype=float)
    ax1.plot(x, df_key["neg_to_nonneg_frac"], marker="o", lw=1.8, color="#2F6C8F", label="Neg->NonNeg")
    ax1.plot(x, df_key["strong_to_notstrong_frac"], marker="o", lw=1.8, color="#B45F5F", label="Strong->NotStrong")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_key["mouse_id"], rotation=40, ha="right")
    ax1.set_ylabel("Fraction")
    ax1.legend(frameon=False)
    style_axis(ax1, grid=True)

    d2 = df_key.melt(
        id_vars=["mouse_id"],
        value_vars=["weak_expand_delta", "neg_shrink_delta", "asymmetry_index_negrelief_minus_strongloss"],
        var_name="metric",
        value_name="value",
    )
    sns.boxplot(data=d2, x="metric", y="value", color="#E5E7EB", width=0.6, fliersize=0, ax=ax2)
    sns.stripplot(data=d2, x="metric", y="value", color="#374151", size=5, jitter=0.16, ax=ax2)
    ax2.axhline(0.0, color="#9CA3AF", lw=1, linestyle="--")
    ax2.set_xticklabels(["WeakExpand", "NegShrink", "Asymmetry"], rotation=18, ha="right")
    ax2.set_ylabel("Delta / Index")
    style_axis(ax2, grid=True)
    save_variants(fig, out_path)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Weak-correlation reorganization analysis. Prefer existing per-mouse pair CSVs; "
            "fallback to raw data computation when missing."
        )
    )
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)
    p.add_argument("--pair-csv-name", type=str, default="sig_noise_pair_values_by_condition.csv")
    p.add_argument("--summary-csv-name", type=str, default="sig_noise_strength_summary_by_condition.csv")
    p.set_defaults(allow_raw_fallback=True)
    p.add_argument(
        "--no-raw-fallback",
        dest="allow_raw_fallback",
        action="store_false",
        help="Disable fallback raw-data computation when pair CSV is missing.",
    )
    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--weak-pos-max", type=float, default=0.10)
    p.add_argument("--strong-quantile", type=float, default=0.90)
    p.add_argument("--group-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)

    all_cond_rows = []
    all_trans_keys = []
    all_trans_mats = []

    for mouse in args.mice:
        df_pair = load_pair_csv_or_fallback(mouse, args)
        if df_pair is None or df_pair.empty:
            print(f"[!] Skip mouse {mouse}: no usable pair data.")
            continue

        res = summarize_mouse_from_pairs(mouse, df_pair, args)
        all_cond_rows.extend(res["condition_rows"])
        if res["transition_key_row"] is not None:
            all_trans_keys.append(res["transition_key_row"])
        if res["transition_matrix_rownorm"] is not None:
            all_trans_mats.append(res["transition_matrix_rownorm"])

    df_cond = pd.DataFrame(all_cond_rows)
    df_trans_key = pd.DataFrame(all_trans_keys)
    if df_cond.empty:
        print("[!] No valid data were collected. Stop.")
        return

    df_cond.to_csv(os.path.join(group_dir, "group_weakcorr_reorg_mouse_metrics.csv"), index=False)
    if not df_trans_key.empty:
        df_trans_key.to_csv(os.path.join(group_dir, "group_weakcorr_reorg_transition_key.csv"), index=False)

    df_stats_3, df_stats_coh_rand, df_stats_transition = build_stats_tables(df_cond, df_trans_key)
    df_stats_all = pd.concat([df_stats_3, df_stats_coh_rand, df_stats_transition], ignore_index=True)
    if not df_stats_all.empty:
        df_stats_all.to_csv(os.path.join(group_dir, "group_weakcorr_reorg_stats.csv"), index=False)

    M_mean = None
    if all_trans_mats:
        M_stack = np.stack(all_trans_mats, axis=0).astype(float)
        M_mean = np.nanmean(M_stack, axis=0)
        rows = []
        for i, s_from in enumerate(STATE_ORDER):
            for j, s_to in enumerate(STATE_ORDER):
                vals = M_stack[:, i, j]
                vals = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "from_state": s_from,
                        "to_state": s_to,
                        "mean_prob": float(np.mean(vals)) if vals.size else np.nan,
                        "sem_prob": float(stats.sem(vals)) if vals.size > 1 else np.nan,
                        "n_mice": int(vals.size),
                    }
                )
        pd.DataFrame(rows).to_csv(os.path.join(group_dir, "group_weakcorr_reorg_transition_matrix.csv"), index=False)

    plot_condition_metrics(df_cond, os.path.join(group_dir, "group_weakcorr_reorg_condition_metrics.png"))
    plot_coherent_random_delta(df_cond, os.path.join(group_dir, "group_weakcorr_reorg_coherent_vs_random_delta.png"))
    if M_mean is not None:
        plot_transition_heatmap(M_mean, os.path.join(group_dir, "group_weakcorr_reorg_transition_heatmap.png"))
    if not df_trans_key.empty:
        plot_transition_keys(df_trans_key, os.path.join(group_dir, "group_weakcorr_reorg_transition_keys.png"))

    report_path = os.path.join(group_dir, "Group_WeakCorr_Reorganization_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Group Weak-Correlation Reorganization Report\n\n")
        f.write(f"- n_mice used: **{df_cond['mouse_id'].nunique()}**\n\n")
        cond_means = (
            df_cond.groupby("Condition", observed=False)[
                ["mean_noise_corr", "neg_frac", "weak_pos_frac", "strong_frac", "strong_mean"]
            ]
            .mean()
            .reset_index()
        )
        f.write("## Condition Means (Mouse-level)\n\n")
        f.write(_to_md(cond_means) + "\n\n")
        f.write("## Three-condition Tests (D/C/R)\n\n")
        f.write(_to_md(df_stats_3) + "\n\n")
        f.write("## Coherent vs Random Tests\n\n")
        f.write(_to_md(df_stats_coh_rand) + "\n\n")
        f.write("## Transition Key Tests\n\n")
        f.write(_to_md(df_stats_transition) + "\n\n")
    print(f"[*] Group report saved: {report_path}")
    print("====== Weak-correlation reorganization analysis completed ======")


if __name__ == "__main__":
    main()
