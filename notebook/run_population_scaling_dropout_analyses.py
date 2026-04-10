import argparse
import os

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

COND_MAP = {1: "Divergent", 2: "Convergent", 3: "Random"}
CONDITIONS = ["Divergent", "Convergent", "Random"]
STRATEGIES = ["random", "spatial_clustered", "spatial_dispersed"]
DROP_METHODS = ["top_response", "spatial_cluster", "spatial_distributed", "random"]

COND_COLORS = {"Divergent": "#7F9C96", "Convergent": "#8B90A8", "Random": "#B98372"}
DROP_COLORS = {
    "top_response": "#6A4C93",
    "spatial_cluster": "#B17457",
    "spatial_distributed": "#4C8A73",
    "random": "#7C7C7C",
}
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(float(t)))
    return vals


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def build_k_values(max_n: int, requested: list[int]) -> np.ndarray:
    req = sorted(set(int(v) for v in requested if int(v) >= 2 and int(v) <= int(max_n)))
    if not req and max_n >= 2:
        req = [max_n]
    if req and req[-1] != max_n:
        req.append(max_n)
    return np.asarray(sorted(set(req)), dtype=int)


def normalize_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Condition" not in df.columns and "condition" in df.columns:
        return df.rename(columns={"condition": "Condition"})
    return df


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return np.nan
    sx = float(np.nanstd(x))
    sy = float(np.nanstd(y))
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= EPS or sy <= EPS:
        return np.nan
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def safe_cosine_rows(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    v = np.asarray(v, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] != v.size:
        return np.full(X.shape[0] if X.ndim == 2 else 0, np.nan, dtype=float)
    v_norm = np.linalg.norm(v)
    if not np.isfinite(v_norm) or v_norm <= EPS:
        return np.full(X.shape[0], np.nan, dtype=float)
    x_norm = np.linalg.norm(X, axis=1)
    denom = x_norm * v_norm + EPS
    vals = (X @ v) / denom
    vals[~np.isfinite(vals)] = np.nan
    return vals


def first_pc_vector(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return np.asarray([], dtype=float)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        if vt.shape[0] < 1:
            return np.asarray([], dtype=float)
        return vt[0]
    except Exception:
        return np.asarray([], dtype=float)


def split_half_metrics(X_trials_by_neuron: np.ndarray, rng: np.random.Generator) -> dict:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    n_trials, n_neurons = X.shape if X.ndim == 2 else (0, 0)
    if n_trials < 4 or n_neurons < 2:
        return {
            "split_half_corr": np.nan,
            "trial_template_cosine": np.nan,
            "pc1_alignment": np.nan,
        }

    perm = rng.permutation(n_trials)
    half = n_trials // 2
    idx_a = perm[:half]
    idx_b = perm[half : half + half]
    if idx_a.size < 2 or idx_b.size < 2:
        return {
            "split_half_corr": np.nan,
            "trial_template_cosine": np.nan,
            "pc1_alignment": np.nan,
        }

    Xa = X[idx_a]
    Xb = X[idx_b]
    ta = np.nanmean(Xa, axis=0)
    tb = np.nanmean(Xb, axis=0)
    split_corr = safe_corr(ta, tb)

    template = 0.5 * (ta + tb)
    trial_cos = safe_cosine_rows(X, template)
    trial_template_cosine = float(np.nanmean(trial_cos)) if np.any(np.isfinite(trial_cos)) else np.nan

    pc_a = first_pc_vector(Xa)
    pc_b = first_pc_vector(Xb)
    if pc_a.size > 0 and pc_b.size > 0 and pc_a.size == pc_b.size:
        pc1_alignment = float(abs(np.dot(pc_a, pc_b)))
    else:
        pc1_alignment = np.nan

    return {
        "split_half_corr": split_corr,
        "trial_template_cosine": trial_template_cosine,
        "pc1_alignment": pc1_alignment,
    }


def select_clustered_local(pos_xy: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(pos_xy.shape[0])
    if k >= n:
        return np.arange(n, dtype=int)
    c = int(rng.integers(0, n))
    d2 = np.sum((pos_xy - pos_xy[c : c + 1]) ** 2, axis=1)
    pick = np.argpartition(d2, kth=k - 1)[:k]
    return np.asarray(pick, dtype=int)


def select_dispersed_local(pos_xy: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(pos_xy.shape[0])
    if k >= n:
        return np.arange(n, dtype=int)
    selected = np.empty(k, dtype=int)
    start = int(rng.integers(0, n))
    selected[0] = start

    d2_min = np.sum((pos_xy - pos_xy[start : start + 1]) ** 2, axis=1)
    d2_min[start] = -1.0
    for i in range(1, k):
        nxt = int(np.argmax(d2_min))
        selected[i] = nxt
        d2_new = np.sum((pos_xy - pos_xy[nxt : nxt + 1]) ** 2, axis=1)
        d2_min = np.minimum(d2_min, d2_new)
        d2_min[selected[: i + 1]] = -1.0
    return selected


def sample_local_by_strategy(strategy: str, pos_xy_pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(pos_xy_pool.shape[0])
    if k >= n:
        return np.arange(n, dtype=int)
    if strategy == "random":
        return np.asarray(rng.choice(n, size=k, replace=False), dtype=int)
    if strategy == "spatial_clustered":
        return select_clustered_local(pos_xy_pool, k, rng)
    if strategy == "spatial_dispersed":
        return select_dispersed_local(pos_xy_pool, k, rng)
    raise ValueError(f"Unknown strategy: {strategy}")


def _to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_empty_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if not np.any(m):
        return out
    pv = p[m]
    order = np.argsort(pv)
    ranked = pv[order]
    k = len(ranked)
    adj_ranked = np.empty(k, dtype=float)
    for i, val in enumerate(ranked):
        adj_ranked[i] = min(1.0, (k - i) * val)
    adj_ranked = np.maximum.accumulate(adj_ranked)
    inv = np.empty(k, dtype=int)
    inv[order] = np.arange(k)
    out_vals = adj_ranked[inv]
    out[m] = out_vals
    return out


def friedman_pairwise(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    piv = (
        df[["mouse_id", "Condition", value_col]]
        .dropna()
        .pivot_table(index="mouse_id", columns="Condition", values=value_col, aggfunc="mean", observed=False)
        .reindex(columns=CONDITIONS)
        .dropna()
    )
    if piv.shape[0] < 3:
        return pd.DataFrame(
            [
                {
                    "metric": value_col,
                    "n_mice": int(piv.shape[0]),
                    "main_test": "Friedman",
                    "main_p": np.nan,
                    "comparison": "insufficient_n",
                    "pairwise_p": np.nan,
                    "pairwise_p_holm": np.nan,
                }
            ]
        )
    _, p_main = stats.friedmanchisquare(piv["Divergent"], piv["Convergent"], piv["Random"])
    comps = [("Divergent", "Convergent"), ("Divergent", "Random"), ("Convergent", "Random")]
    pair_p = []
    rows = []
    for a, b in comps:
        try:
            _, p = stats.wilcoxon(piv[a], piv[b])
            pair_p.append(float(p))
        except Exception:
            pair_p.append(np.nan)
    adj = holm_adjust(np.asarray(pair_p, dtype=float))
    for (a, b), p_raw, p_adj in zip(comps, pair_p, adj):
        rows.append(
            {
                "metric": value_col,
                "n_mice": int(piv.shape[0]),
                "main_test": "Friedman",
                "main_p": float(p_main),
                "comparison": f"{a} vs {b}",
                "pairwise_p": p_raw,
                "pairwise_p_holm": p_adj,
            }
        )
    return pd.DataFrame(rows)


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


def prepare_mouse_data(mouse_id: str, args) -> dict:
    data_path = os.path.join(args.base_dir, mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path, data_type="spikes")
    segments_spi, labels_spi, neuron_pos_spi = preprocess_spike_data(
        neuron_data, neuron_pos, start_edges, stimulus_data, extract_rr=True
    )
    segments = np.asarray(segments_spi, dtype=float)
    labels = np.asarray(labels_spi, dtype=int)
    pos_xy = np.asarray(neuron_pos_spi, dtype=float).T
    if segments.ndim != 3:
        raise ValueError("segments shape must be (trials, neurons, timepoints)")

    response_window = slice(args.response_start, args.response_end)
    baseline_window = slice(args.baseline_start, args.baseline_end)
    X_resp = np.nanmean(segments[:, :, response_window], axis=2)
    X_base = np.nanmean(segments[:, :, baseline_window], axis=2)
    X_delta = np.asarray(X_resp - X_base, dtype=float)

    classes = [c for c in [1, 2, 3] if int(c) in set(labels.tolist())]
    pool_idx = np.arange(X_delta.shape[1], dtype=int)
    return {
        "mouse_id": mouse_id,
        "labels": labels,
        "classes": classes,
        "X_delta": X_delta,
        "pos_xy": pos_xy,
        "pool_idx": pool_idx,
    }


def plot_mouse_scaling(df_scaling: pd.DataFrame, out_path: str):
    if df_scaling.empty:
        return
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4.5 * len(STRATEGIES), 4.2), dpi=180, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, strategy in zip(axes, STRATEGIES):
        sub = df_scaling[df_scaling["strategy"] == strategy].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        grp = (
            sub.groupby(["Condition", "k"], observed=False)["split_half_corr"]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("k")
        )
        for cond in CONDITIONS:
            ss = grp[grp["Condition"] == cond].sort_values("k")
            if ss.empty:
                continue
            x = ss["k"].to_numpy(dtype=float)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=COND_COLORS[cond], label=cond)
            ax.fill_between(x, y - se, y + se, color=COND_COLORS[cond], alpha=0.16, linewidth=0)
        ax.set_title(strategy.replace("_", " "))
        ax.set_xlabel("Neuron count (k)")
        ax.set_ylabel("Split-half template corr")
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def compute_k90_summary(df_scaling: pd.DataFrame, metric: str, target: float = 0.90) -> pd.DataFrame:
    rows = []
    cols = ["mouse_id", "Condition", "strategy", "k", metric]
    sub_all = df_scaling[cols].dropna().copy()
    if sub_all.empty:
        return pd.DataFrame()
    for (mouse_id, cond, strategy), sub in sub_all.groupby(["mouse_id", "Condition", "strategy"], observed=False):
        curve = (
            sub.groupby("k", observed=False)[metric]
            .mean()
            .reset_index()
            .sort_values("k")
        )
        x = curve["k"].to_numpy(dtype=float)
        y = curve[metric].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            rows.append(
                {
                    "mouse_id": mouse_id,
                    "Condition": cond,
                    "strategy": strategy,
                    "metric": metric,
                    "k90": np.nan,
                    "k_min": np.nan,
                    "k_max": np.nan,
                    "n_points": int(x.size),
                }
            )
            continue
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or (y_max - y_min) <= EPS:
            k90 = np.nan
        else:
            y_norm = (y - y_min) / (y_max - y_min + EPS)
            hit = np.where(y_norm >= target)[0]
            k90 = float(x[hit[0]]) if hit.size else float(x[-1])
        rows.append(
            {
                "mouse_id": mouse_id,
                "Condition": cond,
                "strategy": strategy,
                "metric": metric,
                "k90": k90,
                "k_min": float(x[0]),
                "k_max": float(x[-1]),
                "n_points": int(x.size),
            }
        )
    return pd.DataFrame(rows)


def run_scaling_for_mouse(ctx: dict, args, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    mouse_id = ctx["mouse_id"]
    labels = ctx["labels"]
    classes = ctx["classes"]
    X_delta = ctx["X_delta"]
    pos_xy = ctx["pos_xy"]
    pool_idx = ctx["pool_idx"]
    pos_pool = pos_xy[pool_idx]

    k_values = build_k_values(max_n=len(pool_idx), requested=args.k_values)
    rows = []
    for class_id in classes:
        cond = COND_MAP.get(int(class_id), str(class_id))
        X_cond_full = X_delta[labels == int(class_id)][:, pool_idx]
        if X_cond_full.shape[0] < 4:
            continue
        for strategy in STRATEGIES:
            for k in k_values:
                for rep in range(int(args.scaling_repeats)):
                    local_sel = sample_local_by_strategy(strategy, pos_pool, int(k), rng)
                    X_sel = X_cond_full[:, local_sel]
                    met = split_half_metrics(X_sel, rng)
                    rows.append(
                        {
                            "mouse_id": mouse_id,
                            "Class_ID": int(class_id),
                            "Condition": cond,
                            "strategy": strategy,
                            "k": int(k),
                            "repeat": int(rep),
                            "n_pool": int(len(pool_idx)),
                            **met,
                        }
                    )
    df_scaling = pd.DataFrame(rows)
    if df_scaling.empty:
        return df_scaling, pd.DataFrame()

    df_scaling = normalize_condition_column(df_scaling)
    k90_frames = []
    for metric in ["split_half_corr", "trial_template_cosine", "pc1_alignment"]:
        k90_frames.append(compute_k90_summary(df_scaling, metric=metric, target=args.k90_target))
    df_k90 = pd.concat(k90_frames, ignore_index=True) if k90_frames else pd.DataFrame()
    return df_scaling, df_k90


def _top_response_drop_idx(X_cond_full: np.ndarray, n_drop: int) -> np.ndarray:
    score = np.nanmean(X_cond_full, axis=0)
    score_pos = np.clip(score, 0.0, None)
    if np.nansum(score_pos) <= EPS:
        score_pos = np.abs(score)
    order = np.argsort(score_pos)
    return np.asarray(order[-n_drop:], dtype=int)


def run_dropout_for_mouse(ctx: dict, args, rng: np.random.Generator) -> pd.DataFrame:
    mouse_id = ctx["mouse_id"]
    labels = ctx["labels"]
    classes = ctx["classes"]
    X_delta = ctx["X_delta"]
    pos_xy = ctx["pos_xy"]
    pool_idx = ctx["pool_idx"]
    pos_pool = pos_xy[pool_idx]
    n_pool = int(len(pool_idx))
    frac_list = [f for f in args.dropout_fracs if 0 < float(f) < 1]

    rows = []
    for class_id in classes:
        cond = COND_MAP.get(int(class_id), str(class_id))
        X_cond_full = X_delta[labels == int(class_id)][:, pool_idx]
        if X_cond_full.shape[0] < 4 or X_cond_full.shape[1] < 8:
            continue

        baseline_vals = []
        for _ in range(int(args.baseline_repeats)):
            baseline_vals.append(split_half_metrics(X_cond_full, rng))
        base_df = pd.DataFrame(baseline_vals)
        base_mean = base_df.mean(axis=0, numeric_only=True)

        for frac in frac_list:
            n_drop = max(1, int(round(frac * n_pool)))
            n_drop = min(n_drop, n_pool - 2)
            top_drop_local = _top_response_drop_idx(X_cond_full, n_drop)

            for method in DROP_METHODS:
                repeats = int(args.dropout_repeats)
                for rep in range(repeats):
                    if method == "top_response":
                        drop_local = top_drop_local
                    elif method == "random":
                        drop_local = np.asarray(rng.choice(n_pool, size=n_drop, replace=False), dtype=int)
                    elif method == "spatial_cluster":
                        drop_local = select_clustered_local(pos_pool, n_drop, rng)
                    elif method == "spatial_distributed":
                        drop_local = select_dispersed_local(pos_pool, n_drop, rng)
                    else:
                        continue

                    keep_mask = np.ones(n_pool, dtype=bool)
                    keep_mask[drop_local] = False
                    keep_local = np.where(keep_mask)[0]
                    if keep_local.size < 2:
                        continue

                    X_keep = X_cond_full[:, keep_local]
                    met = split_half_metrics(X_keep, rng)
                    rows.append(
                        {
                            "mouse_id": mouse_id,
                            "Class_ID": int(class_id),
                            "Condition": cond,
                            "drop_method": method,
                            "drop_fraction": float(frac),
                            "n_drop": int(n_drop),
                            "repeat": int(rep),
                            "n_pool": int(n_pool),
                            "split_half_corr": met["split_half_corr"],
                            "trial_template_cosine": met["trial_template_cosine"],
                            "pc1_alignment": met["pc1_alignment"],
                            "delta_split_half_corr": met["split_half_corr"] - float(base_mean.get("split_half_corr", np.nan)),
                            "delta_trial_template_cosine": met["trial_template_cosine"] - float(base_mean.get("trial_template_cosine", np.nan)),
                            "delta_pc1_alignment": met["pc1_alignment"] - float(base_mean.get("pc1_alignment", np.nan)),
                        }
                    )
    df_dropout = pd.DataFrame(rows)
    return normalize_condition_column(df_dropout)


def plot_mouse_dropout(df_dropout: pd.DataFrame, out_path: str):
    if df_dropout.empty:
        return
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.8 * len(CONDITIONS), 4.2), dpi=180, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cond in zip(axes, CONDITIONS):
        sub = df_dropout[df_dropout["Condition"] == cond].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        grp = (
            sub.groupby(["drop_method", "drop_fraction"], observed=False)["delta_split_half_corr"]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("drop_fraction")
        )
        for method in DROP_METHODS:
            ss = grp[grp["drop_method"] == method].sort_values("drop_fraction")
            if ss.empty:
                continue
            x = (ss["drop_fraction"].to_numpy(dtype=float) * 100.0)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=DROP_COLORS[method], label=method)
            ax.fill_between(x, y - se, y + se, color=DROP_COLORS[method], alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="#777777", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_title(cond)
        ax.set_xlabel("Drop fraction (%)")
        ax.set_ylabel("Delta split-half corr")
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def run_mouse(mouse_id: str, args, seed_i: int):
    save_root = os.path.join(args.results_dir, mouse_id)
    data_out = os.path.join(save_root, "data")
    fig_out = os.path.join(save_root, "figures")
    ensure_dir(data_out)
    ensure_dir(fig_out)

    print(f"[*] Running mouse: {mouse_id}")
    rng = np.random.default_rng(seed_i)
    ctx = prepare_mouse_data(mouse_id, args)

    df_scaling, df_k90 = run_scaling_for_mouse(ctx, args, rng)
    if not df_scaling.empty:
        scaling_csv = os.path.join(data_out, "popsize_scaling_long.csv")
        df_scaling.to_csv(scaling_csv, index=False)
        print(f"[*] Saved: {scaling_csv}")
        plot_mouse_scaling(df_scaling, os.path.join(fig_out, "popsize_scaling_split_half_by_strategy.png"))
    if not df_k90.empty:
        k90_csv = os.path.join(data_out, "popsize_scaling_k90_summary.csv")
        df_k90.to_csv(k90_csv, index=False)
        print(f"[*] Saved: {k90_csv}")

    df_dropout = run_dropout_for_mouse(ctx, args, rng)
    if not df_dropout.empty:
        dropout_csv = os.path.join(data_out, "dropout_ablation_long.csv")
        df_dropout.to_csv(dropout_csv, index=False)
        print(f"[*] Saved: {dropout_csv}")
        plot_mouse_dropout(df_dropout, os.path.join(fig_out, "dropout_ablation_delta_split_half.png"))

    return {"scaling_df": df_scaling, "k90_df": df_k90, "dropout_df": df_dropout}


def plot_group_scaling(df_scaling: pd.DataFrame, out_path: str):
    if df_scaling.empty:
        return
    dm = (
        df_scaling.groupby(["mouse_id", "Condition", "strategy", "k"], observed=False)["split_half_corr"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4.5 * len(STRATEGIES), 4.2), dpi=180, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, strategy in zip(axes, STRATEGIES):
        sub = dm[dm["strategy"] == strategy].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        grp = (
            sub.groupby(["Condition", "k"], observed=False)["split_half_corr"]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("k")
        )
        for cond in CONDITIONS:
            ss = grp[grp["Condition"] == cond].sort_values("k")
            if ss.empty:
                continue
            x = ss["k"].to_numpy(dtype=float)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=COND_COLORS[cond], label=cond)
            ax.fill_between(x, y - se, y + se, color=COND_COLORS[cond], alpha=0.17, linewidth=0)
        ax.set_title(strategy.replace("_", " "))
        ax.set_xlabel("Neuron count (k)")
        ax.set_ylabel("Split-half template corr")
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def plot_group_dropout(df_dropout: pd.DataFrame, out_path: str):
    if df_dropout.empty:
        return
    dm = (
        df_dropout.groupby(["mouse_id", "Condition", "drop_method", "drop_fraction"], observed=False)["delta_split_half_corr"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.8 * len(CONDITIONS), 4.2), dpi=180, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cond in zip(axes, CONDITIONS):
        sub = dm[dm["Condition"] == cond].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        grp = (
            sub.groupby(["drop_method", "drop_fraction"], observed=False)["delta_split_half_corr"]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("drop_fraction")
        )
        for method in DROP_METHODS:
            ss = grp[grp["drop_method"] == method].sort_values("drop_fraction")
            if ss.empty:
                continue
            x = ss["drop_fraction"].to_numpy(dtype=float) * 100.0
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=DROP_COLORS[method], label=method)
            ax.fill_between(x, y - se, y + se, color=DROP_COLORS[method], alpha=0.16, linewidth=0)
        ax.axhline(0.0, color="#777777", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_title(cond)
        ax.set_xlabel("Drop fraction (%)")
        ax.set_ylabel("Delta split-half corr")
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def dropout_pairwise_stats(df_dropout: pd.DataFrame, value_col: str = "delta_split_half_corr") -> pd.DataFrame:
    if df_dropout.empty:
        return pd.DataFrame()
    dm = (
        df_dropout.groupby(["mouse_id", "Condition", "drop_fraction", "drop_method"], observed=False)[value_col]
        .mean()
        .reset_index()
    )
    pairs = [
        ("spatial_distributed", "spatial_cluster"),
        ("spatial_distributed", "random"),
        ("spatial_cluster", "random"),
        ("top_response", "random"),
        ("top_response", "spatial_distributed"),
    ]
    rows = []
    for (cond, frac), sub in dm.groupby(["Condition", "drop_fraction"], observed=False):
        piv = (
            sub.pivot_table(index="mouse_id", columns="drop_method", values=value_col, aggfunc="mean", observed=False)
            .reindex(columns=DROP_METHODS)
        )
        piv_main = piv.dropna()
        if piv_main.shape[0] >= 3:
            _, p_main = stats.friedmanchisquare(
                piv_main["top_response"],
                piv_main["spatial_cluster"],
                piv_main["spatial_distributed"],
                piv_main["random"],
            )
            p_main = float(p_main)
            n_main = int(piv_main.shape[0])
        else:
            p_main = np.nan
            n_main = int(piv_main.shape[0])

        raw_p = []
        row_buf = []
        for a, b in pairs:
            if a not in piv.columns or b not in piv.columns:
                row_buf.append((a, b, np.nan, np.nan, np.nan, np.nan))
                raw_p.append(np.nan)
                continue
            ab = piv[[a, b]].dropna()
            if ab.shape[0] < 3:
                row_buf.append((a, b, np.nan, float(np.nanmean(ab[a])) if not ab.empty else np.nan, float(np.nanmean(ab[b])) if not ab.empty else np.nan, int(ab.shape[0])))
                raw_p.append(np.nan)
                continue
            try:
                _, p = stats.wilcoxon(ab[a], ab[b])
                p = float(p)
            except Exception:
                p = np.nan
            row_buf.append((a, b, p, float(np.nanmean(ab[a])), float(np.nanmean(ab[b])), int(ab.shape[0])))
            raw_p.append(p)

        adj = holm_adjust(np.asarray(raw_p, dtype=float))
        for (a, b, p_raw, mean_a, mean_b, n_pair), p_adj in zip(row_buf, adj):
            rows.append(
                {
                    "Condition": cond,
                    "drop_fraction": float(frac),
                    "metric": value_col,
                    "main_test": "Friedman(method)",
                    "main_n_mice": n_main,
                    "main_p": p_main,
                    "comparison": f"{a} vs {b}",
                    "n_mice_pair": int(n_pair) if np.isfinite(n_pair) else np.nan,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff_a_minus_b": (mean_a - mean_b) if (np.isfinite(mean_a) and np.isfinite(mean_b)) else np.nan,
                    "pairwise_p": p_raw,
                    "pairwise_p_holm": p_adj,
                }
            )
    return pd.DataFrame(rows)


def run_group(df_scaling: pd.DataFrame, df_k90: pd.DataFrame, df_dropout: pd.DataFrame, args):
    df_scaling = normalize_condition_column(df_scaling)
    df_k90 = normalize_condition_column(df_k90)
    df_dropout = normalize_condition_column(df_dropout)

    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)

    if not df_scaling.empty:
        df_scaling.to_csv(os.path.join(group_dir, "group_popsize_scaling_long.csv"), index=False)
    if not df_dropout.empty:
        df_dropout.to_csv(os.path.join(group_dir, "group_dropout_ablation_long.csv"), index=False)

    if df_k90.empty and not df_scaling.empty:
        k90_frames = []
        for metric in ["split_half_corr", "trial_template_cosine", "pc1_alignment"]:
            k90_frames.append(compute_k90_summary(df_scaling, metric=metric, target=args.k90_target))
        df_k90 = pd.concat(k90_frames, ignore_index=True) if k90_frames else pd.DataFrame()
    if not df_k90.empty:
        df_k90.to_csv(os.path.join(group_dir, "group_popsize_k90_summary.csv"), index=False)

    k90_stat_rows = []
    if not df_k90.empty:
        for strategy in STRATEGIES:
            for metric in ["split_half_corr", "trial_template_cosine", "pc1_alignment"]:
                sub = df_k90[(df_k90["strategy"] == strategy) & (df_k90["metric"] == metric)].copy()
                if sub.empty:
                    continue
                stat_df = friedman_pairwise(sub.rename(columns={"k90": "value"}), "value")
                if not stat_df.empty:
                    stat_df.insert(0, "strategy", strategy)
                    stat_df.insert(1, "k90_metric", metric)
                    k90_stat_rows.append(stat_df)
    df_k90_stats = pd.concat(k90_stat_rows, ignore_index=True) if k90_stat_rows else pd.DataFrame()
    if not df_k90_stats.empty:
        df_k90_stats.to_csv(os.path.join(group_dir, "group_popsize_k90_stats.csv"), index=False)

    df_drop_stats = dropout_pairwise_stats(df_dropout, value_col="delta_split_half_corr")
    if not df_drop_stats.empty:
        df_drop_stats.to_csv(os.path.join(group_dir, "group_dropout_method_stats.csv"), index=False)

    if not df_scaling.empty:
        plot_group_scaling(df_scaling, os.path.join(group_dir, "group_population_scaling_split_half.png"))
    if not df_dropout.empty:
        plot_group_dropout(df_dropout, os.path.join(group_dir, "group_dropout_delta_split_half.png"))

    md_path = os.path.join(group_dir, "Group_PopSize_Dropout_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group Population-Size Scaling + Dropout Report\n\n")
        f.write("## K90 condition tests (Friedman + Wilcoxon, Holm adjusted)\n\n")
        f.write(_to_md(df_k90_stats) + "\n\n")
        f.write("## Dropout method tests on delta split-half corr\n\n")
        f.write(_to_md(df_drop_stats) + "\n\n")
    print(f"[*] Group report saved: {md_path}")


def load_mouse_outputs_from_disk(mouse_id: str, results_dir: str) -> dict:
    data_out = os.path.join(results_dir, mouse_id, "data")
    scaling_path = os.path.join(data_out, "popsize_scaling_long.csv")
    k90_path = os.path.join(data_out, "popsize_scaling_k90_summary.csv")
    dropout_path = os.path.join(data_out, "dropout_ablation_long.csv")

    df_scaling = pd.read_csv(scaling_path) if os.path.isfile(scaling_path) else pd.DataFrame()
    df_k90 = pd.read_csv(k90_path) if os.path.isfile(k90_path) else pd.DataFrame()
    df_dropout = pd.read_csv(dropout_path) if os.path.isfile(dropout_path) else pd.DataFrame()

    df_scaling = normalize_condition_column(df_scaling)
    df_k90 = normalize_condition_column(df_k90)
    df_dropout = normalize_condition_column(df_dropout)

    if not df_scaling.empty and "mouse_id" not in df_scaling.columns:
        df_scaling.insert(0, "mouse_id", mouse_id)
    if not df_k90.empty and "mouse_id" not in df_k90.columns:
        df_k90.insert(0, "mouse_id", mouse_id)
    if not df_dropout.empty and "mouse_id" not in df_dropout.columns:
        df_dropout.insert(0, "mouse_id", mouse_id)

    return {"scaling_df": df_scaling, "k90_df": df_k90, "dropout_df": df_dropout}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run population-size scaling + focal-vs-distributed dropout analyses."
    )
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)

    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--baseline-start", type=int, default=0)
    p.add_argument("--baseline-end", type=int, default=10)

    p.add_argument("--k-values", type=str, default="8,16,24,32,48,64,96,128,192,256")
    p.add_argument("--scaling-repeats", type=int, default=20)
    p.add_argument("--k90-target", type=float, default=0.90)

    p.add_argument("--dropout-fracs", type=str, default="0.05,0.10,0.20,0.30")
    p.add_argument("--dropout-repeats", type=int, default=20)
    p.add_argument("--baseline-repeats", type=int, default=20)

    p.add_argument("--group-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260409)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    args.k_values = parse_int_list(args.k_values)
    args.dropout_fracs = parse_float_list(args.dropout_fracs)

    if args.group_only:
        print("[*] Group-only mode: loading per-mouse outputs from disk.")
        scaling_list = []
        k90_list = []
        dropout_list = []
        for mouse in args.mice:
            out = load_mouse_outputs_from_disk(mouse, args.results_dir)
            if out["scaling_df"] is not None and not out["scaling_df"].empty:
                scaling_list.append(out["scaling_df"])
            else:
                print(f"[!] Missing scaling CSV for {mouse}: results/{mouse}/data/popsize_scaling_long.csv")
            if out["k90_df"] is not None and not out["k90_df"].empty:
                k90_list.append(out["k90_df"])
            if out["dropout_df"] is not None and not out["dropout_df"].empty:
                dropout_list.append(out["dropout_df"])
            else:
                print(f"[!] Missing dropout CSV for {mouse}: results/{mouse}/data/dropout_ablation_long.csv")

        df_scaling = pd.concat(scaling_list, ignore_index=True) if scaling_list else pd.DataFrame()
        df_k90 = pd.concat(k90_list, ignore_index=True) if k90_list else pd.DataFrame()
        df_dropout = pd.concat(dropout_list, ignore_index=True) if dropout_list else pd.DataFrame()
        if df_scaling.empty and df_dropout.empty:
            print("[!] No existing per-mouse outputs found. Stop.")
            return
        run_group(df_scaling, df_k90, df_dropout, args)
        print("====== Population-size scaling + dropout analysis completed (group-only) ======")
        return

    all_scaling = []
    all_k90 = []
    all_dropout = []

    base_seed = int(args.seed)
    for i, mouse in enumerate(args.mice):
        try:
            out = run_mouse(mouse, args, seed_i=int(base_seed + i * 101))
            if out["scaling_df"] is not None and not out["scaling_df"].empty:
                all_scaling.append(out["scaling_df"])
            if out["k90_df"] is not None and not out["k90_df"].empty:
                all_k90.append(out["k90_df"])
            if out["dropout_df"] is not None and not out["dropout_df"].empty:
                all_dropout.append(out["dropout_df"])
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")

    df_scaling = pd.concat(all_scaling, ignore_index=True) if all_scaling else pd.DataFrame()
    df_k90 = pd.concat(all_k90, ignore_index=True) if all_k90 else pd.DataFrame()
    df_dropout = pd.concat(all_dropout, ignore_index=True) if all_dropout else pd.DataFrame()
    if df_scaling.empty and df_dropout.empty:
        print("[!] No valid outputs generated. Stop.")
        return

    run_group(df_scaling, df_k90, df_dropout, args)
    print("====== Population-size scaling + dropout analysis completed ======")


if __name__ == "__main__":
    main()
