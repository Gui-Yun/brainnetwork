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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= EPS:
        return np.nan
    return float(a / b)


def normalize_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Condition" not in df.columns and "condition" in df.columns:
        return df.rename(columns={"condition": "Condition"})
    return df


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def gini_index(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.clip(arr, 0.0, None)
    if np.sum(arr) <= EPS:
        return np.nan
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    cum = np.cumsum(arr_sorted)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)


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


def split_half_template_corr(X_trials_by_neuron: np.ndarray, rng: np.random.Generator) -> float:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    n_trials, n_neurons = X.shape if X.ndim == 2 else (0, 0)
    if n_trials < 4 or n_neurons < 2:
        return np.nan
    perm = rng.permutation(n_trials)
    half = n_trials // 2
    idx_a = perm[:half]
    idx_b = perm[half : half + half]
    if idx_a.size < 2 or idx_b.size < 2:
        return np.nan
    ta = np.nanmean(X[idx_a], axis=0)
    tb = np.nanmean(X[idx_b], axis=0)
    return safe_corr(ta, tb)


def mean_split_half_corr(X_trials_by_neuron: np.ndarray, rng: np.random.Generator, repeats: int) -> float:
    vals = []
    for _ in range(int(repeats)):
        vals.append(split_half_template_corr(X_trials_by_neuron, rng))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def assign_patches(pos_xy: np.ndarray, grid_size: float) -> tuple[np.ndarray, pd.DataFrame, dict]:
    pos = np.asarray(pos_xy, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos_xy must be shaped (n_neurons, 2)")
    x = pos[:, 0]
    y = pos[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    nx = max(1, int(np.ceil((x_max - x_min + EPS) / grid_size)))
    ny = max(1, int(np.ceil((y_max - y_min + EPS) / grid_size)))
    x_idx = np.floor((x - x_min) / grid_size).astype(int)
    y_idx = np.floor((y - y_min) / grid_size).astype(int)
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)
    uid = y_idx * nx + x_idx
    uniq_uid = np.unique(uid)

    uid_to_local = {int(u): i for i, u in enumerate(uniq_uid.tolist())}
    patch_local = np.asarray([uid_to_local[int(u)] for u in uid], dtype=int)

    rows = []
    for local_id, u in enumerate(uniq_uid.tolist()):
        mask = uid == int(u)
        yi = int(u // nx)
        xi = int(u % nx)
        rows.append(
            {
                "patch_local_id": int(local_id),
                "patch_uid": int(u),
                "x_idx": int(xi),
                "y_idx": int(yi),
                "n_neurons": int(np.sum(mask)),
                "x_center": float(x_min + (xi + 0.5) * grid_size),
                "y_center": float(y_min + (yi + 0.5) * grid_size),
            }
        )
    patch_meta = pd.DataFrame(rows).sort_values("patch_local_id").reset_index(drop=True)
    info = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "nx": nx, "ny": ny}
    return patch_local, patch_meta, info


def compute_patch_trial_matrix(
    X_cond: np.ndarray,
    patch_local: np.ndarray,
    valid_patch_ids: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X_cond, dtype=float)
    n_trials = X.shape[0]
    out = np.full((n_trials, valid_patch_ids.size), np.nan, dtype=float)
    for j, pid in enumerate(valid_patch_ids.tolist()):
        mask = patch_local == int(pid)
        if np.sum(mask) == 0:
            continue
        out[:, j] = np.nanmean(X[:, mask], axis=1)
    return out


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


def analyze_condition_patch_tiling(
    mouse_id: str,
    class_id: int,
    condition: str,
    X_cond: np.ndarray,
    patch_local: np.ndarray,
    patch_meta: pd.DataFrame,
    args,
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame]:
    valid_meta = patch_meta[patch_meta["n_neurons"] >= int(args.min_patch_neurons)].copy()
    valid_ids = valid_meta["patch_local_id"].to_numpy(dtype=int)
    n_trials, n_neurons = X_cond.shape if X_cond.ndim == 2 else (0, 0)

    if valid_ids.size < 2 or n_trials < 4 or n_neurons < 2:
        summary = {
            "mouse_id": mouse_id,
            "Class_ID": int(class_id),
            "Condition": condition,
            "n_trials": int(n_trials),
            "n_neurons": int(n_neurons),
            "n_patches_total": int(patch_meta.shape[0]),
            "n_patches_valid": int(valid_ids.size),
            "participation_ratio": np.nan,
            "patch_entropy": np.nan,
            "patch_effective_count": np.nan,
            "patch_effective_prop": np.nan,
            "patch_gini": np.nan,
            "patch_mean_corr": np.nan,
            "patch_mean_abs_corr": np.nan,
            "patch_complementarity_index": np.nan,
            "full_split_half_corr": np.nan,
            "lopo_drop_mean": np.nan,
            "lopo_drop_max": np.nan,
            "lopo_drop_sum": np.nan,
            "lopo_top1_share": np.nan,
            "lopo_effective_patch_count": np.nan,
            "lopo_distributed_support_index": np.nan,
        }
        return summary, pd.DataFrame()

    patch_trial = compute_patch_trial_matrix(X_cond, patch_local, valid_ids)
    pos = np.clip(patch_trial, 0.0, None)
    patch_weight = np.nanmean(pos, axis=0)
    if not np.any(np.isfinite(patch_weight)) or np.nansum(patch_weight) <= EPS:
        patch_weight = np.nanmean(np.abs(patch_trial), axis=0)
    patch_weight = np.nan_to_num(patch_weight, nan=0.0, posinf=0.0, neginf=0.0)

    max_w = float(np.max(patch_weight)) if patch_weight.size else 0.0
    thr = float(args.patch_active_frac_of_max) * max_w
    active_patch = patch_weight >= max(thr, EPS)
    participation_ratio = safe_div(float(np.sum(active_patch)), float(valid_ids.size))

    if np.sum(patch_weight) > EPS:
        pw = patch_weight / np.sum(patch_weight)
        patch_entropy = float(stats.entropy(pw))
        patch_effective_count = float(np.exp(patch_entropy))
        patch_effective_prop = safe_div(patch_effective_count, float(valid_ids.size))
    else:
        patch_entropy = np.nan
        patch_effective_count = np.nan
        patch_effective_prop = np.nan
    patch_gini = gini_index(patch_weight)

    if patch_trial.shape[1] >= 2:
        corr = np.corrcoef(patch_trial, rowvar=False)
        iu = np.triu_indices(corr.shape[0], k=1)
        vals = corr[iu]
        vals = vals[np.isfinite(vals)]
        patch_mean_corr = float(np.mean(vals)) if vals.size else np.nan
        patch_mean_abs_corr = float(np.mean(np.abs(vals))) if vals.size else np.nan
        patch_complementarity_index = (1.0 - patch_mean_abs_corr) if np.isfinite(patch_mean_abs_corr) else np.nan
    else:
        patch_mean_corr = np.nan
        patch_mean_abs_corr = np.nan
        patch_complementarity_index = np.nan

    full_split_half = mean_split_half_corr(X_cond, rng, repeats=args.lopo_repeats)

    detail_rows = []
    for j, pid in enumerate(valid_ids.tolist()):
        keep_mask = patch_local != int(pid)
        n_keep = int(np.sum(keep_mask))
        if n_keep >= int(args.min_neurons_after_drop):
            ablated_split_half = mean_split_half_corr(X_cond[:, keep_mask], rng, repeats=args.lopo_repeats)
            lopo_delta = full_split_half - ablated_split_half if (np.isfinite(full_split_half) and np.isfinite(ablated_split_half)) else np.nan
        else:
            ablated_split_half = np.nan
            lopo_delta = np.nan
        pm = valid_meta.iloc[j]
        detail_rows.append(
            {
                "mouse_id": mouse_id,
                "Class_ID": int(class_id),
                "Condition": condition,
                "patch_local_id": int(pid),
                "x_idx": int(pm["x_idx"]),
                "y_idx": int(pm["y_idx"]),
                "n_neurons_patch": int(pm["n_neurons"]),
                "mean_response_weight": float(patch_weight[j]),
                "active_patch": bool(active_patch[j]),
                "full_split_half_corr": full_split_half,
                "ablated_split_half_corr": ablated_split_half,
                "lopo_delta_split_half": lopo_delta,
            }
        )

    df_detail = pd.DataFrame(detail_rows)
    lopo_vals = df_detail["lopo_delta_split_half"].to_numpy(dtype=float)
    lopo_vals = lopo_vals[np.isfinite(lopo_vals)]
    if lopo_vals.size:
        lopo_drop_mean = float(np.mean(lopo_vals))
        lopo_drop_max = float(np.max(lopo_vals))
        lopo_drop_sum = float(np.sum(lopo_vals))
        lopo_pos = np.clip(lopo_vals, 0.0, None)
        if np.sum(lopo_pos) > EPS:
            p = lopo_pos / np.sum(lopo_pos)
            lopo_top1_share = float(np.max(p))
            lopo_effective_patch_count = float(np.exp(stats.entropy(p)))
            lopo_distributed_support_index = 1.0 - lopo_top1_share
        else:
            lopo_top1_share = np.nan
            lopo_effective_patch_count = np.nan
            lopo_distributed_support_index = np.nan
    else:
        lopo_drop_mean = np.nan
        lopo_drop_max = np.nan
        lopo_drop_sum = np.nan
        lopo_top1_share = np.nan
        lopo_effective_patch_count = np.nan
        lopo_distributed_support_index = np.nan

    summary = {
        "mouse_id": mouse_id,
        "Class_ID": int(class_id),
        "Condition": condition,
        "n_trials": int(n_trials),
        "n_neurons": int(n_neurons),
        "n_patches_total": int(patch_meta.shape[0]),
        "n_patches_valid": int(valid_ids.size),
        "participation_ratio": participation_ratio,
        "patch_entropy": patch_entropy,
        "patch_effective_count": patch_effective_count,
        "patch_effective_prop": patch_effective_prop,
        "patch_gini": patch_gini,
        "patch_mean_corr": patch_mean_corr,
        "patch_mean_abs_corr": patch_mean_abs_corr,
        "patch_complementarity_index": patch_complementarity_index,
        "full_split_half_corr": full_split_half,
        "lopo_drop_mean": lopo_drop_mean,
        "lopo_drop_max": lopo_drop_max,
        "lopo_drop_sum": lopo_drop_sum,
        "lopo_top1_share": lopo_top1_share,
        "lopo_effective_patch_count": lopo_effective_patch_count,
        "lopo_distributed_support_index": lopo_distributed_support_index,
    }
    return summary, df_detail


def patch_grid_matrix(df_patch: pd.DataFrame, value_col: str) -> np.ndarray:
    if df_patch is None or df_patch.empty:
        return np.full((1, 1), np.nan, dtype=float)
    x_max = int(df_patch["x_idx"].max())
    y_max = int(df_patch["y_idx"].max())
    mat = np.full((y_max + 1, x_max + 1), np.nan, dtype=float)
    for _, row in df_patch.iterrows():
        xi = int(row["x_idx"])
        yi = int(row["y_idx"])
        val = float(row[value_col]) if value_col in row and np.isfinite(row[value_col]) else np.nan
        mat[yi, xi] = val
    return mat


def plot_mouse_patch_maps(df_patch: pd.DataFrame, value_col: str, out_path: str, cmap: str = "viridis"):
    if df_patch.empty:
        return
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.3 * len(CONDITIONS), 4.1), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    vmax = np.nanmax(df_patch[value_col].to_numpy(dtype=float)) if np.any(np.isfinite(df_patch[value_col].to_numpy(dtype=float))) else np.nan
    vmin = np.nanmin(df_patch[value_col].to_numpy(dtype=float)) if np.any(np.isfinite(df_patch[value_col].to_numpy(dtype=float))) else np.nan
    for ax, cond in zip(axes, CONDITIONS):
        sub = df_patch[df_patch["Condition"] == cond].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        mat = patch_grid_matrix(sub, value_col=value_col)
        im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(cond)
        ax.set_xlabel("Patch x-index")
        ax.set_ylabel("Patch y-index")
        style_axis(ax, grid=False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_variants(fig, out_path)


def run_mouse(mouse_id: str, args, seed_i: int):
    save_root = os.path.join(args.results_dir, mouse_id)
    data_out = os.path.join(save_root, "data")
    fig_out = os.path.join(save_root, "figures")
    ensure_dir(data_out)
    ensure_dir(fig_out)

    print(f"[*] Running mouse: {mouse_id}")
    data_path = os.path.join(args.base_dir, mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path, data_type="spikes")
    segments_spi, labels_spi, neuron_pos_spi = preprocess_spike_data(
        neuron_data, neuron_pos, start_edges, stimulus_data, extract_rr=True
    )
    labels = np.asarray(labels_spi, dtype=int)
    classes = [c for c in [1, 2, 3] if c in set(labels.tolist())]

    segments = np.asarray(segments_spi, dtype=float)
    response_window = slice(args.response_start, args.response_end)
    baseline_window = slice(args.baseline_start, args.baseline_end)
    X_resp = np.nanmean(segments[:, :, response_window], axis=2)
    X_base = np.nanmean(segments[:, :, baseline_window], axis=2)
    X_delta = np.asarray(X_resp - X_base, dtype=float)
    pos_xy = np.asarray(neuron_pos_spi, dtype=float).T

    patch_local, patch_meta, _ = assign_patches(pos_xy, grid_size=float(args.grid_size))
    rng = np.random.default_rng(seed_i)

    summary_rows = []
    patch_rows = []
    for class_id in classes:
        cond = COND_MAP.get(int(class_id), str(class_id))
        X_cond = X_delta[labels == int(class_id)]
        summary, df_detail = analyze_condition_patch_tiling(
            mouse_id=mouse_id,
            class_id=int(class_id),
            condition=cond,
            X_cond=X_cond,
            patch_local=patch_local,
            patch_meta=patch_meta,
            args=args,
            rng=rng,
        )
        summary_rows.append(summary)
        if df_detail is not None and not df_detail.empty:
            patch_rows.append(df_detail)

    df_summary = pd.DataFrame(summary_rows)
    df_patch = pd.concat(patch_rows, ignore_index=True) if patch_rows else pd.DataFrame()
    df_summary = normalize_condition_column(df_summary)
    df_patch = normalize_condition_column(df_patch)

    if not df_summary.empty:
        p = os.path.join(data_out, "patch_tiling_condition_summary.csv")
        df_summary.to_csv(p, index=False)
        print(f"[*] Saved: {p}")
    if not df_patch.empty:
        p = os.path.join(data_out, "patch_tiling_patch_detail.csv")
        df_patch.to_csv(p, index=False)
        print(f"[*] Saved: {p}")

    if not df_patch.empty:
        plot_mouse_patch_maps(
            df_patch=df_patch,
            value_col="mean_response_weight",
            out_path=os.path.join(fig_out, "patch_tiling_response_weight_map_by_condition.png"),
            cmap="magma",
        )
        plot_mouse_patch_maps(
            df_patch=df_patch,
            value_col="lopo_delta_split_half",
            out_path=os.path.join(fig_out, "patch_tiling_lopo_delta_map_by_condition.png"),
            cmap="viridis",
        )

    return {"summary_df": df_summary, "patch_df": df_patch}


def plot_group_key_metrics(df_summary: pd.DataFrame, out_path: str):
    metrics = [
        ("participation_ratio", "Patch participation ratio"),
        ("patch_effective_prop", "Effective patch proportion"),
        ("patch_complementarity_index", "Patch complementarity index"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 4.2), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df_summary[["mouse_id", "Condition", metric]].dropna().copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
        piv = (
            sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean", observed=False)
            .reindex(columns=CONDITIONS)
        )
        for _, row in piv.iterrows():
            y = row.to_numpy(dtype=float)
            m = np.isfinite(y)
            if m.sum() >= 2:
                ax.plot(np.arange(len(CONDITIONS))[m], y[m], color="#B9B3AA", lw=0.85, alpha=0.65, zorder=1)
        for i, cond in enumerate(CONDITIONS):
            vals = piv[cond].dropna().to_numpy(dtype=float)
            jit = np.linspace(-0.06, 0.06, len(vals)) if len(vals) else np.asarray([])
            ax.scatter(np.full(len(vals), i) + jit, vals, s=24, color=COLORS[cond], edgecolor="white", linewidth=0.5, alpha=0.92, zorder=3)
            if len(vals):
                mu = float(np.nanmean(vals))
                se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                ax.errorbar(i, mu, yerr=se, fmt="D", color="#2F2F2F", markersize=4.8, lw=1.1, capsize=0, zorder=4)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=15)
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    save_variants(fig, out_path)


def plot_group_lopo_metrics(df_summary: pd.DataFrame, out_path: str):
    metrics = [
        ("lopo_drop_sum", "LOPO drop sum"),
        ("lopo_top1_share", "LOPO top1 share"),
        ("lopo_distributed_support_index", "LOPO distributed support index"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 4.2), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df_summary[["mouse_id", "Condition", metric]].dropna().copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
        piv = (
            sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean", observed=False)
            .reindex(columns=CONDITIONS)
        )
        for _, row in piv.iterrows():
            y = row.to_numpy(dtype=float)
            m = np.isfinite(y)
            if m.sum() >= 2:
                ax.plot(np.arange(len(CONDITIONS))[m], y[m], color="#B9B3AA", lw=0.85, alpha=0.65, zorder=1)
        for i, cond in enumerate(CONDITIONS):
            vals = piv[cond].dropna().to_numpy(dtype=float)
            jit = np.linspace(-0.06, 0.06, len(vals)) if len(vals) else np.asarray([])
            ax.scatter(np.full(len(vals), i) + jit, vals, s=24, color=COLORS[cond], edgecolor="white", linewidth=0.5, alpha=0.92, zorder=3)
            if len(vals):
                mu = float(np.nanmean(vals))
                se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                ax.errorbar(i, mu, yerr=se, fmt="D", color="#2F2F2F", markersize=4.8, lw=1.1, capsize=0, zorder=4)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=15)
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    save_variants(fig, out_path)


def run_group(df_summary: pd.DataFrame, df_patch: pd.DataFrame, args):
    df_summary = normalize_condition_column(df_summary)
    df_patch = normalize_condition_column(df_patch)

    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)

    if not df_summary.empty:
        df_summary.to_csv(os.path.join(group_dir, "group_patch_tiling_condition_summary.csv"), index=False)
    if not df_patch.empty:
        df_patch.to_csv(os.path.join(group_dir, "group_patch_tiling_patch_detail.csv"), index=False)

    metrics = [
        "participation_ratio",
        "patch_effective_prop",
        "patch_complementarity_index",
        "patch_gini",
        "full_split_half_corr",
        "lopo_drop_mean",
        "lopo_drop_sum",
        "lopo_top1_share",
        "lopo_distributed_support_index",
    ]
    stat_rows = []
    for m in metrics:
        if m in df_summary.columns:
            stat_rows.append(friedman_pairwise(df_summary, m))
    df_stats = pd.concat(stat_rows, ignore_index=True) if stat_rows else pd.DataFrame()
    if not df_stats.empty:
        df_stats.to_csv(os.path.join(group_dir, "group_patch_tiling_stats.csv"), index=False)

    if not df_summary.empty:
        plot_group_key_metrics(df_summary, os.path.join(group_dir, "group_patch_tiling_key_metrics.png"))
        plot_group_lopo_metrics(df_summary, os.path.join(group_dir, "group_patch_tiling_lopo_metrics.png"))

    md_path = os.path.join(group_dir, "Group_PatchTiling_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group Patch-Tiling Report\n\n")
        f.write("## Condition tests (Friedman + Wilcoxon, Holm adjusted)\n\n")
        f.write(_to_md(df_stats) + "\n\n")
    print(f"[*] Group report saved: {md_path}")


def load_mouse_outputs_from_disk(mouse_id: str, results_dir: str) -> dict:
    data_out = os.path.join(results_dir, mouse_id, "data")
    summary_path = os.path.join(data_out, "patch_tiling_condition_summary.csv")
    patch_path = os.path.join(data_out, "patch_tiling_patch_detail.csv")

    df_summary = pd.read_csv(summary_path) if os.path.isfile(summary_path) else pd.DataFrame()
    df_patch = pd.read_csv(patch_path) if os.path.isfile(patch_path) else pd.DataFrame()

    df_summary = normalize_condition_column(df_summary)
    df_patch = normalize_condition_column(df_patch)
    if not df_summary.empty and "mouse_id" not in df_summary.columns:
        df_summary.insert(0, "mouse_id", mouse_id)
    if not df_patch.empty and "mouse_id" not in df_patch.columns:
        df_patch.insert(0, "mouse_id", mouse_id)
    return {"summary_df": df_summary, "patch_df": df_patch}


def parse_args():
    p = argparse.ArgumentParser(description="Run spatial patch-tiling complementarity analyses.")
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)

    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--baseline-start", type=int, default=0)
    p.add_argument("--baseline-end", type=int, default=10)

    p.add_argument("--grid-size", type=float, default=160.0)
    p.add_argument("--min-patch-neurons", type=int, default=20)
    p.add_argument("--min-neurons-after-drop", type=int, default=10)
    p.add_argument("--patch-active-frac-of-max", type=float, default=0.10)
    p.add_argument("--lopo-repeats", type=int, default=12)

    p.add_argument("--group-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260410)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)

    if args.group_only:
        print("[*] Group-only mode: loading per-mouse outputs from disk.")
        all_summary = []
        all_patch = []
        for mouse in args.mice:
            out = load_mouse_outputs_from_disk(mouse, args.results_dir)
            if out["summary_df"] is not None and not out["summary_df"].empty:
                all_summary.append(out["summary_df"])
            else:
                print(f"[!] Missing summary CSV for {mouse}: results/{mouse}/data/patch_tiling_condition_summary.csv")
            if out["patch_df"] is not None and not out["patch_df"].empty:
                all_patch.append(out["patch_df"])
            else:
                print(f"[!] Missing patch CSV for {mouse}: results/{mouse}/data/patch_tiling_patch_detail.csv")

        df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
        df_patch = pd.concat(all_patch, ignore_index=True) if all_patch else pd.DataFrame()
        if df_summary.empty and df_patch.empty:
            print("[!] No existing per-mouse outputs found. Stop.")
            return
        run_group(df_summary, df_patch, args)
        print("====== Patch-tiling analysis completed (group-only) ======")
        return

    all_summary = []
    all_patch = []
    base_seed = int(args.seed)
    for i, mouse in enumerate(args.mice):
        try:
            out = run_mouse(mouse, args, seed_i=int(base_seed + i * 101))
            if out["summary_df"] is not None and not out["summary_df"].empty:
                all_summary.append(out["summary_df"])
            if out["patch_df"] is not None and not out["patch_df"].empty:
                all_patch.append(out["patch_df"])
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")

    df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    df_patch = pd.concat(all_patch, ignore_index=True) if all_patch else pd.DataFrame()
    if df_summary.empty and df_patch.empty:
        print("[!] No valid outputs generated. Stop.")
        return

    run_group(df_summary, df_patch, args)
    print("====== Patch-tiling analysis completed ======")


if __name__ == "__main__":
    main()
