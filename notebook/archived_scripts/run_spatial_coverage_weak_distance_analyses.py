import argparse
import os
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from scipy.spatial import ConvexHull

from brainnetwork import load_data, preprocess_spike_data, rr_selection_class
from brainnetwork.network import compute_correlation_matrix


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


@dataclass
class SpatialConditionResult:
    class_id: int
    condition: str
    active_method: str
    active_count: int
    total_count: int
    response_mean: float
    baseline_mean: float
    delta_mean: float
    binary_hull_area: float
    binary_mean_pairwise_distance: float
    binary_nn_distance_mean: float
    binary_bin_coverage_prop: float
    binary_occupancy_area: float
    binary_spatial_entropy: float
    weighted_mean_pairwise_distance: float
    weighted_nn_distance_mean: float
    weighted_spatial_entropy: float
    weighted_effective_bin_count: float
    weighted_effective_bin_prop: float


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)


def parse_distance_bins(raw: str) -> np.ndarray:
    vals = []
    for x in str(raw).split(","):
        token = x.strip().lower()
        if token in {"inf", "+inf", "infinity"}:
            vals.append(np.inf)
        else:
            vals.append(float(token))
    arr = np.asarray(vals, dtype=float)
    if arr.size < 2:
        raise ValueError("distance bins require at least two edges")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("distance bin edges must be strictly increasing")
    return arr


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= EPS:
        return np.nan
    return float(a / b)


def upper_tri_values(matrix: np.ndarray) -> np.ndarray:
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    return matrix[mask]


def pairwise_distance_upper(neuron_pos: np.ndarray):
    pos = np.asarray(neuron_pos, dtype=float)
    if pos.ndim != 2 or pos.shape[0] != 2:
        raise ValueError("neuron_pos must be shape (2, n_neurons)")
    x = pos[0]
    y = pos[1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx * dx + dy * dy)
    tri = np.triu_indices(D.shape[0], k=1)
    return D, tri, D[tri]


def nearest_neighbor_distances(pos_xy: np.ndarray) -> np.ndarray:
    if pos_xy.shape[0] < 2:
        return np.asarray([], dtype=float)
    diff = pos_xy[:, None, :] - pos_xy[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)
    return np.min(D, axis=1)


def weighted_pairwise_distance(pos_xy: np.ndarray, weights: np.ndarray) -> float:
    n = pos_xy.shape[0]
    if n < 2:
        return np.nan
    w = np.asarray(weights, dtype=float).reshape(-1)
    d = pos_xy[:, None, :] - pos_xy[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(n, k=1)
    wd = (w[iu[0]] * w[iu[1]]) * dist[iu]
    ww = (w[iu[0]] * w[iu[1]])
    return safe_div(float(np.sum(wd)), float(np.sum(ww)))


def convex_hull_area_2d(pos_xy: np.ndarray) -> float:
    if pos_xy.shape[0] < 3:
        return np.nan
    try:
        hull = ConvexHull(pos_xy)
        return float(hull.volume)
    except Exception:
        return np.nan


def _compute_grid_index(
    points_xy: np.ndarray,
    x_min: float,
    y_min: float,
    nx: int,
    ny: int,
    grid_size: float,
) -> np.ndarray:
    x_idx = np.floor((points_xy[:, 0] - x_min) / grid_size).astype(int)
    y_idx = np.floor((points_xy[:, 1] - y_min) / grid_size).astype(int)
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)
    return y_idx * nx + x_idx


def compute_spatial_metrics_for_condition(
    pos_all: np.ndarray,
    active_idx: np.ndarray,
    weights_active: np.ndarray,
    grid_size: float,
    response_mean: float,
    baseline_mean: float,
    delta_mean: float,
    class_id: int,
    condition: str,
    active_method: str,
) -> SpatialConditionResult:
    total_n = int(pos_all.shape[1])
    act = np.asarray(active_idx, dtype=int)
    if act.size == 0:
        return SpatialConditionResult(
            class_id=class_id,
            condition=condition,
            active_method=active_method,
            active_count=0,
            total_count=total_n,
            response_mean=float(response_mean),
            baseline_mean=float(baseline_mean),
            delta_mean=float(delta_mean),
            binary_hull_area=np.nan,
            binary_mean_pairwise_distance=np.nan,
            binary_nn_distance_mean=np.nan,
            binary_bin_coverage_prop=np.nan,
            binary_occupancy_area=np.nan,
            binary_spatial_entropy=np.nan,
            weighted_mean_pairwise_distance=np.nan,
            weighted_nn_distance_mean=np.nan,
            weighted_spatial_entropy=np.nan,
            weighted_effective_bin_count=np.nan,
            weighted_effective_bin_prop=np.nan,
        )

    pos_xy = np.asarray(pos_all[:, act].T, dtype=float)
    n_active = int(pos_xy.shape[0])

    x_all = np.asarray(pos_all[0], dtype=float)
    y_all = np.asarray(pos_all[1], dtype=float)
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    nx = max(1, int(np.ceil((x_max - x_min + EPS) / grid_size)))
    ny = max(1, int(np.ceil((y_max - y_min + EPS) / grid_size)))
    n_bins = int(nx * ny)

    active_bin_idx = _compute_grid_index(pos_xy, x_min, y_min, nx, ny, grid_size)
    bin_counts = np.bincount(active_bin_idx, minlength=n_bins).astype(float)
    active_bins = bin_counts > 0

    p_bin = bin_counts[active_bins] / np.sum(bin_counts[active_bins]) if np.any(active_bins) else np.asarray([])
    binary_entropy = float(stats.entropy(p_bin)) if p_bin.size > 0 else np.nan
    binary_coverage = safe_div(float(np.sum(active_bins)), float(n_bins))
    occupancy_area = float(np.sum(active_bins) * (grid_size ** 2))

    pair_d = upper_tri_values(np.sqrt(np.sum((pos_xy[:, None, :] - pos_xy[None, :, :]) ** 2, axis=2)))
    mean_pair_d = float(np.mean(pair_d)) if pair_d.size > 0 else np.nan
    nn = nearest_neighbor_distances(pos_xy)
    nn_mean = float(np.mean(nn)) if nn.size > 0 else np.nan
    hull_area = convex_hull_area_2d(pos_xy)

    w = np.asarray(weights_active, dtype=float).reshape(-1)
    w = np.clip(w, 0.0, None)
    if np.sum(w) <= EPS:
        w = np.full(n_active, 1.0 / max(n_active, 1), dtype=float)
    else:
        w = w / np.sum(w)

    weighted_pair_d = weighted_pairwise_distance(pos_xy, w)
    weighted_nn = float(np.sum(w * nn)) if nn.size == n_active else np.nan
    bin_w = np.bincount(active_bin_idx, weights=w, minlength=n_bins).astype(float)
    bin_w_pos = bin_w[bin_w > 0]
    p_w = bin_w_pos / np.sum(bin_w_pos) if bin_w_pos.size > 0 else np.asarray([])
    weighted_entropy = float(stats.entropy(p_w)) if p_w.size > 0 else np.nan
    weighted_eff_bins = float(np.exp(weighted_entropy)) if np.isfinite(weighted_entropy) else np.nan
    weighted_eff_prop = safe_div(weighted_eff_bins, float(n_bins))

    return SpatialConditionResult(
        class_id=class_id,
        condition=condition,
        active_method=active_method,
        active_count=n_active,
        total_count=total_n,
        response_mean=float(response_mean),
        baseline_mean=float(baseline_mean),
        delta_mean=float(delta_mean),
        binary_hull_area=hull_area,
        binary_mean_pairwise_distance=mean_pair_d,
        binary_nn_distance_mean=nn_mean,
        binary_bin_coverage_prop=binary_coverage,
        binary_occupancy_area=occupancy_area,
        binary_spatial_entropy=binary_entropy,
        weighted_mean_pairwise_distance=weighted_pair_d,
        weighted_nn_distance_mean=weighted_nn,
        weighted_spatial_entropy=weighted_entropy,
        weighted_effective_bin_count=weighted_eff_bins,
        weighted_effective_bin_prop=weighted_eff_prop,
    )


def _build_active_idx_and_weights(
    class_id: int,
    rr_by_class: dict,
    delta_per_neuron: np.ndarray,
    min_active: int = 5,
) -> tuple[np.ndarray, np.ndarray, str]:
    rr_set = sorted(list(rr_by_class.get(class_id, set())))
    delta = np.asarray(delta_per_neuron, dtype=float)
    delta_pos = np.clip(delta, 0.0, None)

    if len(rr_set) >= min_active:
        active_idx = np.asarray(rr_set, dtype=int)
        method = "class_rr"
    else:
        q75 = np.nanpercentile(delta_pos, 75.0)
        active_idx = np.where(delta_pos >= q75)[0]
        if active_idx.size < min_active:
            k = min(max(min_active, int(np.ceil(0.1 * delta_pos.size))), delta_pos.size)
            active_idx = np.argsort(delta_pos)[-k:]
        method = "delta_quantile_fallback"

    w = delta_pos[active_idx]
    if np.sum(w) <= EPS:
        w = np.ones_like(w, dtype=float)
    w = w / np.sum(w)
    return active_idx.astype(int), w.astype(float), method


def decile_labels(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    n = vals.size
    order = np.argsort(vals)
    dec = np.empty(n, dtype=int)
    dec[order] = (np.arange(n, dtype=int) * 10 // n) + 1
    return dec


def compute_distance_binned_metrics(
    corr_vals: np.ndarray,
    dist_vals: np.ndarray,
    bins: np.ndarray,
    class_id: int,
    condition: str,
) -> pd.DataFrame:
    dec = decile_labels(corr_vals)
    rows = []
    for i in range(len(bins) - 1):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        m_dist = (dist_vals >= lo) & (dist_vals < hi)
        bin_name = f"{int(lo)}+" if np.isinf(hi) else f"{int(lo)}-{int(hi)}"
        if m_dist.sum() == 0:
            rows.append(
                {
                    "Class_ID": class_id,
                    "Condition": condition,
                    "distance_bin": bin_name,
                    "dist_low": lo,
                    "dist_high": hi,
                    "n_pairs": 0,
                    "mean_corr_all": np.nan,
                    "weak_decile1_mean": np.nan,
                    "weak_decile2_mean": np.nan,
                    "weak_decile12_mean": np.nan,
                    "weak30_mean": np.nan,
                    "strong_decile10_mean": np.nan,
                    "strong_weak_gap": np.nan,
                    "weak30_pair_fraction": np.nan,
                }
            )
            continue
        c = corr_vals[m_dist]
        d = dec[m_dist]
        d1 = c[d == 1]
        d2 = c[d == 2]
        d12 = c[np.isin(d, [1, 2])]
        d30 = c[np.isin(d, [1, 2, 3])]
        d10 = c[d == 10]
        rows.append(
            {
                "Class_ID": class_id,
                "Condition": condition,
                "distance_bin": bin_name,
                "dist_low": lo,
                "dist_high": hi,
                "n_pairs": int(c.size),
                "mean_corr_all": float(np.mean(c)),
                "weak_decile1_mean": float(np.mean(d1)) if d1.size > 0 else np.nan,
                "weak_decile2_mean": float(np.mean(d2)) if d2.size > 0 else np.nan,
                "weak_decile12_mean": float(np.mean(d12)) if d12.size > 0 else np.nan,
                "weak30_mean": float(np.mean(d30)) if d30.size > 0 else np.nan,
                "strong_decile10_mean": float(np.mean(d10)) if d10.size > 0 else np.nan,
                "strong_weak_gap": (float(np.mean(d10) - np.mean(d1)) if (d1.size > 0 and d10.size > 0) else np.nan),
                "weak30_pair_fraction": float(np.mean(np.isin(d, [1, 2, 3]))),
            }
        )
    return pd.DataFrame(rows)


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


def normalize_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Condition" not in df.columns and "condition" in df.columns:
        return df.rename(columns={"condition": "Condition"})
    return df


def load_mouse_outputs_from_disk(mouse_id: str, results_dir: str) -> dict[str, pd.DataFrame]:
    data_out = os.path.join(results_dir, mouse_id, "data")
    spatial_path = os.path.join(data_out, "spatial_coverage_metrics_long.csv")
    dist_path = os.path.join(data_out, "distance_binned_weakcorr_long.csv")

    spatial_df = pd.read_csv(spatial_path) if os.path.isfile(spatial_path) else pd.DataFrame()
    dist_df = pd.read_csv(dist_path) if os.path.isfile(dist_path) else pd.DataFrame()

    spatial_df = normalize_condition_column(spatial_df)
    dist_df = normalize_condition_column(dist_df)

    if not spatial_df.empty and "mouse_id" not in spatial_df.columns:
        spatial_df.insert(0, "mouse_id", mouse_id)
    if not dist_df.empty and "mouse_id" not in dist_df.columns:
        dist_df.insert(0, "mouse_id", mouse_id)

    return {"spatial_df": spatial_df, "distance_df": dist_df}


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
    labels = np.asarray(labels_spi).astype(int)
    classes = [c for c in [1, 2, 3] if c in set(labels.tolist())]
    rr_by_class = rr_selection_class(segments_spi, labels)

    response_window = slice(args.response_start, args.response_end)
    baseline_window = slice(args.baseline_start, args.baseline_end)
    _, tri_idx, dist_vals = pairwise_distance_upper(neuron_pos_spi)

    spatial_rows = []
    dist_rows = []

    fig_cov, axes_cov = plt.subplots(1, len(classes), figsize=(4.6 * len(classes), 4.2), dpi=180)
    axes_cov = np.atleast_1d(axes_cov).ravel()

    for ax, class_id in zip(axes_cov, classes):
        cond_name = COND_MAP.get(class_id, str(class_id))
        x_cond = np.asarray(segments_spi, dtype=float)[labels == class_id]
        if x_cond.shape[0] < 2:
            continue

        resp = np.nanmean(x_cond[:, :, response_window], axis=(0, 2))
        base = np.nanmean(x_cond[:, :, baseline_window], axis=(0, 2))
        delta = resp - base
        active_idx, w_active, method = _build_active_idx_and_weights(
            class_id=class_id, rr_by_class=rr_by_class, delta_per_neuron=delta, min_active=args.min_active_neurons
        )

        spatial_res = compute_spatial_metrics_for_condition(
            pos_all=np.asarray(neuron_pos_spi, dtype=float),
            active_idx=active_idx,
            weights_active=w_active,
            grid_size=args.grid_size,
            response_mean=float(np.nanmean(resp)),
            baseline_mean=float(np.nanmean(base)),
            delta_mean=float(np.nanmean(delta)),
            class_id=class_id,
            condition=cond_name,
            active_method=method,
        )
        spatial_rows.append({"mouse_id": mouse_id, **spatial_res.__dict__})

        corr = compute_correlation_matrix(
            segments=np.asarray(segments_spi, dtype=float),
            labels=labels,
            class_filter=class_id,
            time_range=None,
            zscore=False,
            balance=False,
        )
        corr_vals = corr[tri_idx]
        df_dist = compute_distance_binned_metrics(
            corr_vals=np.asarray(corr_vals, dtype=float),
            dist_vals=np.asarray(dist_vals, dtype=float),
            bins=args.distance_bins,
            class_id=class_id,
            condition=cond_name,
        )
        df_dist.insert(0, "mouse_id", mouse_id)
        dist_rows.append(df_dist)

        pos_all = np.asarray(neuron_pos_spi, dtype=float)
        pos_act = pos_all[:, active_idx]
        ax.scatter(pos_all[0], pos_all[1], s=6, color="#D9D5CE", alpha=0.45, edgecolor="none")
        size = 25 + 260 * w_active
        ax.scatter(pos_act[0], pos_act[1], s=size, color=COLORS.get(cond_name, "#4c4c4c"), alpha=0.9, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{cond_name}\nactive={len(active_idx)} ({method})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        style_axis(ax, grid=False)

    cov_png = os.path.join(fig_out, "spatial_coverage_active_map_by_condition.png")
    save_variants(fig_cov, cov_png)

    df_dist_all = pd.concat(dist_rows, ignore_index=True) if dist_rows else pd.DataFrame()
    df_dist_all = normalize_condition_column(df_dist_all)
    if not df_dist_all.empty:
        order_bins = [f"{int(args.distance_bins[i])}+" if np.isinf(args.distance_bins[i + 1]) else f"{int(args.distance_bins[i])}-{int(args.distance_bins[i + 1])}" for i in range(len(args.distance_bins) - 1)]
        fig_d, ax_d = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=180)
        for cond in CONDITIONS:
            sub = df_dist_all[df_dist_all["Condition"] == cond].copy()
            if sub.empty:
                continue
            sub["distance_bin"] = pd.Categorical(sub["distance_bin"], categories=order_bins, ordered=True)
            sub = sub.sort_values("distance_bin")
            x = np.arange(len(sub))
            ax_d[0].plot(x, sub["weak30_mean"].to_numpy(dtype=float), marker="o", lw=2, color=COLORS[cond], label=cond)
            ax_d[1].plot(x, sub["strong_weak_gap"].to_numpy(dtype=float), marker="o", lw=2, color=COLORS[cond], label=cond)
        for ax, ylab in zip(ax_d, ["Weak-end mean corr (decile 1-3)", "Strong-weak gap (D10 - D1)"]):
            ax.set_xticks(np.arange(len(order_bins)))
            ax.set_xticklabels(order_bins, rotation=25, ha="right")
            ax.set_xlabel(f"Distance bin ({args.distance_unit})")
            ax.set_ylabel(ylab)
            style_axis(ax, grid=True)
        ax_d[0].legend(frameon=False, title="")
        dist_png = os.path.join(fig_out, "distance_binned_weakcorr_profile.png")
        save_variants(fig_d, dist_png)

    df_spatial = pd.DataFrame(spatial_rows)
    df_spatial = normalize_condition_column(df_spatial)
    if not df_spatial.empty:
        spatial_csv = os.path.join(data_out, "spatial_coverage_metrics_long.csv")
        df_spatial.to_csv(spatial_csv, index=False)
        print(f"[*] Saved: {spatial_csv}")

    if not df_dist_all.empty:
        dist_csv = os.path.join(data_out, "distance_binned_weakcorr_long.csv")
        df_dist_all.to_csv(dist_csv, index=False)
        print(f"[*] Saved: {dist_csv}")

    return {"spatial_df": df_spatial, "distance_df": df_dist_all}


def plot_group_spatial_metrics(df_spatial: pd.DataFrame, out_path: str):
    metrics = [
        ("binary_hull_area", "Convex hull area"),
        ("binary_bin_coverage_prop", "Bin coverage proportion"),
        ("binary_spatial_entropy", "Spatial entropy"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df_spatial[["mouse_id", "Condition", metric]].dropna().copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
        piv = sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean", observed=False).reindex(columns=CONDITIONS)
        for _, row in piv.iterrows():
            y = row.to_numpy(dtype=float)
            m = np.isfinite(y)
            if m.sum() >= 2:
                ax.plot(np.arange(len(CONDITIONS))[m], y[m], color="#B9B3AA", lw=0.85, alpha=0.65, zorder=1)
        for i, cond in enumerate(CONDITIONS):
            vals = piv[cond].dropna().to_numpy(dtype=float)
            jit = np.linspace(-0.06, 0.06, len(vals)) if len(vals) else np.asarray([])
            ax.scatter(np.full(len(vals), i) + jit, vals, s=24, color=COLORS[cond], edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
            if len(vals):
                mu = float(np.nanmean(vals))
                se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                ax.errorbar(i, mu, yerr=se, fmt="D", color="#2F2F2F", markersize=4.8, lw=1.1, capsize=0, zorder=4)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=15)
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    save_variants(fig, out_path)


def plot_group_distance_profiles(df_dist: pd.DataFrame, out_path: str):
    order_df = (
        df_dist[["distance_bin", "dist_low", "dist_high"]]
        .drop_duplicates()
        .sort_values("dist_low")
        .copy()
    )
    order_df["distance_bin"] = order_df["distance_bin"].astype(str)
    order_bins = order_df["distance_bin"].tolist()
    bounds = order_df.set_index("distance_bin")[["dist_low", "dist_high"]].to_dict("index")
    metrics = [
        ("weak30_mean", "Weak-end mean corr (decile 1-3)"),
        ("strong_weak_gap", "Strong-weak gap (D10 - D1)"),
    ]
    fig, axes = plt.subplots(2, len(metrics), figsize=(5.6 * len(metrics), 6.2), dpi=180, sharex="col")
    axes = np.atleast_2d(axes)
    cond_handles = [Line2D([0], [0], color=COLORS[c], lw=2, marker="o", label=c) for c in CONDITIONS]
    highlight_range = (100.0, 200.0)
    highlight_bins = {
        bin_name
        for bin_name, bh in bounds.items()
        if (bh["dist_low"] < highlight_range[1]) and (bh["dist_high"] > highlight_range[0])
    }

    for col, (metric, ylab) in enumerate(metrics):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        sub = df_dist[["mouse_id", "Condition", "distance_bin", metric]].dropna().copy()
        sub["distance_bin"] = pd.Categorical(sub["distance_bin"], categories=order_bins, ordered=True)
        grp = (
            sub.groupby(["Condition", "distance_bin"], observed=False)[metric]
            .agg(["mean", "sem"])
            .reset_index()
        )

        for cond in CONDITIONS:
            ss = grp[grp["Condition"] == cond].sort_values("distance_bin")
            x = np.arange(len(ss))
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            if y.size == 0:
                continue
            ax_top.plot(x, y, marker="o", lw=2, color=COLORS[cond])
            ax_top.fill_between(x, y - se, y + se, color=COLORS[cond], alpha=0.18, linewidth=0)

        pivot = (
            sub.pivot_table(
                index=["mouse_id", "distance_bin"],
                columns="Condition",
                values=metric,
                observed=False,
            )
            .reindex(columns=CONDITIONS)
            .reset_index()
        )
        pivot["structured_mean"] = pivot[["Divergent", "Convergent"]].mean(axis=1)
        pivot["delta_random"] = pivot["Random"] - pivot["structured_mean"]
        delta = pivot.dropna(subset=["delta_random"])

        means = delta.groupby("distance_bin", observed=False)["delta_random"].mean()
        ses = delta.groupby("distance_bin", observed=False)["delta_random"].sem()
        x_bins = np.arange(len(order_bins))
        bar_y = np.array([means.get(bin_name, np.nan) for bin_name in order_bins], dtype=float)
        bar_se = np.array([ses.get(bin_name, 0.0) for bin_name in order_bins], dtype=float)
        ax_bottom.bar(x_bins, bar_y, color="#E3D5CA", edgecolor="#9A7B5F", linewidth=0.8, alpha=0.9)
        ax_bottom.errorbar(x_bins, bar_y, yerr=bar_se, fmt="none", ecolor="#6B533D", linewidth=1.1, capsize=3)
        for bin_idx, bin_name in enumerate(order_bins):
            vals = delta[delta["distance_bin"] == bin_name]["delta_random"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            jit = np.linspace(-0.12, 0.12, vals.size) if vals.size > 1 else np.zeros_like(vals)
            ax_bottom.scatter(
                np.full(vals.size, bin_idx) + jit,
                vals,
                s=28,
                color=COLORS["Random"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
            )

        for ax in (ax_top, ax_bottom):
            ax.set_xticks(np.arange(len(order_bins)))
            ax.set_xticklabels(order_bins, rotation=25, ha="right")
            ax.set_xlabel("Distance bin")
            if highlight_bins:
                for bin_name in highlight_bins:
                    if bin_name in order_bins:
                        idx = order_bins.index(bin_name)
                        ax.axvspan(idx - 0.45, idx + 0.45, color="#F6EEE6", alpha=0.35, zorder=0)
            style_axis(ax, grid=(ax is ax_bottom))

        ax_top.set_ylabel(ylab)
        ax_bottom.set_ylabel("Random − structured mean")
        ax_bottom.axhline(0, color="#595959", linestyle="--", linewidth=0.8, alpha=0.8)

    axes[0, 0].legend(handles=cond_handles, frameon=False, title="")
    fig.tight_layout(pad=2.2, h_pad=2.4)
    save_variants(fig, out_path)


def run_group(spatial_all: pd.DataFrame, dist_all: pd.DataFrame, args):
    spatial_all = normalize_condition_column(spatial_all)
    dist_all = normalize_condition_column(dist_all)

    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)
    if not spatial_all.empty:
        spatial_all.to_csv(os.path.join(group_dir, "group_spatial_coverage_metrics_long.csv"), index=False)
    if not dist_all.empty:
        dist_all.to_csv(os.path.join(group_dir, "group_distance_binned_weakcorr_long.csv"), index=False)

    spatial_metrics = [
        "binary_hull_area",
        "binary_mean_pairwise_distance",
        "binary_nn_distance_mean",
        "binary_bin_coverage_prop",
        "binary_spatial_entropy",
        "weighted_mean_pairwise_distance",
        "weighted_nn_distance_mean",
        "weighted_spatial_entropy",
        "weighted_effective_bin_prop",
    ]
    stat_spatial_rows = []
    for m in spatial_metrics:
        if m in spatial_all.columns:
            stat_spatial_rows.append(friedman_pairwise(spatial_all, m))
    stat_spatial = pd.concat(stat_spatial_rows, ignore_index=True) if stat_spatial_rows else pd.DataFrame()
    if not stat_spatial.empty:
        stat_spatial.to_csv(os.path.join(group_dir, "group_spatial_coverage_stats.csv"), index=False)

    dist_stat_rows = []
    if not dist_all.empty:
        for metric in ["weak30_mean", "strong_weak_gap"]:
            for bin_name in sorted(dist_all["distance_bin"].dropna().astype(str).unique()):
                sub = dist_all[dist_all["distance_bin"].astype(str) == bin_name].copy()
                res = friedman_pairwise(sub, metric)
                if not res.empty:
                    res.insert(1, "distance_bin", bin_name)
                    dist_stat_rows.append(res)
    stat_dist = pd.concat(dist_stat_rows, ignore_index=True) if dist_stat_rows else pd.DataFrame()
    if not stat_dist.empty:
        stat_dist.to_csv(os.path.join(group_dir, "group_distance_weakcorr_stats.csv"), index=False)

    if not spatial_all.empty:
        plot_group_spatial_metrics(
            spatial_all,
            os.path.join(group_dir, "group_spatial_coverage_key_metrics.png"),
        )
    if not dist_all.empty:
        plot_group_distance_profiles(
            dist_all,
            os.path.join(group_dir, "group_distance_weakcorr_profiles.png"),
        )

    md_path = os.path.join(group_dir, "Group_Spatial_WeakDistance_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group Spatial Coverage + Distance-binned Weak-Correlation Report\n\n")
        f.write("## Spatial condition tests (Friedman + Wilcoxon, Holm adjusted)\n\n")
        f.write(_to_md(stat_spatial) + "\n\n")
        f.write("## Distance-binned tests (Friedman + Wilcoxon, Holm adjusted)\n\n")
        f.write(_to_md(stat_dist) + "\n\n")
    print(f"[*] Group report saved: {md_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run spatial coverage + distance-binned weak-correlation analyses (per-mouse and group)."
    )
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)
    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--baseline-start", type=int, default=0)
    p.add_argument("--baseline-end", type=int, default=10)
    p.add_argument("--grid-size", type=float, default=80.0)
    p.add_argument("--distance-bins", type=str, default="0,80,160,240,320,400,600,800,inf")
    p.add_argument("--distance-unit", type=str, default="um")
    p.add_argument("--min-active-neurons", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260409)
    p.add_argument(
        "--group-only",
        action="store_true",
        help="Skip per-mouse compute and build group summary from existing per-mouse CSV outputs.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    args.distance_bins = parse_distance_bins(args.distance_bins)

    if args.group_only:
        print("[*] Group-only mode: loading per-mouse outputs from disk.")
        loaded_spatial = []
        loaded_dist = []
        for mouse in args.mice:
            loaded = load_mouse_outputs_from_disk(mouse, args.results_dir)
            if loaded["spatial_df"] is not None and not loaded["spatial_df"].empty:
                loaded_spatial.append(loaded["spatial_df"])
            else:
                print(f"[!] Missing spatial CSV for {mouse}: results/{mouse}/data/spatial_coverage_metrics_long.csv")
            if loaded["distance_df"] is not None and not loaded["distance_df"].empty:
                loaded_dist.append(loaded["distance_df"])
            else:
                print(f"[!] Missing distance CSV for {mouse}: results/{mouse}/data/distance_binned_weakcorr_long.csv")

        spatial_all = pd.concat(loaded_spatial, ignore_index=True) if loaded_spatial else pd.DataFrame()
        dist_all = pd.concat(loaded_dist, ignore_index=True) if loaded_dist else pd.DataFrame()
        if spatial_all.empty and dist_all.empty:
            print("[!] No existing per-mouse outputs found. Stop.")
            return

        run_group(spatial_all, dist_all, args)
        print("====== Spatial coverage + weak-distance analysis completed (group-only) ======")
        return

    all_spatial = []
    all_dist = []
    base_seed = int(args.seed)
    for i, mouse in enumerate(args.mice):
        try:
            seed_i = int(base_seed + i * 101)
            out = run_mouse(mouse, args, seed_i)
            if out["spatial_df"] is not None and not out["spatial_df"].empty:
                all_spatial.append(out["spatial_df"])
            if out["distance_df"] is not None and not out["distance_df"].empty:
                all_dist.append(out["distance_df"])
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")

    spatial_all = pd.concat(all_spatial, ignore_index=True) if all_spatial else pd.DataFrame()
    dist_all = pd.concat(all_dist, ignore_index=True) if all_dist else pd.DataFrame()
    if spatial_all.empty and dist_all.empty:
        print("[!] No valid outputs generated. Stop.")
        return

    run_group(spatial_all, dist_all, args)
    print("====== Spatial coverage + weak-distance analysis completed ======")


if __name__ == "__main__":
    main()
