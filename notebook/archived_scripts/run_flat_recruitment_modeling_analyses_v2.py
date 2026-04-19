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
COND_COLORS = {"Divergent": "#7F9C96", "Convergent": "#8B90A8", "Random": "#B98372"}
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


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def parse_str_list(raw: str) -> list[str]:
    vals = []
    for token in str(raw).split(","):
        t = token.strip()
        if t:
            vals.append(t)
    return vals


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


def normalize_model_variant_column(df: pd.DataFrame, default_variant: str = "tau_only") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "model_variant" not in out.columns:
        out["model_variant"] = default_variant
    out["model_variant"] = out["model_variant"].fillna(default_variant).astype(str)
    return out


def gini_index(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.clip(arr, 0.0, None)
    if np.sum(arr) <= EPS:
        return np.nan
    arr = np.sort(arr)
    n = arr.size
    c = np.cumsum(arr)
    g = (n + 1 - 2 * np.sum(c) / c[-1]) / n
    return float(g)


def entropy_norm(p: np.ndarray) -> float:
    arr = np.asarray(p, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    arr = np.clip(arr, 0.0, None)
    s = np.sum(arr)
    if s <= EPS:
        return np.nan
    arr = arr / s
    h = float(stats.entropy(arr))
    hmax = float(np.log(arr.size))
    return safe_div(h, hmax)


def pairwise_distance_matrix(pos_xy: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_xy, dtype=float)
    d = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


def weighted_radius(q: np.ndarray, pos_xy: np.ndarray) -> float:
    q = np.asarray(q, dtype=float).reshape(-1)
    q = np.clip(q, 0.0, None)
    s = np.sum(q)
    if s <= EPS:
        return np.nan
    q = q / s
    pos = np.asarray(pos_xy, dtype=float)
    cen = q @ pos
    sq = np.sum(pos * pos, axis=1)
    rad2 = float(q @ sq - np.dot(cen, cen))
    return max(rad2, 0.0)


def weighted_pair_span(q: np.ndarray, D: np.ndarray) -> float:
    q = np.asarray(q, dtype=float).reshape(-1)
    q = np.clip(q, 0.0, None)
    s = np.sum(q)
    if s <= EPS:
        return np.nan
    q = q / s
    return float(q @ D @ q)


def orthogonal_expansion_metrics(X_trials_by_neuron: np.ndarray) -> dict:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        return {
            "parallel_variance": np.nan,
            "orthogonal_variance": np.nan,
            "orthogonal_parallel_ratio": np.nan,
            "mean_cosine_similarity": np.nan,
        }

    mu = np.nanmean(X, axis=0)
    mu_norm = float(np.linalg.norm(mu))
    Xc = X - mu[None, :]
    total_var = float(np.nanmean(np.sum(Xc * Xc, axis=1)))

    if not np.isfinite(mu_norm) or mu_norm <= EPS:
        return {
            "parallel_variance": np.nan,
            "orthogonal_variance": np.nan,
            "orthogonal_parallel_ratio": np.nan,
            "mean_cosine_similarity": np.nan,
        }

    u = mu / mu_norm
    proj = Xc @ u
    parallel_var = float(np.nanmean(proj * proj))
    orth_var = max(total_var - parallel_var, 0.0)
    orth_ratio = safe_div(orth_var, parallel_var + EPS)

    xn = np.linalg.norm(X, axis=1)
    cos = (X @ mu) / (xn * mu_norm + EPS)
    cos = cos[np.isfinite(cos)]
    mean_cos = float(np.mean(cos)) if cos.size else np.nan

    return {
        "parallel_variance": parallel_var,
        "orthogonal_variance": orth_var,
        "orthogonal_parallel_ratio": orth_ratio,
        "mean_cosine_similarity": mean_cos,
    }


def tau_flatten_transform(X: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xp = np.clip(np.asarray(X, dtype=float), 0.0, None)
    R = np.sum(Xp, axis=1, keepdims=True)
    valid = (R[:, 0] > EPS)
    Q = np.zeros_like(Xp, dtype=float)
    Q[valid] = Xp[valid] / (R[valid] + EPS)

    power = 1.0 / max(float(tau), EPS)
    Qt = np.zeros_like(Q, dtype=float)
    Qt[valid] = np.power(Q[valid] + EPS, power)
    Qt_sum = np.sum(Qt[valid], axis=1, keepdims=True)
    Qt[valid] = Qt[valid] / (Qt_sum + EPS)

    Xtau = Qt * R
    return Xtau, Qt, R[:, 0]


def model_variant_flags(variant: str) -> tuple[bool, bool, bool]:
    name = str(variant).strip().lower()
    if name == "tau_only":
        return False, False, False
    if name == "tau_core":
        return True, False, False
    if name == "tau_spatial":
        return False, True, False
    if name == "tau_weak":
        return False, False, True
    if name == "tau_spatial_weak":
        return False, True, True
    if name == "tau_core_spatial_weak":
        return True, True, True
    raise ValueError(f"Unknown model variant: {variant}")


def _trial_structured_q(
    q: np.ndarray,
    pos_xy: np.ndarray,
    tau: float,
    use_core: bool,
    use_spatial: bool,
    core_top_frac: float,
    core_alpha: float,
    spatial_beta: float,
) -> np.ndarray:
    q0 = np.asarray(q, dtype=float).reshape(-1)
    q0 = np.clip(q0, 0.0, None)
    s0 = np.sum(q0)
    if s0 <= EPS:
        return np.full_like(q0, 1.0 / max(q0.size, 1), dtype=float)
    q0 = q0 / s0

    n = q0.size
    if use_core:
        k = max(1, int(np.ceil(float(core_top_frac) * n)))
        idx_core = np.argsort(q0)[-k:]
        core_mask = np.zeros(n, dtype=bool)
        core_mask[idx_core] = True
        tail_mask = ~core_mask

        q_core = np.zeros(n, dtype=float)
        q_core[core_mask] = q0[core_mask]
        q_core_sum = np.sum(q_core)
        q_core = q_core / (q_core_sum + EPS)
    else:
        core_mask = np.zeros(n, dtype=bool)
        tail_mask = np.ones(n, dtype=bool)
        q_core = np.zeros(n, dtype=float)

    src_mask = tail_mask.copy()
    if not np.any(src_mask):
        src_mask = np.ones(n, dtype=bool)

    src = q0[src_mask]
    src_flat = np.power(src + EPS, 1.0 / max(float(tau), EPS))
    if use_spatial:
        if use_core and np.sum(q_core) > EPS:
            center = q_core @ pos_xy
        else:
            center = q0 @ pos_xy
        dist = np.sqrt(np.sum((pos_xy[src_mask] - center[None, :]) ** 2, axis=1))
        dmax = float(np.max(dist))
        if dmax > EPS:
            dist = dist / dmax
        src_flat = src_flat * np.exp(float(spatial_beta) * dist)
    src_flat = src_flat / (np.sum(src_flat) + EPS)

    q_expand = np.zeros(n, dtype=float)
    q_expand[src_mask] = src_flat

    if use_core:
        alpha = float(np.clip(core_alpha, 0.0, 1.0))
        q_new = (1.0 - alpha) * q_core + alpha * q_expand
    else:
        q_new = q_expand
    q_new = np.clip(q_new, 0.0, None)
    return q_new / (np.sum(q_new) + EPS)


def structured_recruitment_transform(
    X: np.ndarray,
    pos_xy: np.ndarray,
    tau: float,
    variant: str,
    core_top_frac: float,
    core_alpha: float,
    spatial_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_core, use_spatial, _ = model_variant_flags(variant)
    Xp = np.clip(np.asarray(X, dtype=float), 0.0, None)
    R = np.sum(Xp, axis=1, keepdims=True)
    valid = R[:, 0] > EPS
    Q0 = np.zeros_like(Xp, dtype=float)
    Q0[valid] = Xp[valid] / (R[valid] + EPS)

    Qt = np.zeros_like(Q0, dtype=float)
    for i in np.where(valid)[0]:
        Qt[i] = _trial_structured_q(
            q=Q0[i],
            pos_xy=pos_xy,
            tau=tau,
            use_core=use_core,
            use_spatial=use_spatial,
            core_top_frac=core_top_frac,
            core_alpha=core_alpha,
            spatial_beta=spatial_beta,
        )
    X_new = Qt * R
    return X_new, Qt, R[:, 0]


def build_longrange_kernel_from_patch_meta(patch_meta: pd.DataFrame, power: float = 1.0) -> np.ndarray:
    if patch_meta is None or patch_meta.empty or patch_meta.shape[0] < 2:
        return np.zeros((0, 0), dtype=float)
    coords = patch_meta[["x_idx", "y_idx"]].to_numpy(dtype=float)
    d = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(D, 0.0)
    dmax = float(np.max(D))
    if dmax <= EPS:
        return np.zeros_like(D)
    W = np.power(D / dmax, max(float(power), 0.0))
    np.fill_diagonal(W, 0.0)
    row_sum = np.sum(W, axis=1, keepdims=True)
    W = np.where(row_sum > EPS, W / (row_sum + EPS), 0.0)
    return W


def apply_weak_longrange_coordination(F: np.ndarray, K_long: np.ndarray, lam: float) -> np.ndarray:
    X = np.asarray(F, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        return X
    if K_long.shape[0] != X.shape[1]:
        return X
    lam = float(max(lam, 0.0))
    if lam <= EPS:
        return X
    return X + lam * (X @ K_long)


def weak_edge_fisher_proxy(
    F: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    reg: float,
    weak_quantile: float,
) -> tuple[float, float]:
    X = np.asarray(F, dtype=float)
    if X.ndim != 2 or X.shape[1] < 3 or X.shape[0] < 8:
        return np.nan, np.nan
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + EPS
    Z = (X - mu) / sd
    iu = np.triu_indices(Z.shape[1], k=1)
    if iu[0].size < 3:
        return np.nan, np.nan
    E = Z[:, iu[0]] * Z[:, iu[1]]
    mean_abs = np.nanmean(np.abs(E), axis=0)
    if not np.any(np.isfinite(mean_abs)):
        return np.nan, np.nan
    q = float(np.clip(weak_quantile, 0.05, 0.8))
    thr = float(np.nanquantile(mean_abs, q))
    weak_idx = np.where(mean_abs <= thr)[0]
    if weak_idx.size < 3:
        weak_idx = np.argsort(mean_abs)[: min(10, mean_abs.size)]

    m = np.isin(labels, [1, 2, 3])
    if np.sum(m) < 8:
        return np.nan, np.nan
    y = (labels[m] != 3).astype(int)
    Ew = E[m][:, weak_idx]
    Ef = E[m]

    vals_w = []
    vals_f = []
    for _ in range(int(repeats)):
        vals_w.append(fisher_separability(Ew, y, rng=rng, reg=reg, balance=True))
        vals_f.append(fisher_separability(Ef, y, rng=rng, reg=reg, balance=True))
    vals_w = np.asarray(vals_w, dtype=float)
    vals_f = np.asarray(vals_f, dtype=float)
    vals_w = vals_w[np.isfinite(vals_w)]
    vals_f = vals_f[np.isfinite(vals_f)]
    out_w = float(np.mean(vals_w)) if vals_w.size else np.nan
    out_f = float(np.mean(vals_f)) if vals_f.size else np.nan
    return out_w, out_f


def assign_patches(pos_xy: np.ndarray, grid_size: float, min_patch_neurons: int):
    pos = np.asarray(pos_xy, dtype=float)
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

    rows = []
    uniq = np.unique(uid)
    for u in uniq.tolist():
        m = uid == int(u)
        count = int(np.sum(m))
        if count < int(min_patch_neurons):
            continue
        yi = int(u // nx)
        xi = int(u % nx)
        rows.append(
            {
                "patch_uid": int(u),
                "x_idx": int(xi),
                "y_idx": int(yi),
                "n_neurons": count,
            }
        )
    patch_meta = pd.DataFrame(rows).sort_values(["y_idx", "x_idx"]).reset_index(drop=True)
    if patch_meta.empty:
        return patch_meta, np.zeros((pos.shape[0], 0), dtype=float), np.full(pos.shape[0], -1, dtype=int)

    patch_meta.insert(0, "patch_local_id", np.arange(patch_meta.shape[0], dtype=int))
    uid_to_local = {int(r.patch_uid): int(r.patch_local_id) for _, r in patch_meta.iterrows()}
    patch_local = np.asarray([uid_to_local.get(int(u), -1) for u in uid], dtype=int)

    n = pos.shape[0]
    k = patch_meta.shape[0]
    P = np.zeros((n, k), dtype=float)
    for _, r in patch_meta.iterrows():
        j = int(r["patch_local_id"])
        m = patch_local == j
        P[m, j] = 1.0 / max(int(np.sum(m)), 1)
    return patch_meta, P, patch_local


def fisher_separability(F: np.ndarray, y: np.ndarray, rng: np.random.Generator, reg: float = 1e-3, balance: bool = True) -> float:
    X = np.asarray(F, dtype=float)
    y = np.asarray(y).reshape(-1)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[ok]
    y = y[ok]
    if X.shape[0] < 6 or X.shape[1] < 1:
        return np.nan

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size < 3 or idx1.size < 3:
        return np.nan

    if balance:
        n = int(min(idx0.size, idx1.size))
        idx0 = rng.choice(idx0, size=n, replace=False)
        idx1 = rng.choice(idx1, size=n, replace=False)

    X0 = X[idx0]
    X1 = X[idx1]
    d = X.shape[1]
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    delta = mu1 - mu0

    if d == 1:
        v0 = float(np.var(X0[:, 0], ddof=1))
        v1 = float(np.var(X1[:, 0], ddof=1))
        v = 0.5 * (v0 + v1)
        v = v + reg * (abs(v) + 1.0)
        return float((delta[0] ** 2) / max(v, EPS))

    c0 = np.cov(X0, rowvar=False)
    c1 = np.cov(X1, rowvar=False)
    c = 0.5 * (c0 + c1)
    tr = float(np.trace(c)) if np.all(np.isfinite(c)) else np.nan
    scale = tr / d if (np.isfinite(tr) and abs(tr) > EPS) else 1.0
    c_reg = c + reg * scale * np.eye(d, dtype=float)
    try:
        w = np.linalg.solve(c_reg, delta)
    except Exception:
        w = np.linalg.pinv(c_reg) @ delta
    j = float(delta @ w)
    return j if np.isfinite(j) else np.nan


def latent_fisher_metrics(F: np.ndarray, labels: np.ndarray, rng: np.random.Generator, repeats: int, reg: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    rows = {"fisher_coherent_vs_random": np.nan, "fisher_divergent_vs_convergent": np.nan}

    vals_cr = []
    m_cr = np.isin(labels, [1, 2, 3])
    if np.sum(m_cr) >= 8:
        y_cr = (labels[m_cr] != 3).astype(int)
        X_cr = F[m_cr]
        for _ in range(int(repeats)):
            vals_cr.append(fisher_separability(X_cr, y_cr, rng=rng, reg=reg, balance=True))
    vals_cr = np.asarray(vals_cr, dtype=float)
    vals_cr = vals_cr[np.isfinite(vals_cr)]
    rows["fisher_coherent_vs_random"] = float(np.mean(vals_cr)) if vals_cr.size else np.nan

    vals_dc = []
    m_dc = np.isin(labels, [1, 2])
    if np.sum(m_dc) >= 8:
        y_dc = (labels[m_dc] == 1).astype(int)
        X_dc = F[m_dc]
        for _ in range(int(repeats)):
            vals_dc.append(fisher_separability(X_dc, y_dc, rng=rng, reg=reg, balance=True))
    vals_dc = np.asarray(vals_dc, dtype=float)
    vals_dc = vals_dc[np.isfinite(vals_dc)]
    rows["fisher_divergent_vs_convergent"] = float(np.mean(vals_dc)) if vals_dc.size else np.nan
    return rows


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


def slope_rows(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    need_cols = ["mouse_id", "tau", value_col] + list(group_cols)
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        return pd.DataFrame()
    rows = []
    for key, sub in df[need_cols].dropna().groupby(["mouse_id"] + group_cols, observed=False):
        x = sub["tau"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        if x.size < 2 or np.unique(x).size < 2:
            slope = np.nan
        else:
            slope = float(np.polyfit(x, y, deg=1)[0])
        if isinstance(key, tuple):
            mouse = key[0]
            others = key[1:]
        else:
            mouse = key
            others = ()
        row = {"mouse_id": mouse, "metric": value_col, "slope": slope}
        for c, v in zip(group_cols, others):
            row[c] = v
        rows.append(row)
    return pd.DataFrame(rows)


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


def mouse_plot_condition_metrics(df_cond: pd.DataFrame, out_path: str):
    metrics = [
        ("flatness_entropy_norm", "Flatness entropy (norm)"),
        ("integration_radius", "Integration radius"),
        ("integration_pairwise_span", "Integration pairwise span"),
        ("effective_participation_prop", "Effective participation prop"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df_cond[["Condition", "tau", metric]].dropna().copy()
        for cond in CONDITIONS:
            ss = sub[sub["Condition"] == cond].sort_values("tau")
            if ss.empty:
                continue
            x = ss["tau"].to_numpy(dtype=float)
            y = ss[metric].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=COND_COLORS[cond], label=cond)
        ax.set_xlabel("tau")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def mouse_plot_latent_metrics(df_latent: pd.DataFrame, out_path: str):
    metrics = [
        ("fisher_coherent_vs_random", "Fisher (coherent vs random)"),
        ("fisher_divergent_vs_convergent", "Fisher (divergent vs convergent)"),
        ("robustness_loss_coherent_vs_random", "Drop loss (coherent vs random)"),
        ("robustness_loss_divergent_vs_convergent", "Drop loss (divergent vs convergent)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df_latent[["tau", metric]].dropna().sort_values("tau")
        if not sub.empty:
            ax.plot(sub["tau"], sub[metric], marker="o", lw=2, color="#4F6B8A")
        ax.set_xlabel("tau")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
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
    n_neurons = int(pos_xy.shape[0])
    D = pairwise_distance_matrix(pos_xy)
    sqnorm = np.sum(pos_xy * pos_xy, axis=1)

    patch_meta, P_patch, patch_local = assign_patches(
        pos_xy=pos_xy,
        grid_size=float(args.grid_size),
        min_patch_neurons=int(args.min_patch_neurons),
    )
    n_patches = int(P_patch.shape[1])

    cond_rows = []
    latent_rows = []
    dropout_rows = []
    shuffle_rows = []
    rng = np.random.default_rng(seed_i)
    K_long = build_longrange_kernel_from_patch_meta(patch_meta, power=float(args.longrange_power))

    for variant in args.model_variants:
        use_core, use_spatial, use_weak = model_variant_flags(variant)
        for tau in args.tau_values:
            tau_val = float(tau)
            if use_core or use_spatial:
                X_tau, Q_tau, R_tau = structured_recruitment_transform(
                    X=X_delta,
                    pos_xy=pos_xy,
                    tau=tau_val,
                    variant=variant,
                    core_top_frac=float(args.core_top_frac),
                    core_alpha=float(args.core_alpha),
                    spatial_beta=float(args.spatial_beta),
                )
            else:
                X_tau, Q_tau, R_tau = tau_flatten_transform(X_delta, tau=tau_val)

            F_tau = X_tau @ P_patch if n_patches > 0 else np.zeros((X_tau.shape[0], 0), dtype=float)
            if use_weak:
                F_tau = apply_weak_longrange_coordination(F_tau, K_long, lam=float(args.weak_lambda))

            latent = latent_fisher_metrics(
                F=F_tau,
                labels=labels,
                rng=rng,
                repeats=int(args.latent_repeats),
                reg=float(args.fisher_reg),
            )
            weak_edge_fisher, full_edge_fisher = weak_edge_fisher_proxy(
                F=F_tau,
                labels=labels,
                rng=rng,
                repeats=max(2, int(args.latent_repeats // 2)),
                reg=float(args.fisher_reg),
                weak_quantile=float(args.weak_edge_quantile),
            )

            drop_loss_cr = np.nan
            drop_loss_dc = np.nan
            if n_patches >= 3:
                drop_cr = []
                drop_dc = []
                for j in range(n_patches):
                    n_drop = int(patch_meta.iloc[j]["n_neurons"])
                    if int(n_neurons - n_drop) < int(args.min_neurons_after_drop):
                        continue
                    keep_cols = np.ones(n_patches, dtype=bool)
                    keep_cols[j] = False
                    F_drop = F_tau[:, keep_cols]
                    drop_lat = latent_fisher_metrics(
                        F=F_drop,
                        labels=labels,
                        rng=rng,
                        repeats=max(2, int(args.latent_repeats // 2)),
                        reg=float(args.fisher_reg),
                    )
                    j_cr = drop_lat["fisher_coherent_vs_random"]
                    j_dc = drop_lat["fisher_divergent_vs_convergent"]
                    drop_cr.append(j_cr)
                    drop_dc.append(j_dc)
                    pm = patch_meta.iloc[j]
                    dropout_rows.append(
                        {
                            "mouse_id": mouse_id,
                            "model_variant": variant,
                            "tau": tau_val,
                            "patch_local_id": int(pm["patch_local_id"]),
                            "x_idx": int(pm["x_idx"]),
                            "y_idx": int(pm["y_idx"]),
                            "n_neurons_patch": int(pm["n_neurons"]),
                            "fisher_drop_coherent_vs_random": j_cr,
                            "fisher_drop_divergent_vs_convergent": j_dc,
                        }
                    )
                drop_cr = np.asarray(drop_cr, dtype=float)
                drop_dc = np.asarray(drop_dc, dtype=float)
                m_cr = np.isfinite(drop_cr)
                m_dc = np.isfinite(drop_dc)
                if m_cr.any() and np.isfinite(latent["fisher_coherent_vs_random"]):
                    drop_loss_cr = float(latent["fisher_coherent_vs_random"] - np.mean(drop_cr[m_cr]))
                if m_dc.any() and np.isfinite(latent["fisher_divergent_vs_convergent"]):
                    drop_loss_dc = float(latent["fisher_divergent_vs_convergent"] - np.mean(drop_dc[m_dc]))

            latent_rows.append(
                {
                    "mouse_id": mouse_id,
                    "model_variant": variant,
                    "tau": tau_val,
                    "n_patches": int(n_patches),
                    "fisher_coherent_vs_random": latent["fisher_coherent_vs_random"],
                    "fisher_divergent_vs_convergent": latent["fisher_divergent_vs_convergent"],
                    "weak_edge_fisher_coherent_vs_random": weak_edge_fisher,
                    "full_edge_fisher_coherent_vs_random": full_edge_fisher,
                    "robustness_loss_coherent_vs_random": drop_loss_cr,
                    "robustness_loss_divergent_vs_convergent": drop_loss_dc,
                }
            )

            for class_id in classes:
                cond = COND_MAP.get(int(class_id), str(class_id))
                m = labels == int(class_id)
                if np.sum(m) < 2:
                    continue
                q_cond = Q_tau[m]
                r_cond = R_tau[m]
                x_cond = X_tau[m]
                valid = r_cond > EPS
                if np.sum(valid) < 2:
                    continue
                q_mean = np.mean(q_cond[valid], axis=0)
                q_mean = np.clip(q_mean, 0.0, None)
                q_sum = np.sum(q_mean)
                if q_sum <= EPS:
                    continue
                q_mean = q_mean / q_sum

                h_norm = entropy_norm(q_mean)
                eff_prop = safe_div(float(np.exp(stats.entropy(q_mean))), float(q_mean.size))
                gini = gini_index(q_mean)
                rad = weighted_radius(q_mean, pos_xy)
                span = weighted_pair_span(q_mean, D)
                mean_total = float(np.mean(r_cond[valid]))
                geo = orthogonal_expansion_metrics(x_cond[valid])

                cond_rows.append(
                    {
                        "mouse_id": mouse_id,
                        "model_variant": variant,
                        "Class_ID": int(class_id),
                        "Condition": cond,
                        "tau": tau_val,
                        "n_trials_valid": int(np.sum(valid)),
                        "n_neurons": int(n_neurons),
                        "mean_total_activity": mean_total,
                        "flatness_entropy_norm": h_norm,
                        "flatness_gini": gini,
                        "effective_participation_prop": eff_prop,
                        "integration_radius": rad,
                        "integration_pairwise_span": span,
                        "parallel_variance": geo["parallel_variance"],
                        "orthogonal_variance": geo["orthogonal_variance"],
                        "orthogonal_parallel_ratio": geo["orthogonal_parallel_ratio"],
                        "mean_cosine_similarity": geo["mean_cosine_similarity"],
                        "core_top_frac": float(args.core_top_frac) if use_core else 0.0,
                        "core_alpha": float(args.core_alpha) if use_core else 1.0,
                        "spatial_beta": float(args.spatial_beta) if use_spatial else 0.0,
                        "weak_lambda": float(args.weak_lambda) if use_weak else 0.0,
                    }
                )

                if int(args.shuffle_repeats) > 0:
                    rad_sh = []
                    span_sh = []
                    for _ in range(int(args.shuffle_repeats)):
                        perm = rng.permutation(n_neurons)
                        pos_perm = pos_xy[perm]
                        sq_perm = sqnorm[perm]
                        q_perm = q_mean[perm]
                        cen_p = q_mean @ pos_perm
                        rad_p = float(q_mean @ sq_perm - np.dot(cen_p, cen_p))
                        rad_sh.append(max(rad_p, 0.0))
                        span_sh.append(float(q_perm @ D @ q_perm))
                    rad_sh = np.asarray(rad_sh, dtype=float)
                    span_sh = np.asarray(span_sh, dtype=float)
                    shuffle_rows.append(
                        {
                            "mouse_id": mouse_id,
                            "model_variant": variant,
                            "Class_ID": int(class_id),
                            "Condition": cond,
                            "tau": tau_val,
                            "integration_radius_true": rad,
                            "integration_radius_shuffle_mean": float(np.mean(rad_sh)),
                            "integration_radius_gap_true_minus_shuffle": float(rad - np.mean(rad_sh)),
                            "integration_pairwise_span_true": span,
                            "integration_pairwise_span_shuffle_mean": float(np.mean(span_sh)),
                            "integration_pairwise_span_gap_true_minus_shuffle": float(span - np.mean(span_sh)),
                        }
                    )

    df_cond = normalize_condition_column(pd.DataFrame(cond_rows))
    df_latent = pd.DataFrame(latent_rows)
    df_dropout = pd.DataFrame(dropout_rows)
    df_shuffle = normalize_condition_column(pd.DataFrame(shuffle_rows))

    if not df_cond.empty:
        p = os.path.join(data_out, "modeling_tau_condition_metrics.csv")
        df_cond.to_csv(p, index=False)
        print(f"[*] Saved: {p}")
        for variant in sorted(df_cond["model_variant"].dropna().astype(str).unique()):
            sub = df_cond[df_cond["model_variant"].astype(str) == variant].copy()
            if sub.empty:
                continue
            mouse_plot_condition_metrics(sub, os.path.join(fig_out, f"modeling_tau_condition_metrics_{variant}.png"))
    if not df_latent.empty:
        p = os.path.join(data_out, "modeling_tau_latent_metrics.csv")
        df_latent.to_csv(p, index=False)
        print(f"[*] Saved: {p}")
        for variant in sorted(df_latent["model_variant"].dropna().astype(str).unique()):
            sub = df_latent[df_latent["model_variant"].astype(str) == variant].copy()
            if sub.empty:
                continue
            mouse_plot_latent_metrics(sub, os.path.join(fig_out, f"modeling_tau_latent_metrics_{variant}.png"))
    if not df_dropout.empty:
        p = os.path.join(data_out, "modeling_tau_dropout_patch_detail.csv")
        df_dropout.to_csv(p, index=False)
        print(f"[*] Saved: {p}")
    if not df_shuffle.empty:
        p = os.path.join(data_out, "modeling_tau_spatial_shuffle_control.csv")
        df_shuffle.to_csv(p, index=False)
        print(f"[*] Saved: {p}")

    return {
        "condition_df": df_cond,
        "latent_df": df_latent,
        "dropout_df": df_dropout,
        "shuffle_df": df_shuffle,
    }


def group_plot_condition_curves(df_cond: pd.DataFrame, out_path: str):
    metrics = [
        ("flatness_entropy_norm", "Flatness entropy (norm)"),
        ("integration_radius", "Integration radius"),
        ("integration_pairwise_span", "Integration pairwise span"),
        ("effective_participation_prop", "Effective participation prop"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.8), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = (
            df_cond[["mouse_id", "Condition", "tau", metric]]
            .dropna()
            .groupby(["mouse_id", "Condition", "tau"], observed=False)[metric]
            .mean()
            .reset_index()
        )
        for cond in CONDITIONS:
            ss = (
                sub[sub["Condition"] == cond]
                .groupby("tau", observed=False)[metric]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("tau")
            )
            if ss.empty:
                continue
            x = ss["tau"].to_numpy(dtype=float)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=COND_COLORS[cond], label=cond)
            ax.fill_between(x, y - se, y + se, color=COND_COLORS[cond], alpha=0.16, linewidth=0)
        ax.set_xlabel("tau")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def group_plot_latent_curves(df_latent: pd.DataFrame, out_path: str):
    metrics = [
        ("fisher_coherent_vs_random", "Fisher (coherent vs random)"),
        ("fisher_divergent_vs_convergent", "Fisher (divergent vs convergent)"),
        ("robustness_loss_coherent_vs_random", "Drop loss (coherent vs random)"),
        ("robustness_loss_divergent_vs_convergent", "Drop loss (divergent vs convergent)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.8), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = (
            df_latent[["mouse_id", "tau", metric]]
            .dropna()
            .groupby(["mouse_id", "tau"], observed=False)[metric]
            .mean()
            .reset_index()
        )
        ss = sub.groupby("tau", observed=False)[metric].agg(["mean", "sem"]).reset_index().sort_values("tau")
        if not ss.empty:
            x = ss["tau"].to_numpy(dtype=float)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color="#4F6B8A")
            ax.fill_between(x, y - se, y + se, color="#4F6B8A", alpha=0.16, linewidth=0)
        ax.set_xlabel("tau")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    save_variants(fig, out_path)


def run_group(df_cond: pd.DataFrame, df_latent: pd.DataFrame, df_dropout: pd.DataFrame, df_shuffle: pd.DataFrame, args):
    default_variant = "tau_only"
    if hasattr(args, "model_variants") and args.model_variants:
        default_variant = str(args.model_variants[0])

    df_cond = normalize_model_variant_column(normalize_condition_column(df_cond), default_variant=default_variant)
    df_latent = normalize_model_variant_column(df_latent, default_variant=default_variant)
    df_dropout = normalize_model_variant_column(df_dropout, default_variant=default_variant)
    df_shuffle = normalize_model_variant_column(
        normalize_condition_column(df_shuffle), default_variant=default_variant
    )
    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)

    if not df_cond.empty:
        df_cond.to_csv(os.path.join(group_dir, "group_modeling_tau_condition_metrics.csv"), index=False)
    if not df_latent.empty:
        df_latent.to_csv(os.path.join(group_dir, "group_modeling_tau_latent_metrics.csv"), index=False)
    if not df_dropout.empty:
        df_dropout.to_csv(os.path.join(group_dir, "group_modeling_tau_dropout_patch_detail.csv"), index=False)
    if not df_shuffle.empty:
        df_shuffle.to_csv(os.path.join(group_dir, "group_modeling_tau_spatial_shuffle_control.csv"), index=False)

    cond_metrics = [
        "flatness_entropy_norm",
        "integration_radius",
        "integration_pairwise_span",
        "effective_participation_prop",
        "parallel_variance",
        "orthogonal_variance",
        "orthogonal_parallel_ratio",
        "mean_cosine_similarity",
    ]
    latent_metrics = [
        "fisher_coherent_vs_random",
        "fisher_divergent_vs_convergent",
        "weak_edge_fisher_coherent_vs_random",
        "full_edge_fisher_coherent_vs_random",
        "robustness_loss_coherent_vs_random",
        "robustness_loss_divergent_vs_convergent",
    ]

    slope_rows_all = []
    for metric in cond_metrics:
        s = slope_rows(df_cond, value_col=metric, group_cols=["model_variant", "Condition"])
        if not s.empty:
            slope_rows_all.append(s)
    for metric in latent_metrics:
        s = slope_rows(df_latent, value_col=metric, group_cols=["model_variant"])
        if not s.empty:
            slope_rows_all.append(s)
    df_slopes = pd.concat(slope_rows_all, ignore_index=True) if slope_rows_all else pd.DataFrame()
    if not df_slopes.empty:
        df_slopes.to_csv(os.path.join(group_dir, "group_modeling_tau_slopes_by_mouse.csv"), index=False)

    stat_rows = []
    if not df_slopes.empty:
        slope_valid = df_slopes.dropna(subset=["slope"]).copy()
        if "Condition" in slope_valid.columns:
            cond_sub = slope_valid[slope_valid["Condition"].notna()].copy()
        else:
            cond_sub = pd.DataFrame()

        if not cond_sub.empty:
            for (metric, variant, cond), sub in cond_sub.groupby(
                ["metric", "model_variant", "Condition"], observed=False
            ):
                p, n = wilcoxon_zero(sub["slope"].to_numpy(dtype=float))
                stat_rows.append(
                    {
                        "metric": metric,
                        "model_variant": variant,
                        "Condition": cond,
                        "n_mice": n,
                        "mean_slope": float(np.nanmean(sub["slope"])),
                        "test": "Wilcoxon(slope vs 0)",
                        "p_value": p,
                    }
                )
        latent_sub = slope_valid.copy()
        if "Condition" in latent_sub.columns:
            latent_sub = latent_sub[latent_sub["Condition"].isna()].copy()
        if not latent_sub.empty:
            for (metric, variant), sub in latent_sub.groupby(["metric", "model_variant"], observed=False):
                p, n = wilcoxon_zero(sub["slope"].to_numpy(dtype=float))
                stat_rows.append(
                    {
                        "metric": metric,
                        "model_variant": variant,
                        "Condition": "all",
                        "n_mice": n,
                        "mean_slope": float(np.nanmean(sub["slope"])),
                        "test": "Wilcoxon(slope vs 0)",
                        "p_value": p,
                    }
                )
    df_stats = pd.DataFrame(stat_rows)
    if not df_stats.empty:
        df_stats.to_csv(os.path.join(group_dir, "group_modeling_tau_slope_stats.csv"), index=False)

    if not df_cond.empty:
        for variant in sorted(df_cond["model_variant"].dropna().astype(str).unique()):
            sub = df_cond[df_cond["model_variant"].astype(str) == variant].copy()
            if sub.empty:
                continue
            group_plot_condition_curves(
                sub, os.path.join(group_dir, f"group_modeling_tau_condition_curves_{variant}.png")
            )
    if not df_latent.empty:
        for variant in sorted(df_latent["model_variant"].dropna().astype(str).unique()):
            sub = df_latent[df_latent["model_variant"].astype(str) == variant].copy()
            if sub.empty:
                continue
            group_plot_latent_curves(sub, os.path.join(group_dir, f"group_modeling_tau_latent_curves_{variant}.png"))

    md_path = os.path.join(group_dir, "Group_FlatRecruitment_Modeling_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group Flat-Recruitment Modeling Report (v2)\n\n")
        variants = []
        for dfi in [df_cond, df_latent]:
            if dfi is not None and not dfi.empty and "model_variant" in dfi.columns:
                variants.extend(dfi["model_variant"].dropna().astype(str).unique().tolist())
        variants = sorted(set(variants))
        f.write("## Model variants\n\n")
        f.write((", ".join(variants) if variants else "_none_") + "\n\n")
        f.write("## Tau slope tests (Wilcoxon against 0)\n\n")
        f.write(_to_md(df_stats) + "\n\n")
    print(f"[*] Group report saved: {md_path}")


def load_mouse_outputs_from_disk(mouse_id: str, results_dir: str) -> dict:
    data_out = os.path.join(results_dir, mouse_id, "data")
    cond_path = os.path.join(data_out, "modeling_tau_condition_metrics.csv")
    latent_path = os.path.join(data_out, "modeling_tau_latent_metrics.csv")
    dropout_path = os.path.join(data_out, "modeling_tau_dropout_patch_detail.csv")
    shuffle_path = os.path.join(data_out, "modeling_tau_spatial_shuffle_control.csv")

    df_cond = pd.read_csv(cond_path) if os.path.isfile(cond_path) else pd.DataFrame()
    df_latent = pd.read_csv(latent_path) if os.path.isfile(latent_path) else pd.DataFrame()
    df_dropout = pd.read_csv(dropout_path) if os.path.isfile(dropout_path) else pd.DataFrame()
    df_shuffle = pd.read_csv(shuffle_path) if os.path.isfile(shuffle_path) else pd.DataFrame()
    df_cond = normalize_model_variant_column(normalize_condition_column(df_cond), default_variant="tau_only")
    df_latent = normalize_model_variant_column(df_latent, default_variant="tau_only")
    df_dropout = normalize_model_variant_column(df_dropout, default_variant="tau_only")
    df_shuffle = normalize_model_variant_column(normalize_condition_column(df_shuffle), default_variant="tau_only")
    return {
        "condition_df": df_cond,
        "latent_df": df_latent,
        "dropout_df": df_dropout,
        "shuffle_df": df_shuffle,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-1 modeling: flatten recruitment while preserving total activity, and evaluate integration-related readouts."
    )
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)

    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--baseline-start", type=int, default=0)
    p.add_argument("--baseline-end", type=int, default=10)

    p.add_argument("--tau-values", type=str, default="0.6,0.8,1.0,1.2,1.5,2.0")
    p.add_argument("--grid-size", type=float, default=160.0)
    p.add_argument("--min-patch-neurons", type=int, default=20)
    p.add_argument("--min-neurons-after-drop", type=int, default=10)
    p.add_argument("--latent-repeats", type=int, default=6)
    p.add_argument("--fisher-reg", type=float, default=1e-3)
    p.add_argument("--shuffle-repeats", type=int, default=20)
    p.add_argument(
        "--model-variants",
        type=str,
        default="tau_only,tau_core,tau_spatial,tau_weak,tau_spatial_weak,tau_core_spatial_weak",
    )
    p.add_argument("--core-top-frac", type=float, default=0.15)
    p.add_argument("--core-alpha", type=float, default=0.65)
    p.add_argument("--spatial-beta", type=float, default=1.5)
    p.add_argument("--weak-lambda", type=float, default=0.15)
    p.add_argument("--longrange-power", type=float, default=1.0)
    p.add_argument("--weak-edge-quantile", type=float, default=0.2)

    p.add_argument("--group-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260410)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    args.tau_values = sorted(parse_float_list(args.tau_values))
    raw_variants = parse_str_list(args.model_variants)
    if not raw_variants:
        raw_variants = ["tau_only"]
    uniq_variants = []
    for v in raw_variants:
        model_variant_flags(v)
        if v not in uniq_variants:
            uniq_variants.append(v)
    args.model_variants = uniq_variants

    if args.group_only:
        print("[*] Group-only mode: loading per-mouse outputs from disk.")
        all_cond = []
        all_lat = []
        all_drop = []
        all_shuf = []
        for mouse in args.mice:
            out = load_mouse_outputs_from_disk(mouse, args.results_dir)
            if out["condition_df"] is not None and not out["condition_df"].empty:
                all_cond.append(out["condition_df"])
            else:
                print(f"[!] Missing condition CSV for {mouse}: results/{mouse}/data/modeling_tau_condition_metrics.csv")
            if out["latent_df"] is not None and not out["latent_df"].empty:
                all_lat.append(out["latent_df"])
            else:
                print(f"[!] Missing latent CSV for {mouse}: results/{mouse}/data/modeling_tau_latent_metrics.csv")
            if out["dropout_df"] is not None and not out["dropout_df"].empty:
                all_drop.append(out["dropout_df"])
            if out["shuffle_df"] is not None and not out["shuffle_df"].empty:
                all_shuf.append(out["shuffle_df"])

        df_cond = pd.concat(all_cond, ignore_index=True) if all_cond else pd.DataFrame()
        df_lat = pd.concat(all_lat, ignore_index=True) if all_lat else pd.DataFrame()
        df_drop = pd.concat(all_drop, ignore_index=True) if all_drop else pd.DataFrame()
        df_shuf = pd.concat(all_shuf, ignore_index=True) if all_shuf else pd.DataFrame()
        if df_cond.empty and df_lat.empty:
            print("[!] No existing outputs found. Stop.")
            return
        run_group(df_cond, df_lat, df_drop, df_shuf, args)
        print("====== Flat-recruitment modeling (group-only) completed ======")
        return

    all_cond = []
    all_lat = []
    all_drop = []
    all_shuf = []
    base_seed = int(args.seed)
    for i, mouse in enumerate(args.mice):
        try:
            out = run_mouse(mouse, args, seed_i=int(base_seed + i * 101))
            if out["condition_df"] is not None and not out["condition_df"].empty:
                all_cond.append(out["condition_df"])
            if out["latent_df"] is not None and not out["latent_df"].empty:
                all_lat.append(out["latent_df"])
            if out["dropout_df"] is not None and not out["dropout_df"].empty:
                all_drop.append(out["dropout_df"])
            if out["shuffle_df"] is not None and not out["shuffle_df"].empty:
                all_shuf.append(out["shuffle_df"])
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")

    df_cond = pd.concat(all_cond, ignore_index=True) if all_cond else pd.DataFrame()
    df_lat = pd.concat(all_lat, ignore_index=True) if all_lat else pd.DataFrame()
    df_drop = pd.concat(all_drop, ignore_index=True) if all_drop else pd.DataFrame()
    df_shuf = pd.concat(all_shuf, ignore_index=True) if all_shuf else pd.DataFrame()
    if df_cond.empty and df_lat.empty:
        print("[!] No valid outputs generated. Stop.")
        return

    run_group(df_cond, df_lat, df_drop, df_shuf, args)
    print("====== Flat-recruitment modeling completed ======")


if __name__ == "__main__":
    main()
