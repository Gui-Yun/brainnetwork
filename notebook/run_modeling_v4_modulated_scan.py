import argparse
import os
from dataclasses import dataclass

# Use a headless backend by default and keep matplotlib cache off slow network homes.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns = None
if os.environ.get("BRAINNETWORK_USE_SEABORN", "0") == "1":
    try:
        import seaborn as sns
    except BaseException as exc:
        sns = None
        print(f"[!] seaborn unavailable ({type(exc).__name__}: {exc}); fallback to matplotlib-only styling.")


DEFAULT_RESULTS_DIR = "./results"
GROUP_DIR_NAME = "group_summary"
EPS = 1e-12


if sns is not None:
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
class ModelParams:
    n_neurons: int
    n_core: int
    n_peri_modules: int
    core_prob: float
    peri_prob: float
    w_core: float
    w_peri_mean: float
    w_peri_std: float
    w_bal: float
    target_radius: float
    alpha: float
    lambda_global: float
    lambda_core: float
    lambda_peri: float
    gamma_norm: float
    kappa_coh_relief: float
    relief_cross_only: bool
    n_cond_modes: int
    g_c_global: float
    g_c_module: float
    beta_core: float
    beta_peri: float
    module_selectivity: float
    sigma_ind: float
    g_n: float
    g_coh_extra_noise: float
    module_noise_rho: float
    stim_dim: int
    stim_scale: float
    stim_rho: float
    cond_rho: float
    noise_rho: float
    n_trials: int
    t_steps: int
    response_start: int
    response_end: int
    weak_pos_max: float
    strong_quantile: float
    active_frac_thresh: float


@dataclass
class SearchSpace:
    core_prob: tuple[float, float]
    peri_prob: tuple[float, float]
    w_core: tuple[float, float]
    w_peri_mean: tuple[float, float]
    w_peri_std: tuple[float, float]
    w_bal: tuple[float, float]
    target_radius: tuple[float, float]
    alpha: tuple[float, float]
    lambda_global: tuple[float, float]
    lambda_core: tuple[float, float]
    lambda_peri: tuple[float, float]
    kappa_coh_relief: tuple[float, float]
    g_c_global: tuple[float, float]
    g_c_module: tuple[float, float]
    beta_core: tuple[float, float]
    beta_peri: tuple[float, float]
    module_selectivity: tuple[float, float]
    sigma_ind: tuple[float, float]
    g_n: tuple[float, float]
    g_coh_extra_noise: tuple[float, float]
    module_noise_rho: tuple[float, float]
    stim_scale: tuple[float, float]
    stim_rho: tuple[float, float]
    cond_rho: tuple[float, float]
    noise_rho: tuple[float, float]
    active_frac_thresh: tuple[float, float]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _checked_range(name: str, lo: float, hi: float) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if (not np.isfinite(lo_f)) or (not np.isfinite(hi_f)):
        raise ValueError(f"Range for {name} must be finite, got ({lo}, {hi}).")
    if lo_f > hi_f:
        raise ValueError(f"Range for {name} invalid: min({lo_f}) > max({hi_f}).")
    return lo_f, hi_f


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi - lo <= EPS:
        return lo
    return float(rng.uniform(lo, hi))


def style_axis(ax, grid=False):
    if sns is not None:
        sns.despine(ax=ax, trim=False)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
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


def participation_ratio_vector(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    num = float(np.sum(arr**2) ** 2)
    den = float(np.sum(arr**4))
    if den <= EPS:
        return 0.0
    return float(num / den)


def rr_selection_class_for_model(
    trials: np.ndarray,
    labels: np.ndarray,
    t_stimulus: int,
    l_stimulus: int,
    reliability_threshold: float = 0.75,
    snr_threshold: float = 0.8,
    effect_size_threshold: float = 0.5,
    response_ratio_threshold: float = 0.8,
) -> dict[int, set[int]]:
    arr = np.asarray(trials, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if arr.ndim != 3 or arr.shape[0] != y.size:
        return {}
    n_trials, n_neurons, n_time = arr.shape
    if n_trials < 2 or n_neurons < 2 or n_time < 3:
        return {}

    t0 = int(np.clip(int(t_stimulus), 0, n_time - 1))
    t1 = int(np.clip(int(t_stimulus) + int(l_stimulus), t0 + 1, n_time))
    baseline_pre = np.arange(0, t0)
    baseline_post = np.arange(t1, n_time)
    stimulus_window = np.arange(t0, t1)
    if stimulus_window.size == 0:
        return {}

    rr_neurons: dict[int, set[int]] = {}
    for cls in sorted(np.unique(y).astype(int).tolist()):
        class_trials = arr[y == cls]
        if class_trials.shape[0] < 2:
            continue

        base_means = []
        base_stds = []
        if baseline_pre.size > 0:
            base_means.append(np.nanmean(class_trials[:, :, baseline_pre], axis=2))
            base_stds.append(np.nanstd(class_trials[:, :, baseline_pre], axis=2))
        if baseline_post.size > 0:
            base_means.append(np.nanmean(class_trials[:, :, baseline_post], axis=2))
            base_stds.append(np.nanstd(class_trials[:, :, baseline_post], axis=2))
        if not base_means:
            continue
        if len(base_means) == 1:
            baseline_mean = base_means[0]
            baseline_std = base_stds[0]
        else:
            baseline_mean = 0.5 * (base_means[0] + base_means[1])
            baseline_std = 0.5 * (base_stds[0] + base_stds[1])

        stimulus_mean = np.nanmean(class_trials[:, :, stimulus_window], axis=2)
        stimulus_std = np.nanstd(class_trials[:, :, stimulus_window], axis=2)

        pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2.0)
        effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)

        response_ratio = np.nanmean(effect_size > float(effect_size_threshold), axis=0)
        is_enhanced_mean = (
            np.nanmean(stimulus_mean > baseline_mean, axis=0) > float(response_ratio_threshold)
        )
        is_responsive = (response_ratio > float(response_ratio_threshold)) & is_enhanced_mean
        class_enhanced_idx = np.where(is_responsive)[0]

        signal_strength = np.abs(stimulus_mean - baseline_mean)
        noise_level = baseline_std + 1e-8
        snr = signal_strength / noise_level
        reliability_ratio = np.nanmean(snr > float(snr_threshold), axis=0)
        class_reliable_idx = np.where(reliability_ratio >= float(reliability_threshold))[0]

        class_rr_idx = np.intersect1d(class_enhanced_idx, class_reliable_idx)
        rr_neurons[int(cls)] = set(map(int, class_rr_idx.tolist()))
    return rr_neurons


def participants_ratio_from_two_conditions(
    trials_random: np.ndarray,
    trials_coherent: np.ndarray,
    response_start: int,
    response_end: int,
) -> dict[str, float]:
    def _safe_window(start: int, end: int, n_time: int) -> slice:
        s = int(np.clip(int(start), 0, max(0, n_time - 1)))
        e = int(np.clip(int(end), s + 1, n_time))
        return slice(s, e)

    def _avg_response(
        c_trials: np.ndarray,
        neuron_idx: list[int],
        rw: slice,
    ) -> float:
        if c_trials.size == 0 or len(neuron_idx) == 0:
            return np.nan
        v = float(np.nanmean(c_trials[:, neuron_idx, rw]))
        return v if np.isfinite(v) else np.nan

    def _fallback_rr_sets_from_response(
        all_trials: np.ndarray,
        labels: np.ndarray,
        t0: int,
        t1: int,
    ) -> dict[int, set[int]]:
        n_trials, n_neurons, n_time = all_trials.shape
        rw = _safe_window(t0, t1, n_time)
        pre = np.arange(0, rw.start)
        post = np.arange(rw.stop, n_time)
        out: dict[int, set[int]] = {}
        topk = max(3, int(np.ceil(0.05 * n_neurons)))
        for cls in sorted(np.unique(labels).astype(int).tolist()):
            c_trials = all_trials[labels == cls]
            if c_trials.size == 0:
                out[int(cls)] = set()
                continue
            resp = np.nanmean(c_trials[:, :, rw], axis=(0, 2))
            base_parts = []
            if pre.size > 0:
                base_parts.append(np.nanmean(c_trials[:, :, pre], axis=(0, 2)))
            if post.size > 0:
                base_parts.append(np.nanmean(c_trials[:, :, post], axis=(0, 2)))
            if base_parts:
                baseline = np.nanmean(np.stack(base_parts, axis=0), axis=0)
            else:
                baseline = np.zeros(n_neurons, dtype=float)
            gain = np.asarray(resp - baseline, dtype=float)
            gain = np.nan_to_num(gain, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            pos = np.where(gain > 0)[0]
            if pos.size > 0:
                thr = float(np.quantile(gain[pos], 0.80))
                chosen = np.where(gain >= thr)[0]
            else:
                chosen = np.array([], dtype=int)
            if chosen.size < topk:
                order = np.argsort(gain)[::-1]
                chosen = order[:topk]
            out[int(cls)] = set(map(int, np.asarray(chosen, dtype=int).tolist()))
        return out

    tr = np.asarray(trials_random, dtype=float)
    tc = np.asarray(trials_coherent, dtype=float)
    if tr.ndim != 3 or tc.ndim != 3 or tr.shape[1] != tc.shape[1] or tr.shape[2] != tc.shape[2]:
        return {"random": np.nan, "coherent": np.nan}

    # Align with historical design in run_batch/run_geometry scripts:
    # class-specific RR vs other-RR within RR union.
    all_trials = np.concatenate([tc, tr], axis=0)
    labels = np.concatenate(
        [
            np.full(tc.shape[0], 1, dtype=int),  # coherent -> class 1 slot
            np.full(tr.shape[0], 3, dtype=int),  # random -> class 3 slot
        ]
    )
    rr_sets_strict = rr_selection_class_for_model(
        all_trials,
        labels,
        t_stimulus=int(response_start),
        l_stimulus=max(1, int(response_end) - int(response_start)),
    )
    rr_sets_relaxed = rr_selection_class_for_model(
        all_trials,
        labels,
        t_stimulus=int(response_start),
        l_stimulus=max(1, int(response_end) - int(response_start)),
        reliability_threshold=0.60,
        snr_threshold=0.55,
        effect_size_threshold=0.30,
        response_ratio_threshold=0.65,
    )
    rr_sets_fallback = _fallback_rr_sets_from_response(
        all_trials,
        labels,
        t0=int(response_start),
        t1=int(response_end),
    )
    rw = _safe_window(int(response_start), int(response_end), tr.shape[2])

    rr_sets: dict[int, set[int]] = {}
    for cls in [1, 3]:
        strict_set = rr_sets_strict.get(cls, set()) if rr_sets_strict else set()
        relaxed_set = rr_sets_relaxed.get(cls, set()) if rr_sets_relaxed else set()
        fallback_set = rr_sets_fallback.get(cls, set())
        if len(strict_set) > 0:
            rr_sets[cls] = set(strict_set)
        elif len(relaxed_set) > 0:
            rr_sets[cls] = set(relaxed_set)
        else:
            rr_sets[cls] = set(fallback_set)
    rr_union = set().union(*rr_sets.values()) if rr_sets else set()

    out = {}
    for cond_name, cls, c_trials in [("coherent", 1, tc), ("random", 3, tr)]:
        c_rr = sorted(rr_sets.get(int(cls), set()))
        oth = sorted(rr_union - set(c_rr))
        if len(c_rr) == 0 or c_trials.size == 0:
            out[cond_name] = np.nan
            continue
        m_rr = _avg_response(c_trials, c_rr, rw)
        if len(oth) == 0:
            out[cond_name] = m_rr
            continue
        den = _avg_response(c_trials, oth, rw)
        if not np.isfinite(den) or abs(den) <= EPS:
            out[cond_name] = m_rr
        else:
            out[cond_name] = float(m_rr / den)
    return out


def row_normalize_rows(X: np.ndarray) -> np.ndarray:
    Xp = np.clip(np.asarray(X, dtype=float), 0.0, None)
    if Xp.ndim != 2:
        return np.asarray(Xp, dtype=float)
    s = np.sum(Xp, axis=1, keepdims=True)
    out = np.zeros_like(Xp, dtype=float)
    valid = s[:, 0] > EPS
    if np.any(valid):
        out[valid] = Xp[valid] / (s[valid] + EPS)
    if np.any(~valid) and Xp.shape[1] > 0:
        out[~valid] = 1.0 / Xp.shape[1]
    return out


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


def mean_rsm_similarity(X_trials_by_neuron: np.ndarray) -> float:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        return np.nan
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + EPS)
    S = Xn @ Xn.T
    iu = np.triu_indices(S.shape[0], k=1)
    vals = S[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def orthogonal_expansion_metrics(X_trials_by_neuron: np.ndarray) -> dict:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        return {
            "var_parallel": np.nan,
            "var_orthogonal": np.nan,
            "orth_parallel_ratio": np.nan,
            "mean_rsm_sim": np.nan,
        }
    mu = np.nanmean(X, axis=0)
    mu_norm = float(np.linalg.norm(mu))
    Xc = X - mu[None, :]
    total_var = float(np.nanmean(np.sum(Xc * Xc, axis=1)))
    if mu_norm <= EPS:
        return {
            "var_parallel": np.nan,
            "var_orthogonal": np.nan,
            "orth_parallel_ratio": np.nan,
            "mean_rsm_sim": np.nan,
        }
    u = mu / mu_norm
    proj = Xc @ u
    var_parallel = float(np.nanvar(proj))
    var_orthogonal = max(total_var - var_parallel, 0.0)
    return {
        "var_parallel": var_parallel,
        "var_orthogonal": var_orthogonal,
        "orth_parallel_ratio": float(var_orthogonal / (var_parallel + EPS)),
        "mean_rsm_sim": mean_rsm_similarity(X),
    }


def ar1_series(length: int, rho: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(length, dtype=float)
    if length <= 0:
        return x
    for t in range(1, length):
        x[t] = rho * x[t - 1] + scale * rng.normal()
    return x


def ar1_matrix(length: int, dim: int, rho: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    Z = np.zeros((length, dim), dtype=float)
    if length <= 0 or dim <= 0:
        return Z
    for k in range(dim):
        Z[:, k] = ar1_series(length, rho=rho, scale=scale, rng=rng)
    return Z


def normalize_spectral_radius(W: np.ndarray, target_radius: float) -> np.ndarray:
    arr = np.asarray(W, dtype=float)
    if arr.shape[0] < 2:
        return arr
    try:
        eigs = np.linalg.eigvals(arr)
        sr = float(np.max(np.abs(eigs)))
    except Exception:
        sr = np.nan
    if not np.isfinite(sr) or sr <= EPS:
        return arr
    return arr * (float(target_radius) / (sr + EPS))


def build_network(params: ModelParams, rng: np.random.Generator) -> dict:
    n = int(params.n_neurons)
    n_core = int(params.n_core)
    idx_all = np.arange(n, dtype=int)
    core_idx = idx_all[:n_core]
    peri_idx = idx_all[n_core:]

    # Build periphery modules (v4: structured periphery, not a uniform pool).
    n_mod = int(np.clip(int(params.n_peri_modules), 1, max(1, peri_idx.size if peri_idx.size > 0 else 1)))
    peri_module_ids = np.full(n, -1, dtype=int)
    if peri_idx.size > 0:
        perm = rng.permutation(peri_idx)
        chunks = np.array_split(perm, n_mod)
        for m, ids in enumerate(chunks):
            peri_module_ids[np.asarray(ids, dtype=int)] = int(m)

    core_vec = np.zeros(n, dtype=bool)
    core_vec[core_idx] = True
    peri_vec = peri_module_ids >= 0
    offdiag = ~np.eye(n, dtype=bool)

    mod_i = peri_module_ids[:, None]
    mod_j = peri_module_ids[None, :]
    both_peri = (mod_i >= 0) & (mod_j >= 0)
    same_module = both_peri & (mod_i == mod_j) & offdiag
    cross_module = both_peri & (mod_i != mod_j) & offdiag

    # Strong core scaffold.
    W_core = np.zeros((n, n), dtype=float)
    core_mask = np.zeros((n, n), dtype=bool)
    core_mask[np.ix_(core_idx, core_idx)] = True
    core_mask &= offdiag
    core_conn = (rng.random((n, n)) < float(params.core_prob)) & core_mask
    core_scale = max(0.06, 0.25 * abs(float(params.w_core)))
    W_core[core_conn] = np.abs(
        rng.normal(loc=float(params.w_core), scale=core_scale, size=int(np.sum(core_conn)))
    )

    # Weak periphery edges with module-aware biases.
    W_peri = np.zeros((n, n), dtype=float)
    peri_conn = (rng.random((n, n)) < float(params.peri_prob)) & offdiag
    Wp = rng.normal(loc=float(params.w_peri_mean), scale=float(params.w_peri_std), size=(n, n))
    if np.any(same_module):
        Wp[same_module] += 0.35 * float(params.w_peri_std)
    if np.any(cross_module):
        Wp[cross_module] -= 0.20 * float(params.w_peri_std)
    W_peri[peri_conn] = Wp[peri_conn]

    # Baseline competition / inhibitory structure.
    bal_vec = np.abs(rng.normal(size=n))
    bal_vec = bal_vec / (np.linalg.norm(bal_vec) + EPS)
    W_bal_base = float(params.w_bal) * np.outer(bal_vec, bal_vec)
    if np.any(cross_module):
        W_bal_base[cross_module] += 0.25 * float(params.w_bal)
    core_peri = (np.outer(core_vec, peri_vec) | np.outer(peri_vec, core_vec)) & offdiag
    W_bal_base[core_peri] += 0.10 * float(params.w_bal)
    np.fill_diagonal(W_bal_base, 0.0)

    # Normalize full recurrent matrix and keep component scales consistent.
    W_base = W_core + W_peri - W_bal_base
    np.fill_diagonal(W_base, 0.0)
    scale = 1.0
    try:
        eigs = np.linalg.eigvals(W_base)
        sr = float(np.max(np.abs(eigs)))
        if np.isfinite(sr) and sr > EPS:
            scale = float(params.target_radius) / (sr + EPS)
    except Exception:
        scale = 1.0
    W_core *= scale
    W_peri *= scale
    W_bal_base *= scale
    W_base *= scale

    B = rng.normal(scale=1.0 / np.sqrt(max(1, params.stim_dim)), size=(n, params.stim_dim))
    u_shared = rng.normal(size=n)
    u_shared = u_shared / (np.linalg.norm(u_shared) + EPS)

    # Coherent condition modes: one broad mode + module modes.
    v_core = np.zeros(n, dtype=float)
    v_peri = np.zeros(n, dtype=float)
    v_core[core_idx] = np.abs(rng.normal(loc=1.0, scale=0.25, size=n_core))
    if peri_idx.size > 0:
        v_peri[peri_idx] = np.abs(rng.normal(loc=1.0, scale=0.35, size=peri_idx.size))
    v_global = float(params.beta_core) * v_core + float(params.beta_peri) * v_peri
    v_global = v_global / (np.linalg.norm(v_global) + EPS)

    V_mod = np.zeros((n_mod, n), dtype=float)
    Q_coh = np.zeros((n_mod, n), dtype=float)
    for m in range(n_mod):
        m_idx = np.where(peri_module_ids == m)[0]
        if m_idx.size == 0:
            continue
        vm = np.zeros(n, dtype=float)
        vm[m_idx] = np.abs(rng.normal(loc=1.0, scale=0.30, size=m_idx.size))
        if core_idx.size > 0 and float(params.module_selectivity) > 0:
            vm[core_idx] += float(params.module_selectivity) * np.abs(
                rng.normal(loc=0.14, scale=0.08, size=core_idx.size)
            )
        vm = vm / (np.linalg.norm(vm) + EPS)
        V_mod[m] = vm

        q = np.zeros(n, dtype=float)
        q[m_idx] = rng.normal(size=m_idx.size)
        q += 0.2 * rng.normal(size=n)
        q = q - float(np.dot(q, v_global)) * v_global
        q = q / (np.linalg.norm(q) + EPS)
        Q_coh[m] = q

    if bool(params.relief_cross_only):
        relief_gate = cross_module.astype(float)
    else:
        relief_gate = np.clip(cross_module.astype(float) + 0.35 * core_peri.astype(float), 0.0, 1.0)
    np.fill_diagonal(relief_gate, 0.0)

    return {
        "W_core": W_core,
        "W_peri": W_peri,
        "W_bal_base": W_bal_base,
        "W_base": W_base,
        "B": B,
        "u_shared": u_shared,
        "v_global": v_global,
        "V_mod": V_mod,
        "Q_coh": Q_coh,
        "core_idx": core_idx,
        "peri_idx": peri_idx,
        "peri_module_ids": peri_module_ids,
        "relief_gate": relief_gate,
        "n_modules": n_mod,
    }


def build_effective_balance(network_info: dict, condition: str, params: ModelParams) -> np.ndarray:
    W_bal = np.asarray(network_info["W_bal_base"], dtype=float).copy()
    if condition == "coherent":
        gate = np.asarray(network_info["relief_gate"], dtype=float)
        kappa = float(np.clip(float(params.kappa_coh_relief), 0.0, 0.95))
        W_bal = W_bal * (1.0 - kappa * gate)
    return W_bal


def generate_condition_latents_v4(
    params: ModelParams,
    condition: str,
    t_steps: int,
    n_modules: int,
    rng: np.random.Generator,
) -> dict:
    if condition != "coherent":
        return {
            "a0": np.zeros(t_steps, dtype=float),
            "amod": np.zeros((t_steps, n_modules), dtype=float),
            "extra": np.zeros((t_steps, n_modules), dtype=float),
        }

    a0 = ar1_series(t_steps, rho=params.cond_rho, scale=1.0, rng=rng)
    amod = ar1_matrix(t_steps, n_modules, rho=params.cond_rho, scale=1.0, rng=rng)
    if n_modules > 0:
        sel = 1.0 + float(params.module_selectivity) * rng.normal(loc=0.0, scale=0.30, size=n_modules)
        amod = amod * sel[None, :]
        extra = ar1_matrix(
            t_steps,
            n_modules,
            rho=float(np.clip(params.module_noise_rho, 0.0, 0.99)),
            scale=1.0,
            rng=rng,
        )
    else:
        extra = np.zeros((t_steps, 0), dtype=float)

    return {"a0": a0, "amod": amod, "extra": extra}


def compute_norm_vector_v4(r: np.ndarray, network_info: dict, params: ModelParams) -> np.ndarray:
    rr = np.asarray(r, dtype=float).reshape(-1)
    n = rr.size
    out = np.full(n, float(params.lambda_global) * float(np.mean(rr)), dtype=float)

    core_idx = np.asarray(network_info["core_idx"], dtype=int)
    if core_idx.size > 0:
        out[core_idx] += float(params.lambda_core) * float(np.mean(rr[core_idx]))

    module_ids = np.asarray(network_info["peri_module_ids"], dtype=int)
    valid_mods = np.unique(module_ids[module_ids >= 0]).astype(int)
    for m in valid_mods:
        m_idx = np.where(module_ids == m)[0]
        if m_idx.size > 0:
            out[m_idx] += float(params.lambda_peri) * float(np.mean(rr[m_idx]))

    return float(params.gamma_norm) * out


def coherent_extra_fluctuation_v4(
    t: int,
    condition: str,
    network_info: dict,
    latents: dict,
    params: ModelParams,
) -> np.ndarray:
    n = int(network_info["W_base"].shape[0])
    if condition != "coherent":
        return np.zeros(n, dtype=float)
    Q = np.asarray(network_info["Q_coh"], dtype=float)
    extra = np.asarray(latents.get("extra", np.zeros((1, 0))), dtype=float)
    if Q.ndim != 2 or Q.shape[0] == 0 or extra.ndim != 2 or extra.shape[1] == 0:
        return np.zeros(n, dtype=float)
    k = int(min(Q.shape[0], extra.shape[1]))
    if k <= 0:
        return np.zeros(n, dtype=float)
    coeff = extra[t, :k]
    vec = coeff @ Q[:k]
    return float(params.g_coh_extra_noise) * vec


def simulate_condition(
    params: ModelParams,
    network_info: dict,
    condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    n = params.n_neurons
    T = params.t_steps
    n_trials = params.n_trials
    n_mod_total = int(network_info.get("n_modules", 0))
    if n_mod_total <= 0:
        n_mod_active = 0
    else:
        n_mod_active = int(np.clip(int(params.n_cond_modes) - 1, 0, n_mod_total))
    V_mod = np.asarray(network_info["V_mod"], dtype=float)

    W_bal_eff = build_effective_balance(network_info, condition=condition, params=params)
    W_eff = np.asarray(network_info["W_core"], dtype=float) + np.asarray(network_info["W_peri"], dtype=float) - W_bal_eff

    B = np.asarray(network_info["B"], dtype=float)
    u_shared = np.asarray(network_info["u_shared"], dtype=float)
    v_global = np.asarray(network_info["v_global"], dtype=float)

    out = np.zeros((n_trials, n, T), dtype=float)
    coh_extra_noise_energy = []
    for tr in range(n_trials):
        z = ar1_matrix(T, params.stim_dim, rho=params.stim_rho, scale=params.stim_scale, rng=rng)
        eta = ar1_series(T, rho=params.noise_rho, scale=1.0, rng=rng)
        lat = generate_condition_latents_v4(
            params,
            condition=condition,
            t_steps=T,
            n_modules=n_mod_active,
            rng=rng,
        )
        r = np.zeros(n, dtype=float)
        extra_energy_trial = 0.0
        for t in range(T):
            i_stim = B @ z[t]
            if condition == "coherent":
                i_cond = float(params.g_c_global) * lat["a0"][t] * v_global
                if n_mod_active > 0 and V_mod.size > 0:
                    i_cond = i_cond + float(params.g_c_module) * (lat["amod"][t] @ V_mod[:n_mod_active])
            else:
                i_cond = np.zeros(n, dtype=float)

            xi_ind = float(params.sigma_ind) * rng.normal(size=n)
            xi_shared = float(params.g_n) * u_shared * eta[t]
            xi_coh_extra = coherent_extra_fluctuation_v4(t, condition=condition, network_info=network_info, latents=lat, params=params)
            extra_energy_trial += float(np.mean(xi_coh_extra**2))

            norm_vec = compute_norm_vector_v4(r, network_info=network_info, params=params)
            inp = W_eff @ r + i_stim + i_cond + xi_ind + xi_shared + xi_coh_extra - norm_vec
            rr = np.maximum(inp, 0.0)
            r = (1.0 - float(params.alpha)) * r + float(params.alpha) * rr
            out[tr, :, t] = r
        coh_extra_noise_energy.append(extra_energy_trial / max(1, T))

    diag = {"coh_extra_noise_energy": float(np.mean(coh_extra_noise_energy)) if coh_extra_noise_energy else np.nan}
    return out, diag


def fc_metrics_from_trial_matrix(
    X_trials_by_neuron: np.ndarray,
    weak_thr: float,
    strong_thr: float | None,
    strong_quantile: float,
) -> tuple[dict, float]:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    residual = X - np.nanmean(X, axis=0, keepdims=True)
    C = robust_corrcoef(residual, rowvar=False)
    vals = upper_tri_values(C)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return {
            "mean_noise_corr": np.nan,
            "neg_frac": np.nan,
            "weak_pos_frac": np.nan,
            "strong_frac": np.nan,
            "strong_mean": np.nan,
        }, np.nan

    if strong_thr is None or (not np.isfinite(strong_thr)):
        q = float(np.clip(float(strong_quantile), 0.5, 0.995))
        strong_thr_use = float(np.quantile(vals, q))
    else:
        strong_thr_use = float(strong_thr)
    strong_mask = vals >= strong_thr_use

    out = {
        "mean_noise_corr": float(np.mean(vals)),
        "neg_frac": float(np.mean(vals < 0)),
        "weak_pos_frac": float(np.mean((vals > 0) & (vals <= weak_thr))),
        "strong_frac": float(np.mean(strong_mask)),
        "strong_mean": float(np.mean(vals[strong_mask])) if np.any(strong_mask) else np.nan,
    }
    return out, strong_thr_use


def allocation_metrics_from_trial_matrix(
    X_trials_by_neuron: np.ndarray,
    n_core: int,
    active_frac_thresh: float,
) -> dict:
    X = np.asarray(X_trials_by_neuron, dtype=float)
    m = np.nanmean(X, axis=0)
    m = np.clip(m, 0.0, None)
    s = float(np.sum(m))
    if s <= EPS:
        return {
            "gini": np.nan,
            "top10_frac": np.nan,
            "active_count": np.nan,
            "pr_mean": np.nan,
        }
    n = m.size
    thr = float(active_frac_thresh) * float(np.max(m))
    active_count = int(np.sum(m > thr))
    topk = max(1, int(np.ceil(0.10 * n)))
    idx = np.argsort(m)[-topk:]
    top10_frac = float(np.sum(m[idx]) / (s + EPS))
    # participants_ratio is injected later by paired RR computation across
    # random/coherent conditions.
    gini_trials = np.asarray([gini_index(X[i]) for i in range(X.shape[0])], dtype=float)
    out = {
        "gini": float(np.nanmean(gini_trials)) if np.any(np.isfinite(gini_trials)) else np.nan,
        "top10_frac": top10_frac,
        "active_count": float(active_count),
        "pr_mean": float(np.nanmean([participation_ratio_vector(X[i]) for i in range(X.shape[0])])),
    }
    return out


def extract_condition_metrics(params: ModelParams, trials: np.ndarray, strong_thr: float | None) -> tuple[dict, float]:
    win = slice(params.response_start, params.response_end)
    X = np.nanmean(np.asarray(trials, dtype=float)[:, :, win], axis=2)
    fc, strong_thr_out = fc_metrics_from_trial_matrix(
        X,
        weak_thr=params.weak_pos_max,
        strong_thr=strong_thr,
        strong_quantile=params.strong_quantile,
    )
    alloc = allocation_metrics_from_trial_matrix(X, n_core=params.n_core, active_frac_thresh=params.active_frac_thresh)
    geom = orthogonal_expansion_metrics(X)
    out = {**fc, **alloc, **geom}
    return out, strong_thr_out


def module_recruitment_index_from_trials(
    trials: np.ndarray,
    peri_module_ids: np.ndarray,
    response_start: int,
    response_end: int,
) -> float:
    arr = np.asarray(trials, dtype=float)
    module_ids = np.asarray(peri_module_ids, dtype=int).reshape(-1)
    if arr.ndim != 3 or arr.shape[1] != module_ids.size:
        return np.nan
    valid_mods = np.unique(module_ids[module_ids >= 0]).astype(int)
    if valid_mods.size == 0:
        return np.nan

    win = slice(int(response_start), int(response_end))
    X = np.nanmean(arr[:, :, win], axis=2)  # trial x neuron
    mod_resp = []
    for m in valid_mods:
        idx = np.where(module_ids == int(m))[0]
        if idx.size == 0:
            continue
        mod_resp.append(np.nanmean(X[:, idx], axis=1))
    if not mod_resp:
        return np.nan
    M = np.stack(mod_resp, axis=1)  # trial x module
    peak = np.nanmax(M, axis=1, keepdims=True)
    thr = 0.20 * peak
    active = M > (thr + EPS)
    return float(np.nanmean(np.sum(active, axis=1) / max(1, M.shape[1])))


def default_targets() -> dict:
    return {
        "fc": {
            "neg_frac_delta": -0.091475,
            "weak_pos_frac_delta": 0.044388,
            "strong_frac_delta": -0.043404,
            "strong_mean_delta": 0.013299,
            "mean_noise_corr_delta": 0.011880,
        },
        "alloc": {
            "participants_ratio_delta": -0.833325,
            "gini_delta": -0.023393,
            "top10_frac_delta": np.nan,
        },
        "geom": {
            "var_orthogonal_delta": 0.377860,
            "var_parallel_delta": 0.051113,
            "mean_rsm_sim_delta": -0.067825,
            "orth_parallel_ratio_delta": -0.308074,
        },
    }


def _coherent_mouse_average(df: pd.DataFrame, metric: str) -> pd.Series:
    piv = df[["mouse_id", "Condition", metric]].pivot(index="mouse_id", columns="Condition", values=metric)
    ok = piv.dropna(subset=["Random", "Divergent", "Convergent"])
    coh = 0.5 * (ok["Divergent"] + ok["Convergent"])
    return coh - ok["Random"]


def load_empirical_targets(results_dir: str) -> tuple[dict, dict]:
    group_dir = os.path.join(results_dir, GROUP_DIR_NAME)
    t = default_targets()
    meta = {"target_source": "default_constants"}

    weak_path = os.path.join(group_dir, "group_weakcorr_reorg_mouse_metrics.csv")
    if os.path.isfile(weak_path):
        try:
            df = pd.read_csv(weak_path)
            d_neg = _coherent_mouse_average(df, "neg_frac")
            d_weak = _coherent_mouse_average(df, "weak_pos_frac")
            d_sfrac = _coherent_mouse_average(df, "strong_frac")
            d_smean = _coherent_mouse_average(df, "strong_mean")
            d_mean = _coherent_mouse_average(df, "mean_noise_corr")
            t["fc"] = {
                "neg_frac_delta": float(np.nanmean(d_neg)),
                "weak_pos_frac_delta": float(np.nanmean(d_weak)),
                "strong_frac_delta": float(np.nanmean(d_sfrac)),
                "strong_mean_delta": float(np.nanmean(d_smean)),
                "mean_noise_corr_delta": float(np.nanmean(d_mean)),
            }
            meta["target_source"] = "group_weakcorr_reorg_mouse_metrics.csv"
        except Exception as exc:
            meta["weak_target_error"] = str(exc)

    master_path = os.path.join(group_dir, "group_master_metrics.csv")
    if os.path.isfile(master_path):
        try:
            dfm = pd.read_csv(master_path)
            d_pr = _coherent_mouse_average(dfm, "Participants_Ratio")
            d_gini = _coherent_mouse_average(dfm, "Gini_Mean")
            d_rsm = _coherent_mouse_average(dfm, "Mean_RSM_Sim")
            d_vp = _coherent_mouse_average(dfm, "Geom_VarParallel")
            d_vo = _coherent_mouse_average(dfm, "Geom_VarOrthogonal")
            d_or = _coherent_mouse_average(dfm, "Geom_OrthParallelRatio")
            t["alloc"] = {
                "participants_ratio_delta": float(np.nanmean(d_pr)),
                "gini_delta": float(np.nanmean(d_gini)),
                "top10_frac_delta": np.nan,
            }
            t["geom"] = {
                "var_orthogonal_delta": float(np.nanmean(d_vo)),
                "var_parallel_delta": float(np.nanmean(d_vp)),
                "mean_rsm_sim_delta": float(np.nanmean(d_rsm)),
                "orth_parallel_ratio_delta": float(np.nanmean(d_or)),
            }
            if meta["target_source"] == "default_constants":
                meta["target_source"] = "group_master_metrics.csv + defaults_for_fc"
            else:
                meta["target_source"] = meta["target_source"] + " + group_master_metrics.csv"
        except Exception as exc:
            meta["master_target_error"] = str(exc)

    return t, meta


def _target_metric_value(targets: dict, metric: str) -> float:
    for group in ("fc", "alloc", "geom"):
        grp = targets.get(group, {})
        if metric in grp:
            return float(grp.get(metric, np.nan))
    return np.nan


def score_group(
    model_delta: dict,
    target_delta: dict,
    metrics: list[str],
    metric_weights: dict[str, float] | None = None,
    error_mode: dict[str, str] | None = None,
    band_limits: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, dict]:
    errs = {}
    for m in metrics:
        tm = target_delta.get(m, np.nan)
        mm = model_delta.get(m, np.nan)
        if not np.isfinite(tm) or not np.isfinite(mm):
            errs[m] = np.nan
            continue
        mode = "relative"
        if error_mode is not None:
            mode = str(error_mode.get(m, "relative"))
        if mode == "tolerance_band":
            if band_limits is None or m not in band_limits:
                errs[m] = float(abs(mm - tm) / (abs(tm) + 1e-6))
            else:
                lo, hi = band_limits[m]
                lo_f = float(min(lo, hi))
                hi_f = float(max(lo, hi))
                width = max(hi_f - lo_f, 1e-3)
                if mm < lo_f:
                    errs[m] = float((lo_f - mm) / width)
                elif mm > hi_f:
                    errs[m] = float((mm - hi_f) / width)
                else:
                    errs[m] = 0.0
        elif mode == "log_ratio":
            sign_tm = np.sign(tm) if abs(tm) > EPS else 0.0
            sign_mm = np.sign(mm) if abs(mm) > EPS else 0.0
            mag_err = abs(np.log((abs(mm) + 1e-6) / (abs(tm) + 1e-6)))
            if sign_tm != 0 and sign_mm != sign_tm:
                errs[m] = float(1.0 + mag_err)
            else:
                errs[m] = float(mag_err)
        elif mode == "direction_only":
            sign_tm = np.sign(tm) if abs(tm) > EPS else 0.0
            sign_mm = np.sign(mm) if abs(mm) > EPS else 0.0
            if sign_tm == 0.0:
                errs[m] = 0.0 if abs(mm) <= EPS else 1.0
            else:
                errs[m] = 0.0 if sign_mm == sign_tm else 1.0
        else:
            errs[m] = float(abs(mm - tm) / (abs(tm) + 1e-6))
    vals = []
    wts = []
    for m in metrics:
        v = errs.get(m, np.nan)
        if not np.isfinite(v):
            continue
        w = 1.0
        if metric_weights is not None:
            w = float(metric_weights.get(m, 1.0))
        if not np.isfinite(w) or w <= 0:
            continue
        vals.append(float(v))
        wts.append(float(w))
    if not vals:
        return np.nan, errs
    arr = np.asarray(vals, dtype=float)
    warr = np.asarray(wts, dtype=float)
    return float(np.sum(arr * warr) / (np.sum(warr) + EPS)), errs


def stability_score_from_repeats(
    rep_deltas: list[dict],
    targets: dict,
    metric_weights: dict[str, float],
    sign_penalty_w: float,
    rel_std_cap: float,
) -> tuple[float, dict]:
    errs: dict[str, float] = {}
    vals = []
    wts = []
    for metric, weight in metric_weights.items():
        w = float(weight)
        if not np.isfinite(w) or w <= 0:
            continue
        tm = float(_target_metric_value(targets, metric))
        x = np.asarray([d.get(metric, np.nan) for d in rep_deltas], dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2 or (not np.isfinite(tm)):
            errs[metric] = np.nan
            continue

        rel_std = float(np.nanstd(x, ddof=1) / (abs(tm) + 1e-6))
        rel_std = float(np.clip(rel_std, 0.0, max(0.0, float(rel_std_cap))))
        if abs(tm) <= EPS:
            sign_mismatch = float(np.nanmean(np.abs(x) > EPS))
        else:
            sign_mismatch = float(np.nanmean(np.sign(x) != np.sign(tm)))
        err = rel_std + float(sign_penalty_w) * sign_mismatch
        errs[metric] = float(err)
        vals.append(float(err))
        wts.append(w)

    if not vals:
        return np.nan, errs
    arr = np.asarray(vals, dtype=float)
    warr = np.asarray(wts, dtype=float)
    return float(np.sum(arr * warr) / (np.sum(warr) + EPS)), errs


def _band_amp_penalty_one(
    model_val: float,
    target_val: float,
    min_ratio: float,
    max_ratio: float,
    w_under: float,
    w_over: float,
) -> tuple[float, float]:
    if (not np.isfinite(model_val)) or (not np.isfinite(target_val)):
        return 0.0, 0.0
    if abs(target_val) <= EPS:
        return 0.0, 0.0
    if min_ratio < 0:
        min_ratio = 0.0
    if max_ratio <= 0:
        max_ratio = np.inf
    if max_ratio < min_ratio:
        max_ratio = min_ratio

    # Signed progress ratio: 1.0 means exactly at target magnitude and sign.
    progress = float(model_val / target_val)

    p_under = 0.0
    p_over = 0.0
    if w_under > 0 and progress < min_ratio:
        p_under = float(w_under) * float((min_ratio - progress) / (min_ratio + EPS))
    if w_over > 0 and np.isfinite(max_ratio) and progress > max_ratio:
        # Log-growth over-penalty avoids exploding score when an outlier overshoots heavily.
        excess = float((progress - max_ratio) / (max_ratio + EPS))
        p_over = float(w_over) * float(np.log1p(max(0.0, excess)))
    return p_under, p_over


def amplitude_band_penalty(model_delta: dict, targets: dict, args) -> tuple[float, float, float]:
    p_under = 0.0
    p_over = 0.0

    pu, po = _band_amp_penalty_one(
        float(model_delta.get("participants_ratio_delta", np.nan)),
        float(targets["alloc"].get("participants_ratio_delta", np.nan)),
        float(args.min_amp_ratio_participants),
        float(args.max_amp_ratio_participants),
        float(args.min_amp_penalty_w_participants),
        float(args.max_amp_penalty_w_participants),
    )
    p_under += pu
    p_over += po

    pu, po = _band_amp_penalty_one(
        float(model_delta.get("orth_parallel_ratio_delta", np.nan)),
        float(targets["geom"].get("orth_parallel_ratio_delta", np.nan)),
        float(args.min_amp_ratio_orth_ratio),
        float(args.max_amp_ratio_orth_ratio),
        float(args.min_amp_penalty_w_orth_ratio),
        float(args.max_amp_penalty_w_orth_ratio),
    )
    p_under += pu
    p_over += po

    pu, po = _band_amp_penalty_one(
        float(model_delta.get("neg_frac_delta", np.nan)),
        float(targets["fc"].get("neg_frac_delta", np.nan)),
        float(args.min_amp_ratio_neg_frac),
        float(args.max_amp_ratio_neg_frac),
        float(args.min_amp_penalty_w_neg_frac),
        float(args.max_amp_penalty_w_neg_frac),
    )
    p_under += pu
    p_over += po

    pu, po = _band_amp_penalty_one(
        float(model_delta.get("weak_pos_frac_delta", np.nan)),
        float(targets["fc"].get("weak_pos_frac_delta", np.nan)),
        float(args.min_amp_ratio_weak_pos),
        float(args.max_amp_ratio_weak_pos),
        float(args.min_amp_penalty_w_weak_pos),
        float(args.max_amp_penalty_w_weak_pos),
    )
    p_under += pu
    p_over += po

    pu, po = _band_amp_penalty_one(
        float(model_delta.get("mean_rsm_sim_delta", np.nan)),
        float(targets["geom"].get("mean_rsm_sim_delta", np.nan)),
        float(args.min_amp_ratio_rsm),
        float(args.max_amp_ratio_rsm),
        float(args.min_amp_penalty_w_rsm),
        float(args.max_amp_penalty_w_rsm),
    )
    p_under += pu
    p_over += po
    return float(p_under + p_over), float(p_under), float(p_over)


def directional_penalty(model_delta: dict, args) -> float:
    p = 0.0
    if model_delta.get("neg_frac_delta", np.inf) >= float(args.penalty_neg_frac_max):
        p += float(args.penalty_neg_frac_w)
    if model_delta.get("weak_pos_frac_delta", -np.inf) <= float(args.penalty_weak_pos_min):
        p += float(args.penalty_weak_pos_w)
    if model_delta.get("participants_ratio_delta", np.inf) >= float(args.penalty_participants_max):
        p += float(args.penalty_participants_w)
    if model_delta.get("mean_rsm_sim_delta", np.inf) >= float(args.penalty_rsm_max):
        p += float(args.penalty_rsm_w)
    if model_delta.get("strong_frac_delta", np.inf) >= float(args.penalty_strong_frac_max):
        p += float(args.penalty_strong_frac_w)

    # v4 hard constraints: geometry raw effects + stable strong core.
    if model_delta.get("var_orthogonal_delta", -np.inf) <= float(args.penalty_var_orth_min):
        p += float(args.penalty_var_orth_w)
    if model_delta.get("mean_rsm_sim_delta", np.inf) >= float(args.penalty_rsm_hard_max):
        p += float(args.penalty_rsm_hard_w)
    if model_delta.get("strong_mean_delta", np.inf) > float(args.penalty_strong_mean_upper):
        p += float(args.penalty_strong_mean_w)

    # If FC hierarchy compresses but allocation does not transmit, add mild penalty.
    neg_ok = model_delta.get("neg_frac_delta", np.inf) < float(args.bridge_neg_frac_max)
    weak_ok = model_delta.get("weak_pos_frac_delta", -np.inf) > float(args.bridge_weak_pos_min)
    participants_delta = model_delta.get("participants_ratio_delta", np.nan)
    if neg_ok and weak_ok and np.isfinite(participants_delta):
        if participants_delta > float(args.bridge_participants_max):
            p += float(args.bridge_penalty_w)

    # Mean noise-corr tolerance band (avoid overfitting to point target).
    mean_nc = model_delta.get("mean_noise_corr_delta", np.nan)
    if np.isfinite(mean_nc):
        lo = float(min(args.mean_noise_band_low, args.mean_noise_band_high))
        hi = float(max(args.mean_noise_band_low, args.mean_noise_band_high))
        if mean_nc < lo:
            p += float(args.mean_noise_band_w) * float((lo - mean_nc) / (hi - lo + EPS))
        elif mean_nc > hi:
            p += float(args.mean_noise_band_w) * float((mean_nc - hi) / (hi - lo + EPS))
    return p


def simulate_one_param_set(params: ModelParams, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    network = build_network(params, rng=rng)

    trials_rand, diag_rand = simulate_condition(params, network_info=network, condition="random", rng=rng)
    trials_coh, diag_coh = simulate_condition(params, network_info=network, condition="coherent", rng=rng)

    met_rand, strong_thr = extract_condition_metrics(params, trials_rand, strong_thr=None)
    met_coh, _ = extract_condition_metrics(params, trials_coh, strong_thr=strong_thr)

    rr_participants = participants_ratio_from_two_conditions(
        trials_random=trials_rand,
        trials_coherent=trials_coh,
        response_start=params.response_start,
        response_end=params.response_end,
    )
    pr_rand = float(rr_participants.get("random", np.nan))
    pr_coh = float(rr_participants.get("coherent", np.nan))
    # Always keep the key to avoid missing-key penalties during scoring.
    met_rand["participants_ratio"] = pr_rand
    met_coh["participants_ratio"] = pr_coh

    all_metric_keys = sorted(set(met_rand.keys()).union(set(met_coh.keys())))
    delta = {f"{k}_delta": float(met_coh.get(k, np.nan) - met_rand.get(k, np.nan)) for k in all_metric_keys}

    # v4 diagnostics: help check whether FC compression is actually transmitted
    # to allocation/geometry through the intended mechanism.
    mod_recruit_rand = module_recruitment_index_from_trials(
        trials_rand,
        peri_module_ids=np.asarray(network["peri_module_ids"], dtype=int),
        response_start=params.response_start,
        response_end=params.response_end,
    )
    mod_recruit_coh = module_recruitment_index_from_trials(
        trials_coh,
        peri_module_ids=np.asarray(network["peri_module_ids"], dtype=int),
        response_start=params.response_start,
        response_end=params.response_end,
    )
    delta["module_recruitment_delta"] = float(mod_recruit_coh - mod_recruit_rand)
    delta["coh_extra_noise_energy_delta"] = float(
        float(diag_coh.get("coh_extra_noise_energy", np.nan))
        - float(diag_rand.get("coh_extra_noise_energy", np.nan))
    )
    delta["neg_relief_index_delta"] = float(
        -float(delta.get("neg_frac_delta", np.nan)) + float(delta.get("weak_pos_frac_delta", np.nan))
    )
    delta["orth_abs_delta"] = float(delta.get("var_orthogonal_delta", np.nan))

    out = {
        "metrics_random": met_rand,
        "metrics_coherent": met_coh,
        "diag_random": diag_rand,
        "diag_coherent": diag_coh,
        "delta": delta,
    }
    return out


def sample_param_set(base: ModelParams, rng: np.random.Generator, space: SearchSpace) -> ModelParams:
    return ModelParams(
        n_neurons=base.n_neurons,
        n_core=base.n_core,
        n_peri_modules=base.n_peri_modules,
        core_prob=_sample_range(rng, space.core_prob),
        peri_prob=_sample_range(rng, space.peri_prob),
        w_core=_sample_range(rng, space.w_core),
        w_peri_mean=_sample_range(rng, space.w_peri_mean),
        w_peri_std=_sample_range(rng, space.w_peri_std),
        w_bal=_sample_range(rng, space.w_bal),
        target_radius=_sample_range(rng, space.target_radius),
        alpha=_sample_range(rng, space.alpha),
        lambda_global=_sample_range(rng, space.lambda_global),
        lambda_core=_sample_range(rng, space.lambda_core),
        lambda_peri=_sample_range(rng, space.lambda_peri),
        gamma_norm=base.gamma_norm,
        kappa_coh_relief=_sample_range(rng, space.kappa_coh_relief),
        relief_cross_only=base.relief_cross_only,
        n_cond_modes=base.n_cond_modes,
        g_c_global=_sample_range(rng, space.g_c_global),
        g_c_module=_sample_range(rng, space.g_c_module),
        beta_core=_sample_range(rng, space.beta_core),
        beta_peri=_sample_range(rng, space.beta_peri),
        module_selectivity=_sample_range(rng, space.module_selectivity),
        sigma_ind=_sample_range(rng, space.sigma_ind),
        g_n=_sample_range(rng, space.g_n),
        g_coh_extra_noise=_sample_range(rng, space.g_coh_extra_noise),
        module_noise_rho=_sample_range(rng, space.module_noise_rho),
        stim_dim=base.stim_dim,
        stim_scale=_sample_range(rng, space.stim_scale),
        stim_rho=_sample_range(rng, space.stim_rho),
        cond_rho=_sample_range(rng, space.cond_rho),
        noise_rho=_sample_range(rng, space.noise_rho),
        n_trials=base.n_trials,
        t_steps=base.t_steps,
        response_start=base.response_start,
        response_end=base.response_end,
        weak_pos_max=base.weak_pos_max,
        strong_quantile=base.strong_quantile,
        active_frac_thresh=_sample_range(rng, space.active_frac_thresh),
    )


def model_params_to_row(prefix: str, p: ModelParams) -> dict:
    d = {
        f"{prefix}n_neurons": p.n_neurons,
        f"{prefix}n_core": p.n_core,
        f"{prefix}n_peri_modules": p.n_peri_modules,
        f"{prefix}core_prob": p.core_prob,
        f"{prefix}peri_prob": p.peri_prob,
        f"{prefix}w_core": p.w_core,
        f"{prefix}w_peri_mean": p.w_peri_mean,
        f"{prefix}w_peri_std": p.w_peri_std,
        f"{prefix}w_bal": p.w_bal,
        f"{prefix}target_radius": p.target_radius,
        f"{prefix}alpha": p.alpha,
        f"{prefix}lambda_global": p.lambda_global,
        f"{prefix}lambda_core": p.lambda_core,
        f"{prefix}lambda_peri": p.lambda_peri,
        f"{prefix}kappa_coh_relief": p.kappa_coh_relief,
        f"{prefix}g_c_global": p.g_c_global,
        f"{prefix}g_c_module": p.g_c_module,
        f"{prefix}beta_core": p.beta_core,
        f"{prefix}beta_peri": p.beta_peri,
        f"{prefix}module_selectivity": p.module_selectivity,
        f"{prefix}sigma_ind": p.sigma_ind,
        f"{prefix}g_n": p.g_n,
        f"{prefix}g_coh_extra_noise": p.g_coh_extra_noise,
        f"{prefix}module_noise_rho": p.module_noise_rho,
        f"{prefix}stim_scale": p.stim_scale,
        f"{prefix}stim_rho": p.stim_rho,
        f"{prefix}cond_rho": p.cond_rho,
        f"{prefix}noise_rho": p.noise_rho,
        f"{prefix}active_frac_thresh": p.active_frac_thresh,
    }
    return d


def _score_with_cap(score: float, cap: float, nan_fallback: float) -> float:
    if not np.isfinite(score):
        return float(nan_fallback)
    if np.isfinite(cap) and cap > 0:
        return float(np.clip(score, 0.0, cap))
    return float(score)


def run_search(
    args,
    base_params: ModelParams,
    targets: dict,
    search_space: SearchSpace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(args.seed))
    fc_keys = ["neg_frac_delta", "weak_pos_frac_delta", "strong_frac_delta", "strong_mean_delta", "mean_noise_corr_delta"]
    alloc_keys = ["participants_ratio_delta", "gini_delta"]
    geom_keys = ["var_orthogonal_delta", "var_parallel_delta", "mean_rsm_sim_delta", "orth_parallel_ratio_delta"]
    fc_error_mode = {"mean_noise_corr_delta": "tolerance_band"}
    fc_band_limits = {
        "mean_noise_corr_delta": (
            float(args.mean_noise_band_low),
            float(args.mean_noise_band_high),
        )
    }
    geom_error_mode = {
        "var_orthogonal_delta": "direction_only",
        "var_parallel_delta": "direction_only",
        "orth_parallel_ratio_delta": "direction_only",
    }
    alloc_metric_weights = {
        "participants_ratio_delta": float(args.metric_w_participants_ratio),
        "gini_delta": float(args.metric_w_gini),
    }
    geom_metric_weights = {
        "var_orthogonal_delta": float(args.metric_w_var_orthogonal),
        "var_parallel_delta": float(args.metric_w_var_parallel),
        "mean_rsm_sim_delta": float(args.metric_w_mean_rsm),
        "orth_parallel_ratio_delta": float(args.metric_w_orth_parallel_ratio),
    }
    stability_metric_weights = {
        "participants_ratio_delta": float(args.metric_w_stability_participants),
        "mean_rsm_sim_delta": float(args.metric_w_stability_rsm),
        "neg_frac_delta": float(args.metric_w_stability_neg_frac),
        "weak_pos_frac_delta": float(args.metric_w_stability_weak_pos),
    }

    rows = []
    detail_rows = []
    for idx in range(int(args.num_samples)):
        pset = sample_param_set(base_params, rng, search_space)
        rep_deltas = []
        for r in range(int(args.repeats_per_param)):
            seed_i = int(args.seed + idx * 1009 + r * 131)
            out = simulate_one_param_set(pset, seed=seed_i)
            rep_deltas.append(out["delta"])
            detail = {"sample_id": idx, "repeat_id": r, **model_params_to_row("", pset), **out["delta"]}
            detail_rows.append(detail)

        avg_delta = {}
        keys = sorted(set().union(*[d.keys() for d in rep_deltas]))
        for k in keys:
            vals = np.asarray([d.get(k, np.nan) for d in rep_deltas], dtype=float)
            avg_delta[k] = float(np.nanmean(vals))

        s_fc, e_fc = score_group(
            avg_delta,
            targets["fc"],
            fc_keys,
            error_mode=fc_error_mode,
            band_limits=fc_band_limits,
        )
        s_alloc, e_alloc = score_group(
            avg_delta, targets["alloc"], alloc_keys, metric_weights=alloc_metric_weights
        )
        s_geom, e_geom = score_group(
            avg_delta,
            targets["geom"],
            geom_keys,
            metric_weights=geom_metric_weights,
            error_mode=geom_error_mode,
        )
        s_stability, e_stability = stability_score_from_repeats(
            rep_deltas=rep_deltas,
            targets=targets,
            metric_weights=stability_metric_weights,
            sign_penalty_w=float(args.stability_sign_penalty_w),
            rel_std_cap=float(args.stability_rel_std_cap),
        )
        s_fc_use = _score_with_cap(s_fc, cap=float(args.score_cap_fc), nan_fallback=float(args.score_nan_fallback))
        s_alloc_use = _score_with_cap(
            s_alloc, cap=float(args.score_cap_alloc), nan_fallback=float(args.score_nan_fallback)
        )
        s_geom_use = _score_with_cap(s_geom, cap=float(args.score_cap_geom), nan_fallback=float(args.score_nan_fallback))
        s_stability_use = _score_with_cap(
            s_stability,
            cap=float(args.score_cap_stability),
            nan_fallback=float(args.score_stability_nan_fallback),
        )
        score_base_core = (
            float(args.w_fc) * s_fc_use
            + float(args.w_alloc) * s_alloc_use
            + float(args.w_geom) * s_geom_use
        )
        score_base = score_base_core + float(args.w_stability) * s_stability_use
        penalty_directional = directional_penalty(avg_delta, args)
        penalty_min_amp, penalty_amp_under, penalty_amp_over = amplitude_band_penalty(avg_delta, targets, args)
        score_total = score_base + penalty_directional + penalty_min_amp

        row = {
            "sample_id": idx,
            "score_total": score_total,
            "score_base": score_base,
            "score_base_core": score_base_core,
            "penalty_directional": penalty_directional,
            "penalty_min_amp": penalty_min_amp,
            "penalty_amp_under": penalty_amp_under,
            "penalty_amp_over": penalty_amp_over,
            "score_fc": s_fc,
            "score_alloc": s_alloc,
            "score_geom": s_geom,
            "score_stability": s_stability,
            "score_fc_capped": s_fc_use,
            "score_alloc_capped": s_alloc_use,
            "score_geom_capped": s_geom_use,
            "score_stability_capped": s_stability_use,
            **model_params_to_row("", pset),
            **avg_delta,
        }
        for k, v in e_fc.items():
            row[f"err_fc__{k}"] = v
        for k, v in e_alloc.items():
            row[f"err_alloc__{k}"] = v
        for k, v in e_geom.items():
            row[f"err_geom__{k}"] = v
        for k, v in e_stability.items():
            row[f"err_stability__{k}"] = v
        rows.append(row)

        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[*] search progress: {idx + 1}/{args.num_samples} (best score so far may improve)")

    df = pd.DataFrame(rows).sort_values("score_total").reset_index(drop=True)
    df_detail = pd.DataFrame(detail_rows)
    return df, df_detail


def plot_search_overview(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return
    top = df.head(min(200, df.shape[0])).copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), dpi=180)

    ax = axes[0]
    ax.scatter(np.arange(top.shape[0]), top["score_total"], s=22, color="#4F6B8A", alpha=0.85)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Total score")
    style_axis(ax, grid=True)

    ax = axes[1]
    ax.scatter(df["g_c_global"], df["lambda_global"], c=df["score_total"], s=18, cmap="viridis", alpha=0.85)
    ax.set_xlabel("g_c_global")
    ax.set_ylabel("lambda_global")
    style_axis(ax, grid=True)

    ax = axes[2]
    ratio = df["beta_peri"] / (df["beta_core"] + EPS)
    ax.scatter(ratio, df["score_total"], s=18, color="#8B90A8", alpha=0.8)
    ax.set_xlabel("beta_peri / beta_core")
    ax.set_ylabel("Total score")
    style_axis(ax, grid=True)

    save_variants(fig, os.path.join(out_dir, "model_v4_search_overview.png"))


def plot_best_vs_target(best_row: pd.Series, targets: dict, out_dir: str):
    pairs = [
        ("neg_frac_delta", "FC"),
        ("weak_pos_frac_delta", "FC"),
        ("strong_frac_delta", "FC"),
        ("strong_mean_delta", "FC"),
        ("mean_noise_corr_delta", "FC"),
        ("participants_ratio_delta", "Alloc"),
        ("gini_delta", "Alloc"),
        ("var_orthogonal_delta", "Geom"),
        ("var_parallel_delta", "Geom"),
        ("mean_rsm_sim_delta", "Geom"),
        ("orth_parallel_ratio_delta", "Geom"),
    ]
    labels, mvals, tvals = [], [], []
    for key, grp in pairs:
        if grp == "FC":
            tv = targets["fc"].get(key, np.nan)
        elif grp == "Alloc":
            tv = targets["alloc"].get(key, np.nan)
        else:
            tv = targets["geom"].get(key, np.nan)
        mv = float(best_row.get(key, np.nan))
        if not (np.isfinite(tv) and np.isfinite(mv)):
            continue
        labels.append(key.replace("_delta", ""))
        mvals.append(mv)
        tvals.append(tv)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.6), dpi=180)
    ax.bar(x - w / 2, tvals, width=w, color="#9CA3AF", edgecolor="#1F2937", label="Empirical target")
    ax.bar(x + w / 2, mvals, width=w, color="#4F6B8A", edgecolor="#1F2937", label="Best model")
    ax.axhline(0.0, color="#6B7280", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Coherent - Random")
    ax.legend(frameon=False)
    style_axis(ax, grid=True)
    save_variants(fig, os.path.join(out_dir, "model_v4_best_vs_target_deltas.png"))


def parse_args():
    p = argparse.ArgumentParser(description="Mechanistic v4 modeling scan (modular periphery + conditional relief + asymmetric normalization).")
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--num-samples", type=int, default=480)
    p.add_argument("--repeats-per-param", type=int, default=7)
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument("--log-every", type=int, default=25)

    p.add_argument("--n-neurons", type=int, default=180)
    p.add_argument("--core-frac", type=float, default=0.16)
    p.add_argument("--n-peri-modules", type=int, default=3)
    p.add_argument("--relief-cross-only", type=int, default=1)
    p.add_argument("--n-cond-modes", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=90)
    p.add_argument("--t-steps", type=int, default=40)
    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=30)
    p.add_argument("--stim-dim", type=int, default=6)
    p.add_argument("--weak-pos-max", type=float, default=0.10)
    p.add_argument("--strong-quantile", type=float, default=0.90)
    p.add_argument("--active-frac-thresh", type=float, default=0.15)

    p.add_argument("--w-fc", type=float, default=0.25)
    p.add_argument("--w-alloc", type=float, default=0.45)
    p.add_argument("--w-geom", type=float, default=0.30)
    p.add_argument("--w-stability", type=float, default=0.80)
    p.add_argument("--metric-w-participants-ratio", type=float, default=6.0)
    p.add_argument("--metric-w-gini", type=float, default=0.5)
    p.add_argument("--metric-w-orth-parallel-ratio", type=float, default=0.2)
    p.add_argument("--metric-w-var-orthogonal", type=float, default=0.3)
    p.add_argument("--metric-w-var-parallel", type=float, default=0.1)
    p.add_argument("--metric-w-mean-rsm", type=float, default=6.0)
    p.add_argument("--metric-w-stability-participants", type=float, default=6.0)
    p.add_argument("--metric-w-stability-rsm", type=float, default=6.0)
    p.add_argument("--metric-w-stability-neg-frac", type=float, default=1.0)
    p.add_argument("--metric-w-stability-weak-pos", type=float, default=1.0)
    p.add_argument("--stability-sign-penalty-w", type=float, default=1.5)
    p.add_argument("--stability-rel-std-cap", type=float, default=6.0)

    p.add_argument("--score-cap-fc", type=float, default=8.0)
    p.add_argument("--score-cap-alloc", type=float, default=8.0)
    p.add_argument("--score-cap-geom", type=float, default=8.0)
    p.add_argument("--score-cap-stability", type=float, default=8.0)
    p.add_argument("--score-nan-fallback", type=float, default=2.0)
    p.add_argument("--score-stability-nan-fallback", type=float, default=2.5)

    p.add_argument("--penalty-neg-frac-max", type=float, default=0.0)
    p.add_argument("--penalty-neg-frac-w", type=float, default=1.25)
    p.add_argument("--penalty-weak-pos-min", type=float, default=0.0)
    p.add_argument("--penalty-weak-pos-w", type=float, default=1.25)
    p.add_argument("--penalty-participants-max", type=float, default=0.0)
    p.add_argument("--penalty-participants-w", type=float, default=1.8)
    p.add_argument("--penalty-rsm-max", type=float, default=0.0)
    p.add_argument("--penalty-rsm-w", type=float, default=1.5)
    p.add_argument("--penalty-strong-frac-max", type=float, default=0.0)
    p.add_argument("--penalty-strong-frac-w", type=float, default=0.5)
    p.add_argument("--penalty-var-orth-min", type=float, default=0.0)
    p.add_argument("--penalty-var-orth-w", type=float, default=1.3)
    p.add_argument("--penalty-rsm-hard-max", type=float, default=0.0)
    p.add_argument("--penalty-rsm-hard-w", type=float, default=1.8)
    p.add_argument("--penalty-strong-mean-upper", type=float, default=0.020)
    p.add_argument("--penalty-strong-mean-w", type=float, default=0.8)
    p.add_argument("--bridge-neg-frac-max", type=float, default=0.0)
    p.add_argument("--bridge-weak-pos-min", type=float, default=0.0)
    p.add_argument("--bridge-participants-max", type=float, default=-0.20)
    p.add_argument("--bridge-penalty-w", type=float, default=0.9)
    p.add_argument("--mean-noise-band-low", type=float, default=0.0)
    p.add_argument("--mean-noise-band-high", type=float, default=0.02)
    p.add_argument("--mean-noise-band-w", type=float, default=0.5)

    p.add_argument("--min-amp-ratio-participants", type=float, default=0.45)
    p.add_argument("--min-amp-penalty-w-participants", type=float, default=1.8)
    p.add_argument("--max-amp-ratio-participants", type=float, default=1.30)
    p.add_argument("--max-amp-penalty-w-participants", type=float, default=0.9)
    p.add_argument("--min-amp-ratio-orth-ratio", type=float, default=0.30)
    p.add_argument("--min-amp-penalty-w-orth-ratio", type=float, default=0.0)
    p.add_argument("--max-amp-ratio-orth-ratio", type=float, default=2.00)
    p.add_argument("--max-amp-penalty-w-orth-ratio", type=float, default=0.0)
    p.add_argument("--min-amp-ratio-neg-frac", type=float, default=0.20)
    p.add_argument("--min-amp-penalty-w-neg-frac", type=float, default=0.6)
    p.add_argument("--max-amp-ratio-neg-frac", type=float, default=1.60)
    p.add_argument("--max-amp-penalty-w-neg-frac", type=float, default=0.3)
    p.add_argument("--min-amp-ratio-weak-pos", type=float, default=0.20)
    p.add_argument("--min-amp-penalty-w-weak-pos", type=float, default=0.6)
    p.add_argument("--max-amp-ratio-weak-pos", type=float, default=1.60)
    p.add_argument("--max-amp-penalty-w-weak-pos", type=float, default=0.3)
    p.add_argument("--min-amp-ratio-rsm", type=float, default=0.25)
    p.add_argument("--min-amp-penalty-w-rsm", type=float, default=0.8)
    p.add_argument("--max-amp-ratio-rsm", type=float, default=1.80)
    p.add_argument("--max-amp-penalty-w-rsm", type=float, default=0.6)

    p.add_argument("--core-prob-min", type=float, default=0.12)
    p.add_argument("--core-prob-max", type=float, default=0.55)
    p.add_argument("--peri-prob-min", type=float, default=0.005)
    p.add_argument("--peri-prob-max", type=float, default=0.16)
    p.add_argument("--w-core-min", type=float, default=0.12)
    p.add_argument("--w-core-max", type=float, default=1.80)
    p.add_argument("--w-peri-mean-min", type=float, default=-0.12)
    p.add_argument("--w-peri-mean-max", type=float, default=0.08)
    p.add_argument("--w-peri-std-min", type=float, default=0.02)
    p.add_argument("--w-peri-std-max", type=float, default=0.40)
    p.add_argument("--w-bal-min", type=float, default=0.02)
    p.add_argument("--w-bal-max", type=float, default=1.80)
    p.add_argument("--target-radius-min", type=float, default=0.62)
    p.add_argument("--target-radius-max", type=float, default=0.99)
    p.add_argument("--alpha-min", type=float, default=0.08)
    p.add_argument("--alpha-max", type=float, default=0.60)
    p.add_argument("--lambda-global-min", type=float, default=0.03)
    p.add_argument("--lambda-global-max", type=float, default=2.50)
    p.add_argument("--lambda-core-min", type=float, default=0.10)
    p.add_argument("--lambda-core-max", type=float, default=3.20)
    p.add_argument("--lambda-peri-min", type=float, default=0.02)
    p.add_argument("--lambda-peri-max", type=float, default=2.20)
    p.add_argument("--kappa-coh-relief-min", type=float, default=0.00)
    p.add_argument("--kappa-coh-relief-max", type=float, default=0.85)
    p.add_argument("--g-c-global-min", type=float, default=0.05)
    p.add_argument("--g-c-global-max", type=float, default=4.50)
    p.add_argument("--g-c-module-min", type=float, default=0.05)
    p.add_argument("--g-c-module-max", type=float, default=4.50)
    p.add_argument("--beta-core-min", type=float, default=0.05)
    p.add_argument("--beta-core-max", type=float, default=1.60)
    p.add_argument("--beta-peri-min", type=float, default=0.20)
    p.add_argument("--beta-peri-max", type=float, default=4.50)
    p.add_argument("--module-selectivity-min", type=float, default=0.00)
    p.add_argument("--module-selectivity-max", type=float, default=1.80)
    p.add_argument("--sigma-ind-min", type=float, default=0.02)
    p.add_argument("--sigma-ind-max", type=float, default=1.20)
    p.add_argument("--g-n-min", type=float, default=0.00)
    p.add_argument("--g-n-max", type=float, default=1.20)
    p.add_argument("--g-coh-extra-noise-min", type=float, default=0.00)
    p.add_argument("--g-coh-extra-noise-max", type=float, default=1.20)
    p.add_argument("--module-noise-rho-min", type=float, default=0.20)
    p.add_argument("--module-noise-rho-max", type=float, default=0.99)
    p.add_argument("--stim-scale-min", type=float, default=0.10)
    p.add_argument("--stim-scale-max", type=float, default=1.80)
    p.add_argument("--stim-rho-min", type=float, default=0.20)
    p.add_argument("--stim-rho-max", type=float, default=0.95)
    p.add_argument("--cond-rho-min", type=float, default=0.20)
    p.add_argument("--cond-rho-max", type=float, default=0.99)
    p.add_argument("--noise-rho-min", type=float, default=0.05)
    p.add_argument("--noise-rho-max", type=float, default=0.95)
    p.add_argument("--active-frac-thresh-min", type=float, default=0.05)
    p.add_argument("--active-frac-thresh-max", type=float, default=0.50)

    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def build_search_space(args) -> SearchSpace:
    return SearchSpace(
        core_prob=_checked_range("core_prob", args.core_prob_min, args.core_prob_max),
        peri_prob=_checked_range("peri_prob", args.peri_prob_min, args.peri_prob_max),
        w_core=_checked_range("w_core", args.w_core_min, args.w_core_max),
        w_peri_mean=_checked_range("w_peri_mean", args.w_peri_mean_min, args.w_peri_mean_max),
        w_peri_std=_checked_range("w_peri_std", args.w_peri_std_min, args.w_peri_std_max),
        w_bal=_checked_range("w_bal", args.w_bal_min, args.w_bal_max),
        target_radius=_checked_range("target_radius", args.target_radius_min, args.target_radius_max),
        alpha=_checked_range("alpha", args.alpha_min, args.alpha_max),
        lambda_global=_checked_range("lambda_global", args.lambda_global_min, args.lambda_global_max),
        lambda_core=_checked_range("lambda_core", args.lambda_core_min, args.lambda_core_max),
        lambda_peri=_checked_range("lambda_peri", args.lambda_peri_min, args.lambda_peri_max),
        kappa_coh_relief=_checked_range("kappa_coh_relief", args.kappa_coh_relief_min, args.kappa_coh_relief_max),
        g_c_global=_checked_range("g_c_global", args.g_c_global_min, args.g_c_global_max),
        g_c_module=_checked_range("g_c_module", args.g_c_module_min, args.g_c_module_max),
        beta_core=_checked_range("beta_core", args.beta_core_min, args.beta_core_max),
        beta_peri=_checked_range("beta_peri", args.beta_peri_min, args.beta_peri_max),
        module_selectivity=_checked_range("module_selectivity", args.module_selectivity_min, args.module_selectivity_max),
        sigma_ind=_checked_range("sigma_ind", args.sigma_ind_min, args.sigma_ind_max),
        g_n=_checked_range("g_n", args.g_n_min, args.g_n_max),
        g_coh_extra_noise=_checked_range("g_coh_extra_noise", args.g_coh_extra_noise_min, args.g_coh_extra_noise_max),
        module_noise_rho=_checked_range("module_noise_rho", args.module_noise_rho_min, args.module_noise_rho_max),
        stim_scale=_checked_range("stim_scale", args.stim_scale_min, args.stim_scale_max),
        stim_rho=_checked_range("stim_rho", args.stim_rho_min, args.stim_rho_max),
        cond_rho=_checked_range("cond_rho", args.cond_rho_min, args.cond_rho_max),
        noise_rho=_checked_range("noise_rho", args.noise_rho_min, args.noise_rho_max),
        active_frac_thresh=_checked_range("active_frac_thresh", args.active_frac_thresh_min, args.active_frac_thresh_max),
    )


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)
    search_space = build_search_space(args)

    targets, target_meta = load_empirical_targets(args.results_dir)
    print("[*] target source:", target_meta.get("target_source", "unknown"))

    n_core = max(4, int(round(float(args.core_frac) * int(args.n_neurons))))
    base = ModelParams(
        n_neurons=int(args.n_neurons),
        n_core=int(n_core),
        n_peri_modules=max(1, int(args.n_peri_modules)),
        core_prob=0.28,
        peri_prob=0.05,
        w_core=0.6,
        w_peri_mean=0.0,
        w_peri_std=0.1,
        w_bal=0.25,
        target_radius=0.90,
        alpha=0.25,
        lambda_global=0.6,
        lambda_core=1.0,
        lambda_peri=0.4,
        gamma_norm=1.0,
        kappa_coh_relief=0.3,
        relief_cross_only=bool(int(args.relief_cross_only)),
        n_cond_modes=max(1, int(args.n_cond_modes)),
        g_c_global=0.8,
        g_c_module=1.2,
        beta_core=0.8,
        beta_peri=1.2,
        module_selectivity=0.6,
        sigma_ind=0.3,
        g_n=0.15,
        g_coh_extra_noise=0.2,
        module_noise_rho=0.75,
        stim_dim=int(args.stim_dim),
        stim_scale=0.7,
        stim_rho=0.72,
        cond_rho=0.85,
        noise_rho=0.55,
        n_trials=int(args.n_trials),
        t_steps=int(args.t_steps),
        response_start=int(args.response_start),
        response_end=int(args.response_end),
        weak_pos_max=float(args.weak_pos_max),
        strong_quantile=float(args.strong_quantile),
        active_frac_thresh=float(args.active_frac_thresh),
    )

    df, df_detail = run_search(args=args, base_params=base, targets=targets, search_space=search_space)
    if df.empty:
        print("[!] No valid scan result. Stop.")
        return

    scan_csv = os.path.join(group_dir, "group_model_v4_scan_summary.csv")
    detail_csv = os.path.join(group_dir, "group_model_v4_scan_repeats_long.csv")
    df.to_csv(scan_csv, index=False)
    df_detail.to_csv(detail_csv, index=False)
    print(f"[*] Saved: {scan_csv}")
    print(f"[*] Saved: {detail_csv}")

    top_k = max(1, min(int(args.top_k), df.shape[0]))
    top = df.head(top_k).copy()
    top_csv = os.path.join(group_dir, "group_model_v4_top_params.csv")
    top.to_csv(top_csv, index=False)
    print(f"[*] Saved: {top_csv}")

    plot_search_overview(df, out_dir=group_dir)
    plot_best_vs_target(df.iloc[0], targets=targets, out_dir=group_dir)

    report_path = os.path.join(group_dir, "Group_Modeling_v4_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Group Modeling v4 Report\n\n")
        f.write("## Target Source\n\n")
        f.write(f"- source: `{target_meta.get('target_source', 'unknown')}`\n\n")
        f.write("## Search Hyperparameters\n\n")
        hp = pd.DataFrame(
            [
                {
                    "num_samples": int(args.num_samples),
                    "repeats_per_param": int(args.repeats_per_param),
                    "w_fc": float(args.w_fc),
                    "w_alloc": float(args.w_alloc),
                    "w_geom": float(args.w_geom),
                    "w_stability": float(args.w_stability),
                    "n_peri_modules": int(args.n_peri_modules),
                    "relief_cross_only": int(args.relief_cross_only),
                    "metric_w_participants_ratio": float(args.metric_w_participants_ratio),
                    "metric_w_var_orthogonal": float(args.metric_w_var_orthogonal),
                    "metric_w_mean_rsm": float(args.metric_w_mean_rsm),
                    "metric_w_orth_parallel_ratio": float(args.metric_w_orth_parallel_ratio),
                    "metric_w_stability_participants": float(args.metric_w_stability_participants),
                    "metric_w_stability_rsm": float(args.metric_w_stability_rsm),
                    "metric_w_stability_neg_frac": float(args.metric_w_stability_neg_frac),
                    "metric_w_stability_weak_pos": float(args.metric_w_stability_weak_pos),
                    "stability_sign_penalty_w": float(args.stability_sign_penalty_w),
                    "stability_rel_std_cap": float(args.stability_rel_std_cap),
                    "score_cap_fc": float(args.score_cap_fc),
                    "score_cap_alloc": float(args.score_cap_alloc),
                    "score_cap_geom": float(args.score_cap_geom),
                    "score_cap_stability": float(args.score_cap_stability),
                    "score_nan_fallback": float(args.score_nan_fallback),
                    "score_stability_nan_fallback": float(args.score_stability_nan_fallback),
                    "mean_noise_band_low": float(args.mean_noise_band_low),
                    "mean_noise_band_high": float(args.mean_noise_band_high),
                    "min_amp_ratio_participants": float(args.min_amp_ratio_participants),
                    "max_amp_ratio_participants": float(args.max_amp_ratio_participants),
                    "min_amp_ratio_orth_ratio": float(args.min_amp_ratio_orth_ratio),
                    "max_amp_ratio_orth_ratio": float(args.max_amp_ratio_orth_ratio),
                }
            ]
        )
        f.write(_to_md(hp) + "\n\n")
        f.write("## Empirical Targets (Coherent - Random)\n\n")
        f.write("### FC\n\n")
        f.write(_to_md(pd.DataFrame([targets["fc"]])) + "\n\n")
        f.write("### Allocation\n\n")
        f.write(_to_md(pd.DataFrame([targets["alloc"]])) + "\n\n")
        f.write("### Geometry\n\n")
        f.write(_to_md(pd.DataFrame([targets["geom"]])) + "\n\n")
        f.write("## Best Parameter Row\n\n")
        f.write(_to_md(df.head(1)) + "\n\n")
        f.write("## Top Parameter Rows\n\n")
        f.write(_to_md(top) + "\n\n")
    print(f"[*] Report saved: {report_path}")
    print("====== Modeling v4 modulated scan completed ======")


if __name__ == "__main__":
    main()
