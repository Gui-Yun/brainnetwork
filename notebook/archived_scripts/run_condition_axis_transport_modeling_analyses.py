import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from brainnetwork import load_data, preprocess_spike_data
from run_flat_recruitment_modeling_analyses_v2 import (
    CONDITIONS,
    COND_COLORS,
    COND_MAP,
    DEFAULT_BASE_DIR,
    DEFAULT_MOUSE_IDS,
    DEFAULT_RESULTS_DIR,
    EPS,
    GROUP_DIR_NAME,
    _to_md,
    assign_patches,
    entropy_norm,
    ensure_dir,
    gini_index,
    latent_fisher_metrics,
    normalize_condition_column,
    orthogonal_expansion_metrics,
    pairwise_distance_matrix,
    parse_float_list,
    parse_str_list,
    safe_div,
    save_variants,
    style_axis,
    weak_edge_fisher_proxy,
    weighted_pair_span,
    weighted_radius,
    wilcoxon_zero,
)


VARIANT_FLAGS = {
    "axis_only": (False, False, False),
    "axis_core": (True, False, False),
    "axis_spatial": (False, True, False),
    "axis_weak": (False, False, True),
    "axis_spatial_core": (True, True, False),
    "axis_spatial_weak": (False, True, True),
    "full": (True, True, True),
}
VARIANT_ORDER = [
    "axis_only",
    "axis_core",
    "axis_spatial",
    "axis_weak",
    "axis_spatial_core",
    "axis_spatial_weak",
    "full",
]


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


def model_variant_flags(variant: str) -> tuple[bool, bool, bool]:
    key = str(variant).strip().lower()
    if key not in VARIANT_FLAGS:
        raise ValueError(f"Unknown model variant: {variant}")
    return VARIANT_FLAGS[key]


def normalize_model_variant_column(df: pd.DataFrame, default_variant: str = "axis_only") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "model_variant" not in out.columns:
        out["model_variant"] = default_variant
    out["model_variant"] = out["model_variant"].fillna(default_variant).astype(str)
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size != y.size or x.size == 0:
        return np.nan
    xn = float(np.linalg.norm(x))
    yn = float(np.linalg.norm(y))
    if xn <= EPS or yn <= EPS:
        return np.nan
    return float(np.dot(x, y) / (xn * yn + EPS))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    x = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(q, dtype=float).reshape(-1)
    if x.size != y.size or x.size == 0:
        return np.nan
    x = np.clip(x, EPS, None)
    y = np.clip(y, EPS, None)
    x = x / (np.sum(x) + EPS)
    y = y / (np.sum(y) + EPS)
    return float(np.sum(x * np.log((x + EPS) / (y + EPS))))


def template_mean(Q: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if np.sum(m) < 2:
        return None
    q = np.mean(np.asarray(Q, dtype=float)[m], axis=0)
    q = np.clip(q, 0.0, None)
    s = np.sum(q)
    if s <= EPS:
        return None
    return q / s


def estimate_axis_components(Q: np.ndarray, labels: np.ndarray, pos_xy: np.ndarray) -> dict | None:
    q_r = template_mean(Q, labels == 3)
    q_d = template_mean(Q, labels == 1)
    q_c = template_mean(Q, labels == 2)
    if q_r is None or q_d is None or q_c is None:
        return None

    q_coh = 0.5 * (q_d + q_c)
    q_coh = q_coh / (np.sum(q_coh) + EPS)

    axis = np.log(q_coh + EPS) - np.log(q_r + EPS)
    axis = axis - float(np.sum(q_r * axis))

    center_r = q_r @ pos_xy
    d = np.sqrt(np.sum((pos_xy - center_r[None, :]) ** 2, axis=1))
    d = d / (float(np.max(d)) + EPS)

    return {
        "q_random": q_r,
        "q_divergent": q_d,
        "q_convergent": q_c,
        "q_coherent": q_coh,
        "axis": axis,
        "dist_to_random_center": d,
        "random_center_x": float(center_r[0]),
        "random_center_y": float(center_r[1]),
    }


def build_core_mask(q_ref: np.ndarray, top_frac: float) -> np.ndarray:
    q = np.asarray(q_ref, dtype=float).reshape(-1)
    n = q.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.ceil(float(np.clip(top_frac, 1e-3, 1.0)) * n)))
    idx = np.argsort(q)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def build_patch_longrange_kernel(
    patch_meta: pd.DataFrame,
    d0_quantile: float = 0.6,
    sigma_scale: float = 0.35,
) -> np.ndarray:
    if patch_meta is None or patch_meta.empty or patch_meta.shape[0] < 2:
        return np.zeros((0, 0), dtype=float)
    coords = patch_meta[["x_idx", "y_idx"]].to_numpy(dtype=float)
    dxy = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(dxy * dxy, axis=2))
    np.fill_diagonal(D, 0.0)
    iu = np.triu_indices(D.shape[0], k=1)
    vals = D[iu]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.zeros_like(D)

    q = float(np.clip(d0_quantile, 0.1, 0.95))
    d0 = float(np.quantile(vals, q))
    dmax = float(np.max(vals))
    sigma = max(float(sigma_scale) * dmax, EPS)

    W = np.exp(-(D * D) / (2.0 * sigma * sigma))
    W[D <= d0] = 0.0
    np.fill_diagonal(W, 0.0)

    row_sum = np.sum(W, axis=1, keepdims=True)
    W = np.where(row_sum > EPS, W / (row_sum + EPS), 0.0)
    return W


def apply_weak_coordination_neuron_via_patch(
    X: np.ndarray,
    P_patch: np.ndarray,
    patch_local: np.ndarray,
    patch_meta: pd.DataFrame,
    K_patch: np.ndarray,
    lam: float,
) -> np.ndarray:
    X0 = np.asarray(X, dtype=float)
    if X0.ndim != 2 or X0.shape[1] < 2:
        return X0
    if P_patch.shape[1] < 2 or K_patch.shape[0] != P_patch.shape[1]:
        return X0
    lam = float(max(lam, 0.0))
    if lam <= EPS:
        return X0

    F = X0 @ P_patch
    F_long = F @ K_patch

    back = np.zeros_like(X0, dtype=float)
    idx = np.where(np.asarray(patch_local, dtype=int) >= 0)[0]
    if idx.size > 0:
        back[:, idx] = F_long[:, patch_local[idx]]

    X_new = np.clip(X0 + lam * back, 0.0, None)
    r0 = np.sum(X0, axis=1, keepdims=True)
    r1 = np.sum(X_new, axis=1, keepdims=True)
    valid = (r0[:, 0] > EPS) & (r1[:, 0] > EPS)
    valid_idx = np.where(valid)[0]
    if valid_idx.size > 0:
        X_new[valid_idx] = X_new[valid_idx] * (r0[valid_idx] / (r1[valid_idx] + EPS))
    return X_new


def transform_random_trials(
    X: np.ndarray,
    labels: np.ndarray,
    axis: np.ndarray,
    spatial_d: np.ndarray,
    core_mask: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    lam: float,
    use_core: bool,
    use_spatial: bool,
    use_weak: bool,
    P_patch: np.ndarray,
    patch_local: np.ndarray,
    patch_meta: pd.DataFrame,
    K_patch: np.ndarray,
) -> np.ndarray:
    X_new = np.asarray(X, dtype=float).copy()
    m_rand = np.asarray(labels) == 3
    if np.sum(m_rand) < 1:
        return X_new

    X_rand = X_new[m_rand]
    R_rand = np.sum(X_rand, axis=1, keepdims=True)
    Q_rand = row_normalize_rows(X_rand)

    logits = float(alpha) * np.asarray(axis, dtype=float)
    if use_spatial:
        logits = logits + float(beta) * np.asarray(spatial_d, dtype=float)
    weights = np.exp(np.clip(logits, -20.0, 20.0))
    Q_star = row_normalize_rows(Q_rand * weights[None, :])

    if use_core:
        rho = float(np.clip(rho, 0.0, 1.0))
        cm = np.asarray(core_mask, dtype=float)[None, :]
        core_mass = np.sum(Q_rand * cm, axis=1)
        Q_core = row_normalize_rows(Q_rand * cm)
        zero_core = core_mass <= EPS
        if np.any(zero_core):
            Q_core[zero_core] = Q_rand[zero_core]
        Q_mix = row_normalize_rows((1.0 - rho) * Q_star + rho * Q_core)
    else:
        Q_mix = Q_star

    X_rand_new = Q_mix * R_rand

    if use_weak:
        X_rand_new = apply_weak_coordination_neuron_via_patch(
            X=X_rand_new,
            P_patch=P_patch,
            patch_local=patch_local,
            patch_meta=patch_meta,
            K_patch=K_patch,
            lam=float(lam),
        )

    X_new[m_rand] = X_rand_new
    return X_new


def template_proximity_metrics(q_rand_mean: np.ndarray, q_r: np.ndarray, q_coh: np.ndarray) -> dict:
    return {
        "cosine_random_to_random_template": cosine_similarity(q_rand_mean, q_r),
        "cosine_random_to_coherent_template": cosine_similarity(q_rand_mean, q_coh),
        "kl_random_to_random_template": kl_divergence(q_rand_mean, q_r),
        "kl_random_to_coherent_template": kl_divergence(q_rand_mean, q_coh),
    }


def slope_rows_alpha(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    need = ["mouse_id", "alpha", value_col] + list(group_cols)
    miss = [c for c in need if c not in df.columns]
    if miss:
        return pd.DataFrame()

    rows = []
    use = df[need].dropna()
    if use.empty:
        return pd.DataFrame()
    for key, sub in use.groupby(["mouse_id"] + group_cols, observed=False):
        x = sub["alpha"].to_numpy(dtype=float)
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


def mouse_plot_random_condition_metrics(df_cond: pd.DataFrame, out_path: str):
    sub = df_cond[df_cond["Condition"].astype(str) == "Random"].copy()
    if sub.empty:
        return
    metrics = [
        ("flatness_entropy_norm", "Random entropy (norm)"),
        ("effective_participation_prop", "Random participation"),
        ("integration_radius", "Random integration radius"),
        ("integration_pairwise_span", "Random pairwise span"),
    ]
    variants = [v for v in VARIANT_ORDER if v in set(sub["model_variant"].astype(str).tolist())]
    palette = sns.color_palette("tab10", n_colors=max(3, len(variants)))
    color_map = {v: palette[i] for i, v in enumerate(variants)}

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        for variant in variants:
            ss = (
                sub[sub["model_variant"].astype(str) == variant][["alpha", metric]]
                .dropna()
                .groupby("alpha", observed=False)[metric]
                .mean()
                .reset_index()
                .sort_values("alpha")
            )
            if ss.empty:
                continue
            ax.plot(ss["alpha"], ss[metric], marker="o", lw=2, color=color_map[variant], label=variant)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def mouse_plot_latent_metrics(df_latent: pd.DataFrame, out_path: str):
    if df_latent.empty:
        return
    metrics = [
        ("fisher_coherent_vs_random", "Fisher (coherent vs random)"),
        ("weak_edge_fisher_coherent_vs_random", "Weak-edge Fisher (coh vs rand)"),
        ("cosine_random_to_coherent_template", "Cos(Random, Coherent template)"),
        ("kl_random_to_coherent_template", "KL(Random -> Coherent template)"),
    ]
    variants = [v for v in VARIANT_ORDER if v in set(df_latent["model_variant"].astype(str).tolist())]
    palette = sns.color_palette("tab10", n_colors=max(3, len(variants)))
    color_map = {v: palette[i] for i, v in enumerate(variants)}

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        for variant in variants:
            ss = (
                df_latent[df_latent["model_variant"].astype(str) == variant][["alpha", metric]]
                .dropna()
                .groupby("alpha", observed=False)[metric]
                .mean()
                .reset_index()
                .sort_values("alpha")
            )
            if ss.empty:
                continue
            ax.plot(ss["alpha"], ss[metric], marker="o", lw=2, color=color_map[variant], label=variant)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
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
    data_path = os.path.join(args.base_dir, mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path, data_type="spikes")
    segments_spi, labels_spi, neuron_pos_spi = preprocess_spike_data(
        neuron_data, neuron_pos, start_edges, stimulus_data, extract_rr=True
    )
    labels = np.asarray(labels_spi, dtype=int)
    classes = [c for c in [1, 2, 3] if c in set(labels.tolist())]
    if 3 not in classes or 1 not in classes or 2 not in classes:
        raise RuntimeError(f"{mouse_id}: missing one or more required classes 1/2/3 in labels.")

    segments = np.asarray(segments_spi, dtype=float)
    response_window = slice(args.response_start, args.response_end)
    baseline_window = slice(args.baseline_start, args.baseline_end)
    X_resp = np.nanmean(segments[:, :, response_window], axis=2)
    X_base = np.nanmean(segments[:, :, baseline_window], axis=2)
    X_delta = np.clip(np.asarray(X_resp - X_base, dtype=float), 0.0, None)

    pos_xy = np.asarray(neuron_pos_spi, dtype=float).T
    n_neurons = int(pos_xy.shape[0])
    D_neuron = pairwise_distance_matrix(pos_xy)

    Q0 = row_normalize_rows(X_delta)
    axis_parts = estimate_axis_components(Q0, labels, pos_xy)
    if axis_parts is None:
        raise RuntimeError(f"{mouse_id}: failed to estimate axis templates.")
    q_r = axis_parts["q_random"]
    q_coh = axis_parts["q_coherent"]
    axis_vec = axis_parts["axis"]
    spatial_d = axis_parts["dist_to_random_center"]

    core_mask = build_core_mask(q_r, top_frac=float(args.core_top_frac))

    patch_meta, P_patch, patch_local = assign_patches(
        pos_xy=pos_xy,
        grid_size=float(args.grid_size),
        min_patch_neurons=int(args.min_patch_neurons),
    )
    n_patches = int(P_patch.shape[1])
    K_patch = build_patch_longrange_kernel(
        patch_meta=patch_meta,
        d0_quantile=float(args.longrange_d0_quantile),
        sigma_scale=float(args.longrange_sigma_scale),
    )

    cond_rows = []
    latent_rows = []
    dropout_rows = []
    rng = np.random.default_rng(seed_i)

    for variant in args.model_variants:
        use_core, use_spatial, use_weak = model_variant_flags(variant)
        beta = float(args.spatial_beta) if use_spatial else 0.0
        rho = float(args.core_rho) if use_core else 0.0
        lam = float(args.weak_lambda) if use_weak else 0.0

        for alpha in args.alpha_values:
            alpha_val = float(alpha)
            X_mod = transform_random_trials(
                X=X_delta,
                labels=labels,
                axis=axis_vec,
                spatial_d=spatial_d,
                core_mask=core_mask,
                alpha=alpha_val,
                beta=beta,
                rho=rho,
                lam=lam,
                use_core=use_core,
                use_spatial=use_spatial,
                use_weak=use_weak,
                P_patch=P_patch,
                patch_local=patch_local,
                patch_meta=patch_meta,
                K_patch=K_patch,
            )

            R_mod = np.sum(X_mod, axis=1)
            Q_mod = row_normalize_rows(X_mod)

            if n_patches > 0:
                F_mod = X_mod @ P_patch
            else:
                F_mod = np.zeros((X_mod.shape[0], 0), dtype=float)

            latent = latent_fisher_metrics(
                F=F_mod,
                labels=labels,
                rng=rng,
                repeats=int(args.latent_repeats),
                reg=float(args.fisher_reg),
            )
            weak_edge_fisher, full_edge_fisher = weak_edge_fisher_proxy(
                F=F_mod,
                labels=labels,
                rng=rng,
                repeats=max(2, int(args.latent_repeats // 2)),
                reg=float(args.fisher_reg),
                weak_quantile=float(args.weak_edge_quantile),
            )

            m_rand = labels == 3
            if np.sum(m_rand) >= 2:
                q_rand_mean = np.mean(Q_mod[m_rand], axis=0)
                q_rand_mean = np.clip(q_rand_mean, 0.0, None)
                q_rand_mean = q_rand_mean / (np.sum(q_rand_mean) + EPS)
                prox = template_proximity_metrics(q_rand_mean, q_r, q_coh)
            else:
                prox = {
                    "cosine_random_to_random_template": np.nan,
                    "cosine_random_to_coherent_template": np.nan,
                    "kl_random_to_random_template": np.nan,
                    "kl_random_to_coherent_template": np.nan,
                }

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
                    F_drop = F_mod[:, keep_cols]
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
                            "alpha": alpha_val,
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
                    "alpha": alpha_val,
                    "n_patches": int(n_patches),
                    "fisher_coherent_vs_random": latent["fisher_coherent_vs_random"],
                    "fisher_divergent_vs_convergent": latent["fisher_divergent_vs_convergent"],
                    "weak_edge_fisher_coherent_vs_random": weak_edge_fisher,
                    "full_edge_fisher_coherent_vs_random": full_edge_fisher,
                    "robustness_loss_coherent_vs_random": drop_loss_cr,
                    "robustness_loss_divergent_vs_convergent": drop_loss_dc,
                    "cosine_random_to_random_template": prox["cosine_random_to_random_template"],
                    "cosine_random_to_coherent_template": prox["cosine_random_to_coherent_template"],
                    "kl_random_to_random_template": prox["kl_random_to_random_template"],
                    "kl_random_to_coherent_template": prox["kl_random_to_coherent_template"],
                    "core_top_frac": float(args.core_top_frac) if use_core else 0.0,
                    "core_rho": rho,
                    "spatial_beta": beta,
                    "weak_lambda": lam,
                    "axis_norm_l2": float(np.linalg.norm(axis_vec)),
                }
            )

            for class_id in classes:
                cond = COND_MAP.get(int(class_id), str(class_id))
                m = labels == int(class_id)
                if np.sum(m) < 2:
                    continue

                q_cond = Q_mod[m]
                r_cond = R_mod[m]
                x_cond = X_mod[m]
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
                span = weighted_pair_span(q_mean, D_neuron)
                mean_total = float(np.mean(r_cond[valid]))
                geo = orthogonal_expansion_metrics(x_cond[valid])

                row = {
                    "mouse_id": mouse_id,
                    "model_variant": variant,
                    "Class_ID": int(class_id),
                    "Condition": cond,
                    "alpha": alpha_val,
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
                    "core_rho": rho,
                    "spatial_beta": beta,
                    "weak_lambda": lam,
                }
                if cond == "Random":
                    row.update(prox)
                else:
                    row.update(
                        {
                            "cosine_random_to_random_template": np.nan,
                            "cosine_random_to_coherent_template": np.nan,
                            "kl_random_to_random_template": np.nan,
                            "kl_random_to_coherent_template": np.nan,
                        }
                    )
                cond_rows.append(row)

    df_cond = normalize_model_variant_column(normalize_condition_column(pd.DataFrame(cond_rows)))
    df_latent = normalize_model_variant_column(pd.DataFrame(latent_rows))
    df_dropout = normalize_model_variant_column(pd.DataFrame(dropout_rows))

    if not df_cond.empty:
        p = os.path.join(data_out, "axis_transport_condition_metrics.csv")
        df_cond.to_csv(p, index=False)
        print(f"[*] Saved: {p}")
        mouse_plot_random_condition_metrics(
            df_cond=df_cond,
            out_path=os.path.join(fig_out, "axis_transport_random_condition_curves.png"),
        )
    if not df_latent.empty:
        p = os.path.join(data_out, "axis_transport_latent_metrics.csv")
        df_latent.to_csv(p, index=False)
        print(f"[*] Saved: {p}")
        mouse_plot_latent_metrics(
            df_latent=df_latent,
            out_path=os.path.join(fig_out, "axis_transport_latent_curves.png"),
        )
    if not df_dropout.empty:
        p = os.path.join(data_out, "axis_transport_dropout_patch_detail.csv")
        df_dropout.to_csv(p, index=False)
        print(f"[*] Saved: {p}")

    return {
        "condition_df": df_cond,
        "latent_df": df_latent,
        "dropout_df": df_dropout,
    }


def group_plot_random_condition_metrics(df_cond: pd.DataFrame, out_path: str):
    sub = df_cond[df_cond["Condition"].astype(str) == "Random"].copy()
    if sub.empty:
        return
    metrics = [
        ("flatness_entropy_norm", "Random entropy (norm)"),
        ("effective_participation_prop", "Random participation"),
        ("integration_radius", "Random integration radius"),
        ("integration_pairwise_span", "Random pairwise span"),
    ]
    variants = [v for v in VARIANT_ORDER if v in set(sub["model_variant"].astype(str).tolist())]
    palette = sns.color_palette("tab10", n_colors=max(3, len(variants)))
    color_map = {v: palette[i] for i, v in enumerate(variants)}

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        by_mouse = (
            sub[["mouse_id", "model_variant", "alpha", metric]]
            .dropna()
            .groupby(["mouse_id", "model_variant", "alpha"], observed=False)[metric]
            .mean()
            .reset_index()
        )
        for variant in variants:
            ss = (
                by_mouse[by_mouse["model_variant"].astype(str) == variant]
                .groupby("alpha", observed=False)[metric]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("alpha")
            )
            if ss.empty:
                continue
            x = ss["alpha"].to_numpy(dtype=float)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=color_map[variant], label=variant)
            ax.fill_between(x, y - se, y + se, color=color_map[variant], alpha=0.16, linewidth=0)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def group_plot_latent_metrics(df_latent: pd.DataFrame, out_path: str):
    if df_latent.empty:
        return
    metrics = [
        ("fisher_coherent_vs_random", "Fisher (coherent vs random)"),
        ("weak_edge_fisher_coherent_vs_random", "Weak-edge Fisher"),
        ("cosine_random_to_coherent_template", "Cos(Random, Coherent template)"),
        ("kl_random_to_coherent_template", "KL(Random -> Coherent template)"),
    ]
    variants = [v for v in VARIANT_ORDER if v in set(df_latent["model_variant"].astype(str).tolist())]
    palette = sns.color_palette("tab10", n_colors=max(3, len(variants)))
    color_map = {v: palette[i] for i, v in enumerate(variants)}

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, metrics):
        by_mouse = (
            df_latent[["mouse_id", "model_variant", "alpha", metric]]
            .dropna()
            .groupby(["mouse_id", "model_variant", "alpha"], observed=False)[metric]
            .mean()
            .reset_index()
        )
        for variant in variants:
            ss = (
                by_mouse[by_mouse["model_variant"].astype(str) == variant]
                .groupby("alpha", observed=False)[metric]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("alpha")
            )
            if ss.empty:
                continue
            x = ss["alpha"].to_numpy(dtype=float)
            y = ss["mean"].to_numpy(dtype=float)
            se = ss["sem"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=2, color=color_map[variant], label=variant)
            ax.fill_between(x, y - se, y + se, color=color_map[variant], alpha=0.16, linewidth=0)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        style_axis(ax, grid=True)
    axes[0].legend(frameon=False, title="")
    save_variants(fig, out_path)


def run_group(df_cond: pd.DataFrame, df_latent: pd.DataFrame, df_dropout: pd.DataFrame, args):
    default_variant = "axis_only"
    if hasattr(args, "model_variants") and args.model_variants:
        default_variant = str(args.model_variants[0])

    df_cond = normalize_model_variant_column(normalize_condition_column(df_cond), default_variant=default_variant)
    df_latent = normalize_model_variant_column(df_latent, default_variant=default_variant)
    df_dropout = normalize_model_variant_column(df_dropout, default_variant=default_variant)

    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)

    if not df_cond.empty:
        df_cond.to_csv(os.path.join(group_dir, "group_axis_transport_condition_metrics.csv"), index=False)
    if not df_latent.empty:
        df_latent.to_csv(os.path.join(group_dir, "group_axis_transport_latent_metrics.csv"), index=False)
    if not df_dropout.empty:
        df_dropout.to_csv(os.path.join(group_dir, "group_axis_transport_dropout_patch_detail.csv"), index=False)

    cond_metrics = [
        "flatness_entropy_norm",
        "effective_participation_prop",
        "integration_radius",
        "integration_pairwise_span",
        "mean_cosine_similarity",
        "orthogonal_variance",
        "orthogonal_parallel_ratio",
    ]
    latent_metrics = [
        "fisher_coherent_vs_random",
        "fisher_divergent_vs_convergent",
        "weak_edge_fisher_coherent_vs_random",
        "full_edge_fisher_coherent_vs_random",
        "robustness_loss_coherent_vs_random",
        "robustness_loss_divergent_vs_convergent",
        "cosine_random_to_random_template",
        "cosine_random_to_coherent_template",
        "kl_random_to_random_template",
        "kl_random_to_coherent_template",
    ]

    slope_rows_all = []
    for metric in cond_metrics:
        s = slope_rows_alpha(df_cond, value_col=metric, group_cols=["model_variant", "Condition"])
        if not s.empty:
            slope_rows_all.append(s)
    for metric in latent_metrics:
        s = slope_rows_alpha(df_latent, value_col=metric, group_cols=["model_variant"])
        if not s.empty:
            slope_rows_all.append(s)

    df_slopes = pd.concat(slope_rows_all, ignore_index=True) if slope_rows_all else pd.DataFrame()
    if not df_slopes.empty:
        df_slopes.to_csv(os.path.join(group_dir, "group_axis_transport_slopes_by_mouse.csv"), index=False)

    stat_rows = []
    if not df_slopes.empty:
        slope_valid = df_slopes.dropna(subset=["slope"]).copy()
        cond_sub = slope_valid[slope_valid["Condition"].notna()].copy() if "Condition" in slope_valid.columns else pd.DataFrame()
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
        df_stats.to_csv(os.path.join(group_dir, "group_axis_transport_slope_stats.csv"), index=False)

    if not df_cond.empty:
        group_plot_random_condition_metrics(
            df_cond=df_cond,
            out_path=os.path.join(group_dir, "group_axis_transport_random_condition_curves.png"),
        )
    if not df_latent.empty:
        group_plot_latent_metrics(
            df_latent=df_latent,
            out_path=os.path.join(group_dir, "group_axis_transport_latent_curves.png"),
        )

    md_path = os.path.join(group_dir, "Group_AxisTransport_Modeling_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group Axis-Transport Modeling Report\n\n")
        variants = []
        for dfi in [df_cond, df_latent]:
            if dfi is not None and not dfi.empty and "model_variant" in dfi.columns:
                variants.extend(dfi["model_variant"].dropna().astype(str).unique().tolist())
        variants = sorted(set(variants))
        f.write("## Model variants\n\n")
        f.write((", ".join(variants) if variants else "_none_") + "\n\n")
        f.write("## Alpha slope tests (Wilcoxon against 0)\n\n")
        f.write(_to_md(df_stats) + "\n\n")
    print(f"[*] Group report saved: {md_path}")


def load_mouse_outputs_from_disk(mouse_id: str, results_dir: str) -> dict:
    data_out = os.path.join(results_dir, mouse_id, "data")
    cond_path = os.path.join(data_out, "axis_transport_condition_metrics.csv")
    latent_path = os.path.join(data_out, "axis_transport_latent_metrics.csv")
    dropout_path = os.path.join(data_out, "axis_transport_dropout_patch_detail.csv")

    df_cond = pd.read_csv(cond_path) if os.path.isfile(cond_path) else pd.DataFrame()
    df_latent = pd.read_csv(latent_path) if os.path.isfile(latent_path) else pd.DataFrame()
    df_dropout = pd.read_csv(dropout_path) if os.path.isfile(dropout_path) else pd.DataFrame()

    df_cond = normalize_model_variant_column(normalize_condition_column(df_cond), default_variant="axis_only")
    df_latent = normalize_model_variant_column(df_latent, default_variant="axis_only")
    df_dropout = normalize_model_variant_column(df_dropout, default_variant="axis_only")
    return {
        "condition_df": df_cond,
        "latent_df": df_latent,
        "dropout_df": df_dropout,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Condition-level axis-transport modeling: transform Random trials toward a coherent recruitment axis, "
            "with optional core retention, spatial complementarity, and weak long-range coordination."
        )
    )
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)

    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=13)
    p.add_argument("--baseline-start", type=int, default=0)
    p.add_argument("--baseline-end", type=int, default=10)

    p.add_argument("--alpha-values", type=str, default="-1.0,-0.5,0.0,0.5,1.0,1.5,2.0")
    p.add_argument(
        "--model-variants",
        type=str,
        default="axis_only,axis_core,axis_spatial,axis_weak,axis_spatial_core,axis_spatial_weak,full",
    )
    p.add_argument("--core-top-frac", type=float, default=0.15)
    p.add_argument("--core-rho", type=float, default=0.65)
    p.add_argument("--spatial-beta", type=float, default=1.5)
    p.add_argument("--weak-lambda", type=float, default=0.15)
    p.add_argument("--longrange-d0-quantile", type=float, default=0.6)
    p.add_argument("--longrange-sigma-scale", type=float, default=0.35)

    p.add_argument("--grid-size", type=float, default=160.0)
    p.add_argument("--min-patch-neurons", type=int, default=20)
    p.add_argument("--min-neurons-after-drop", type=int, default=10)
    p.add_argument("--latent-repeats", type=int, default=6)
    p.add_argument("--fisher-reg", type=float, default=1e-3)
    p.add_argument("--weak-edge-quantile", type=float, default=0.2)

    p.add_argument("--group-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260410)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    args.alpha_values = sorted(parse_float_list(args.alpha_values))
    raw_variants = parse_str_list(args.model_variants)
    if not raw_variants:
        raw_variants = ["axis_only"]
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
        for mouse in args.mice:
            out = load_mouse_outputs_from_disk(mouse, args.results_dir)
            if out["condition_df"] is not None and not out["condition_df"].empty:
                all_cond.append(out["condition_df"])
            else:
                print(f"[!] Missing condition CSV for {mouse}: results/{mouse}/data/axis_transport_condition_metrics.csv")
            if out["latent_df"] is not None and not out["latent_df"].empty:
                all_lat.append(out["latent_df"])
            else:
                print(f"[!] Missing latent CSV for {mouse}: results/{mouse}/data/axis_transport_latent_metrics.csv")
            if out["dropout_df"] is not None and not out["dropout_df"].empty:
                all_drop.append(out["dropout_df"])

        df_cond = pd.concat(all_cond, ignore_index=True) if all_cond else pd.DataFrame()
        df_lat = pd.concat(all_lat, ignore_index=True) if all_lat else pd.DataFrame()
        df_drop = pd.concat(all_drop, ignore_index=True) if all_drop else pd.DataFrame()
        if df_cond.empty and df_lat.empty:
            print("[!] No existing outputs found. Stop.")
            return
        run_group(df_cond, df_lat, df_drop, args)
        print("====== Axis-transport modeling (group-only) completed ======")
        return

    all_cond = []
    all_lat = []
    all_drop = []
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
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")

    df_cond = pd.concat(all_cond, ignore_index=True) if all_cond else pd.DataFrame()
    df_lat = pd.concat(all_lat, ignore_index=True) if all_lat else pd.DataFrame()
    df_drop = pd.concat(all_drop, ignore_index=True) if all_drop else pd.DataFrame()
    if df_cond.empty and df_lat.empty:
        print("[!] No valid outputs generated. Stop.")
        return

    run_group(df_cond, df_lat, df_drop, args)
    print("====== Axis-transport modeling completed ======")


if __name__ == "__main__":
    main()
