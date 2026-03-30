import argparse
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from brainnetwork import load_data, preprocess_spike_data, rr_selection_class


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
DEFAULT_RESULTS_DIR = "./result"
GROUP_DIR_NAME = "group_summary"
COND_MAP = {1: "Divergent", 2: "Convergent", 3: "Random"}
LABEL_NAMES = {1: "Divergent", 2: "Convergent", 3: "Random"}
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


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)


def _balanced_indices(y_vec, rng):
    classes = np.sort(np.unique(y_vec))
    min_count = min(int((y_vec == c).sum()) for c in classes)
    keep = []
    for c in classes:
        idx_c = np.where(y_vec == c)[0]
        keep.extend(rng.choice(idx_c, size=min_count, replace=False).tolist())
    return np.asarray(sorted(keep), dtype=int)


def _trial_fc_upper_triangle(trial_neuron_time):
    cmat = np.corrcoef(trial_neuron_time)
    cmat = np.nan_to_num(cmat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(cmat, 1.0)
    tri = np.triu_indices(cmat.shape[0], k=1)
    return cmat[tri]


def _fc_model(n_components, random_state=123):
    return Pipeline(
        [
            ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
            ("svc", SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale")),
        ]
    )


def _fc_cv_with_pred(X_mat, y_vec, n_components, n_splits=3, random_state=123):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.empty_like(y_vec)
    fold_acc = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_mat, y_vec)):
        model = _fc_model(n_components=n_components, random_state=random_state + fold_i)
        model.fit(X_mat[tr_idx], y_vec[tr_idx])
        pred = model.predict(X_mat[te_idx])
        y_pred[te_idx] = pred
        fold_acc.append(float((pred == y_vec[te_idx]).mean()))
    fold_acc = np.asarray(fold_acc, dtype=float)
    return float(fold_acc.mean()), float(fold_acc.std(ddof=1)), y_pred


def _decoder_eval_with_shuffle(X_mat, y_vec, n_splits, max_components, shuffle_repeats, random_state):
    n_samples, n_features = X_mat.shape
    n_components = int(min(max_components, n_samples - 1, n_features - 1))
    if n_components < 2:
        return {
            "n_features": int(n_features),
            "n_components": int(n_components),
            "accuracy_mean": np.nan,
            "accuracy_std": np.nan,
            "shuffle_accuracy_mean": np.nan,
            "shuffle_accuracy_std": np.nan,
            "accuracy_minus_shuffle": np.nan,
            "shuffle_scores": np.asarray([], dtype=float),
        }

    acc, acc_std, _ = _fc_cv_with_pred(
        X_mat,
        y_vec,
        n_components=n_components,
        n_splits=n_splits,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state + 1000)
    shuf_scores = []
    for rep in range(shuffle_repeats):
        y_shuf = rng.permutation(y_vec)
        rep_acc, _, _ = _fc_cv_with_pred(
            X_mat,
            y_shuf,
            n_components=n_components,
            n_splits=n_splits,
            random_state=random_state + 100 + rep,
        )
        shuf_scores.append(rep_acc)
    shuf_scores = np.asarray(shuf_scores, dtype=float)
    return {
        "n_features": int(n_features),
        "n_components": int(n_components),
        "accuracy_mean": float(acc),
        "accuracy_std": float(acc_std),
        "shuffle_accuracy_mean": float(np.nanmean(shuf_scores)),
        "shuffle_accuracy_std": float(np.nanstd(shuf_scores, ddof=1)),
        "accuracy_minus_shuffle": float(acc - np.nanmean(shuf_scores)),
        "shuffle_scores": shuf_scores,
    }


def _fc_cv_acc_edge_mask(
    X_mat,
    y_vec,
    n_components,
    drop_edge_idx=None,
    n_splits=3,
    random_state=123,
):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_acc = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_mat, y_vec)):
        X_tr = X_mat[tr_idx].copy()
        X_te = X_mat[te_idx].copy()
        if drop_edge_idx is not None and len(drop_edge_idx) > 0:
            X_tr[:, drop_edge_idx] = 0.0
            X_te[:, drop_edge_idx] = 0.0
        pipe = Pipeline(
            [
                ("svd", TruncatedSVD(n_components=n_components, random_state=random_state + 101 + fold_i)),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=5000,
                        random_state=random_state + 201 + fold_i,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_vec[tr_idx])
        pred = pipe.predict(X_te)
        fold_acc.append(float((pred == y_vec[te_idx]).mean()))
    fold_acc = np.asarray(fold_acc, dtype=float)
    return float(fold_acc.mean()), float(fold_acc.std(ddof=1))


def _linear_cv_acc(Z_mat, y_vec, n_splits=3, random_state=123, drop_components=None):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_acc = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(Z_mat, y_vec)):
        X_tr = Z_mat[tr_idx].copy()
        X_te = Z_mat[te_idx].copy()
        if drop_components is not None and len(drop_components) > 0:
            X_tr[:, drop_components] = 0.0
            X_te[:, drop_components] = 0.0
        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=random_state + fold_i,
        )
        model.fit(X_tr, y_vec[tr_idx])
        pred = model.predict(X_te)
        fold_acc.append(float((pred == y_vec[te_idx]).mean()))
    fold_acc = np.asarray(fold_acc, dtype=float)
    return float(fold_acc.mean()), float(fold_acc.std(ddof=1))


def _stratified_subsample_idx(y_vec, ratio, rng_obj):
    keep = []
    for c in np.sort(np.unique(y_vec)):
        idx_c = np.where(y_vec == c)[0]
        n_keep_c = max(1, int(np.floor(idx_c.size * ratio)))
        keep.extend(rng_obj.choice(idx_c, size=n_keep_c, replace=False).tolist())
    return np.asarray(sorted(keep), dtype=int)


def _quantile_bins(values, n_bins=10):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=int)
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, q)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    if np.allclose(edges, edges[0]):
        return np.zeros(values.size, dtype=int)
    bins = np.digitize(values, edges[1:-1], right=True)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins.astype(int)


def _sample_variance_matched(
    rng,
    edge_std,
    target_idx,
    n_draw,
    n_bins=10,
    pool_mask=None,
):
    edge_std = np.asarray(edge_std, dtype=float)
    target_idx = np.asarray(target_idx, dtype=int)
    n_features = edge_std.size
    bins = _quantile_bins(edge_std, n_bins=n_bins)
    if pool_mask is None:
        pool_mask = np.ones(n_features, dtype=bool)
    pool_mask = np.asarray(pool_mask, dtype=bool)
    base_pool = np.where(pool_mask)[0]
    if base_pool.size < n_draw:
        return rng.choice(base_pool, size=n_draw, replace=True)

    chosen = []
    used = set()
    for b in np.unique(bins[target_idx]):
        n_need = int(np.sum(bins[target_idx] == b))
        cand = np.where((bins == b) & pool_mask)[0]
        cand = np.asarray([idx for idx in cand if idx not in used], dtype=int)
        if cand.size >= n_need:
            pick = rng.choice(cand, size=n_need, replace=False)
        elif cand.size > 0:
            remain = n_need - cand.size
            pick2 = rng.choice(base_pool, size=remain, replace=False)
            pick = np.concatenate([cand, pick2])
        else:
            pick = rng.choice(base_pool, size=n_need, replace=False)
        chosen.extend(pick.tolist())
        used.update(pick.tolist())

    chosen = np.asarray(chosen, dtype=int)
    if chosen.size < n_draw:
        remain_pool = np.asarray([idx for idx in base_pool if idx not in set(chosen.tolist())], dtype=int)
        remain = n_draw - chosen.size
        if remain_pool.size >= remain:
            extra = rng.choice(remain_pool, size=remain, replace=False)
        else:
            extra = rng.choice(base_pool, size=remain, replace=True)
        chosen = np.concatenate([chosen, extra])
    return chosen[:n_draw]


def _project_edge_scores(
    X_fc,
    y_fc,
    n_components_fc,
    n_splits,
    random_state,
    stability_repeats,
    subsample_ratio,
    topk_components,
):
    svd_ref = TruncatedSVD(n_components=n_components_fc, random_state=random_state + 999)
    Z_fc = svd_ref.fit_transform(X_fc)

    rng = np.random.default_rng(random_state + 4000)
    component_select_count = np.zeros(n_components_fc, dtype=int)
    component_abscoef_sum = np.zeros(n_components_fc, dtype=float)

    for rep in range(stability_repeats):
        sub_idx = _stratified_subsample_idx(y_fc, subsample_ratio, rng)
        X_sub = Z_fc[sub_idx]
        y_sub = y_fc[sub_idx]
        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=random_state + 5000 + rep,
        )
        model.fit(X_sub, y_sub)
        coef_abs = np.abs(model.coef_)
        comp_score = coef_abs if coef_abs.ndim == 1 else coef_abs.mean(axis=0)
        component_abscoef_sum += comp_score
        top_idx = np.argsort(comp_score)[::-1][:topk_components]
        component_select_count[top_idx] += 1

    selection_freq = component_select_count / max(1, stability_repeats)
    mean_abs_coef = component_abscoef_sum / max(1, stability_repeats)
    stability_df = pd.DataFrame(
        {
            "component_idx": np.arange(n_components_fc, dtype=int),
            "selection_frequency": selection_freq,
            "mean_abs_coef": mean_abs_coef,
        }
    ).sort_values(["selection_frequency", "mean_abs_coef"], ascending=False, ignore_index=True)

    base_acc, base_std = _linear_cv_acc(
        Z_fc,
        y_fc,
        n_splits=n_splits,
        random_state=random_state + 6000,
        drop_components=None,
    )
    top_components = stability_df["component_idx"].to_numpy(dtype=int)[:topk_components]
    ablation_rows = []
    for comp_idx in top_components:
        drop_acc, drop_std = _linear_cv_acc(
            Z_fc,
            y_fc,
            n_splits=n_splits,
            random_state=random_state + 6000,
            drop_components=[int(comp_idx)],
        )
        ablation_rows.append(
            {
                "component_idx": int(comp_idx),
                "base_accuracy_mean": float(base_acc),
                "base_accuracy_std": float(base_std),
                "ablation_accuracy_mean": float(drop_acc),
                "ablation_accuracy_std": float(drop_std),
                "delta_vs_base": float(base_acc - drop_acc),
                "selection_frequency": float(selection_freq[int(comp_idx)]),
                "mean_abs_coef": float(mean_abs_coef[int(comp_idx)]),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows).sort_values("delta_vs_base", ascending=False, ignore_index=True)

    comp_ids = ablation_df["component_idx"].to_numpy(dtype=int)
    comp_w = ablation_df["delta_vs_base"].to_numpy(dtype=float)
    if np.all(np.abs(comp_w) < EPS):
        comp_w = ablation_df["selection_frequency"].to_numpy(dtype=float)
    edge_importance_raw = np.zeros(X_fc.shape[1], dtype=float)
    for comp_idx, w in zip(comp_ids, comp_w):
        edge_importance_raw += float(abs(w)) * np.abs(svd_ref.components_[int(comp_idx)])
    edge_importance = edge_importance_raw / (edge_importance_raw.sum() + EPS)
    return edge_importance, edge_importance_raw, stability_df, ablation_df


def _edge_scores_with_variance_control(edge_importance, edge_importance_raw, edge_std):
    imp_raw = np.asarray(edge_importance, dtype=float)
    edge_std = np.asarray(edge_std, dtype=float)
    imp_z = imp_raw / (edge_std + EPS)
    imp_z = imp_z / (imp_z.sum() + EPS)

    log_imp = np.log(np.asarray(edge_importance_raw, dtype=float) + EPS)
    log_std = np.log(edge_std + EPS)
    valid = np.isfinite(log_std) & np.isfinite(log_imp)
    if valid.sum() >= 3 and np.nanstd(log_std[valid]) > EPS:
        slope, intercept = np.polyfit(log_std[valid], log_imp[valid], deg=1)
    else:
        slope, intercept = 0.0, float(np.nanmean(log_imp[valid])) if valid.any() else 0.0
    resid = log_imp - (slope * log_std + intercept)
    imp_resid_ratio = np.exp(resid)
    imp_resid = imp_resid_ratio / (imp_resid_ratio.sum() + EPS)

    score_map = {
        "raw": imp_raw,
        "zscore": imp_z,
        "residual": imp_resid,
    }
    info = {"logimp_vs_logstd_slope": float(slope), "logimp_vs_logstd_intercept": float(intercept)}
    return score_map, info


def _save_variants(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    for ax in fig.axes:
        ax.set_title("")
    fig.savefig(path.replace(".png", "_notitle.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def load_fc_focus_outputs_from_files(results_dir, mice):
    def _pick_existing(base_dir, candidates):
        for fn in candidates:
            p = os.path.join(base_dir, fn)
            if _nonempty(p):
                return p
        return None

    summary_list, weak_list, ablation_list, decile02_list = [], [], [], []
    for mouse_id in mice:
        data_out = os.path.join(results_dir, mouse_id, "data")
        summary_path = _pick_existing(data_out, ["fc_decoder_summary_focus.csv", "fc_decoder_summary.csv"])
        weak_path = _pick_existing(data_out, ["fc_weak_edge_summary_focus.csv"])
        ablation_path = _pick_existing(data_out, ["fc_edge_ablation_focus.csv", "fc_edge_ablation_delta_acc.csv"])
        decile02_path = _pick_existing(data_out, ["fc_decoder_decile0_2_vs_baseline.csv"])

        if summary_path is not None:
            df = pd.read_csv(summary_path)
            if "mouse_id" not in df.columns:
                df["mouse_id"] = mouse_id
            summary_list.append(df)
        else:
            print(f"[!] Missing summary file for {mouse_id}: tried fc_decoder_summary_focus.csv / fc_decoder_summary.csv")

        if weak_path is not None:
            df = pd.read_csv(weak_path)
            if "mouse_id" not in df.columns:
                df["mouse_id"] = mouse_id
            weak_list.append(df)

        if ablation_path is not None:
            df = pd.read_csv(ablation_path)
            if "mouse_id" not in df.columns:
                df.insert(0, "mouse_id", mouse_id)
            ablation_list.append(df)

        if decile02_path is not None:
            df = pd.read_csv(decile02_path)
            if "mouse_id" not in df.columns:
                df["mouse_id"] = mouse_id
            decile02_list.append(df)

    df_summary = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()
    df_weak = pd.concat(weak_list, ignore_index=True) if weak_list else pd.DataFrame()
    df_ablation = pd.concat(ablation_list, ignore_index=True) if ablation_list else pd.DataFrame()
    df_decile0_2 = pd.concat(decile02_list, ignore_index=True) if decile02_list else pd.DataFrame()
    return df_summary, df_weak, df_ablation, df_decile0_2


def _build_fc_inputs(segments_spi, labels_spi, rr_union, args, seed_i):
    segments_all = np.asarray(segments_spi, dtype=float)
    labels_all = np.asarray(labels_spi).astype(int)
    valid_mask = labels_all != 0
    segments_valid = segments_all[valid_mask]
    labels_valid = labels_all[valid_mask]

    rng = np.random.default_rng(seed_i)
    keep_idx = _balanced_indices(labels_valid, rng)
    segments_fc = segments_valid[keep_idx]
    y_fc = labels_valid[keep_idx]

    if args.neuron_mode == "rr_union" and len(rr_union) > 1:
        rr_idx = np.asarray(sorted([i for i in rr_union if i < segments_fc.shape[1]]), dtype=int)
        if rr_idx.size >= 5:
            segments_fc = segments_fc[:, rr_idx, :]
            neuron_mode = "rr_union"
        else:
            rr_idx = np.arange(segments_fc.shape[1], dtype=int)
            neuron_mode = "all_neurons_fallback"
    else:
        rr_idx = np.arange(segments_fc.shape[1], dtype=int)
        neuron_mode = "all_neurons"

    n_time_fc = segments_fc.shape[2]
    fc_start = max(0, min(args.response_start, n_time_fc - 1))
    fc_end = max(fc_start + 1, min(args.response_end, n_time_fc))

    X_fc_list = []
    for t_idx in range(segments_fc.shape[0]):
        trial_nt = segments_fc[t_idx, :, fc_start:fc_end]
        X_fc_list.append(_trial_fc_upper_triangle(trial_nt))
    X_fc = np.vstack(X_fc_list)
    X_fc = np.clip(X_fc, -0.999999, 0.999999)
    X_fc = np.arctanh(X_fc)

    tri_i, tri_j = np.triu_indices(segments_fc.shape[1], k=1)
    return {
        "X_fc": X_fc,
        "y_fc": y_fc,
        "rr_idx": rr_idx,
        "segments_fc": segments_fc,
        "fc_start": fc_start,
        "fc_end": fc_end,
        "neuron_mode": neuron_mode,
        "tri_i": tri_i,
        "tri_j": tri_j,
    }


def _build_edge_tables(mouse_id, score_map, edge_importance_raw, edge_mean_corr, edge_std, strength_decile, rr_idx, tri_i, tri_j):
    n_features = edge_mean_corr.size
    rows = []
    for edge_idx in range(n_features):
        i_local = int(tri_i[edge_idx])
        j_local = int(tri_j[edge_idx])
        rows.append(
            {
                "mouse_id": mouse_id,
                "edge_idx": int(edge_idx),
                "neuron_i_local": i_local,
                "neuron_j_local": j_local,
                "neuron_i_global": int(rr_idx[i_local]),
                "neuron_j_global": int(rr_idx[j_local]),
                "mean_corr": float(edge_mean_corr[edge_idx]),
                "edge_std": float(edge_std[edge_idx]),
                "strength_decile_abs": int(strength_decile[edge_idx]),
                "importance_raw": float(score_map["raw"][edge_idx]),
                "importance_zscore": float(score_map["zscore"][edge_idx]),
                "importance_residual": float(score_map["residual"][edge_idx]),
                "importance_raw_unscaled": float(edge_importance_raw[edge_idx]),
            }
        )
    edge_df = pd.DataFrame(rows).sort_values("importance_raw", ascending=False, ignore_index=True)
    edge_df.insert(0, "rank_raw", np.arange(1, edge_df.shape[0] + 1, dtype=int))
    return edge_df


def _decile_enrichment(mouse_id, score_map, strength_decile, edge_mean_corr, edge_std):
    rows = []
    weak_deciles = {1, 2, 3}
    strong_deciles = {8, 9, 10}
    for score_type in ["raw", "zscore", "residual"]:
        score = np.asarray(score_map[score_type], dtype=float)
        for dec in range(1, 11):
            mask = strength_decile == dec
            rows.append(
                {
                    "mouse_id": mouse_id,
                    "score_type": score_type,
                    "strength_decile_abs": int(dec),
                    "n_edges": int(mask.sum()),
                    "importance_sum": float(score[mask].sum()),
                    "importance_mean": float(score[mask].mean()),
                    "mean_corr": float(edge_mean_corr[mask].mean()),
                    "mean_std": float(edge_std[mask].mean()),
                }
            )
        mask_weak = np.isin(strength_decile, list(weak_deciles))
        mask_strong = np.isin(strength_decile, list(strong_deciles))
        rows.append(
            {
                "mouse_id": mouse_id,
                "score_type": score_type,
                "strength_decile_abs": -1,
                "n_edges": int(mask_weak.sum()),
                "importance_sum": float(score[mask_weak].sum()),
                "importance_mean": float(score[mask_weak].mean()),
                "mean_corr": float(edge_mean_corr[mask_weak].mean()),
                "mean_std": float(edge_std[mask_weak].mean()),
            }
        )
        rows.append(
            {
                "mouse_id": mouse_id,
                "score_type": score_type,
                "strength_decile_abs": -2,
                "n_edges": int(mask_strong.sum()),
                "importance_sum": float(score[mask_strong].sum()),
                "importance_mean": float(score[mask_strong].mean()),
                "mean_corr": float(edge_mean_corr[mask_strong].mean()),
                "mean_std": float(edge_std[mask_strong].mean()),
            }
        )
    return pd.DataFrame(rows)


def _run_edge_ablations(
    X_fc,
    y_fc,
    n_components_fc,
    n_splits,
    random_state,
    score_map,
    edge_std,
    strength_decile,
    edge_fracs,
    random_repeats,
    weak_deciles,
    var_bins,
):
    base_acc, base_std = _fc_cv_acc_edge_mask(
        X_fc,
        y_fc,
        n_components=n_components_fc,
        drop_edge_idx=None,
        n_splits=n_splits,
        random_state=random_state + 9000,
    )
    rng = np.random.default_rng(random_state + 9500)
    n_features = X_fc.shape[1]
    weak_mask = np.isin(strength_decile, np.asarray(weak_deciles, dtype=int))
    rows = []
    for score_type, frac in product(["raw", "zscore", "residual"], edge_fracs):
        score = np.asarray(score_map[score_type], dtype=float)
        n_drop = int(max(1, np.ceil(n_features * float(frac))))
        rank_desc = np.argsort(score)[::-1]

        global_top_idx = rank_desc[:n_drop]
        top_acc, top_std = _fc_cv_acc_edge_mask(
            X_fc,
            y_fc,
            n_components=n_components_fc,
            drop_edge_idx=global_top_idx,
            n_splits=n_splits,
            random_state=random_state + 9000,
        )
        rows.append(
            {
                "score_type": score_type,
                "ablation_set": "global_top",
                "drop_fraction": float(frac),
                "n_edges_dropped": int(n_drop),
                "repeat_idx": -1,
                "base_accuracy_mean": float(base_acc),
                "base_accuracy_std": float(base_std),
                "accuracy_mean": float(top_acc),
                "accuracy_std": float(top_std),
                "delta_vs_base": float(base_acc - top_acc),
            }
        )

        for rep in range(random_repeats):
            rnd_idx = _sample_variance_matched(
                rng=rng,
                edge_std=edge_std,
                target_idx=global_top_idx,
                n_draw=n_drop,
                n_bins=var_bins,
                pool_mask=np.ones(n_features, dtype=bool),
            )
            rnd_acc, rnd_std = _fc_cv_acc_edge_mask(
                X_fc,
                y_fc,
                n_components=n_components_fc,
                drop_edge_idx=rnd_idx,
                n_splits=n_splits,
                random_state=random_state + 9200 + rep,
            )
            rows.append(
                {
                    "score_type": score_type,
                    "ablation_set": "global_var_matched_random",
                    "drop_fraction": float(frac),
                    "n_edges_dropped": int(n_drop),
                    "repeat_idx": int(rep),
                    "base_accuracy_mean": float(base_acc),
                    "base_accuracy_std": float(base_std),
                    "accuracy_mean": float(rnd_acc),
                    "accuracy_std": float(rnd_std),
                    "delta_vs_base": float(base_acc - rnd_acc),
                }
            )

        weak_pool = np.where(weak_mask)[0]
        if weak_pool.size < n_drop:
            continue
        weak_order = weak_pool[np.argsort(score[weak_pool])[::-1]]
        weak_top_idx = weak_order[:n_drop]
        weak_top_acc, weak_top_std = _fc_cv_acc_edge_mask(
            X_fc,
            y_fc,
            n_components=n_components_fc,
            drop_edge_idx=weak_top_idx,
            n_splits=n_splits,
            random_state=random_state + 9300,
        )
        rows.append(
            {
                "score_type": score_type,
                "ablation_set": "weak_top",
                "drop_fraction": float(frac),
                "n_edges_dropped": int(n_drop),
                "repeat_idx": -1,
                "base_accuracy_mean": float(base_acc),
                "base_accuracy_std": float(base_std),
                "accuracy_mean": float(weak_top_acc),
                "accuracy_std": float(weak_top_std),
                "delta_vs_base": float(base_acc - weak_top_acc),
            }
        )
        for rep in range(random_repeats):
            weak_rnd_idx = _sample_variance_matched(
                rng=rng,
                edge_std=edge_std,
                target_idx=weak_top_idx,
                n_draw=n_drop,
                n_bins=var_bins,
                pool_mask=weak_mask,
            )
            weak_rnd_acc, weak_rnd_std = _fc_cv_acc_edge_mask(
                X_fc,
                y_fc,
                n_components=n_components_fc,
                drop_edge_idx=weak_rnd_idx,
                n_splits=n_splits,
                random_state=random_state + 9400 + rep,
            )
            rows.append(
                {
                    "score_type": score_type,
                    "ablation_set": "weak_var_matched_random",
                    "drop_fraction": float(frac),
                    "n_edges_dropped": int(n_drop),
                    "repeat_idx": int(rep),
                    "base_accuracy_mean": float(base_acc),
                    "base_accuracy_std": float(base_std),
                    "accuracy_mean": float(weak_rnd_acc),
                    "accuracy_std": float(weak_rnd_std),
                    "delta_vs_base": float(base_acc - weak_rnd_acc),
                }
            )
    return pd.DataFrame(rows), float(base_acc), float(base_std)


def run_mouse(mouse_id, args, seed_i):
    save_root = os.path.join(args.results_dir, mouse_id)
    data_out = os.path.join(save_root, "data")
    fig_out = os.path.join(save_root, "figures")
    ensure_dir(data_out)
    ensure_dir(fig_out)
    print(f"[*] Running FC decoder focus for {mouse_id}")

    data_path = os.path.join(args.base_dir, mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)
    segments_spi, labels_spi, _ = preprocess_spike_data(neuron_data, neuron_pos, start_edges, stimulus_data)
    rr_raw = rr_selection_class(segments_spi, labels_spi)
    rr_sets = {int(k): set(map(int, v)) for k, v in rr_raw.items()}
    rr_union = set().union(*rr_sets.values()) if rr_sets else set()

    fc_input = _build_fc_inputs(segments_spi, labels_spi, rr_union, args, seed_i)
    X_fc = fc_input["X_fc"]
    y_fc = fc_input["y_fc"]
    rr_idx = fc_input["rr_idx"]
    tri_i = fc_input["tri_i"]
    tri_j = fc_input["tri_j"]
    n_samples_fc, n_features_fc = X_fc.shape
    n_components_fc = int(min(args.max_components, n_samples_fc - 1, n_features_fc - 1))
    if n_components_fc < 2:
        raise ValueError(f"n_components too small ({n_components_fc}); check trials/neuron count.")

    fc_acc, fc_std, y_fc_pred = _fc_cv_with_pred(
        X_fc,
        y_fc,
        n_components=n_components_fc,
        n_splits=args.n_splits,
        random_state=seed_i,
    )

    rng = np.random.default_rng(seed_i + 200)
    shuffle_scores = []
    for rep in range(args.shuffle_repeats):
        y_shuf = rng.permutation(y_fc)
        rep_acc, _, _ = _fc_cv_with_pred(
            X_fc,
            y_shuf,
            n_components=n_components_fc,
            n_splits=args.n_splits,
            random_state=seed_i + 200 + rep,
        )
        shuffle_scores.append(rep_acc)
    shuffle_scores = np.asarray(shuffle_scores, dtype=float)

    fc_classes = np.sort(np.unique(y_fc))
    fc_class_names = [LABEL_NAMES.get(int(c), str(c)) for c in fc_classes]
    cm_norm = confusion_matrix(y_fc, y_fc_pred, labels=fc_classes, normalize="true")
    cm_raw = confusion_matrix(y_fc, y_fc_pred, labels=fc_classes, normalize=None)

    fig, ax = plt.subplots(figsize=(6.2, 5.1), dpi=180)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=fc_class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
    ax.set_title(
        f"FC Decoder (SVD+SVC, n_comp={n_components_fc})\n"
        f"Acc={fc_acc:.3f}+/-{fc_std:.3f} | Shuffle={shuffle_scores.mean():.3f}+/-{shuffle_scores.std(ddof=1):.3f}"
    )
    _save_variants(fig, os.path.join(fig_out, "fc_decoder_confusion_matrix_focus.png"))

    summary = {
        "mouse_id": mouse_id,
        "window_start": int(fc_input["fc_start"]),
        "window_end": int(fc_input["fc_end"]),
        "n_trials": int(n_samples_fc),
        "n_neurons_used": int(fc_input["segments_fc"].shape[1]),
        "neuron_mode": fc_input["neuron_mode"],
        "n_features_fc": int(n_features_fc),
        "n_components_svd": int(n_components_fc),
        "n_splits": int(args.n_splits),
        "accuracy_mean": float(fc_acc),
        "accuracy_std": float(fc_std),
        "shuffle_accuracy_mean": float(shuffle_scores.mean()),
        "shuffle_accuracy_std": float(shuffle_scores.std(ddof=1)),
        "accuracy_minus_shuffle": float(fc_acc - shuffle_scores.mean()),
    }
    for idx, cname in enumerate(fc_class_names):
        summary[f"recall_{cname}"] = float(cm_norm[idx, idx])
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(data_out, "fc_decoder_summary_focus.csv"), index=False)
    pd.DataFrame(cm_raw, index=fc_class_names, columns=fc_class_names).to_csv(
        os.path.join(data_out, "fc_decoder_confusion_matrix_focus.csv")
    )
    pd.DataFrame({"repeat_idx": np.arange(shuffle_scores.size), "accuracy_mean": shuffle_scores}).to_csv(
        os.path.join(data_out, "fc_decoder_shuffle_repeats_focus.csv"),
        index=False,
    )

    edge_importance, edge_importance_raw, comp_stability_df, comp_ablation_df = _project_edge_scores(
        X_fc=X_fc,
        y_fc=y_fc,
        n_components_fc=n_components_fc,
        n_splits=args.n_splits,
        random_state=seed_i,
        stability_repeats=args.stability_repeats,
        subsample_ratio=args.subsample_ratio,
        topk_components=min(args.component_topk, n_components_fc),
    )
    comp_stability_df.to_csv(os.path.join(data_out, "fc_component_stability_selection_focus.csv"), index=False)
    comp_ablation_df.to_csv(os.path.join(data_out, "fc_component_ablation_delta_acc_focus.csv"), index=False)

    edge_mean_corr = np.tanh(np.nanmean(X_fc, axis=0))
    edge_std = np.nanstd(X_fc, axis=0, ddof=1)
    strength_abs = np.abs(edge_mean_corr)
    order = np.argsort(strength_abs)
    strength_decile = np.empty(n_features_fc, dtype=int)
    strength_decile[order] = (np.arange(n_features_fc, dtype=int) * 10 // n_features_fc) + 1
    strength_decile0 = strength_decile - 1

    weak_mask_0_2 = strength_decile0 <= 2
    X_fc_weak_0_2 = X_fc[:, weak_mask_0_2]
    weak_eval = _decoder_eval_with_shuffle(
        X_mat=X_fc_weak_0_2,
        y_vec=y_fc,
        n_splits=args.n_splits,
        max_components=args.max_components,
        shuffle_repeats=args.shuffle_repeats,
        random_state=seed_i + 12000,
    )
    weak_decoder_row = {
        "mouse_id": mouse_id,
        "subset_name": "decile0_2_zero_based",
        "subset_desc": "weakest 30% edges by |mean corr|",
        "subset_decile0_min": 0,
        "subset_decile0_max": 2,
        "n_edges_subset": int(weak_mask_0_2.sum()),
        "edge_fraction_subset": float(weak_mask_0_2.mean()),
        "full_n_edges": int(n_features_fc),
        "full_accuracy_mean": float(fc_acc),
        "full_accuracy_std": float(fc_std),
        "full_shuffle_accuracy_mean": float(shuffle_scores.mean()),
        "full_shuffle_accuracy_std": float(shuffle_scores.std(ddof=1)),
        "full_accuracy_minus_shuffle": float(fc_acc - shuffle_scores.mean()),
        "subset_n_components_svd": int(weak_eval["n_components"]),
        "subset_accuracy_mean": float(weak_eval["accuracy_mean"]),
        "subset_accuracy_std": float(weak_eval["accuracy_std"]),
        "subset_shuffle_accuracy_mean": float(weak_eval["shuffle_accuracy_mean"]),
        "subset_shuffle_accuracy_std": float(weak_eval["shuffle_accuracy_std"]),
        "subset_accuracy_minus_shuffle": float(weak_eval["accuracy_minus_shuffle"]),
        "subset_minus_full_accuracy": float(weak_eval["accuracy_mean"] - fc_acc),
        "subset_minus_full_delta": float(weak_eval["accuracy_minus_shuffle"] - (fc_acc - shuffle_scores.mean())),
    }
    weak_decoder_df = pd.DataFrame([weak_decoder_row])
    weak_decoder_df.to_csv(os.path.join(data_out, "fc_decoder_decile0_2_vs_baseline.csv"), index=False)
    subset_shuffle_scores = np.asarray(weak_eval["shuffle_scores"], dtype=float)
    if subset_shuffle_scores.size == 0:
        subset_shuffle_scores = np.full(args.shuffle_repeats, np.nan, dtype=float)
    elif subset_shuffle_scores.size != args.shuffle_repeats:
        subset_shuffle_scores = np.resize(subset_shuffle_scores, args.shuffle_repeats)
    pd.DataFrame(
        {
            "repeat_idx": np.arange(args.shuffle_repeats, dtype=int),
            "full_shuffle_accuracy": np.asarray(shuffle_scores, dtype=float),
            "subset_shuffle_accuracy": subset_shuffle_scores,
        }
    ).to_csv(os.path.join(data_out, "fc_decoder_decile0_2_shuffle_repeats.csv"), index=False)

    score_map, var_info = _edge_scores_with_variance_control(edge_importance, edge_importance_raw, edge_std)
    edge_df = _build_edge_tables(
        mouse_id=mouse_id,
        score_map=score_map,
        edge_importance_raw=edge_importance_raw,
        edge_mean_corr=edge_mean_corr,
        edge_std=edge_std,
        strength_decile=strength_decile,
        rr_idx=rr_idx,
        tri_i=tri_i,
        tri_j=tri_j,
    )
    edge_df.to_csv(os.path.join(data_out, "fc_edge_importance_focus.csv"), index=False)

    decile_df = _decile_enrichment(
        mouse_id=mouse_id,
        score_map=score_map,
        strength_decile=strength_decile,
        edge_mean_corr=edge_mean_corr,
        edge_std=edge_std,
    )
    decile_df.to_csv(os.path.join(data_out, "fc_edge_decile_enrichment_focus.csv"), index=False)

    edge_ablation_df, base_edge_acc, base_edge_std = _run_edge_ablations(
        X_fc=X_fc,
        y_fc=y_fc,
        n_components_fc=n_components_fc,
        n_splits=args.n_splits,
        random_state=seed_i,
        score_map=score_map,
        edge_std=edge_std,
        strength_decile=strength_decile,
        edge_fracs=args.edge_fracs,
        random_repeats=args.edge_random_repeats,
        weak_deciles=args.weak_deciles,
        var_bins=args.var_bins,
    )
    edge_ablation_df.to_csv(os.path.join(data_out, "fc_edge_ablation_focus.csv"), index=False)

    var_info_df = pd.DataFrame(
        [
            {
                "mouse_id": mouse_id,
                "base_edge_accuracy_mean": base_edge_acc,
                "base_edge_accuracy_std": base_edge_std,
                **var_info,
            }
        ]
    )
    var_info_df.to_csv(os.path.join(data_out, "fc_variance_bias_fit_focus.csv"), index=False)

    decile_plot = decile_df[decile_df["strength_decile_abs"] > 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)
    sns.lineplot(
        data=decile_plot,
        x="strength_decile_abs",
        y="importance_sum",
        hue="score_type",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_xlabel("Strength decile by |mean corr| (1=weak, 10=strong)")
    axes[0].set_ylabel("Importance sum")
    axes[0].set_title("Edge importance across strength deciles")
    style_axis(axes[0], grid=True)
    axes[0].legend(frameon=False, title="")

    abl_plot = (
        edge_ablation_df.groupby(["score_type", "ablation_set", "drop_fraction"], as_index=False)
        .agg(delta_vs_base=("delta_vs_base", "mean"))
        .sort_values(["score_type", "ablation_set", "drop_fraction"], ignore_index=True)
    )
    sns.barplot(
        data=abl_plot,
        x="drop_fraction",
        y="delta_vs_base",
        hue="ablation_set",
        ax=axes[1],
    )
    axes[1].set_xlabel("Drop fraction")
    axes[1].set_ylabel("Mean accuracy drop")
    axes[1].set_title("Ablation: weak-edge focus vs variance-matched random")
    style_axis(axes[1], grid=True)
    axes[1].legend(frameon=False, title="")
    _save_variants(fig, os.path.join(fig_out, "fc_weak_edge_importance_focus.png"))

    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=180)
    comp_df = pd.DataFrame(
        {
            "model": ["full_edges", "full_shuffle", "weak_decile0_2", "weak_decile0_2_shuffle"],
            "accuracy": [
                float(fc_acc),
                float(shuffle_scores.mean()),
                float(weak_eval["accuracy_mean"]),
                float(weak_eval["shuffle_accuracy_mean"]),
            ],
        }
    )
    sns.barplot(data=comp_df, x="model", y="accuracy", color="#4C78A8", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("CV accuracy")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Decoder with weakest decile0-2 edges only")
    style_axis(ax, grid=True)
    _save_variants(fig, os.path.join(fig_out, "fc_decoder_decile0_2_vs_baseline.png"))

    weak_summary_rows = []
    for score_type in ["raw", "zscore", "residual"]:
        sub = decile_df[(decile_df["score_type"] == score_type) & (decile_df["strength_decile_abs"] > 0)]
        weak_sum = float(sub[sub["strength_decile_abs"].isin([1, 2, 3])]["importance_sum"].sum())
        strong_sum = float(sub[sub["strength_decile_abs"].isin([8, 9, 10])]["importance_sum"].sum())
        weak_summary_rows.append(
            {
                "mouse_id": mouse_id,
                "score_type": score_type,
                "weak_importance_sum_decile123": weak_sum,
                "strong_importance_sum_decile8910": strong_sum,
                "weak_minus_strong": weak_sum - strong_sum,
            }
        )
    weak_summary_df = pd.DataFrame(weak_summary_rows)
    weak_summary_df.to_csv(os.path.join(data_out, "fc_weak_edge_summary_focus.csv"), index=False)

    print(f"[*] Mouse done: {mouse_id}")
    return summary_df, weak_summary_df, edge_ablation_df, weak_decoder_df


def run_group(df_summary, df_weak, df_ablation, df_decile0_2, args):
    group_dir = os.path.join(args.results_dir, GROUP_DIR_NAME)
    ensure_dir(group_dir)
    df_summary.to_csv(os.path.join(group_dir, "group_fc_decoder_summary_focus.csv"), index=False)
    df_weak.to_csv(os.path.join(group_dir, "group_fc_weak_edge_summary_focus.csv"), index=False)
    df_ablation.to_csv(os.path.join(group_dir, "group_fc_edge_ablation_focus.csv"), index=False)
    df_decile0_2.to_csv(os.path.join(group_dir, "group_fc_decoder_decile0_2_vs_baseline.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=180)
    sns.barplot(data=df_summary, x="mouse_id", y="accuracy_minus_shuffle", color="#4C78A8", ax=axes[0])
    axes[0].axhline(0.0, color="#555555", linewidth=1.0)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("FC accuracy - shuffle")
    axes[0].tick_params(axis="x", rotation=30)
    style_axis(axes[0], grid=True)

    if not df_weak.empty:
        sns.boxplot(data=df_weak, x="score_type", y="weak_importance_sum_decile123", color="#72B7B2", ax=axes[1])
        sns.stripplot(data=df_weak, x="score_type", y="weak_importance_sum_decile123", color="#1f1f1f", size=4, ax=axes[1])
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Weak-edge importance sum (decile 1-3)")
        style_axis(axes[1], grid=True)
    else:
        axes[1].text(0.5, 0.5, "No weak-edge summary files", ha="center", va="center")
        axes[1].set_axis_off()
    _save_variants(fig, os.path.join(group_dir, "group_fc_weak_edge_focus.png"))

    if not df_decile0_2.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=180)
        long_decile = pd.melt(
            df_decile0_2[
                [
                    "mouse_id",
                    "full_accuracy_mean",
                    "full_shuffle_accuracy_mean",
                    "subset_accuracy_mean",
                    "subset_shuffle_accuracy_mean",
                ]
            ],
            id_vars=["mouse_id"],
            value_vars=[
                "full_accuracy_mean",
                "full_shuffle_accuracy_mean",
                "subset_accuracy_mean",
                "subset_shuffle_accuracy_mean",
            ],
            var_name="model",
            value_name="accuracy",
        )
        name_map = {
            "full_accuracy_mean": "full_edges",
            "full_shuffle_accuracy_mean": "full_shuffle",
            "subset_accuracy_mean": "weak_decile0_2",
            "subset_shuffle_accuracy_mean": "weak_decile0_2_shuffle",
        }
        long_decile["model"] = long_decile["model"].map(name_map)
        sns.boxplot(data=long_decile, x="model", y="accuracy", color="#72B7B2", ax=ax)
        sns.stripplot(data=long_decile, x="model", y="accuracy", color="#1f1f1f", size=4, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("CV accuracy")
        ax.tick_params(axis="x", rotation=20)
        style_axis(ax, grid=True)
        _save_variants(fig, os.path.join(group_dir, "group_fc_decoder_decile0_2_vs_baseline.png"))

    agg = (
        df_weak.groupby("score_type", as_index=False).agg(
            weak_importance_mean=("weak_importance_sum_decile123", "mean"),
            weak_importance_std=("weak_importance_sum_decile123", "std"),
            n_mice=("mouse_id", "nunique"),
        )
        if not df_weak.empty
        else pd.DataFrame(columns=["score_type", "weak_importance_mean", "weak_importance_std", "n_mice"])
    )
    agg.to_csv(os.path.join(group_dir, "group_fc_weak_edge_by_score_focus.csv"), index=False)

    md_path = os.path.join(group_dir, "Group_FC_Decoder_Focus_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group FC Decoder Focus Report\n\n")
        f.write(f"Number of mice: {df_summary['mouse_id'].nunique()}\n\n")
        f.write("## Decoder summary\n\n")
        f.write(df_summary.to_markdown(index=False) + "\n\n")
        f.write("## Weak-edge summary by score\n\n")
        f.write(df_weak.to_markdown(index=False) + "\n\n")
        f.write("## Decoder using decile0-2 edges only\n\n")
        f.write(df_decile0_2.to_markdown(index=False) + "\n\n")
        f.write("## Group weak-edge aggregate\n\n")
        f.write(agg.to_markdown(index=False) + "\n\n")
        f.write("## Figure\n\n")
        f.write("![group_fc_weak_edge_focus](./group_fc_weak_edge_focus.png)\n")
        f.write("![group_fc_decoder_decile0_2_vs_baseline](./group_fc_decoder_decile0_2_vs_baseline.png)\n")
    print(f"[*] Group report saved: {md_path}")


def parse_args():
    p = argparse.ArgumentParser(description="FC decoder focused pipeline with weak-edge and variance-bias controls.")
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)
    p.add_argument("--response-start", type=int, default=10)
    p.add_argument("--response-end", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--shuffle-repeats", type=int, default=30)
    p.add_argument("--max-components", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260330)
    p.add_argument("--neuron-mode", type=str, default="rr_union", choices=["rr_union", "all_neurons"])
    p.add_argument("--stability-repeats", type=int, default=25)
    p.add_argument("--subsample-ratio", type=float, default=0.80)
    p.add_argument("--component-topk", type=int, default=10)
    p.add_argument("--edge-fracs", nargs="*", type=float, default=[0.01, 0.03, 0.05])
    p.add_argument("--edge-random-repeats", type=int, default=10)
    p.add_argument("--weak-deciles", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--var-bins", type=int, default=10)
    p.add_argument("--group-only-from-files", action="store_true", help="Skip per-mouse recompute and build group report from existing per-mouse FC focus CSV files.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.results_dir)

    if args.group_only_from_files:
        df_summary, df_weak, df_ablation, df_decile0_2 = load_fc_focus_outputs_from_files(args.results_dir, args.mice)
        if df_summary is None or df_summary.empty:
            print("[!] No fc_decoder_summary_focus.csv found. Stop.")
            return
        run_group(df_summary, df_weak, df_ablation, df_decile0_2, args)
        print("====== FC decoder focus group-only integration completed ======")
        return

    all_summary, all_weak, all_ablation, all_decile0_2 = [], [], [], []
    base_seed = int(args.seed)
    for i, mouse in enumerate(args.mice):
        try:
            seed_i = int(base_seed + i * 101)
            df_summary, df_weak, df_ablation, df_decile0_2 = run_mouse(mouse, args, seed_i=seed_i)
            if df_summary is not None and not df_summary.empty:
                all_summary.append(df_summary)
            if df_weak is not None and not df_weak.empty:
                all_weak.append(df_weak)
            if df_ablation is not None and not df_ablation.empty:
                df_ablation = df_ablation.copy()
                df_ablation.insert(0, "mouse_id", mouse)
                all_ablation.append(df_ablation)
            if df_decile0_2 is not None and not df_decile0_2.empty:
                all_decile0_2.append(df_decile0_2)
        except Exception as exc:
            print(f"[!] Mouse {mouse} failed: {exc}")
    if not all_summary:
        print("[!] No valid mouse outputs. Stop.")
        return
    df_summary = pd.concat(all_summary, ignore_index=True)
    df_weak = pd.concat(all_weak, ignore_index=True) if all_weak else pd.DataFrame()
    df_ablation = pd.concat(all_ablation, ignore_index=True) if all_ablation else pd.DataFrame()
    df_decile0_2 = pd.concat(all_decile0_2, ignore_index=True) if all_decile0_2 else pd.DataFrame()
    run_group(df_summary, df_weak, df_ablation, df_decile0_2, args)
    print("====== FC decoder focus pipeline completed ======")


if __name__ == "__main__":
    main()
