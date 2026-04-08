import os
from typing import Dict, Iterable, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


import matplotlib.pyplot as plt


def _build_trial_features(segments: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 3:
        raise ValueError("segments must be (n_trials, n_neurons, n_timepoints)")
    start, end = window
    if not (0 <= start < end <= segments.shape[2]):
        raise ValueError(f"invalid window={window} for n_timepoints={segments.shape[2]}")
    return np.nanmean(segments[:, :, start:end], axis=2)


def _build_decoder() -> Pipeline:
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale")),
        ]
    )


def _evaluate_decoder_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> Dict[str, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = _build_decoder()
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    return {
        "acc_mean": float(scores.mean()),
        "acc_std": float(scores.std(ddof=1)),
        "y_pred": y_pred,
        "n_folds": int(cv.get_n_splits()),
    }


def _estimate_shuffled_baseline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    repeats: int,
    n_splits: int,
    random_state: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    values = []
    for rep in range(repeats):
        y_shuffled = rng.permutation(y)
        result = _evaluate_decoder_cv(
            X,
            y_shuffled,
            n_splits=n_splits,
            random_state=random_state + rep + 1,
        )
        values.append(result["acc_mean"])
    arr = np.asarray(values, dtype=float)
    return {
        "all": arr,
        "acc_mean": float(arr.mean()),
        "acc_std": float(arr.std(ddof=1)),
    }


def _select_top_response_neurons(X: np.ndarray, ratio: float) -> np.ndarray:
    if not (0 < ratio < 1):
        raise ValueError("ratio must be in (0, 1)")
    n_neurons = X.shape[1]
    n_remove = max(1, int(np.ceil(n_neurons * ratio)))
    neuron_response = np.nanmean(X, axis=0)
    top_idx = np.argsort(neuron_response)[::-1][:n_remove]
    return top_idx


def _run_random_drop_control(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_remove: int,
    repeats: int,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_neurons = X.shape[1]
    all_acc = []
    for rep in range(repeats):
        remove_idx = rng.choice(n_neurons, size=n_remove, replace=False)
        keep_mask = np.ones(n_neurons, dtype=bool)
        keep_mask[remove_idx] = False
        result = _evaluate_decoder_cv(
            X[:, keep_mask],
            y,
            n_splits=n_splits,
            random_state=random_state + rep + 1000,
        )
        all_acc.append(result["acc_mean"])
    return np.asarray(all_acc, dtype=float)


def run_decoder_task1_task2(
    segments: np.ndarray,
    labels: Iterable[int],
    *,
    data_out_dir: str,
    fig_out_dir: str,
    label_names: Dict[int, str] | None = None,
    window: Tuple[int, int] = (10, 13),
    n_splits: int = 5,
    random_state: int = 42,
    shuffle_repeats: int = 200,
    ablation_ratio: float = 0.10,
    random_drop_repeats: int = 100,
    drop_zero_label: bool = True,
) -> Dict[str, str]:
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(fig_out_dir, exist_ok=True)

    X = _build_trial_features(segments, window=window)
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be 1D")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of trials")

    if drop_zero_label:
        mask = y != 0
        X = X[mask]
        y = y[mask]

    classes = np.sort(np.unique(y))
    if classes.size < 2:
        raise ValueError("need at least two classes for decoding")

    if label_names is None:
        label_names = {int(c): str(c) for c in classes}
    class_labels = [label_names.get(int(c), str(c)) for c in classes]

    # Task 1: decoder + confusion matrix + shuffled baseline
    full_result = _evaluate_decoder_cv(X, y, n_splits=n_splits, random_state=random_state)
    shuffled = _estimate_shuffled_baseline(
        X,
        y,
        repeats=shuffle_repeats,
        n_splits=n_splits,
        random_state=random_state,
    )

    cm_norm = confusion_matrix(y, full_result["y_pred"], labels=classes, normalize="true")
    cm_raw = confusion_matrix(y, full_result["y_pred"], labels=classes, normalize=None)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=180)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
    ax.set_title(
        "Task1 Decoder Confusion Matrix\n"
        f"Acc={full_result['acc_mean']:.3f}±{full_result['acc_std']:.3f} | "
        f"Shuffle={shuffled['acc_mean']:.3f}±{shuffled['acc_std']:.3f}"
    )
    fig.tight_layout()
    confusion_fig_path = os.path.join(fig_out_dir, "decoder_confusion_matrix.png")
    fig.savefig(confusion_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    decoder_summary = pd.DataFrame(
        [
            {
                "window_start": int(window[0]),
                "window_end": int(window[1]),
                "n_trials": int(X.shape[0]),
                "n_neurons": int(X.shape[1]),
                "n_classes": int(classes.size),
                "n_folds": int(full_result["n_folds"]),
                "accuracy_mean": float(full_result["acc_mean"]),
                "accuracy_std": float(full_result["acc_std"]),
                "shuffle_accuracy_mean": float(shuffled["acc_mean"]),
                "shuffle_accuracy_std": float(shuffled["acc_std"]),
                "accuracy_minus_shuffle": float(full_result["acc_mean"] - shuffled["acc_mean"]),
            }
        ]
    )
    for idx, class_name in enumerate(class_labels):
        decoder_summary[f"recall_{class_name}"] = float(cm_norm[idx, idx])

    decoder_summary_path = os.path.join(data_out_dir, "decoder_summary.csv")
    decoder_summary.to_csv(decoder_summary_path, index=False)

    cm_path = os.path.join(data_out_dir, "decoder_confusion_matrix.csv")
    pd.DataFrame(cm_raw, index=class_labels, columns=class_labels).to_csv(cm_path)

    # Task 2: top-10% ablation + random-drop control
    top_idx = _select_top_response_neurons(X, ratio=ablation_ratio)
    keep_mask = np.ones(X.shape[1], dtype=bool)
    keep_mask[top_idx] = False
    X_ablate = X[:, keep_mask]
    ablate_result = _evaluate_decoder_cv(
        X_ablate,
        y,
        n_splits=n_splits,
        random_state=random_state + 500,
    )

    rand_acc = _run_random_drop_control(
        X,
        y,
        n_remove=top_idx.size,
        repeats=random_drop_repeats,
        n_splits=n_splits,
        random_state=random_state + 900,
    )
    rand_mean = float(rand_acc.mean())
    rand_std = float(rand_acc.std(ddof=1))

    ablation_summary = pd.DataFrame(
        [
            {
                "window_start": int(window[0]),
                "window_end": int(window[1]),
                "n_trials": int(X.shape[0]),
                "n_neurons_total": int(X.shape[1]),
                "n_neurons_removed": int(top_idx.size),
                "removed_ratio": float(top_idx.size / X.shape[1]),
                "full_accuracy_mean": float(full_result["acc_mean"]),
                "full_accuracy_std": float(full_result["acc_std"]),
                "top10_ablation_accuracy_mean": float(ablate_result["acc_mean"]),
                "top10_ablation_accuracy_std": float(ablate_result["acc_std"]),
                "random_drop_mean_accuracy": rand_mean,
                "random_drop_std_accuracy": rand_std,
                "delta_full_minus_top10": float(full_result["acc_mean"] - ablate_result["acc_mean"]),
                "delta_top10_minus_random_mean": float(ablate_result["acc_mean"] - rand_mean),
                "random_drop_q05": float(np.quantile(rand_acc, 0.05)),
                "random_drop_q95": float(np.quantile(rand_acc, 0.95)),
                "ablation_rank_in_random": float((rand_acc <= ablate_result["acc_mean"]).mean()),
            }
        ]
    )
    ablation_summary_path = os.path.join(data_out_dir, "decoder_ablation_summary.csv")
    ablation_summary.to_csv(ablation_summary_path, index=False)

    rand_repeats_path = os.path.join(data_out_dir, "decoder_random_drop_repeats.csv")
    pd.DataFrame(
        {
            "repeat_idx": np.arange(rand_acc.size, dtype=int),
            "accuracy_mean": rand_acc,
        }
    ).to_csv(rand_repeats_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    bar_labels = ["Full model", "Top10% ablation", "Random drop\n(mean±sd)"]
    means = [full_result["acc_mean"], ablate_result["acc_mean"], rand_mean]
    errs = [full_result["acc_std"], ablate_result["acc_std"], rand_std]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(3)
    ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=4, edgecolor="#333333")

    rng = np.random.default_rng(random_state + 2024)
    jitter = rng.uniform(-0.14, 0.14, size=rand_acc.size)
    ax.scatter(np.full(rand_acc.size, x[2]) + jitter, rand_acc, s=18, alpha=0.55, color="#2E7D32")

    chance = 1.0 / classes.size
    ax.axhline(chance, color="#777777", linestyle="--", linewidth=1.2, label=f"Chance={chance:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title(
        "Task2 Top10% Neuron Ablation\n"
        f"Δ(full-top10)={full_result['acc_mean'] - ablate_result['acc_mean']:.3f} | "
        f"Δ(top10-rand)={ablate_result['acc_mean'] - rand_mean:.3f}"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    ablation_fig_path = os.path.join(fig_out_dir, "decoder_ablation_top10.png")
    fig.savefig(ablation_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[*] Task1 decoder summary saved to: {decoder_summary_path}")
    print(f"[*] Task1 confusion figure saved to: {confusion_fig_path}")
    print(f"[*] Task2 ablation summary saved to: {ablation_summary_path}")
    print(f"[*] Task2 ablation figure saved to: {ablation_fig_path}")

    return {
        "decoder_summary_csv": decoder_summary_path,
        "decoder_confusion_csv": cm_path,
        "decoder_confusion_fig": confusion_fig_path,
        "decoder_ablation_summary_csv": ablation_summary_path,
        "decoder_random_drop_repeats_csv": rand_repeats_path,
        "decoder_ablation_fig": ablation_fig_path,
    }
