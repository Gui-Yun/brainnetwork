#!/usr/bin/env python3
"""Focused RSM analysis for one mouse.

This script reproduces the core RSM outputs already used in the project:
- Condition-wise mean/std/entropy of trial-to-trial RSM similarity
- Similarity distribution plot
- Per-condition RSM heatmaps

And adds:
- Condition-wise mean RSM timecourse across frames
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brainnetwork import load_data, preprocess_spike_data, t_stimulus, l_stimulus  # noqa: E402


DEFAULT_BASE_DIR = "/beegfs_hdd/data/nfs_share/users/guiyun/nishome/Micedata"
DEFAULT_LABEL_MAP = {
    1: "Divergent",
    2: "Convergent",
    3: "Random",
}
DEFAULT_COLORS = {
    "Divergent": "#FF4B4B",
    "Convergent": "#1C75BC",
    "Random": "#7AC143",
}


def parse_conditions(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def condition_name(class_id: int) -> str:
    return DEFAULT_LABEL_MAP.get(int(class_id), f"Class_{int(class_id)}")


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    vals = matrix[mask]
    return vals[np.isfinite(vals)]


def compute_rsm_stats(x_cond: np.ndarray, n_bins: int) -> tuple[dict, np.ndarray, np.ndarray]:
    rsm = cosine_similarity(x_cond)
    sim_vals = upper_triangle_values(rsm)
    if sim_vals.size == 0:
        return (
            {
                "Mean_RSM": np.nan,
                "RSM_STD": np.nan,
                "RSM_Entropy": np.nan,
                "Pair_Count": 0,
                "N_Trials": int(x_cond.shape[0]),
            },
            sim_vals,
            rsm,
        )

    counts, _ = np.histogram(sim_vals, bins=int(n_bins), range=(-1, 1), density=False)
    probs = counts.astype(float)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    probs = probs[probs > 0]

    return (
        {
            "Mean_RSM": float(np.mean(sim_vals)),
            "RSM_STD": float(np.std(sim_vals)),
            "RSM_Entropy": float(entropy(probs, base=2)) if probs.size else np.nan,
            "Pair_Count": int(sim_vals.size),
            "N_Trials": int(x_cond.shape[0]),
        },
        sim_vals,
        rsm,
    )


def balance_trials(
    segments: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    counts = {c: int(np.sum(labels == c)) for c in class_ids}
    min_count = min(counts.values())

    picked = []
    for c in class_ids:
        idx = np.flatnonzero(labels == c)
        if idx.size == min_count:
            picked.append(idx)
        else:
            picked.append(rng.choice(idx, size=min_count, replace=False))

    all_idx = np.concatenate(picked)
    all_idx = np.sort(all_idx)
    return segments[all_idx], labels[all_idx]


def build_response_matrix(segments: np.ndarray, start: int, end: int) -> np.ndarray:
    if not (0 <= start < end <= segments.shape[2]):
        raise ValueError(
            f"Invalid response window [{start}, {end}) for time length={segments.shape[2]}"
        )
    return np.nanmean(segments[:, :, start:end], axis=2)


def analyze_static_rsm(
    x_resp: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
    n_bins: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = []
    rsm_dict: dict[str, np.ndarray] = {}
    sim_dict: dict[str, np.ndarray] = {}

    for class_id in class_ids:
        name = condition_name(class_id)
        x_cond = x_resp[labels == class_id]
        stat, sim_vals, rsm = compute_rsm_stats(x_cond, n_bins=n_bins)
        rows.append(
            {
                "Class_ID": int(class_id),
                "Condition": name,
                **stat,
            }
        )
        rsm_dict[name] = rsm
        sim_dict[name] = sim_vals

    return pd.DataFrame(rows), rsm_dict, sim_dict


def analyze_rsm_timecourse(
    segments: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
) -> pd.DataFrame:
    rows = []
    n_frames = int(segments.shape[2])

    for class_id in class_ids:
        name = condition_name(class_id)
        seg_cond = segments[labels == class_id]
        n_trials = int(seg_cond.shape[0])

        for frame_idx in range(n_frames):
            x_frame = seg_cond[:, :, frame_idx]
            rsm = cosine_similarity(x_frame)
            sim_vals = upper_triangle_values(rsm)
            if sim_vals.size == 0:
                mean_rsm = np.nan
                std_rsm = np.nan
                sem_rsm = np.nan
                pair_count = 0
            else:
                mean_rsm = float(np.mean(sim_vals))
                std_rsm = float(np.std(sim_vals))
                pair_count = int(sim_vals.size)
                sem_rsm = float(std_rsm / np.sqrt(pair_count)) if pair_count > 0 else np.nan

            rows.append(
                {
                    "Class_ID": int(class_id),
                    "Condition": name,
                    "Frame": frame_idx,
                    "Mean_RSM": mean_rsm,
                    "RSM_STD": std_rsm,
                    "RSM_SEM": sem_rsm,
                    "Pair_Count": pair_count,
                    "N_Trials": n_trials,
                }
            )

    return pd.DataFrame(rows)


def palette_for_conditions(condition_names: list[str]) -> dict[str, str]:
    out = {}
    for i, name in enumerate(condition_names):
        out[name] = DEFAULT_COLORS.get(name, sns.color_palette("tab10")[i % 10])
    return out


def save_similarity_distribution(
    sim_dict: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    out_path: str,
    palette: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.6), dpi=300)
    for cond in summary_df["Condition"].tolist():
        vals = np.asarray(sim_dict.get(cond, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        ent = summary_df.loc[summary_df["Condition"] == cond, "RSM_Entropy"].iloc[0]
        sns.kdeplot(
            vals,
            label=f"{cond} (H={ent:.2f})" if np.isfinite(ent) else cond,
            color=palette.get(cond, "#333333"),
            fill=True,
            alpha=0.25,
            linewidth=2,
            ax=ax,
        )

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.5, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Condition", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rsm_heatmaps(
    rsm_dict: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    fig_out_dir: str,
) -> None:
    for cond in summary_df["Condition"].tolist():
        rsm = rsm_dict.get(cond)
        if rsm is None or rsm.size == 0:
            continue
        ent = summary_df.loc[summary_df["Condition"] == cond, "RSM_Entropy"].iloc[0]

        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
        sns.heatmap(rsm, cmap="viridis", vmin=-1, vmax=1, ax=ax)
        if np.isfinite(ent):
            ax.set_title(f"RSM Heatmap - {cond} (H={ent:.2f} bits)")
        else:
            ax.set_title(f"RSM Heatmap - {cond}")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Trials")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_out_dir, f"rsm_heatmap_{cond}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_rsm_timecourse(
    time_df: pd.DataFrame,
    out_path: str,
    palette: dict[str, str],
    stim_onset: int,
    stim_len: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=300)

    for cond in time_df["Condition"].drop_duplicates().tolist():
        sub = time_df[time_df["Condition"] == cond].sort_values("Frame")
        x = sub["Frame"].to_numpy(dtype=float)
        y = sub["Mean_RSM"].to_numpy(dtype=float)
        sem = sub["RSM_SEM"].to_numpy(dtype=float)
        color = palette.get(cond, "#333333")

        ax.plot(x, y, lw=2.0, color=color, label=cond)
        valid = np.isfinite(y) & np.isfinite(sem)
        if np.any(valid):
            ax.fill_between(x[valid], y[valid] - sem[valid], y[valid] + sem[valid], color=color, alpha=0.18, linewidth=0)

    stim_end = stim_onset + stim_len
    ax.axvline(stim_onset, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(stim_end, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvspan(stim_onset, stim_end, color="#BBBBBB", alpha=0.15)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean RSM (off-diagonal cosine similarity)")
    ax.set_title("Condition-wise Mean RSM Timecourse")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Condition", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused RSM analysis + RSM timecourse.")
    parser.add_argument("--mouse-id", type=str, required=True, help="Mouse ID, e.g. M73_1128")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="Directory containing raw mouse folders")
    parser.add_argument("--results-dir", type=str, default="./results", help="Output root directory")
    parser.add_argument("--conditions", type=str, default="1,2,3", help="Comma-separated condition IDs to analyze")
    parser.add_argument("--response-start", type=int, default=10, help="Response window start frame (inclusive)")
    parser.add_argument("--response-end", type=int, default=13, help="Response window end frame (exclusive)")
    parser.add_argument("--n-bins", type=int, default=50, help="Histogram bins for RSM entropy")
    parser.add_argument("--seed", type=int, default=20260328, help="Random seed for trial balancing")
    parser.add_argument("--no-extract-rr", action="store_true", help="Disable RR-neuron selection in preprocessing")
    args = parser.parse_args()

    data_path = os.path.join(args.base_dir, args.mouse_id)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Mouse directory not found: {data_path}")

    mouse_out = os.path.join(args.results_dir, args.mouse_id)
    data_out = os.path.join(mouse_out, "data")
    fig_out = os.path.join(mouse_out, "figures")
    os.makedirs(data_out, exist_ok=True)
    os.makedirs(fig_out, exist_ok=True)

    print(f"[*] Loading data: {data_path}")
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)
    segments, labels, _ = preprocess_spike_data(
        neuron_data,
        neuron_pos,
        start_edges,
        stimulus_data,
        extract_rr=(not args.no_extract_rr),
    )
    segments = np.asarray(segments, dtype=float)
    labels = np.asarray(labels).astype(int)

    requested_class_ids = parse_conditions(args.conditions)
    present_class_ids = sorted(np.unique(labels).astype(int).tolist())
    class_ids = [c for c in requested_class_ids if c in present_class_ids]
    if len(class_ids) < 2:
        raise ValueError(
            f"Need at least 2 valid classes. Requested={requested_class_ids}, present={present_class_ids}, used={class_ids}"
        )

    keep_mask = np.isin(labels, class_ids)
    segments = segments[keep_mask]
    labels = labels[keep_mask]

    counts_before = {condition_name(c): int(np.sum(labels == c)) for c in class_ids}
    print(f"[*] Trial counts before balancing: {counts_before}")

    segments_bal, labels_bal = balance_trials(segments, labels, class_ids=class_ids, seed=args.seed)
    counts_after = {condition_name(c): int(np.sum(labels_bal == c)) for c in class_ids}
    print(f"[*] Trial counts after balancing: {counts_after}")

    x_resp = build_response_matrix(segments_bal, start=args.response_start, end=args.response_end)

    summary_df, rsm_dict, sim_dict = analyze_static_rsm(
        x_resp=x_resp,
        labels=labels_bal,
        class_ids=class_ids,
        n_bins=args.n_bins,
    )
    summary_df.to_csv(os.path.join(data_out, "rsm_summary.csv"), index=False)

    sim_rows = []
    for cond, vals in sim_dict.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        for v in vals:
            sim_rows.append({"Condition": cond, "Similarity": float(v)})
    pd.DataFrame(sim_rows).to_csv(os.path.join(data_out, "rsm_similarity_long.csv"), index=False)

    time_df = analyze_rsm_timecourse(
        segments=segments_bal,
        labels=labels_bal,
        class_ids=class_ids,
    )
    time_df.to_csv(os.path.join(data_out, "rsm_timecourse_long.csv"), index=False)

    pivot_df = time_df.pivot_table(index="Frame", columns="Condition", values="Mean_RSM", aggfunc="mean")
    pivot_df.to_csv(os.path.join(data_out, "rsm_timecourse_wide.csv"))

    condition_names = [condition_name(c) for c in class_ids]
    palette = palette_for_conditions(condition_names)

    save_similarity_distribution(
        sim_dict=sim_dict,
        summary_df=summary_df,
        out_path=os.path.join(fig_out, "similarity_distribution.png"),
        palette=palette,
    )
    save_rsm_heatmaps(rsm_dict=rsm_dict, summary_df=summary_df, fig_out_dir=fig_out)
    save_rsm_timecourse(
        time_df=time_df,
        out_path=os.path.join(fig_out, "rsm_timecourse_mean_similarity.png"),
        palette=palette,
        stim_onset=int(t_stimulus),
        stim_len=int(l_stimulus),
    )

    manifest = {
        "mouse_id": args.mouse_id,
        "data_path": data_path,
        "output_data_dir": data_out,
        "output_figure_dir": fig_out,
        "conditions_used": class_ids,
        "condition_names": condition_names,
        "response_window": [int(args.response_start), int(args.response_end)],
        "n_bins": int(args.n_bins),
        "seed": int(args.seed),
        "extract_rr": bool(not args.no_extract_rr),
        "stimulus_onset_frame": int(t_stimulus),
        "stimulus_length_frames": int(l_stimulus),
    }
    with open(os.path.join(data_out, "rsm_analysis_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[*] Done.")
    print(f"    - Summary CSV: {os.path.join(data_out, 'rsm_summary.csv')}")
    print(f"    - Timecourse CSV: {os.path.join(data_out, 'rsm_timecourse_long.csv')}")
    print(f"    - Timecourse figure: {os.path.join(fig_out, 'rsm_timecourse_mean_similarity.png')}")


if __name__ == "__main__":
    main()
