import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainnetwork import classify_by_timepoints, load_data, preprocess_spike_data, rr_selection_class


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

CLASS_TO_CONDITION = {1: "Divergent", 2: "Convergent", 3: "Random"}
CONDITION_ORDER = ["Divergent", "Convergent", "Random"]
COLORS = {"Divergent": "#7F9C96", "Convergent": "#8B90A8", "Random": "#B98372", "Coherent": "#5F7088"}
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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def style_axis(ax, grid=False):
    sns.despine(ax=ax, trim=False)
    if grid:
        ax.grid(axis="y", linestyle=":", alpha=0.55)
    else:
        ax.grid(False)


def save_variants(fig: plt.Figure, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    base = out_png.with_suffix("")
    for ax in fig.axes:
        ax.set_title("")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    fig.savefig(str(base) + "_notitle.png", dpi=300, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def get_roots(args):
    out_root = Path(args.results_dir).resolve()
    fallback_root = Path(args.fallback_results_dir).resolve()
    ensure_dir(out_root)
    roots = [out_root]
    if fallback_root != out_root:
        roots.append(fallback_root)
    return out_root, roots


def find_existing_rel(roots, rel_path: str):
    rel = Path(rel_path)
    for r in roots:
        p = r / rel
        if p.exists():
            return p
    return None


def maybe_copy_from_other_roots(out_root: Path, roots, rel_path: str):
    rel = Path(rel_path)
    dst = out_root / rel
    if dst.exists():
        return dst
    src = find_existing_rel([r for r in roots if r != out_root], rel_path)
    if src is None:
        return None
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)
    return dst


def parse_markdown_table_after_heading(md_text: str, heading: str) -> pd.DataFrame:
    lines = md_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Heading not found: {heading}")

    table_lines = []
    for line in lines[start_idx + 1 :]:
        if line.strip().startswith("|"):
            table_lines.append(line.rstrip())
        elif table_lines:
            break

    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found after heading: {heading}")

    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) > len(header):
            overflow = len(cells) - len(header)
            merge_start = 2
            merge_end = merge_start + overflow + 1
            merged = "|".join(cells[merge_start:merge_end]).strip()
            cells = cells[:merge_start] + [merged] + cells[merge_end:]
        if len(cells) == len(header):
            rows.append(cells)

    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


def load_rr_segments_union(base_dir: str, mouse_id: str):
    data_path = str(Path(base_dir) / mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)
    segments_all, labels, neuron_pos_all = preprocess_spike_data(
        neuron_data, neuron_pos, start_edges, stimulus_data, extract_rr=False
    )
    segments_all = np.asarray(segments_all, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rr_map_orig = rr_selection_class(segments_all, labels)
    rr_union_orig = sorted(set().union(*rr_map_orig.values())) if rr_map_orig else []
    if len(rr_union_orig) == 0:
        raise RuntimeError(f"[{mouse_id}] rr_selection_class returned empty RR union.")
    segments_rr = segments_all[:, rr_union_orig, :]
    neuron_pos_rr = np.asarray(neuron_pos_all)[:, rr_union_orig]
    idx_map = {orig: i for i, orig in enumerate(rr_union_orig)}
    rr_map_local = {}
    for cls, idxs in rr_map_orig.items():
        rr_map_local[int(cls)] = sorted(idx_map[i] for i in idxs if i in idx_map)
    return segments_rr, labels, neuron_pos_rr, rr_map_local


def load_rr_segments_default(base_dir: str, mouse_id: str):
    data_path = str(Path(base_dir) / mouse_id)
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)
    segments_rr, labels, neuron_pos_rr = preprocess_spike_data(
        neuron_data, neuron_pos, start_edges, stimulus_data, extract_rr=True
    )
    return np.asarray(segments_rr, dtype=float), np.asarray(labels, dtype=int), np.asarray(neuron_pos_rr, dtype=float)


def balanced_trial_indices(labels: np.ndarray, classes: list[int], rng: np.random.Generator):
    counts = [int(np.sum(labels == c)) for c in classes]
    if any(c <= 0 for c in counts):
        raise RuntimeError(f"Cannot balance classes with empty class: {dict(zip(classes, counts))}")
    min_count = int(min(counts))
    picks = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        choose = rng.choice(idx, size=min_count, replace=False)
        picks[c] = np.asarray(choose, dtype=int)
    return picks, min_count


def extract_time_decoder_curves(args, out_root: Path, group_backfill_dir: Path):
    print("[*] Step 1/5: extracting time-decoding curves...")
    rows = []
    rng = np.random.default_rng(args.seed)
    for mouse_id in args.mice:
        out_csv = out_root / mouse_id / "data" / "time_decoder_curve_backfill.csv"
        ensure_dir(out_csv.parent)

        if out_csv.exists() and (not args.overwrite):
            df_mouse = pd.read_csv(out_csv)
            rows.append(df_mouse)
            print(f"  - reuse existing: {out_csv}")
            continue

        try:
            segments_rr, labels, _ = load_rr_segments_default(args.base_dir, mouse_id)
            acc, t_sec, acc_std, n_folds = classify_by_timepoints(segments_rr, labels)

            shuf_acc_all = []
            shuf_std_all = []
            for rep in range(max(1, int(args.time_shuffle_repeats))):
                y_perm = rng.permutation(labels)
                sh_acc, _, sh_std, _ = classify_by_timepoints(segments_rr, y_perm)
                shuf_acc_all.append(sh_acc)
                shuf_std_all.append(sh_std)
            shuf_acc_all = np.asarray(shuf_acc_all, dtype=float)
            shuf_std_all = np.asarray(shuf_std_all, dtype=float)
            shuf_acc = np.nanmean(shuf_acc_all, axis=0)
            shuf_acc_sem = (
                stats.sem(shuf_acc_all, axis=0, nan_policy="omit")
                if shuf_acc_all.shape[0] > 1
                else np.zeros_like(shuf_acc)
            )
            shuf_fold_std = np.nanmean(shuf_std_all, axis=0)

            df_mouse = pd.DataFrame(
                {
                    "mouse_id": mouse_id,
                    "frame_idx": np.arange(len(acc), dtype=int),
                    "time_sec": t_sec,
                    "accuracy": acc,
                    "accuracy_std_cv": acc_std,
                    "shuffle_accuracy": shuf_acc,
                    "shuffle_accuracy_sem_repeat": shuf_acc_sem,
                    "shuffle_accuracy_std_cv": shuf_fold_std,
                    "n_folds": int(n_folds),
                    "shuffle_repeats": int(shuf_acc_all.shape[0]),
                }
            )
            df_mouse.to_csv(out_csv, index=False)
            rows.append(df_mouse)
            print(f"  - wrote: {out_csv}")
        except Exception as exc:
            print(f"[!] time-decoding failed for {mouse_id}: {exc}")

    if not rows:
        print("[!] Step 1 skipped: no mouse decoded.")
        return

    df_all = pd.concat(rows, ignore_index=True)
    all_csv = group_backfill_dir / "time_decoder_curve_by_mouse_long.csv"
    df_all.to_csv(all_csv, index=False)

    agg = (
        df_all.groupby("time_sec", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sem=("accuracy", lambda x: stats.sem(x, nan_policy="omit")),
            shuffle_mean=("shuffle_accuracy", "mean"),
            shuffle_sem=("shuffle_accuracy", lambda x: stats.sem(x, nan_policy="omit")),
            n_mice=("mouse_id", "nunique"),
        )
        .sort_values("time_sec")
        .reset_index(drop=True)
    )
    agg_csv = group_backfill_dir / "time_decoder_curve_group_summary.csv"
    agg.to_csv(agg_csv, index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    t = agg["time_sec"].to_numpy(float)
    y = agg["accuracy_mean"].to_numpy(float)
    se = np.nan_to_num(agg["accuracy_sem"].to_numpy(float), nan=0.0)
    ys = agg["shuffle_mean"].to_numpy(float)
    ses = np.nan_to_num(agg["shuffle_sem"].to_numpy(float), nan=0.0)
    ax.plot(t, y, color="#5F7088", lw=2.1, label="True labels")
    ax.fill_between(t, y - se, y + se, color="#5F7088", alpha=0.17, linewidth=0)
    ax.plot(t, ys, color="#D2CCC3", lw=1.9, ls="--", label="Shuffled labels")
    ax.fill_between(t, ys - ses, ys + ses, color="#D2CCC3", alpha=0.20, linewidth=0)
    ax.axvspan((args.response_start - 10) / 4.0, (args.response_end - 10) / 4.0, color="#BFBFBF", alpha=0.12)
    ax.set_xlabel("Time (s, relative to stimulus onset)")
    ax.set_ylabel("Decoder accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, grid=False)
    save_variants(fig, group_backfill_dir / "Fig1_PanelD_time_decoder_group_true_vs_shuffle.png")
    print(f"  - wrote: {all_csv}")
    print(f"  - wrote: {agg_csv}")


def extract_fig1_c1_c2(args, out_root: Path, group_backfill_dir: Path):
    print("[*] Step 2/5: extracting Fig1 C1/C2 gap data...")
    mouse_id = args.representative_mouse
    rng = np.random.default_rng(args.seed + 11)
    try:
        segments_rr, labels, _, _ = load_rr_segments_union(args.base_dir, mouse_id)
    except Exception as exc:
        print(f"[!] Step 2 skipped ({mouse_id}): {exc}")
        return

    classes = [c for c in [1, 2, 3] if np.sum(labels == c) > 0]
    if len(classes) < 2:
        print(f"[!] Step 2 skipped ({mouse_id}): fewer than 2 valid classes.")
        return
    rw = slice(int(args.response_start), int(args.response_end))
    n_trials, n_neurons, n_time = segments_rr.shape
    time_idx = np.arange(n_time, dtype=int)
    time_sec = (time_idx - int(args.response_start)) / 4.0

    # ----- C1-1: unsorted trial x neuron response matrix (existing method style) -----
    X_trial = np.nanmean(segments_rr[:, :, rw], axis=2)  # (trial, neuron)
    picks, min_count = balanced_trial_indices(labels, classes, rng)
    row_blocks = []
    meta_rows = []
    row_cursor = 0
    for c in classes:
        idx = picks[c]
        x_c = X_trial[idx]
        row_blocks.append(x_c)
        for j, trial_idx in enumerate(idx.tolist()):
            meta_rows.append(
                {
                    "row_idx": int(row_cursor + j),
                    "mouse_id": mouse_id,
                    "class_id": int(c),
                    "condition": CLASS_TO_CONDITION.get(int(c), f"class_{c}"),
                    "orig_trial_idx": int(trial_idx),
                }
            )
        row_cursor += len(idx)
    X_bal = np.vstack(row_blocks).astype(float)
    y_bal = np.concatenate([np.full(min_count, c, dtype=int) for c in classes])

    X_bal_z = stats.zscore(X_bal, axis=0, nan_policy="omit")
    X_bal_z = np.nan_to_num(X_bal_z, nan=0.0, posinf=0.0, neginf=0.0)

    mat_npy = group_backfill_dir / f"fig1_c1_unsorted_trial_neuron_matrix_{mouse_id}.npy"
    meta_csv = group_backfill_dir / f"fig1_c1_unsorted_trial_neuron_matrix_{mouse_id}_meta.csv"
    np.save(mat_npy, X_bal_z.astype(np.float32))
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    sns.heatmap(
        X_bal_z,
        ax=ax,
        cmap="viridis",
        vmin=-2.0,
        vmax=2.0,
        cbar_kws={"label": "Z-scored trial response"},
        xticklabels=False,
        yticklabels=False,
    )
    boundaries = np.cumsum([np.sum(y_bal == c) for c in classes])[:-1]
    for b in boundaries:
        ax.hlines(b, *ax.get_xlim(), colors="white", linestyles="--", linewidth=1.0)
    y_centers = []
    s = 0
    for c in classes:
        cnt = int(np.sum(y_bal == c))
        y_centers.append(s + cnt / 2.0)
        s += cnt
    ax.set_yticks(y_centers)
    ax.set_yticklabels([CLASS_TO_CONDITION.get(c, str(c)) for c in classes], rotation=0)
    ax.set_xlabel("Neurons (unsorted)")
    ax.set_ylabel("Trials (balanced by condition)")
    save_variants(fig, group_backfill_dir / f"Fig1_PanelC1_unsorted_trial_response_matrix_{mouse_id}.png")

    # ----- C1-2: representative single-trial neuron x time heatmaps -----
    rep_rows = []
    rep_fig, rep_axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4.2), dpi=300, sharey=True)
    rep_axes = np.atleast_1d(rep_axes)
    for k, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        trial_resp = np.nanmean(segments_rr[idx, :, rw], axis=(1, 2))
        med = np.nanmedian(trial_resp)
        rep_idx = int(idx[np.argmin(np.abs(trial_resp - med))])
        mat = np.asarray(segments_rr[rep_idx], dtype=float)  # (neuron, time)
        mat_z = stats.zscore(mat, axis=1, nan_policy="omit")
        mat_z = np.nan_to_num(mat_z, nan=0.0, posinf=0.0, neginf=0.0)

        ax = rep_axes[k]
        im = ax.imshow(mat_z, aspect="auto", cmap="mako", vmin=-2.2, vmax=2.2, interpolation="nearest")
        ax.set_title(f"{CLASS_TO_CONDITION.get(c, c)}\ntrial={rep_idx}")
        ax.set_xlabel("Frame")
        if k == 0:
            ax.set_ylabel("RR neuron (unsorted)")
        ax.axvspan(int(args.response_start), int(args.response_end), color="white", alpha=0.12, lw=0)

        ii, jj = np.meshgrid(np.arange(mat.shape[0], dtype=int), np.arange(mat.shape[1], dtype=int), indexing="ij")
        rep_rows.append(
            pd.DataFrame(
                {
                    "mouse_id": mouse_id,
                    "class_id": int(c),
                    "condition": CLASS_TO_CONDITION.get(c, str(c)),
                    "orig_trial_idx": int(rep_idx),
                    "neuron_idx": ii.ravel(),
                    "time_idx": jj.ravel(),
                    "time_sec": (jj.ravel() - int(args.response_start)) / 4.0,
                    "value": mat.ravel(),
                    "z_value": mat_z.ravel(),
                }
            )
        )
    cbar = rep_fig.colorbar(im, ax=rep_axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Z-scored activity")
    save_variants(rep_fig, group_backfill_dir / f"Fig1_PanelC1b_unsorted_neuron_time_representative_trials_{mouse_id}.png")
    rep_csv = group_backfill_dir / f"fig1_c1_representative_trial_neuron_time_long_{mouse_id}.csv"
    pd.concat(rep_rows, ignore_index=True).to_csv(rep_csv, index=False)

    # ----- C2: population + single-cell mean traces -----
    pop_rows = []
    for c in classes:
        cond = CLASS_TO_CONDITION.get(c, str(c))
        cls_trials = segments_rr[labels == c]
        pop_mean = np.nanmean(cls_trials, axis=(0, 1))
        trial_means = np.nanmean(cls_trials, axis=1)
        pop_sem = stats.sem(trial_means, axis=0, nan_policy="omit")
        pop_rows.append(
            pd.DataFrame(
                {
                    "mouse_id": mouse_id,
                    "class_id": int(c),
                    "condition": cond,
                    "time_idx": time_idx,
                    "time_sec": time_sec,
                    "mean": pop_mean,
                    "sem": pop_sem,
                    "n_trials": int(cls_trials.shape[0]),
                    "trace_type": "population_rr",
                }
            )
        )
    pop_df = pd.concat(pop_rows, ignore_index=True)
    pop_csv = group_backfill_dir / f"fig1_c2_population_trace_long_{mouse_id}.csv"
    pop_df.to_csv(pop_csv, index=False)

    mean_resp = np.nanmean(segments_rr[:, :, rw], axis=(0, 2))
    top_n = max(1, int(args.single_cell_count))
    top_idx = np.argsort(mean_resp)[::-1][:top_n]
    cell_rows = []
    for rank, neuron_idx in enumerate(top_idx.tolist(), start=1):
        for c in classes:
            cond = CLASS_TO_CONDITION.get(c, str(c))
            cls_trials = segments_rr[labels == c, neuron_idx, :]
            c_mean = np.nanmean(cls_trials, axis=0)
            c_sem = stats.sem(cls_trials, axis=0, nan_policy="omit")
            cell_rows.append(
                pd.DataFrame(
                    {
                        "mouse_id": mouse_id,
                        "class_id": int(c),
                        "condition": cond,
                        "neuron_idx": int(neuron_idx),
                        "rank_by_response": int(rank),
                        "time_idx": time_idx,
                        "time_sec": time_sec,
                        "mean": c_mean,
                        "sem": c_sem,
                        "n_trials": int(cls_trials.shape[0]),
                        "trace_type": "single_cell",
                    }
                )
            )
    cell_df = pd.concat(cell_rows, ignore_index=True)
    cell_csv = group_backfill_dir / f"fig1_c2_singlecell_trace_long_{mouse_id}.csv"
    cell_df.to_csv(cell_csv, index=False)

    fig = plt.figure(figsize=(4.6 * top_n, 6.8), dpi=300)
    gs = fig.add_gridspec(2, max(1, top_n), height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.20)

    ax_top = fig.add_subplot(gs[0, :])
    for c in classes:
        cond = CLASS_TO_CONDITION.get(c, str(c))
        sub = pop_df[pop_df["condition"] == cond].sort_values("time_idx")
        x = sub["time_idx"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        se = np.nan_to_num(sub["sem"].to_numpy(float), nan=0.0)
        ax_top.plot(x, y, lw=2.1, color=COLORS.get(cond, "#444444"), label=cond)
        ax_top.fill_between(x, y - se, y + se, color=COLORS.get(cond, "#444444"), alpha=0.16, linewidth=0)
    ax_top.axvspan(int(args.response_start), int(args.response_end), color="#BFBFBF", alpha=0.14)
    ax_top.set_ylabel("Mean activity")
    ax_top.set_xlabel("Frame")
    ax_top.legend(frameon=False, ncol=min(3, len(classes)), loc="upper right")
    style_axis(ax_top, grid=False)

    for j, neuron_idx in enumerate(top_idx.tolist()):
        ax = fig.add_subplot(gs[1, j])
        sub_n = cell_df[cell_df["neuron_idx"] == neuron_idx]
        for c in classes:
            cond = CLASS_TO_CONDITION.get(c, str(c))
            ss = sub_n[sub_n["condition"] == cond].sort_values("time_idx")
            x = ss["time_idx"].to_numpy(float)
            y = ss["mean"].to_numpy(float)
            se = np.nan_to_num(ss["sem"].to_numpy(float), nan=0.0)
            ax.plot(x, y, lw=1.8, color=COLORS.get(cond, "#444444"), label=cond)
            ax.fill_between(x, y - se, y + se, color=COLORS.get(cond, "#444444"), alpha=0.16, linewidth=0)
        ax.axvspan(int(args.response_start), int(args.response_end), color="#BFBFBF", alpha=0.14)
        ax.set_title(f"Neuron {neuron_idx} (top {j+1})", fontsize=10)
        ax.set_xlabel("Frame")
        if j == 0:
            ax.set_ylabel("Activity")
        style_axis(ax, grid=False)
    save_variants(fig, group_backfill_dir / f"Fig1_PanelC2_population_singlecell_traces_{mouse_id}.png")
    print(f"  - wrote C1/C2 outputs for {mouse_id}")


def ensure_weakcorr_metrics(args, out_root: Path, roots, group_backfill_dir: Path):
    print("[*] Step 3/5: ensuring weak-correlation reorganization tables...")
    rel_mouse = "group_summary/group_weakcorr_reorg_mouse_metrics.csv"
    rel_stats = "group_summary/group_weakcorr_reorg_stats.csv"

    weak_mouse = maybe_copy_from_other_roots(out_root, roots, rel_mouse) or (out_root / rel_mouse)
    weak_stats = maybe_copy_from_other_roots(out_root, roots, rel_stats) or (out_root / rel_stats)

    if (not weak_mouse.exists()) and args.auto_run_weakcorr:
        script_path = ROOT / "notebook" / "run_weakcorr_reorganization_analyses.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--base-dir",
            str(args.base_dir),
            "--results-dir",
            str(out_root),
            "--mice",
            *args.mice,
        ]
        if args.weakcorr_no_raw_fallback:
            cmd.append("--no-raw-fallback")
        print(f"  - running existing weakcorr script: {' '.join(cmd)}")
        run = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if run.returncode != 0:
            print("[!] weakcorr script returned non-zero status.")
            if run.stdout:
                print(run.stdout[-2000:])
            if run.stderr:
                print(run.stderr[-2000:])

    if not weak_mouse.exists():
        print("[!] Step 3 warning: missing group_weakcorr_reorg_mouse_metrics.csv")
        return

    df = pd.read_csv(weak_mouse)
    out_mouse = group_backfill_dir / "fig2_weakcorr_mouse_metrics_for_plot.csv"
    df.to_csv(out_mouse, index=False)
    print(f"  - wrote: {out_mouse}")

    if weak_stats.exists():
        df_stats = pd.read_csv(weak_stats)
        out_stats = group_backfill_dir / "fig2_weakcorr_stats_for_plot.csv"
        df_stats.to_csv(out_stats, index=False)
        print(f"  - wrote: {out_stats}")

    # Dedicated paired panel for neg_frac and weak_pos_frac (Coherent vs Random)
    cond_set = set(df["Condition"].astype(str).tolist())
    coh_name = "Coherent" if "Coherent" in cond_set else ("Convergent" if "Convergent" in cond_set else None)
    rand_name = "Random" if "Random" in cond_set else None
    if coh_name is None or rand_name is None:
        print("[!] Step 3 warning: cannot build neg/weak paired panel (missing Coherent/Random).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=300)
    metrics = [("neg_frac", "Negative fraction"), ("weak_pos_frac", "Weak positive fraction")]
    delta_rows = []
    for ax, (metric, ylab) in zip(axes, metrics):
        piv = (
            df[df["Condition"].isin([coh_name, rand_name])]
            .pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean", observed=False)
            .dropna()
        )
        if piv.empty:
            ax.set_title(f"{metric}: no data")
            continue
        x = np.array([0.0, 1.0])
        for mouse_id, row in piv.iterrows():
            a = float(row[coh_name])
            b = float(row[rand_name])
            ax.plot(x, [a, b], color="#A9A39A", lw=1.0, alpha=0.8)
            ax.scatter(x, [a, b], color=[COLORS.get(coh_name, "#5F7088"), COLORS.get(rand_name, "#B98372")], s=26, zorder=3)
            delta_rows.append({"mouse_id": mouse_id, "metric": metric, "coherent_minus_random": a - b})
        try:
            stat = stats.wilcoxon(piv[coh_name], piv[rand_name])
            p = float(stat.pvalue)
        except Exception:
            p = np.nan
        ax.set_xticks([0, 1], [coh_name, rand_name])
        ax.set_ylabel(ylab)
        ax.text(
            0.5,
            0.98,
            f"p={p:.3g}" if np.isfinite(p) else "p=n/a",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D2CCC3", alpha=0.85),
        )
        style_axis(ax, grid=False)
    save_variants(fig, group_backfill_dir / "Fig2_PanelD_neg_weak_paired_group.png")

    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(group_backfill_dir / "fig2_neg_weak_mouse_deltas.csv", index=False)


def geometry_metrics(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return {
            "mean_norm": np.nan,
            "angle_deg": np.nan,
            "var_parallel": np.nan,
            "var_orthogonal": np.nan,
            "orth_parallel_ratio": np.nan,
            "anisotropy_index": np.nan,
            "lambda1": np.nan,
            "lambda2": np.nan,
        }
    mu = np.mean(X, axis=0)
    mu_norm = float(np.linalg.norm(mu))
    mu_hat = mu / (mu_norm + EPS)
    Y = X - mu
    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    v1 = vt[0]
    v1n = float(np.linalg.norm(v1))
    if mu_norm <= EPS or v1n <= EPS:
        angle_deg = np.nan
    else:
        angle_deg = float(np.degrees(np.arccos(np.clip(np.abs(np.dot(mu_hat, v1 / v1n)), 0.0, 1.0))))
    if mu_norm <= EPS:
        var_parallel = np.nan
        var_orth = np.nan
        ratio = np.nan
    else:
        a = Y @ mu_hat
        var_parallel = float(np.mean(a**2))
        r = Y - np.outer(a, mu_hat)
        var_orth = float(np.mean(np.sum(r**2, axis=1)))
        ratio = float(var_orth / (var_parallel + EPS))
    eig = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(Y, rowvar=False)), 0.0))[::-1]
    lam1 = float(eig[0]) if eig.size >= 1 else np.nan
    lam2 = float(eig[1]) if eig.size >= 2 else np.nan
    anis = float(lam1 / (np.sum(eig) + EPS)) if eig.size > 0 else np.nan
    return {
        "mean_norm": mu_norm,
        "angle_deg": angle_deg,
        "var_parallel": var_parallel,
        "var_orthogonal": var_orth,
        "orth_parallel_ratio": ratio,
        "anisotropy_index": anis,
        "lambda1": lam1,
        "lambda2": lam2,
    }


def plot_cov_ellipse(ax, xy, color):
    arr = np.asarray(xy, dtype=float)
    if arr.shape[0] < 3:
        return
    cov = np.cov(arr.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * np.sqrt(np.maximum(vals[:2], EPS))
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(
        Ellipse(
            (np.mean(arr[:, 0]), np.mean(arr[:, 1])),
            width,
            height,
            angle=ang,
            edgecolor=color,
            facecolor=color,
            alpha=0.14,
            lw=1.7,
        )
    )


def extract_geometry_projection(args, group_backfill_dir: Path):
    print("[*] Step 4/5: extracting representative geometry projection data...")
    mouse_id = args.representative_mouse
    rng = np.random.default_rng(args.seed + 29)
    try:
        segments_rr, labels, _, _ = load_rr_segments_union(args.base_dir, mouse_id)
    except Exception as exc:
        print(f"[!] Step 4 skipped ({mouse_id}): {exc}")
        return

    classes = [c for c in [1, 2, 3] if np.sum(labels == c) > 0]
    if len(classes) < 2:
        print(f"[!] Step 4 skipped ({mouse_id}): fewer than 2 valid classes.")
        return

    X_trial = np.nanmean(segments_rr[:, :, int(args.response_start) : int(args.response_end)], axis=2)
    picks, _ = balanced_trial_indices(labels, classes, rng)
    X_resp = np.vstack([X_trial[picks[c]] for c in classes]).astype(float)
    y_resp = np.concatenate([np.full(len(picks[c]), c, dtype=int) for c in classes])

    rows = []
    cond_metrics = []
    fig, axes = plt.subplots(1, len(classes), figsize=(4.8 * len(classes), 4.2), dpi=300)
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, classes):
        cond = CLASS_TO_CONDITION.get(int(c), f"class_{c}")
        Xc = X_resp[y_resp == c]
        mu = np.mean(Xc, axis=0)
        Y = Xc - mu
        _, _, vt = np.linalg.svd(Y, full_matrices=False)
        basis = vt[:2].T
        z = Y @ basis
        mu_proj = mu @ basis

        gm = geometry_metrics(Xc)
        gm["mouse_id"] = mouse_id
        gm["class_id"] = int(c)
        gm["condition"] = cond
        cond_metrics.append(gm)

        scale = max(float(np.nanstd(z[:, 0])), float(np.nanstd(z[:, 1])), 1e-3)
        ax.scatter(z[:, 0], z[:, 1], s=22, alpha=0.42, color=COLORS.get(cond, "#555555"), edgecolor="none")
        plot_cov_ellipse(ax, z, COLORS.get(cond, "#555555"))
        ax.arrow(0, 0, scale, 0, color="#111111", head_width=0.06 * scale, length_includes_head=True)
        ax.arrow(0, 0, mu_proj[0], mu_proj[1], color="#8C4A3E", head_width=0.06 * scale, length_includes_head=True)
        ax.set_title(f"{cond}\nangle={gm['angle_deg']:.2f} deg")
        ax.set_xlabel("PC1")
        if ax is axes[0]:
            ax.set_ylabel("PC2")
        style_axis(ax, grid=False)

        for i in range(Xc.shape[0]):
            rows.append(
                {
                    "mouse_id": mouse_id,
                    "class_id": int(c),
                    "condition": cond,
                    "trial_rank_in_condition": int(i),
                    "z1": float(z[i, 0]),
                    "z2": float(z[i, 1]),
                    "mu_proj1": float(mu_proj[0]),
                    "mu_proj2": float(mu_proj[1]),
                }
            )

    proj_csv = group_backfill_dir / f"fig3_d2_geometry_projection_trials_{mouse_id}.csv"
    pd.DataFrame(rows).to_csv(proj_csv, index=False)
    cond_csv = group_backfill_dir / f"fig3_d2_geometry_projection_condition_metrics_{mouse_id}.csv"
    pd.DataFrame(cond_metrics).to_csv(cond_csv, index=False)
    save_variants(fig, group_backfill_dir / f"Fig3_PanelD2_state_space_projection_{mouse_id}.png")
    print(f"  - wrote: {proj_csv}")
    print(f"  - wrote: {cond_csv}")


def extract_model_and_fc_tables(out_root: Path, roots, group_backfill_dir: Path):
    print("[*] Step 5/5: extracting markdown report tables to CSV...")

    # ---- Modeling v4 ----
    model_report = find_existing_rel(roots, "group_summary/Group_Modeling_v4_Report.md")
    if model_report is not None:
        txt = model_report.read_text(encoding="utf-8")
        try:
            df_hyper = parse_markdown_table_after_heading(txt, "## Search Hyperparameters")
            df_hyper.to_csv(group_backfill_dir / "model_v4_search_hyperparameters.csv", index=False)
        except Exception:
            pass
        target_dict = {}
        for head, name in [("### FC", "fc"), ("### Allocation", "allocation"), ("### Geometry", "geometry")]:
            try:
                df_t = parse_markdown_table_after_heading(txt, head)
                df_t.to_csv(group_backfill_dir / f"model_v4_targets_{name}.csv", index=False)
                if not df_t.empty:
                    row = df_t.iloc[0].to_dict()
                    for k, v in row.items():
                        try:
                            target_dict[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception:
                continue
        best_df = None
        try:
            best_df = parse_markdown_table_after_heading(txt, "## Best Parameter Row")
            best_df.to_csv(group_backfill_dir / "model_v4_best_parameter_row.csv", index=False)
        except Exception:
            pass
        try:
            top_df = parse_markdown_table_after_heading(txt, "## Top Parameter Rows")
            top_df.to_csv(group_backfill_dir / "model_v4_top_parameter_rows.csv", index=False)
        except Exception:
            pass

        if best_df is not None and (not best_df.empty) and target_dict:
            best_row = best_df.iloc[0].to_dict()
            tidy_rows = []
            for metric, target in target_dict.items():
                if metric in best_row:
                    try:
                        model_val = float(best_row[metric])
                        tidy_rows.append(
                            {
                                "metric": metric,
                                "target_value": float(target),
                                "model_value_best_row": model_val,
                                "model_minus_target": model_val - float(target),
                            }
                        )
                    except Exception:
                        pass
            if tidy_rows:
                pd.DataFrame(tidy_rows).to_csv(group_backfill_dir / "model_v4_best_vs_target_tidy.csv", index=False)
        print(f"  - parsed modeling report: {model_report}")
    else:
        print("[!] modeling v4 report not found.")

    # ---- FC decoder focus ----
    fc_report = find_existing_rel(roots, "group_summary/Group_FC_Decoder_Focus_Report.md")
    if fc_report is not None:
        txt = fc_report.read_text(encoding="utf-8")
        heading_map = [
            ("## Decoder summary", "fc_focus_decoder_summary.csv"),
            ("## Weak-edge summary by score", "fc_focus_weak_edge_by_score.csv"),
            ("## Decoder using decile0-2 edges only", "fc_focus_decile0_2_decoder.csv"),
            ("## Group weak-edge aggregate", "fc_focus_group_weak_edge_aggregate.csv"),
        ]
        for heading, filename in heading_map:
            try:
                df = parse_markdown_table_after_heading(txt, heading)
                df.to_csv(group_backfill_dir / filename, index=False)
            except Exception:
                continue
        print(f"  - parsed FC focus report: {fc_report}")
    else:
        print("[!] FC decoder focus report not found.")


def write_manifest(group_backfill_dir: Path):
    rows = []
    for p in sorted(group_backfill_dir.glob("*")):
        if p.is_file():
            rows.append(
                {
                    "filename": p.name,
                    "size_bytes": int(p.stat().st_size),
                    "relative_path": str(p.relative_to(group_backfill_dir.parent.parent)),
                }
            )
    df = pd.DataFrame(rows)
    manifest_csv = group_backfill_dir / "backfill_manifest.csv"
    df.to_csv(manifest_csv, index=False)
    md = group_backfill_dir / "backfill_manifest.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# 20260419 Gap Backfill Manifest\n\n")
        if df.empty:
            f.write("_empty_\n")
        else:
            try:
                f.write(df.to_markdown(index=False) + "\n")
            except Exception:
                f.write("```\n" + df.to_string(index=False) + "\n```\n")


def write_plot_data_index(group_backfill_dir: Path, representative_mouse: str):
    """Write an explicit figure->data mapping so panels can be re-drawn quickly."""
    m = representative_mouse
    mapping = [
        (
            f"Fig1_PanelC1_unsorted_trial_response_matrix_{m}.png",
            [
                f"fig1_c1_unsorted_trial_neuron_matrix_{m}.npy",
                f"fig1_c1_unsorted_trial_neuron_matrix_{m}_meta.csv",
            ],
            "Fig1 C1 (unsorted trial x neuron matrix)",
        ),
        (
            f"Fig1_PanelC1b_unsorted_neuron_time_representative_trials_{m}.png",
            [
                f"fig1_c1_representative_trial_neuron_time_long_{m}.csv",
            ],
            "Fig1 C1b (representative trial neuron x time heatmaps)",
        ),
        (
            f"Fig1_PanelC2_population_singlecell_traces_{m}.png",
            [
                f"fig1_c2_population_trace_long_{m}.csv",
                f"fig1_c2_singlecell_trace_long_{m}.csv",
            ],
            "Fig1 C2 (population + single-cell traces)",
        ),
        (
            "Fig1_PanelD_time_decoder_group_true_vs_shuffle.png",
            [
                "time_decoder_curve_by_mouse_long.csv",
                "time_decoder_curve_group_summary.csv",
            ],
            "Fig1 D (time decoding true vs shuffle)",
        ),
        (
            "Fig2_PanelD_neg_weak_paired_group.png",
            [
                "fig2_weakcorr_mouse_metrics_for_plot.csv",
                "fig2_weakcorr_stats_for_plot.csv",
                "fig2_neg_weak_mouse_deltas.csv",
            ],
            "Fig2 D (neg_frac + weak_pos_frac paired panel)",
        ),
        (
            f"Fig3_PanelD2_state_space_projection_{m}.png",
            [
                f"fig3_d2_geometry_projection_trials_{m}.csv",
                f"fig3_d2_geometry_projection_condition_metrics_{m}.csv",
            ],
            "Fig3 D2 (state-space projection)",
        ),
    ]

    rows = []
    for image_name, data_names, note in mapping:
        image_path = group_backfill_dir / image_name
        data_exist = []
        data_missing = []
        for dn in data_names:
            p = group_backfill_dir / dn
            if p.exists():
                data_exist.append(dn)
            else:
                data_missing.append(dn)
        rows.append(
            {
                "panel_image": image_name,
                "image_exists": bool(image_path.exists()),
                "data_files_expected": "; ".join(data_names),
                "data_files_present": "; ".join(data_exist),
                "data_files_missing": "; ".join(data_missing),
                "all_data_present": len(data_missing) == 0,
                "note": note,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = group_backfill_dir / "plot_data_index.csv"
    df.to_csv(csv_path, index=False)

    md_path = group_backfill_dir / "plot_data_index.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Plot Data Index (Figure -> Replot Data)\n\n")
        f.write(
            "This table guarantees each key panel has corresponding raw/long-form data for rapid re-plot.\n\n"
        )
        try:
            f.write(df.to_markdown(index=False) + "\n")
        except Exception:
            f.write("```\n" + df.to_string(index=False) + "\n```\n")

    print(f"[*] wrote plot-data index: {csv_path}")
    missing_rows = df[(df["image_exists"] == True) & (df["all_data_present"] == False)]
    if not missing_rows.empty:
        print("[!] Some panel images exist but mapped data files are missing:")
        for _, r in missing_rows.iterrows():
            print(f"    - {r['panel_image']} missing: {r['data_files_missing']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing analyses/data for the 2026-04-19 paper re-visualization plan. "
            "Methods are aligned to existing pipeline scripts."
        )
    )
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--fallback-results-dir", type=str, default=str(ROOT / "result"))
    parser.add_argument("--mice", nargs="*", default=DEFAULT_MOUSE_IDS)
    parser.add_argument("--representative-mouse", type=str, default="M73_1128")
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--response-start", type=int, default=10)
    parser.add_argument("--response-end", type=int, default=13)
    parser.add_argument("--single-cell-count", type=int, default=3)
    parser.add_argument("--time-shuffle-repeats", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auto-run-weakcorr", action="store_true", default=True)
    parser.add_argument("--weakcorr-no-raw-fallback", action="store_true", default=True)
    parser.add_argument("--skip-time-decoder", action="store_true")
    parser.add_argument("--skip-fig1-gaps", action="store_true")
    parser.add_argument("--skip-weakcorr", action="store_true")
    parser.add_argument("--skip-geometry-projection", action="store_true")
    parser.add_argument("--skip-report-extract", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root, roots = get_roots(args)
    group_backfill_dir = out_root / "group_summary" / "paper_backfill_20260419"
    ensure_dir(group_backfill_dir)

    print("[*] ===== 20260419 Gap Backfill Start =====")
    print(f"[*] Output root: {out_root}")
    print(f"[*] Backfill dir: {group_backfill_dir}")

    if not args.skip_time_decoder:
        extract_time_decoder_curves(args, out_root, group_backfill_dir)
    else:
        print("[*] Step 1 skipped by flag.")

    if not args.skip_fig1_gaps:
        extract_fig1_c1_c2(args, out_root, group_backfill_dir)
    else:
        print("[*] Step 2 skipped by flag.")

    if not args.skip_weakcorr:
        ensure_weakcorr_metrics(args, out_root, roots, group_backfill_dir)
    else:
        print("[*] Step 3 skipped by flag.")

    if not args.skip_geometry_projection:
        extract_geometry_projection(args, group_backfill_dir)
    else:
        print("[*] Step 4 skipped by flag.")

    if not args.skip_report_extract:
        extract_model_and_fc_tables(out_root, roots, group_backfill_dir)
    else:
        print("[*] Step 5 skipped by flag.")

    write_manifest(group_backfill_dir)
    write_plot_data_index(group_backfill_dir, args.representative_mouse)
    print("[*] ===== 20260419 Gap Backfill Done =====")


if __name__ == "__main__":
    main()
