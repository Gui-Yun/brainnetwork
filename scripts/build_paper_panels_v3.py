from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import shutil
from PIL import Image
from scipy import stats


BASE = Path(__file__).resolve().parents[1]
RESULT_ROOTS = [BASE / "result", BASE / "results"]
OUT_DIR = BASE / "result" / "paper_panels_20260331_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOUSE_EXAMPLE = "M73_1128"
CONDITIONS = ["Divergent", "Convergent", "Random"]
COLORS = {
    "Divergent": "#7F9C96",
    "Convergent": "#8B90A8",
    "Random": "#B98372",
}
STRONG_TONES = {
    "Divergent": "#5F7E77",
    "Convergent": "#666C86",
    "Random": "#8E5E50",
}
WEAK_TONES = {
    "Divergent": "#B8CBC6",
    "Convergent": "#C3C6D5",
    "Random": "#D9B7AD",
}
NEUTRAL = {
    "text": "#2F2F2F",
    "paired": "#A9A39A",
    "grid": "#E9E5DF",
}
DECODER_COLORS = {"Full FC": "#5F7088", "Weak-edge FC": "#AAB5C3", "Shuffle": "#D2CCC3"}
GEOM_COLORS = {"Parallel": "#3F6B92", "Orthogonal": "#6C4F9D"}


@dataclass
class PanelResult:
    panel: str
    filename: str
    status: str
    source: str
    note: str = ""


def style_axis(ax, light_grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    if light_grid:
        ax.grid(axis="y", color="#E9E5DF", lw=0.8, alpha=0.8)
    else:
        ax.grid(False)


def p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _condition_pairwise_pvals(piv: pd.DataFrame) -> dict[tuple[str, str], float]:
    out = {}
    pairs = [("Divergent", "Convergent"), ("Divergent", "Random"), ("Convergent", "Random")]
    for a, b in pairs:
        valid = piv[[a, b]].dropna()
        if len(valid) < 3:
            out[(a, b)] = np.nan
            continue
        try:
            _, p = stats.wilcoxon(valid[a], valid[b])
            out[(a, b)] = float(p)
        except Exception:
            out[(a, b)] = np.nan
    return out


def add_sig_brackets(ax, pairs, pvals, x_index: dict[str, float], y_top_pad: float = 0.04, step_frac: float = 0.085):
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if y1 > y0 else 1.0
    base = y1 + y_top_pad * yr
    step = step_frac * yr
    n_drawn = 0
    for i, (a, b) in enumerate(pairs):
        p = pvals.get((a, b), np.nan)
        star = p_to_star(p)
        if star == "ns" or star == "n/a":
            continue
        x1, x2 = x_index[a], x_index[b]
        y = base + n_drawn * step
        ax.plot([x1, x1, x2, x2], [y, y + 0.18 * step, y + 0.18 * step, y], lw=1.1, c=NEUTRAL["text"])
        ax.text((x1 + x2) * 0.5, y + 0.23 * step, star, ha="center", va="bottom", fontsize=9.0, color=NEUTRAL["text"])
        n_drawn += 1
    if n_drawn > 0:
        ax.set_ylim(y0, base + n_drawn * step + 0.9 * step)


def find_first_existing(*relative_paths: str) -> Path:
    for rel in relative_paths:
        for root in RESULT_ROOTS:
            p = root / rel
            if p.exists():
                return p
    raise FileNotFoundError(f"No file found for any of: {relative_paths}")


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
            # Handle inline "|" inside free-text columns (e.g., subset_desc with "|mean corr|").
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


def load_activity_decoder_summary() -> pd.DataFrame:
    rows = []
    for m in sorted((BASE / "result").glob("M*")):
        fp = m / "data" / "decoder_summary.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append(
            {
                "mouse_id": m.name,
                "true_acc": float(r["accuracy_mean"]),
                "shuffle_acc": float(r["shuffle_accuracy_mean"]),
            }
        )
    if not rows:
        raise ValueError("No decoder_summary.csv available under result/*/data")
    return pd.DataFrame(rows)


def load_fc_focus_table() -> pd.DataFrame:
    report = find_first_existing("group_summary/Group_FC_Decoder_Focus_Report.md")
    text = report.read_text(encoding="utf-8")
    df = parse_markdown_table_after_heading(text, "## Decoder using decile0-2 edges only")
    keep = ["mouse_id", "full_accuracy_mean", "full_shuffle_accuracy_mean", "subset_accuracy_mean", "subset_shuffle_accuracy_mean"]
    return df[keep].copy()


def load_geometry_delta_table() -> pd.DataFrame:
    report = find_first_existing("group_summary/Group_Geometry_Report.md")
    text = report.read_text(encoding="utf-8")
    df = parse_markdown_table_after_heading(text, "### Per-mouse delta table")
    keep = ["mouse_id", "delta_parallel_coherent_minus_random", "delta_orthogonal_coherent_minus_random"]
    return df[keep].copy()


def load_rsm_original_means() -> pd.DataFrame:
    rows = []
    for p in sorted((BASE / "result").glob("M*/data/group_rsm_shuffle_long.csv")):
        mouse = p.parts[-3]
        d = pd.read_csv(p)
        d = d[d["data_type"] == "original"].copy()
        if d.empty:
            continue
        d["mouse_id"] = mouse
        rows.append(d[["mouse_id", "condition", "mean_rsm"]].rename(columns={"condition": "Condition"}))
    if not rows:
        raise ValueError("No original rows in group_rsm_shuffle_long.csv")
    return pd.concat(rows, ignore_index=True)


def _rgb_hex(rgb):
    return np.array([int(rgb[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def copy_image(src: Path, dst: Path):
    """Copy image file to dst; ensures parent exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def plot_fig1_panel_a_from_image(src: Path, out_path: Path):
    copy_image(src, out_path)


def plot_fig1_panel_b_from_image(src: Path, out_path: Path):
    copy_image(src, out_path)


def recolor_fig1_panel_c(src: Path, dst: Path):
    img = np.array(Image.open(src).convert("RGB")).astype(np.float32)
    out = img.copy()

    # Heuristic color remapping on line/ribbon pixels.
    # old blue -> Convergent; old orange -> Divergent; old green -> Random
    old_blue = (img[..., 2] > img[..., 1] + 15) & (img[..., 2] > img[..., 0] + 15)
    old_orange = (img[..., 0] > img[..., 1] + 20) & (img[..., 1] > img[..., 2] + 10)
    old_green = (img[..., 1] > img[..., 0] + 8) & (img[..., 1] > img[..., 2] + 8)

    mapping = [
        (old_blue, _rgb_hex(COLORS["Convergent"])),
        (old_orange, _rgb_hex(COLORS["Divergent"])),
        (old_green, _rgb_hex(COLORS["Random"])),
    ]

    white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    for mask, tgt in mapping:
        if not np.any(mask):
            continue
        pix = img[mask]
        # Keep ribbon-lightness differences by mixing with white based on pixel spread.
        sat_like = (np.max(pix, axis=1) - np.min(pix, axis=1)) / 255.0
        alpha = np.clip(0.18 + 1.05 * sat_like, 0.20, 1.0)[:, None]
        out[mask] = white * (1.0 - alpha) + tgt[None, :] * alpha

    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(dst)


def _side_by_side(img_left: Image.Image, img_right: Image.Image, gap: int = 24) -> Image.Image:
    h = max(img_left.height, img_right.height)
    def _resize(im):
        scale = h / im.height
        return im.resize((int(im.width * scale), h), Image.LANCZOS)
    left = _resize(img_left)
    right = _resize(img_right)
    canvas = Image.new("RGBA", (left.width + gap + right.width, h), (255, 255, 255, 255))
    canvas.paste(left, (0, 0), left)
    canvas.paste(right, (left.width + gap, 0), right)
    return canvas


def plot_fig1_panel_d_confusion(csv_path: Path, out_path: Path, extra_path: Path | None = None):
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "True"})
    if "True" in df.columns:
        df = df.set_index("True")
    df = df.reindex(index=CONDITIONS, columns=CONDITIONS)
    mat = df.to_numpy(dtype=float)
    mat = np.divide(mat, mat.sum(axis=1, keepdims=True), where=mat.sum(axis=1, keepdims=True) > 0)

    cmap = LinearSegmentedColormap.from_list("conf", ["#F3F1ED", "#B8C1CE", "#5F7088"])
    fig, ax = plt.subplots(figsize=(3.2, 2.9), dpi=300)
    icon_ticks = ["D←→", "C→←", "R*"]
    sns.heatmap(
        mat,
        vmin=0,
        vmax=1,
        cmap=cmap,
        square=True,
        cbar=True,
        linewidths=0.6,
        linecolor="#E3DED6",
        xticklabels=icon_ticks,
        yticklabels=icon_ticks,
        ax=ax,
    )
    ax.set_xlabel("Pred.")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Optional: combine with external accuracy plot
    if extra_path is not None and Path(extra_path).exists():
        left = Image.open(out_path).convert("RGBA")
        right = Image.open(extra_path).convert("RGBA")
        combo = _side_by_side(left, right, gap=32)
        combo.convert("RGB").save(out_path, dpi=(300, 300))


def plot_fig1_panel_e_group_decoder(dec_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=300)
    x = np.array([0, 1], dtype=float)
    for _, row in dec_df.iterrows():
        ax.plot(x, [row["true_acc"], row["shuffle_acc"]], color=NEUTRAL["paired"], lw=0.9, alpha=0.55, zorder=1)

    vals_true = dec_df["true_acc"].to_numpy(dtype=float)
    vals_shuffle = dec_df["shuffle_acc"].to_numpy(dtype=float)
    jit = np.linspace(-0.035, 0.035, len(dec_df))
    ax.scatter(np.full(len(dec_df), 0.0) + jit, vals_true, s=24, color=DECODER_COLORS["Full FC"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(np.full(len(dec_df), 1.0) + jit, vals_shuffle, s=24, color=DECODER_COLORS["Shuffle"], edgecolor="white", linewidth=0.5, zorder=3)

    for xi, vals, c in [(0, vals_true, DECODER_COLORS["Full FC"]), (1, vals_shuffle, DECODER_COLORS["Shuffle"])]:
        mu = np.nanmean(vals)
        se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
        ax.errorbar(xi, mu, yerr=se, fmt="D", color=NEUTRAL["text"], markersize=4.8, lw=1.2, capsize=0, zorder=4)
        ax.scatter([xi], [mu], s=36, color=c, edgecolor=NEUTRAL["text"], linewidth=0.7, zorder=5)

    p_val = np.nan
    try:
        _, p_val = stats.wilcoxon(vals_true, vals_shuffle)
        ax.text(0.02, 0.98, f"Wilcoxon p={p_val:.2e}", transform=ax.transAxes, ha="left", va="top", fontsize=7.3)
    except Exception:
        pass

    ax.axhline(1.0 / 3.0, lw=1.1, ls="--", color="#8D8A84")
    ax.set_xticks([0, 1], ["True", "Shuffle"])
    ax.set_ylabel("Classification accuracy")
    if np.isfinite(p_val):
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        y = y1 + 0.03 * yr
        ax.plot([0, 0, 1, 1], [y, y + 0.02 * yr, y + 0.02 * yr, y], lw=1.1, c=NEUTRAL["text"])
        ax.text(0.5, y + 0.025 * yr, p_to_star(p_val), ha="center", va="bottom", fontsize=9.0)
        ax.set_ylim(y0, y + 0.12 * yr)
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_metric_box(master_df, metric, ylabel, out_name, palette=None):
    sub = master_df[["mouse_id", "Condition", metric]].dropna().copy()
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
    piv = sub.pivot_table(index="mouse_id", columns="Condition", values=metric, aggfunc="mean", observed=False).reindex(columns=CONDITIONS)

    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)
    pal = palette or COLORS

    sns.boxplot(
        data=sub,
        x="Condition",
        y=metric,
        hue="Condition",
        order=CONDITIONS,
        palette=[pal[c] for c in CONDITIONS],
        legend=False,
        width=0.58,
        fliersize=0,
        linewidth=1.1,
        ax=ax,
        boxprops={"alpha": 0.82},
        medianprops={"color": NEUTRAL["text"], "linewidth": 1.2},
    )

    for i, cond in enumerate(CONDITIONS):
        vals = piv[cond].dropna().to_numpy()
        jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 0 else np.array([])
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=26,
            color=pal[cond],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )

    # Friedman + pairwise stars
    p_main = np.nan
    try:
        valid = piv.dropna()
        _, p_main = stats.friedmanchisquare(valid["Divergent"], valid["Convergent"], valid["Random"])
    except Exception:
        pass
    if np.isfinite(p_main):
        ax.text(0.02, 0.98, f"Friedman p={p_main:.2e}", transform=ax.transAxes, ha="left", va="top", fontsize=7.1)
    pvals = _condition_pairwise_pvals(piv)
    add_sig_brackets(
        ax,
        [("Divergent", "Convergent"), ("Divergent", "Random"), ("Convergent", "Random")],
        pvals,
        {"Divergent": 0, "Convergent": 1, "Random": 2},
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel(ylabel)
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_noise_corr_matrices(pair_csv: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(pair_csv)
    mats = {}
    for cond in CONDITIONS:
        vals = df.loc[df["Class_Name"] == cond, "Noise_Corr"].to_numpy(dtype=float)
        n_pairs = len(vals)
        n = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
        if n * (n - 1) // 2 != n_pairs:
            raise ValueError(f"Cannot infer neuron number for {cond} from n_pairs={n_pairs}")
        m = np.eye(n, dtype=float)
        iu = np.triu_indices(n, k=1)
        m[iu] = vals
        m[(iu[1], iu[0])] = vals
        mats[cond] = m
    return mats


def plot_fig2_panel_a_corr_matrix(pair_csv: Path):
    mats = build_noise_corr_matrices(pair_csv)
    mean_m = np.mean([mats[c] for c in CONDITIONS], axis=0)
    sort_idx = np.argsort(mean_m.mean(axis=1))[::-1]
    mats = {c: mats[c][sort_idx][:, sort_idx] for c in CONDITIONS}
    n = mats[CONDITIONS[0]].shape[0]
    if n > 120:
        show_idx = np.linspace(0, n - 1, 120).astype(int)
        mats = {c: mats[c][show_idx][:, show_idx] for c in CONDITIONS}

    all_vals = np.concatenate([mats[c][np.triu_indices_from(mats[c], k=1)] for c in CONDITIONS])
    lim = float(np.nanquantile(np.abs(all_vals), 0.85))
    lim = float(np.clip(lim, 0.04, 0.18))
    cmap = LinearSegmentedColormap.from_list("corr_div", ["#738394", "#F4F1EC", "#B98577"])

    fig = plt.figure(figsize=(7.4, 2.65), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for ax, cond in zip(axes, CONDITIONS):
        im = ax.imshow(mats[cond], vmin=-lim, vmax=lim, cmap=cmap, interpolation="nearest")
        ax.set_title(cond, fontsize=8.4, color=NEUTRAL["text"])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pairwise correlation", fontsize=8)
    fig.savefig(OUT_DIR / "Fig2_PanelA_representative_corr_matrix_sorted_single_M73.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_panel_b_ridge(pair_csv: Path):
    df = pd.read_csv(pair_csv)
    fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=300)
    x_grid = np.linspace(-1.0, 1.0, 500)
    offsets = np.arange(len(CONDITIONS))[::-1].astype(float)

    for y0, cond in zip(offsets, CONDITIONS):
        vals = df.loc[df["Class_Name"] == cond, "Noise_Corr"].to_numpy(dtype=float)
        if vals.size > 60000:
            rng = np.random.default_rng(20260331)
            vals = rng.choice(vals, size=60000, replace=False)
        kde = stats.gaussian_kde(vals)
        dens = kde(x_grid)
        dens = dens / dens.max() * 0.82
        ax.fill_between(x_grid, y0, y0 + dens, color=COLORS[cond], alpha=0.74, linewidth=0)
        ax.plot(x_grid, y0 + dens, color=COLORS[cond], lw=1.15)

    ax.set_yticks(offsets + 0.38, CONDITIONS)
    ax.set_xlabel("Pairwise correlation")
    ax.set_ylabel("Condition")
    ax.set_xlim(-0.5, 0.75)
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_PanelB_pairwise_corr_ridge_single_M73.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_panel_d_decile_zoom(decile_df):
    decile_df = decile_df.copy()
    decile_df["Condition"] = pd.Categorical(decile_df["Condition"], categories=CONDITIONS, ordered=True)

    agg = (
        decile_df.groupby(["Condition", "Decile_Index"], as_index=False, observed=False)["Mean_Correlation"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    fig, (ax_main, ax_zoom) = plt.subplots(
        2, 1, figsize=(5.0, 5.4), dpi=300, gridspec_kw={"height_ratios": [2.5, 1.6], "hspace": 0.18}
    )

    for cond in CONDITIONS:
        sub = agg[agg["Condition"] == cond].sort_values("Decile_Index")
        x = sub["Decile_Index"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        se = sub["sem"].to_numpy(dtype=float)

        ax_main.plot(x, y, lw=2.0, marker="o", ms=4.0, color=COLORS[cond], label=cond)
        ax_main.fill_between(x, y - se, y + se, color=COLORS[cond], alpha=0.16, linewidth=0)

        # zoom weak tail
        keep = x <= 3
        ax_zoom.plot(x[keep], y[keep], lw=2.0, marker="o", ms=4.0, color=COLORS[cond])
        ax_zoom.fill_between(x[keep], (y - se)[keep], (y + se)[keep], color=COLORS[cond], alpha=0.16, linewidth=0)

    ax_main.set_xlim(1, 10)
    ax_main.set_xticks(range(1, 11))
    ax_main.set_ylabel("Mean correlation")
    style_axis(ax_main, light_grid=False)
    ax_main.legend(frameon=False, loc="upper left")

    # weak-tail amplified range
    weak_vals = agg[agg["Decile_Index"].isin([1, 2, 3])]["mean"].to_numpy(dtype=float)
    if weak_vals.size:
        y_min = np.nanmin(weak_vals) - 0.012
        y_max = np.nanmax(weak_vals) + 0.012
        ax_zoom.set_ylim(y_min, y_max)
    ax_zoom.set_xlim(1, 3)
    ax_zoom.set_xticks([1, 2, 3])
    ax_zoom.set_xlabel("Weak-tail deciles")
    ax_zoom.set_ylabel("Zoomed mean corr")
    style_axis(ax_zoom, light_grid=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_PanelE_decile_profile_group.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_panel_e_endpoint(master_df):
    sub = master_df[["Condition", "Weak_Correlation", "Strong_Correlation"]].dropna().copy()
    g = sub.groupby("Condition", as_index=False).agg(
        weak_mean=("Weak_Correlation", "mean"),
        weak_sem=("Weak_Correlation", "sem"),
        strong_mean=("Strong_Correlation", "mean"),
        strong_sem=("Strong_Correlation", "sem"),
    )
    g["Condition"] = pd.Categorical(g["Condition"], categories=CONDITIONS, ordered=True)
    g = g.sort_values("Condition")

    fig, ax = plt.subplots(figsize=(4.8, 3.1), dpi=300)
    y_pos = np.arange(len(g))
    for i, row in enumerate(g.itertuples(index=False)):
        cond = row.Condition
        c = COLORS[cond]
        weak = row.weak_mean
        strong = row.strong_mean
        ax.plot([weak, strong], [i, i], color=c, lw=2.2, alpha=0.95, zorder=2)
        ax.errorbar(weak, i, xerr=row.weak_sem, fmt="o", color=c, alpha=0.35, markersize=6.0, lw=1.2, capsize=0, zorder=3)
        ax.errorbar(strong, i, xerr=row.strong_sem, fmt="o", color=c, alpha=0.98, markersize=6.4, lw=1.2, capsize=0, zorder=4)
        ax.text(strong + 0.008, i, f"Δ={strong-weak:.3f}", va="center", ha="left", fontsize=7.5, color="#5A554F")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(g["Condition"].tolist())
    ax.set_xlabel("Mean correlation")
    ax.invert_yaxis()
    style_axis(ax, light_grid=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_PanelE_endpoint_dumbbell_group.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_panel_f_fc_decoder(fc_df: pd.DataFrame):
    d = fc_df.copy()
    for c in ["full_accuracy_mean", "full_shuffle_accuracy_mean", "subset_accuracy_mean", "subset_shuffle_accuracy_mean"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["full_accuracy_mean", "full_shuffle_accuracy_mean", "subset_accuracy_mean", "subset_shuffle_accuracy_mean"])
    if d.empty:
        raise ValueError("FC focus table has no valid numeric rows.")
    d["shuffle_baseline"] = (d["full_shuffle_accuracy_mean"] + d["subset_shuffle_accuracy_mean"]) / 2.0
    order = ["Full FC", "Weak-edge FC", "Shuffle"]
    piv = pd.DataFrame(
        {
            "Full FC": d["full_accuracy_mean"].astype(float).values,
            "Weak-edge FC": d["subset_accuracy_mean"].astype(float).values,
            "Shuffle": d["shuffle_baseline"].astype(float).values,
        },
        index=d["mouse_id"].astype(str),
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    x = np.arange(3, dtype=float)
    for _, row in piv.iterrows():
        ax.plot(x, row[order].to_numpy(dtype=float), color=NEUTRAL["paired"], lw=0.9, alpha=0.55, zorder=1)

    for i, label in enumerate(order):
        vals = piv[label].to_numpy(dtype=float)
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=24, color=DECODER_COLORS[label], edgecolor="white", linewidth=0.5, zorder=3)
        mu = np.nanmean(vals)
        se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
        ax.errorbar(i, mu, yerr=se, fmt="D", color=NEUTRAL["text"], markersize=4.8, lw=1.2, capsize=0, zorder=4)

    ax.axhline(1.0 / 3.0, lw=1.1, ls="--", color="#8D8A84")
    try:
        _, p_full = stats.wilcoxon(piv["Full FC"], piv["Shuffle"])
        _, p_weak = stats.wilcoxon(piv["Weak-edge FC"], piv["Shuffle"])
        ax.text(0.02, 0.98, f"Full vs shuffle p={p_full:.2e}\nWeak vs shuffle p={p_weak:.2e}", transform=ax.transAxes, ha="left", va="top", fontsize=7.2)
    except Exception:
        pass

    ax.set_xticks(x, order)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("Classification accuracy")
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_PanelF_fc_decoder_full_weak_shuffle_group.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_panel_a_restyled(src_img: Path, out_path: Path):
    img = np.array(Image.open(src_img).convert("RGB"))
    # Detect colored heatmap bbox, then keep only the matrix (exclude colorbar).
    diff = img.max(axis=2) - img.min(axis=2)
    mask = diff > 30
    ys, xs = np.where(mask)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    # Split potential second segment (colorbar) by column density.
    col_count = mask.sum(axis=0)
    seg_idx = np.where(col_count > 200)[0]
    if len(seg_idx) > 0:
        runs = []
        st = seg_idx[0]
        pr = seg_idx[0]
        for k in seg_idx[1:]:
            if k == pr + 1:
                pr = k
            else:
                runs.append((st, pr))
                st, pr = k, k
        runs.append((st, pr))
        if len(runs) >= 1:
            x0, x1 = int(runs[0][0]), int(runs[0][1])

    heat = img[y0 : y1 + 1, x0 : x1 + 1]
    gray = np.dot(heat[..., :3], [0.299, 0.587, 0.114]) / 255.0
    # Re-color to single-tone response palette.
    low = np.array([244, 241, 236], dtype=float) / 255.0  # #F4F1EC
    high = np.array([109, 124, 115], dtype=float) / 255.0  # #6D7C73
    recol = low[None, None, :] * (1.0 - gray[..., None]) + high[None, None, :] * gray[..., None]

    # Swap x/y orientation for a slim-tall panel.
    recol = np.transpose(recol, (1, 0, 2))

    fig, ax = plt.subplots(figsize=(2.6, 4.8), dpi=300)
    ax.imshow(recol, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Trials (sorted)")
    ax.set_ylabel("Neurons (sorted by preference)")
    # Keep a slim, readable axis tick style.
    ax.set_xticks(np.linspace(0, recol.shape[1], 5))
    ax.set_yticks(np.linspace(0, recol.shape[0], 6))
    ax.tick_params(labelsize=6.7, length=2.5)
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_panel_d_geometry(geom_df: pd.DataFrame):
    d = geom_df.copy()
    d["Parallel expansion"] = pd.to_numeric(d["delta_parallel_coherent_minus_random"], errors="coerce")
    d["Orthogonal expansion"] = pd.to_numeric(d["delta_orthogonal_coherent_minus_random"], errors="coerce")
    d = d.dropna(subset=["Parallel expansion", "Orthogonal expansion"]).copy()
    d["delta_orth_minus_para"] = d["Orthogonal expansion"] - d["Parallel expansion"]
    d = d.sort_values("delta_orth_minus_para")

    fig, ax = plt.subplots(figsize=(4.3, 3.4), dpi=300)

    # Paired slope lines per mouse (contrast-focused).
    for _, r in d.iterrows():
        ax.plot(
            [0, 1],
            [r["Parallel expansion"], r["Orthogonal expansion"]],
            color="#B9B4AC",
            lw=1.0,
            alpha=0.75,
            zorder=1,
        )

    ax.scatter(
        np.zeros(len(d)),
        d["Parallel expansion"].to_numpy(dtype=float),
        s=34,
        color=GEOM_COLORS["Parallel"],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
        label="Parallel",
    )
    ax.scatter(
        np.ones(len(d)),
        d["Orthogonal expansion"].to_numpy(dtype=float),
        s=34,
        color=GEOM_COLORS["Orthogonal"],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
        label="Orthogonal",
    )

    # Group mean ± SEM for each component.
    for x, col, c in [
        (0, "Parallel expansion", GEOM_COLORS["Parallel"]),
        (1, "Orthogonal expansion", GEOM_COLORS["Orthogonal"]),
    ]:
        vals = d[col].to_numpy(dtype=float)
        mu = float(np.nanmean(vals))
        se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.errorbar(
            x,
            mu,
            yerr=se,
            fmt="D",
            color=NEUTRAL["text"],
            markersize=5.5,
            lw=1.25,
            capsize=0,
            zorder=4,
        )
        ax.scatter([x], [mu], s=56, color=c, edgecolor=NEUTRAL["text"], linewidth=0.7, zorder=5)

    p = np.nan
    try:
        a = d["Parallel expansion"].to_numpy(dtype=float)
        b = d["Orthogonal expansion"].to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() >= 3:
            _, p = stats.wilcoxon(a[valid], b[valid], alternative="less")
            ax.text(
                0.02,
                0.98,
                f"Orthogonal > Parallel, p={p:.2e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.3,
            )
    except Exception:
        pass

    # Significance bracket.
    if np.isfinite(p):
        y0, y1 = ax.get_ylim()
        yr = y1 - y0 if y1 > y0 else 1.0
        y = y1 + 0.04 * yr
        ax.plot([0, 0, 1, 1], [y, y + 0.02 * yr, y + 0.02 * yr, y], lw=1.1, c=NEUTRAL["text"])
        ax.text(0.5, y + 0.025 * yr, p_to_star(p), ha="center", va="bottom", fontsize=10, color=NEUTRAL["text"])
        ax.set_ylim(y0, y + 0.14 * yr)

    ax.axhline(0, lw=1.0, ls="--", color="#8D8A84")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Parallel", "Orthogonal"])
    ax.set_ylabel("Expansion (Coherent - Random)")
    style_axis(ax, light_grid=False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig3_PanelD_orthogonal_vs_parallel_expansion_group.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_panel_f_ridge(rsm_df: pd.DataFrame, out_name: str = "Fig3_PanelF_rsm_distribution_group.png"):
    df = rsm_df.copy()
    df["Condition"] = pd.Categorical(df["Condition"], categories=CONDITIONS, ordered=True)
    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=300)
    x_grid = np.linspace(0.2, 0.9, 450)
    offsets = np.arange(len(CONDITIONS))[::-1].astype(float)
    for y0, cond in zip(offsets, CONDITIONS):
        vals = df.loc[df["Condition"] == cond, "mean_rsm"].to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        kde = stats.gaussian_kde(vals)
        dens = kde(x_grid)
        dens = dens / dens.max() * 0.8
        ax.fill_between(x_grid, y0, y0 + dens, color=COLORS[cond], alpha=0.76, linewidth=0)
        ax.plot(x_grid, y0 + dens, color=COLORS[cond], lw=1.15)
        ax.scatter(vals, np.full_like(vals, y0) + 0.03, s=10, color=COLORS[cond], edgecolor="white", linewidth=0.4, zorder=4)
        mu = float(np.mean(vals))
        med = float(np.median(vals))
        ax.plot([mu, mu], [y0, y0 + 0.24], color=NEUTRAL["text"], lw=1.2, zorder=5)
        ax.plot([med, med], [y0, y0 + 0.24], color=NEUTRAL["text"], lw=1.1, ls="--", zorder=5)
    ax.set_yticks(offsets + 0.35, CONDITIONS)
    ax.set_xlabel("Mean trial-to-trial RSM similarity")
    ax.set_ylabel("Condition")
    ax.text(0.98, 0.98, "Solid: mean\nDashed: median", transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=NEUTRAL["text"])
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_panel_g(master_df):
    sub = master_df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=300)

    # condition-colored points
    for cond in CONDITIONS:
        ss = sub[sub["Condition"] == cond]
        ax.scatter(
            ss["Participants_Ratio"],
            ss["Mean_RSM_Sim"],
            s=48,
            color=COLORS[cond],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            label=cond,
            zorder=3
        )
        ax.scatter([ss["Participants_Ratio"].mean()], [ss["Mean_RSM_Sim"].mean()], s=58, marker="D", color=COLORS[cond], edgecolor="#2F2F2F", linewidth=0.7, alpha=0.9, zorder=4)

    # mixed-effect line
    text = "LMM unavailable"
    p = np.nan
    try:
        mdf = smf.mixedlm("Mean_RSM_Sim ~ Participants_Ratio", sub, groups=sub["mouse_id"]).fit()
        b0 = float(mdf.params["Intercept"])
        b1 = float(mdf.params["Participants_Ratio"])
        p = float(mdf.pvalues["Participants_Ratio"])
        xs = np.linspace(sub["Participants_Ratio"].min(), sub["Participants_Ratio"].max(), 200)
        ys = b0 + b1 * xs
        ax.plot(xs, ys, color="#202020", lw=2.2, ls="--", zorder=2)
        text = f"LMM: RSM ~ PR + (1|mouse)\nβ={b1:.4f}, p={p:.2e}, N={sub['mouse_id'].nunique()}"
    except Exception:
        # fallback OLS
        coef = np.polyfit(sub["Participants_Ratio"], sub["Mean_RSM_Sim"], 1)
        xs = np.linspace(sub["Participants_Ratio"].min(), sub["Participants_Ratio"].max(), 200)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, color="#202020", lw=2.0, ls="--", zorder=2)
        text = f"OLS slope={coef[0]:.4f}, N={sub['mouse_id'].nunique()}"

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#D2CCC3", alpha=0.88),
    )

    ax.set_xlabel("Participants ratio")
    ax.set_ylabel("Mean RSM similarity")
    if np.isfinite(p):
        ax.text(0.98, 0.04, f"{p_to_star(p)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=NEUTRAL["text"])
    style_axis(ax, light_grid=False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2, borderpad=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig3_PanelG_lmm_participants_vs_rsm_group.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_index(results: list[PanelResult]):
    lines = [
        "# Paper Panels v4 (Guide-aligned)",
        "",
        "Generated by `scripts/build_paper_panels_v3.py`",
        "",
        "| Panel | File | Status | Source | Note |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        lines.append(f"| {r.panel} | `{r.filename}` | {r.status} | {r.source} | {r.note} |")
    (OUT_DIR / "FIGURE_PANEL_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def run_panel(panel: str, filename: str, source: str, fn, note: str = "") -> PanelResult:
    try:
        fn()
        return PanelResult(panel=panel, filename=filename, status="ok", source=source, note=note)
    except Exception as exc:
        return PanelResult(panel=panel, filename=filename, status="failed", source=source, note=str(exc))


def main():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": NEUTRAL["text"],
            "axes.labelcolor": NEUTRAL["text"],
            "xtick.color": NEUTRAL["text"],
            "ytick.color": NEUTRAL["text"],
        }
    )

    master_csv = find_first_existing("group_summary/group_master_metrics.csv")
    decile_csv = find_first_existing("group_summary/group_corr_deciles_long.csv")
    rr_trace = find_first_existing(f"{MOUSE_EXAMPLE}/figures/rr_population_average.png")
    pref_mat = find_first_existing(f"{MOUSE_EXAMPLE}/figures/neural_patterns_preference_sorted.png")
    confusion_csv = find_first_existing(f"{MOUSE_EXAMPLE}/data/decoder_confusion_matrix.csv")
    pair_csv = find_first_existing(f"{MOUSE_EXAMPLE}/data/sig_noise_pair_values_by_condition.csv")

    master_df = pd.read_csv(master_csv)
    decile_df = pd.read_csv(decile_csv)
    decoder_df = load_activity_decoder_summary()
    fc_focus_df = load_fc_focus_table()
    geom_df = load_geometry_delta_table()
    rsm_df = load_rsm_original_means()

    results = []

    # Fig1
    fig1a_src = BASE / "figure_assets" / "Fig1A_setup_schematic_source.png"
    fig1b_src = find_first_existing("M78_1017/figures/rr_distribution.png")

    f = "Fig1_PanelA_experimental_schematic.png"
    results.append(run_panel("Fig1A", f, str(fig1a_src), lambda: plot_fig1_panel_a_from_image(fig1a_src, OUT_DIR / f)))

    f = "Fig1_PanelB_representative_fov_rr_overlay.png"
    results.append(run_panel("Fig1B", f, str(fig1b_src), lambda: plot_fig1_panel_b_from_image(fig1b_src, OUT_DIR / f)))

    f = "Fig1_PanelC_population_mean_trace_group.png"
    results.append(run_panel("Fig1C", f, str(rr_trace), lambda: recolor_fig1_panel_c(rr_trace, OUT_DIR / f)))

    cls_acc = find_first_existing("results/M21_1107/figures/classification_accuracy.png", "M21_1107/figures/classification_accuracy.png")
    f = "Fig1_PanelD_decoder_confusion_single_M73.png"
    results.append(run_panel("Fig1D", f, str(confusion_csv), lambda: plot_fig1_panel_d_confusion(confusion_csv, OUT_DIR / f, extra_path=cls_acc)))

    f = "Fig1_PanelE_decoder_accuracy_group_true_vs_shuffle.png"
    results.append(run_panel("Fig1E", f, "result/*/data/decoder_summary.csv", lambda: plot_fig1_panel_e_group_decoder(decoder_df, OUT_DIR / f)))

    # Fig2
    results.append(run_panel("Fig2A", "Fig2_PanelA_representative_corr_matrix_sorted_single_M73.png", str(pair_csv), lambda: plot_fig2_panel_a_corr_matrix(pair_csv), note="built from Noise_Corr"))
    results.append(run_panel("Fig2B", "Fig2_PanelB_pairwise_corr_ridge_single_M73.png", str(pair_csv), lambda: plot_fig2_panel_b_ridge(pair_csv), note="Noise_Corr ridge"))
    results.append(run_panel("Fig2C", "Fig2_PanelC_strongest_decile_group.png", str(master_csv), lambda: plot_group_metric_box(master_df, "Strong_Correlation", "Mean correlation (strongest decile)", "Fig2_PanelC_strongest_decile_group.png", palette=STRONG_TONES)))
    results.append(run_panel("Fig2D", "Fig2_PanelD_weakest_decile_group.png", str(master_csv), lambda: plot_group_metric_box(master_df, "Weak_Correlation", "Mean correlation (weakest decile)", "Fig2_PanelD_weakest_decile_group.png", palette=WEAK_TONES)))
    results.append(run_panel("Fig2E", "Fig2_PanelE_decile_profile_group.png", str(decile_csv), lambda: plot_fig2_panel_d_decile_zoom(decile_df)))
    results.append(run_panel("Fig2F", "Fig2_PanelF_fc_decoder_full_weak_shuffle_group.png", "Group_FC_Decoder_Focus_Report.md", lambda: plot_fig2_panel_f_fc_decoder(fc_focus_df)))

    # Fig3
    f = "Fig3_PanelA_preference_sorted_response_matrix_single_M73.png"
    results.append(run_panel("Fig3A", f, str(pref_mat), lambda: plot_fig3_panel_a_restyled(pref_mat, OUT_DIR / f), note="recolored + axis-swapped"))
    results.append(run_panel("Fig3B", "Fig3_PanelB_participants_ratio_group.png", str(master_csv), lambda: plot_group_metric_box(master_df, "Participants_Ratio", "Participants ratio", "Fig3_PanelB_participants_ratio_group.png", palette=COLORS)))
    results.append(run_panel("Fig3C", "Fig3_PanelC_gini_group.png", str(master_csv), lambda: plot_group_metric_box(master_df, "Gini_Mean", "Gini coefficient", "Fig3_PanelC_gini_group.png", palette=COLORS)))
    results.append(run_panel("Fig3D", "Fig3_PanelD_orthogonal_vs_parallel_expansion_group.png", "Group_Geometry_Report.md", lambda: plot_fig3_panel_d_geometry(geom_df)))
    results.append(run_panel("Fig3E", "Fig3_PanelE_rsm_distribution_group.png", "result/*/data/group_rsm_shuffle_long.csv", lambda: plot_fig3_panel_f_ridge(rsm_df, out_name="Fig3_PanelE_rsm_distribution_group.png")))
    results.append(run_panel("Fig3F", "Fig3_PanelF_mean_rsm_group.png", str(master_csv), lambda: plot_group_metric_box(master_df, "Mean_RSM_Sim", "Mean RSM similarity", "Fig3_PanelF_mean_rsm_group.png", palette=COLORS)))
    results.append(run_panel("Fig3G", "Fig3_PanelG_lmm_participants_vs_rsm_group.png", str(master_csv), lambda: plot_fig3_panel_g(master_df)))

    write_index(results)
    ok = sum(1 for r in results if r.status == "ok")
    fail = len(results) - ok
    print(f"[*] New panel folder: {OUT_DIR}")
    print(f"[*] Panels generated: {ok}/{len(results)} ok, {fail} failed")
    if fail:
        for r in results:
            if r.status != "ok":
                print(f"    - {r.panel}: {r.note}")


if __name__ == "__main__":
    main()
