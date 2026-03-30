import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from PIL import Image, ImageEnhance


BASE = Path(__file__).resolve().parents[1]
SINGLE_FIG = BASE / "result" / "M73_1128" / "figures"
SINGLE_DATA = BASE / "result" / "M73_1128" / "data"
GROUP_DIR = BASE / "result" / "group_summary"
OUT_DIR = BASE / "result" / "paper_panels_20260329_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["Divergent", "Convergent", "Random"]
COLORS = {
    "Divergent": "#7F9C96",
    "Convergent": "#8B90A8",
    "Random": "#B98372",
}


def style_axis(ax, light_grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    if light_grid:
        ax.grid(axis="y", color="#E9E5DF", lw=0.8, alpha=0.8)
    else:
        ax.grid(False)


def _rgb_hex(rgb):
    return np.array([int(rgb[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def recolor_fig1_panel_c():
    src = SINGLE_FIG / "rr_population_average.png"
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
    Image.fromarray(out).save(OUT_DIR / "Fig1_PanelC_population_mean_trace_single_M73.png")


def copy_fig1_panel_e():
    src = SINGLE_FIG / "classification_accuracy.png"
    dst = OUT_DIR / "Fig1_PanelE_decoding_timecourse_single_M73.png"
    Image.open(src).save(dst)


def plot_group_metric(master_df, metric, ylabel, out_name):
    sub = master_df[["mouse_id", "Condition", metric]].dropna().copy()
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)
    piv = sub.pivot(index="mouse_id", columns="Condition", values=metric).reindex(columns=CONDITIONS)

    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)

    for _, row in piv.iterrows():
        y = row.to_numpy(dtype=float)
        m = ~np.isnan(y)
        if m.sum() >= 2:
            ax.plot(np.arange(3)[m], y[m], color="#B8B2AA", lw=0.9, alpha=0.65, zorder=1)

    for i, cond in enumerate(CONDITIONS):
        vals = piv[cond].dropna().to_numpy()
        jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 0 else np.array([])
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=24,
            color=COLORS[cond],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        mu = np.nanmean(vals) if len(vals) else np.nan
        se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        if np.isfinite(mu):
            ax.errorbar(
                i,
                mu,
                yerr=se,
                fmt="D",
                color="#2F2F2F",
                markersize=4.8,
                capsize=0,
                lw=1.2,
                zorder=4,
            )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel(ylabel)
    style_axis(ax, light_grid=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_panel_d_decile_zoom(decile_df):
    decile_df = decile_df.copy()
    decile_df["Condition"] = pd.Categorical(decile_df["Condition"], categories=CONDITIONS, ordered=True)

    agg = (
        decile_df.groupby(["Condition", "Decile_Index"], as_index=False)["Mean_Correlation"]
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
    fig.savefig(OUT_DIR / "Fig2_PanelD_decile_profile_with_weak_zoom_group.png", dpi=300, bbox_inches="tight")
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


def plot_fig3_panel_f():
    # Use the condition-wise similarity distribution panel as a more discriminative
    # representation of RSM stability differences.
    src = SINGLE_FIG / "similarity_distribution.png"
    im = Image.open(src).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im.save(OUT_DIR / "Fig3_PanelF_similarity_distribution_single_M73_redraw.png")


def plot_fig3_panel_g(master_df):
    sub = master_df[["mouse_id", "Condition", "Participants_Ratio", "Mean_RSM_Sim"]].dropna().copy()
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=CONDITIONS, ordered=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=300)

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
        # group centroid marker
        ax.scatter(
            [ss["Participants_Ratio"].mean()],
            [ss["Mean_RSM_Sim"].mean()],
            s=86,
            marker="D",
            color=COLORS[cond],
            edgecolor="#2F2F2F",
            linewidth=0.8,
            alpha=0.95,
            zorder=4,
        )

    # mixed-effect line
    text = "LMM unavailable"
    try:
        mdf = smf.mixedlm("Mean_RSM_Sim ~ Participants_Ratio", sub, groups=sub["mouse_id"]).fit()
        b0 = float(mdf.params["Intercept"])
        b1 = float(mdf.params["Participants_Ratio"])
        p = float(mdf.pvalues["Participants_Ratio"])
        xs = np.linspace(sub["Participants_Ratio"].min(), sub["Participants_Ratio"].max(), 200)
        ys = b0 + b1 * xs
        ax.plot(xs, ys, color="#202020", lw=2.2, ls="--", zorder=2)
        text = f"LMM slope β = {b1:.4f}\np = {p:.3e}\nN = {sub['mouse_id'].nunique()} mice"
    except Exception:
        # fallback OLS
        coef = np.polyfit(sub["Participants_Ratio"], sub["Mean_RSM_Sim"], 1)
        xs = np.linspace(sub["Participants_Ratio"].min(), sub["Participants_Ratio"].max(), 200)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, color="#202020", lw=2.0, ls="--", zorder=2)
        text = f"OLS slope = {coef[0]:.4f}\nN = {sub['mouse_id'].nunique()} mice"

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#D2CCC3", alpha=0.92),
    )

    ax.set_xlabel("Participants ratio")
    ax.set_ylabel("Mean RSM similarity")
    style_axis(ax, light_grid=False)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig3_PanelG_state_binding_group_redraw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_existing_good_panels():
    copies = [
        (OUT_DIR / "Fig1_PanelD_decoder_confusion_single_M73.png", SINGLE_FIG / "decoder_confusion_matrix.png"),
        (OUT_DIR / "Fig2_PanelA_example_corr_distribution_single_M73.png", SINGLE_FIG / "pairwise_correlation.png"),
        (OUT_DIR / "Fig3_PanelA_pattern_heatmap_single_M73.png", SINGLE_FIG / "neural_patterns_preference_sorted.png"),
        (OUT_DIR / "Fig3_PanelB_rr_overlap_summary_single_M73.png", SINGLE_FIG / "rr_sets_venn.png"),
        (OUT_DIR / "Fig3_PanelC_participants_ratio_group.png", GROUP_DIR / "group_participants_ratio_notitle.png"),
        (OUT_DIR / "Fig3_PanelD_gini_group.png", GROUP_DIR / "group_gini_mean_notitle.png"),
        (OUT_DIR / "Fig3_PanelE_mean_rsm_group.png", GROUP_DIR / "group_mean_rsm_sim_notitle.png"),
    ]
    for dst, src in copies:
        Image.open(src).save(dst)


def write_index():
    lines = [
        "# Paper Panels v2",
        "",
        "Feedback-driven redraw on 2026-03-29.",
        "",
        "## Panels",
    ]
    for p in sorted(OUT_DIR.glob("*.png")):
        lines.append(f"- `{p.name}`")
    (OUT_DIR / "FIGURE_PANEL_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    sns.set_theme(style="ticks", context="paper")
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
        }
    )

    master_df = pd.read_csv(GROUP_DIR / "group_master_metrics.csv")
    decile_df = pd.read_csv(GROUP_DIR / "group_corr_deciles_long.csv")

    copy_existing_good_panels()
    recolor_fig1_panel_c()
    copy_fig1_panel_e()

    plot_group_metric(master_df, "Weak_Correlation", "Weak correlation (bottom 10%)", "Fig2_PanelB_weak_correlation_group.png")
    plot_group_metric(master_df, "Strong_Correlation", "Strong correlation (top 10%)", "Fig2_PanelC_strong_correlation_group.png")
    plot_fig2_panel_d_decile_zoom(decile_df)
    plot_fig2_panel_e_endpoint(master_df)

    plot_fig3_panel_f()
    plot_fig3_panel_g(master_df)

    write_index()
    print(f"[*] New panel folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
