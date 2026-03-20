import glob
import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


RESULTS_BASE_DIR = "./results/"
GROUP_OUT_DIR = os.path.join(RESULTS_BASE_DIR, "group_summary")
os.makedirs(GROUP_OUT_DIR, exist_ok=True)

CONDITIONS = ["Divergent", "Convergent", "Random"]
COLORS = {"Divergent": "#FF4B4B", "Convergent": "#1C75BC", "Random": "#7AC143"}

ID_TO_COND = {"1": "Divergent", "2": "Convergent", "3": "Random"}
COND_ALIASES = {
    "divergent": "Divergent",
    "convergent": "Convergent",
    "random": "Random",
    "1": "Divergent",
    "2": "Convergent",
    "3": "Random",
}

GRAPH_METRICS = ["efficiency", "modularity", "local_efficiency", "avg_clustering"]


def normalize_condition(value):
    if value is None:
        return None
    s = str(value).strip()
    if s in ID_TO_COND:
        return ID_TO_COND[s]
    return COND_ALIASES.get(s.lower())


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_optional_csv(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[!] Failed to read {path}: {exc}")
        return None


def load_all_mice_bundles(base_dir):
    pattern = os.path.join(base_dir, "*", "data", "*_statistics.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        raise FileNotFoundError(f"No *_statistics.json found under {base_dir}")

    bundles = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        mouse_id = payload.get("mouse_id", os.path.basename(fp).replace("_statistics.json", ""))
        data_dir = os.path.dirname(fp)
        bundles.append(
            {
                "mouse_id": mouse_id,
                "json_path": fp,
                "data_dir": data_dir,
                "payload": payload,
                "trial_shape_summary": load_optional_csv(data_dir, "trial_response_shape_summary.csv"),
                "effective_dim": load_optional_csv(data_dir, "effective_dimensionality_by_class.csv"),
                "graph_sw": load_optional_csv(data_dir, "network_metrics_strong_vs_weak.csv"),
                "rr_overlap": load_optional_csv(data_dir, "rr_overlap_summary.csv"),
                "corr_deciles_csv": load_optional_csv(data_dir, f"{mouse_id}_correlation_deciles.csv"),
            }
        )

    print(f"[*] Loaded {len(bundles)} mice from {base_dir}")
    return bundles


def _build_dict_by_condition(records, name_key):
    out = {}
    for item in records or []:
        cond = normalize_condition(item.get(name_key))
        if cond is not None:
            out[cond] = item
    return out


def _participants_by_condition(raw_dict):
    out = {}
    for k, v in (raw_dict or {}).items():
        cond = normalize_condition(k)
        if cond is not None:
            out[cond] = safe_float(v)
    return out


def _graph_condition_metrics(graph_df, condition):
    cols = {}
    if graph_df is None or graph_df.empty:
        return cols

    g = graph_df.copy()
    g["Class_Name"] = g["Class_Name"].map(normalize_condition)
    g["Network_Type"] = g["Network_Type"].astype(str).str.lower()
    sub = g[g["Class_Name"] == condition]
    if sub.empty:
        return cols

    for metric in GRAPH_METRICS:
        strong = safe_float(sub.loc[sub["Network_Type"] == "strong", metric].iloc[0]) if (sub["Network_Type"] == "strong").any() else np.nan
        weak = safe_float(sub.loc[sub["Network_Type"] == "weak", metric].iloc[0]) if (sub["Network_Type"] == "weak").any() else np.nan
        cols[f"GraphStrong_{metric}"] = strong
        cols[f"GraphWeak_{metric}"] = weak
        cols[f"GraphGap_{metric}"] = strong - weak if pd.notna(strong) and pd.notna(weak) else np.nan
    return cols


def _trial_shape_by_condition(trial_shape_summary_df, condition):
    cols = {}
    if trial_shape_summary_df is None or trial_shape_summary_df.empty:
        return cols

    t = trial_shape_summary_df.copy()
    t["Class_Name"] = t["Class_Name"].map(normalize_condition)
    row = t[t["Class_Name"] == condition]
    if row.empty:
        return cols
    row = row.iloc[0]
    for key in ["Gini_Mean", "Gini_STD", "PR_Mean", "PR_STD", "PR_Norm_Mean", "PR_Norm_STD"]:
        if key in row:
            cols[key] = safe_float(row[key])
    return cols


def _effective_dim_by_condition(effective_dim_df, condition):
    cols = {}
    if effective_dim_df is None or effective_dim_df.empty:
        return cols

    d = effective_dim_df.copy()
    d["Class_Name"] = d["Class_Name"].map(normalize_condition)
    row = d[d["Class_Name"] == condition]
    if row.empty:
        return cols
    row = row.iloc[0]
    for key in ["Effective_Dim_PR", "Effective_Dim_eRank", "Effective_Dim_90Var"]:
        if key in row:
            cols[key] = safe_float(row[key])
    return cols


def build_master_dataframe(bundles):
    rows = []

    for b in bundles:
        mouse_id = b["mouse_id"]
        payload = b["payload"]

        entropy_by_cond = _build_dict_by_condition(payload.get("entropy_metrics", []), "Stimulus")
        corr_by_cond = _build_dict_by_condition(payload.get("network_correlation", []), "Class_Name")
        part_by_cond = _participants_by_condition(payload.get("rr_participants_ratio", {}))

        for cond in CONDITIONS:
            ent = entropy_by_cond.get(cond, {})
            corr = corr_by_cond.get(cond, {})

            row = {
                "mouse_id": mouse_id,
                "Condition": cond,
                "Entropy": safe_float(ent.get("Entropy")),
                "Mean_RSM_Sim": safe_float(ent.get("Mean_Sim")),
                "Mean_Correlation": safe_float(corr.get("Mean_Correlation", corr.get("Mean_Abs_Correlation"))),
                "Strong_Correlation": safe_float(corr.get("Strong_Correlation_Mean", corr.get("Strong_Abs_Correlation_Mean"))),
                "Weak_Correlation": safe_float(corr.get("Weak_Correlation_Mean", corr.get("Weak_Abs_Correlation_Mean"))),
                "Strong_Weak_Gap": safe_float(corr.get("Strong_Weak_Gap")),
                "Participants_Ratio": safe_float(part_by_cond.get(cond)),
            }

            row.update(_trial_shape_by_condition(b["trial_shape_summary"], cond))
            row.update(_effective_dim_by_condition(b["effective_dim"], cond))
            row.update(_graph_condition_metrics(b["graph_sw"], cond))
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_decile_dataframe(bundles):
    rows = []
    for b in bundles:
        mouse_id = b["mouse_id"]
        payload = b["payload"]
        deciles = payload.get("network_correlation_deciles")

        if deciles is None and b["corr_deciles_csv"] is not None:
            deciles = b["corr_deciles_csv"].to_dict(orient="records")

        if not deciles:
            continue

        for item in deciles:
            cond = normalize_condition(item.get("Class_Name") or item.get("Condition") or item.get("Class_ID"))
            if cond is None:
                continue
            rows.append(
                {
                    "mouse_id": mouse_id,
                    "Condition": cond,
                    "Decile_Index": int(item.get("Decile_Index")),
                    "Mean_Correlation": safe_float(item.get("Mean_Correlation", item.get("Mean_Abs_Correlation"))),
                }
            )
    return pd.DataFrame(rows)


def build_graph_sw_long_dataframe(bundles):
    rows = []
    for b in bundles:
        g = b["graph_sw"]
        if g is None or g.empty:
            continue
        tmp = g.copy()
        tmp["mouse_id"] = b["mouse_id"]
        tmp["Condition"] = tmp["Class_Name"].map(normalize_condition)
        tmp["Network_Type"] = tmp["Network_Type"].astype(str).str.lower()
        tmp = tmp[tmp["Condition"].isin(CONDITIONS)]
        for metric in GRAPH_METRICS:
            if metric in tmp.columns:
                keep = tmp[["mouse_id", "Condition", "Network_Type", metric]].copy()
                keep = keep.rename(columns={metric: "Value"})
                keep["Metric"] = metric
                rows.append(keep)

    if not rows:
        return pd.DataFrame(columns=["mouse_id", "Condition", "Network_Type", "Value", "Metric"])
    return pd.concat(rows, ignore_index=True)


def build_rr_overlap_dataframe(bundles):
    rows = []
    for b in bundles:
        rr = b["rr_overlap"]
        if rr is None or rr.empty:
            continue
        tmp = rr.copy()
        tmp["mouse_id"] = b["mouse_id"]
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["mouse_id", "Subset", "Subset_Size"])
    return pd.concat(rows, ignore_index=True)


def perform_statistical_tests(df, metric):
    pivot = df.pivot(index="mouse_id", columns="Condition", values=metric).reindex(columns=CONDITIONS).dropna()
    if len(pivot) < 3:
        return {"main_effect": "N too small for stable test", "p_main": np.nan, "post_hoc": {}}

    stat, p_val = stats.friedmanchisquare(pivot["Divergent"], pivot["Convergent"], pivot["Random"])
    out = {"main_effect": f"Friedman chi2={stat:.3f}, p={p_val:.4e}", "p_main": p_val, "post_hoc": {}}

    for c1, c2 in combinations(CONDITIONS, 2):
        try:
            _, p_pair = stats.wilcoxon(pivot[c1], pivot[c2])
            out["post_hoc"][f"{c1} vs {c2}"] = p_pair
        except Exception:
            out["post_hoc"][f"{c1} vs {c2}"] = np.nan
    return out


def p_to_star(p_val):
    if pd.isna(p_val):
        return "ns"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def plot_group_metric(df, metric, ylabel, title, stat_res, save_name):
    if metric not in df.columns or df[metric].isna().all():
        return None

    n_mice = df["mouse_id"].nunique()
    plt.figure(figsize=(6, 5))
    if n_mice >= 3:
        sns.boxplot(data=df, x="Condition", y=metric, order=CONDITIONS, hue="Condition", palette=COLORS, width=0.5, showfliers=False, legend=False)
        sns.stripplot(data=df, x="Condition", y=metric, order=CONDITIONS, color="black", size=5, alpha=0.7, jitter=True)
    else:
        sns.stripplot(data=df, x="Condition", y=metric, order=CONDITIONS, hue="Condition", palette=COLORS, size=8, alpha=0.85, jitter=False, legend=False)

    plt.title(f"{title}\n{stat_res.get('main_effect', '')}")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine()

    # Add pairwise significance stars when global effect is at least suggestive.
    if n_mice >= 3 and pd.notna(stat_res.get("p_main")) and stat_res.get("p_main", 1.0) < 0.1:
        vals = df[metric].dropna()
        if not vals.empty:
            y_min, y_max = vals.min(), vals.max()
            y_range = y_max - y_min
            if y_range <= 1e-12:
                y_range = max(abs(y_max), 1.0) * 0.1
            step = y_range * 0.10
            base = y_max + step

            for i, (c1, c2) in enumerate(combinations(CONDITIONS, 2)):
                p = stat_res.get("post_hoc", {}).get(f"{c1} vs {c2}", np.nan)
                star = p_to_star(p)
                if star == "ns":
                    continue
                x1, x2 = CONDITIONS.index(c1), CONDITIONS.index(c2)
                y = base + i * step
                plt.plot([x1, x1, x2, x2], [y, y + step * 0.2, y + step * 0.2, y], lw=1.2, c="k")
                plt.text((x1 + x2) * 0.5, y + step * 0.2, star, ha="center", va="bottom", color="k")

    out = os.path.join(GROUP_OUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_decile_curve(decile_df):
    if decile_df.empty:
        return None
    plt.figure(figsize=(7.5, 5))
    sns.lineplot(
        data=decile_df,
        x="Decile_Index",
        y="Mean_Correlation",
        hue="Condition",
        hue_order=CONDITIONS,
        palette=COLORS,
        estimator="mean",
        errorbar="se",
        marker="o",
        linewidth=2,
    )
    plt.xticks(np.arange(1, 11))
    plt.xlabel("Decile (1=lowest, 10=highest)")
    plt.ylabel("Mean correlation")
    plt.title("Decile-wise Correlation Curve (Group Mean ± SE)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine()
    out = os.path.join(GROUP_OUT_DIR, "group_corr_decile_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_graph_sw_comparison(graph_long_df, metric):
    sub = graph_long_df[graph_long_df["Metric"] == metric].copy()
    if sub.empty:
        return None
    plt.figure(figsize=(7.5, 5))
    sns.pointplot(
        data=sub,
        x="Condition",
        y="Value",
        hue="Network_Type",
        order=CONDITIONS,
        hue_order=["strong", "weak"],
        dodge=0.25,
        capsize=0.1,
        errorbar="se",
        palette={"strong": "#1F77B4", "weak": "#D55E00"},
    )
    plt.xlabel("")
    plt.ylabel(metric)
    plt.title(f"Strong vs Weak Graph Metric: {metric}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine()
    out = os.path.join(GROUP_OUT_DIR, f"group_graph_sw_{metric}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def generate_group_markdown(master_df, stat_results, image_paths, rr_overlap_df):
    md_path = os.path.join(GROUP_OUT_DIR, "Group_Analysis_Report.md")

    numeric_cols = [c for c in master_df.columns if c not in ["mouse_id", "Condition"]]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(master_df[c])]
    summary_df = master_df.groupby("Condition")[numeric_cols].agg(["mean", "sem"]).round(4)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Group-level Multi-mouse Analysis Report\n\n")
        f.write(f"**Number of mice**: {master_df['mouse_id'].nunique()}\n\n")
        f.write(f"**Mouse IDs**: {', '.join(sorted(master_df['mouse_id'].unique()))}\n\n")

        f.write("## 1. Descriptive Statistics (Mean ± SEM)\n\n")
        desc = pd.DataFrame(index=summary_df.index)
        for col in summary_df.columns.levels[0]:
            desc[col] = summary_df[col]["mean"].astype(str) + " ± " + summary_df[col]["sem"].astype(str)
        f.write(desc.reset_index().to_markdown(index=False) + "\n\n")

        f.write("## 2. Friedman + Wilcoxon Tests\n\n")
        f.write("| Metric | Main Effect | Div vs Con | Div vs Rand | Con vs Rand |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for metric, res in stat_results.items():
            ph = res.get("post_hoc", {})
            f.write(
                f"| **{metric}** | {res.get('main_effect', 'N/A')} | "
                f"p={ph.get('Divergent vs Convergent', np.nan):.4f} ({p_to_star(ph.get('Divergent vs Convergent', np.nan))}) | "
                f"p={ph.get('Divergent vs Random', np.nan):.4f} ({p_to_star(ph.get('Divergent vs Random', np.nan))}) | "
                f"p={ph.get('Convergent vs Random', np.nan):.4f} ({p_to_star(ph.get('Convergent vs Random', np.nan))}) |\n"
            )
        f.write("\n")

        if not rr_overlap_df.empty:
            f.write("## 3. RR Overlap Summary Across Mice\n\n")
            rr_summary = (
                rr_overlap_df.groupby("Subset", as_index=False)
                .agg(Mean_Size=("Subset_Size", "mean"), SEM_Size=("Subset_Size", "sem"))
            )
            f.write(rr_summary.to_markdown(index=False) + "\n\n")

        f.write("## 4. Figures\n\n")
        for name, path in image_paths.items():
            if path is None:
                continue
            rel = os.path.basename(path)
            f.write(f"### {name}\n")
            f.write(f"![{name}](./{rel})\n\n")

    print(f"[*] Group markdown report written to: {md_path}")


if __name__ == "__main__":
    bundles = load_all_mice_bundles(RESULTS_BASE_DIR)

    master_df = build_master_dataframe(bundles)
    master_csv = os.path.join(GROUP_OUT_DIR, "group_master_metrics.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"[*] Saved master condition table: {master_csv}")

    decile_df = build_decile_dataframe(bundles)
    if not decile_df.empty:
        decile_csv = os.path.join(GROUP_OUT_DIR, "group_corr_deciles_long.csv")
        decile_df.to_csv(decile_csv, index=False)
        print(f"[*] Saved decile long table: {decile_csv}")

    graph_long_df = build_graph_sw_long_dataframe(bundles)
    if not graph_long_df.empty:
        graph_long_csv = os.path.join(GROUP_OUT_DIR, "group_graph_strong_vs_weak_long.csv")
        graph_long_df.to_csv(graph_long_csv, index=False)
        print(f"[*] Saved strong-vs-weak graph long table: {graph_long_csv}")

    rr_overlap_df = build_rr_overlap_dataframe(bundles)
    if not rr_overlap_df.empty:
        rr_csv = os.path.join(GROUP_OUT_DIR, "group_rr_overlap_long.csv")
        rr_overlap_df.to_csv(rr_csv, index=False)
        print(f"[*] Saved RR overlap long table: {rr_csv}")

    candidate_metrics = [
        "Entropy",
        "Mean_RSM_Sim",
        "Mean_Correlation",
        "Strong_Correlation",
        "Weak_Correlation",
        "Strong_Weak_Gap",
        "Participants_Ratio",
        "Gini_Mean",
        "PR_Mean",
        "PR_Norm_Mean",
        "Effective_Dim_PR",
        "Effective_Dim_eRank",
        "Effective_Dim_90Var",
        "GraphGap_efficiency",
        "GraphGap_modularity",
        "GraphGap_local_efficiency",
        "GraphGap_avg_clustering",
    ]
    metrics_to_test = [m for m in candidate_metrics if m in master_df.columns and not master_df[m].isna().all()]

    stat_results = {m: perform_statistical_tests(master_df, m) for m in metrics_to_test}

    image_paths = {}
    image_paths["Entropy"] = plot_group_metric(master_df, "Entropy", "Entropy (bits)", "Representation Entropy", stat_results.get("Entropy", {}), "group_entropy.png")
    image_paths["RSM Mean Similarity"] = plot_group_metric(master_df, "Mean_RSM_Sim", "Cosine similarity", "RSM Mean Similarity", stat_results.get("Mean_RSM_Sim", {}), "group_rsm_mean.png")
    image_paths["Mean Correlation"] = plot_group_metric(master_df, "Mean_Correlation", "Correlation", "Mean Pairwise Correlation", stat_results.get("Mean_Correlation", {}), "group_mean_corr.png")
    image_paths["Strong Correlation (Top 10%)"] = plot_group_metric(master_df, "Strong_Correlation", "Correlation", "Strong Connections (Top 10%)", stat_results.get("Strong_Correlation", {}), "group_strong_corr.png")
    image_paths["Weak Correlation (Bottom 10%)"] = plot_group_metric(master_df, "Weak_Correlation", "Correlation", "Weak Connections (Bottom 10%)", stat_results.get("Weak_Correlation", {}), "group_weak_corr.png")
    image_paths["Strong-Weak Gap"] = plot_group_metric(master_df, "Strong_Weak_Gap", "Correlation gap", "Strong-Weak Correlation Gap", stat_results.get("Strong_Weak_Gap", {}), "group_corr_gap.png")
    image_paths["RR Participants Ratio"] = plot_group_metric(master_df, "Participants_Ratio", "Ratio", "RR Participants Ratio", stat_results.get("Participants_Ratio", {}), "group_participants.png")
    image_paths["Gini (Mean)"] = plot_group_metric(master_df, "Gini_Mean", "Gini", "Trial Response Gini (Mean)", stat_results.get("Gini_Mean", {}), "group_gini_mean.png")
    image_paths["Participation Ratio (Mean)"] = plot_group_metric(master_df, "PR_Mean", "PR", "Trial Participation Ratio (Mean)", stat_results.get("PR_Mean", {}), "group_pr_mean.png")
    image_paths["Effective Dim (PR)"] = plot_group_metric(master_df, "Effective_Dim_PR", "Dimension", "Effective Dimension (PR)", stat_results.get("Effective_Dim_PR", {}), "group_effdim_pr.png")
    image_paths["Effective Dim (eRank)"] = plot_group_metric(master_df, "Effective_Dim_eRank", "Dimension", "Effective Dimension (eRank)", stat_results.get("Effective_Dim_eRank", {}), "group_effdim_erank.png")

    image_paths["Decile Correlation Curve"] = plot_decile_curve(decile_df)
    for gm in GRAPH_METRICS:
        image_paths[f"Graph Strong vs Weak - {gm}"] = plot_graph_sw_comparison(graph_long_df, gm)

    generate_group_markdown(master_df, stat_results, image_paths, rr_overlap_df)
    print("====== Group integration completed ======")
