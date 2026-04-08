# %% [markdown]
# # Mouse Neural Activity Integrated Analysis
# 

# %% [markdown]
# ## Table of Contents
# 
# - [0. Runtime and Imports](#sec-0)
# - [1. Dataset and Output Paths](#sec-1)
# - [2. Fluorescence Preprocessing](#sec-2)
# - [2.1 RR Distribution and Population Response](#sec-21)
# - [2.2 Timepoint Classification and Shuffled Baseline](#sec-22)
# - [3. Spike Preprocessing and Network Construction](#sec-3)
# - [3.1 Network Metrics Visualization](#sec-31)
# - [4. Trial Aggregation and Class Balancing](#sec-4)
# - [5. Neural Pattern Visualization](#sec-5)
# - [6. RSM and Entropy](#sec-6)
# - [7. Class-wise RR Selection](#sec-7)
# - [8. Class-RR vs Other-RR](#sec-8)
# - [9. Strong vs Weak Connectivity](#sec-9)
# - [10. Export and Report Generation](#sec-10)
# - [11. Decoder Robustness (Task 1-2)](#sec-11)
# - [12. FC Decoder Chain (Tasks 3-6)](#sec-12)
# - [13. Population-Pattern Shuffle Dependence](#sec-13)
# - [14. Shuffled Condition Differences](#sec-14)
# - [15. Final Integrated Export](#sec-15)
# 
# 

# %% [markdown]
# <a id="sec-0"></a>
# ## 0. Runtime and Imports
# 
# Import analysis functions and plotting libraries.
# 

# %%
# 导入必要的库
import os
from brainnetwork import load_data, preprocess_data, preprocess_spike_data, rr_selection_class
from brainnetwork import classify_by_timepoints, FI_by_timepoints_v2, FI_by_neuron_count
from brainnetwork import construct_correlation_network, compute_network_metrics_by_class
from brainnetwork.visualization import *
# %% 可视化相关函数定义
from tifffile import imread 
import numpy as np
import pandas as pd
import scipy.stats as stats


# %%

# 可视化需要显示
import matplotlib.pyplot as plt
plt.ion()  # 开启交互式绘图

# %% [markdown]
# <a id="sec-1"></a>
# ## 1. Dataset and Output Paths
# 
# Select one mouse dataset and initialize output directories.
# 

# %%
base_dir = "/beegfs_hdd/data/nfs_share/users/guiyun/nishome/Micedata/"
data_paths = ["M21_1107", "M71_1024", "M73_1128", "M77_1031", "M77_1107", "M78_1017", "M79_1128", "M91_1017"]

# ==========================================================
# Run control:
# - default all analyses
# - if selected analyses already have non-empty output files, skip that mouse
# Options: "core", "decoder_task1_2", "fc_task3_6", "shuffle", "rsm_geometry", "all"
# ==========================================================
RUN_ANALYSES = ["all"]

ANALYSIS_GROUPS = ("core", "decoder_task1_2", "fc_task3_6", "shuffle", "rsm_geometry")


def _nonempty_file(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _resolve_selected_analyses(run_analyses):
    vals = [str(x).strip().lower() for x in (run_analyses or ["all"]) if str(x).strip()]
    if not vals or "all" in vals:
        return list(ANALYSIS_GROUPS)
    unknown = sorted(set(vals) - set(ANALYSIS_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown RUN_ANALYSES values: {unknown}. "
            f"Allowed: {list(ANALYSIS_GROUPS)} + ['all']"
        )
    return vals


def _analysis_marker_paths(mouse_id, data_out_dir, fig_out_dir):
    return {
        "core": [
            os.path.join(data_out_dir, f"{mouse_id}_statistics.json"),
            os.path.join(data_out_dir, f"{mouse_id}_analysis_report.md"),
            os.path.join(data_out_dir, "correlation_deciles.csv"),
            os.path.join(data_out_dir, "trial_response_shape_summary.csv"),
            os.path.join(data_out_dir, "sig_noise_strength_summary_by_condition.csv"),
            os.path.join(data_out_dir, "noise_corr_decile_coupling.csv"),
            os.path.join(fig_out_dir, "pairwise_correlation.png"),
            os.path.join(fig_out_dir, "effective_dimensionality_by_class.png"),
            os.path.join(fig_out_dir, "signal_noise_coupling_scatter_by_condition.png"),
        ],
        "decoder_task1_2": [
            os.path.join(data_out_dir, "decoder_summary.csv"),
            os.path.join(data_out_dir, "decoder_ablation_summary.csv"),
            os.path.join(fig_out_dir, "decoder_confusion_matrix.png"),
            os.path.join(fig_out_dir, "decoder_ablation_top10.png"),
        ],
        "fc_task3_6": [
            os.path.join(data_out_dir, "fc_decoder_summary.csv"),
            os.path.join(data_out_dir, "neuron_decoder_linking_detail.csv"),
            os.path.join(data_out_dir, "fc_edge_decile_enrichment.csv"),
            os.path.join(fig_out_dir, "fc_decoder_confusion_matrix.png"),
            os.path.join(fig_out_dir, "neuron_decoder_linking_panel.png"),
        ],
        "shuffle": [
            os.path.join(data_out_dir, "population_pattern_shuffle_manifest.csv"),
            os.path.join(data_out_dir, "group_corr_shuffle_long.csv"),
            os.path.join(data_out_dir, "group_shuffle_condition_stats.csv"),
            os.path.join(fig_out_dir, "shuffle_condition_difference_by_metric.png"),
        ],
        "rsm_geometry": [
            os.path.join(data_out_dir, "geometry_condition_level_long.csv"),
            os.path.join(data_out_dir, "geometry_condition_pairwise.csv"),
            os.path.join(data_out_dir, "geometry_condition_stats.md"),
            os.path.join(fig_out_dir, "geometry_example_mouse_pc_scatter.png"),
            os.path.join(fig_out_dir, "geometry_angle_condition.png"),
            os.path.join(fig_out_dir, "geometry_orth_parallel_condition.png"),
        ],
    }


def _selected_analyses_done(mouse_id, data_out_dir, fig_out_dir, run_analyses):
    selected = _resolve_selected_analyses(run_analyses)
    marker_map = _analysis_marker_paths(mouse_id, data_out_dir, fig_out_dir)
    done_map = {}
    for key in selected:
        paths = marker_map[key]
        done_map[key] = all(_nonempty_file(p) for p in paths)
    return selected, done_map

for idx, path in enumerate(data_paths):
    data_path = os.path.join(base_dir, path)
    print(f"Processing data from: {data_path}")
    save_dir = "./results/" + path

    data_out_dir = os.path.join(save_dir, "data")
    fig_out_dir = os.path.join(save_dir, "figures")

    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(fig_out_dir, exist_ok=True)

    current_mouse_id = os.path.basename(os.path.normpath(data_path))
    selected_groups, done_map = _selected_analyses_done(
        current_mouse_id, data_out_dir, fig_out_dir, RUN_ANALYSES
    )
    if all(done_map.values()):
        print(
            f"[*] Skip {current_mouse_id}: selected analyses already finished -> "
            f"{', '.join(selected_groups)}"
        )
        continue
    missing_groups = [k for k in selected_groups if not done_map[k]]
    print(
        f"[*] Run {current_mouse_id}: missing analyses -> "
        f"{', '.join(missing_groups)}"
    )

    # %% [markdown]
    # ## Baseline Response and Encoding
    # 
    # This section summarizes baseline response characteristics and stimulus encoding performance.
    # 

    # %% [markdown]
    # <a id="sec-2"></a>
    # ## 2. Fluorescence Preprocessing
    # 
    # Load fluorescence data, segment trials, and prepare labels.
    # 

    # %%
    neuron_data_flo, neuron_pos_flo, start_edges_flo, stimulus_data_flo = load_data(data_path, data_type='fluorescence')
    segments_flo, labels_flo, neuron_pos_rr = preprocess_data(
        neuron_data_flo,
        neuron_pos_flo,
        start_edges_flo,
        stimulus_data_flo,
    )

    # %% [markdown]
    # <a id="sec-21"></a>
    # ### 2.1 RR Distribution and Population Response
    # 
    # Visualize RR neuron spatial distribution and condition-averaged responses.
    # 

    # %%
    try:
        rr_distribution_plot_pretty(neuron_pos_flo, neuron_pos_rr, data_path, gamma=1.1, fig_out_dir=fig_out_dir)
    except Exception as e:
        print(f"Error occurred while plotting RR distribution for spike data: {e}")

    plot_rr_population_average_pretty(segments_flo, labels_flo, fig_out_dir=fig_out_dir)

    # %% [markdown]
    # <a id="sec-22"></a>
    # ### 2.2 Timepoint Classification and Shuffled Baseline
    # 
    # Compare classification accuracy against a label-shuffled baseline.
    # 

    # %%
    accuracies, time_points, accuracy_std, n_folds = classify_by_timepoints(segments_flo, labels_flo)

    # 进行标签随机打乱的分类分析，作为基线比较
    labels_shuffled = np.random.permutation(labels_flo)
    accuracies_shu, time_points_shu, accuracy_std_shu, n_folds_shu = classify_by_timepoints(segments_flo, labels_shuffled)



    plot_accuracy(time_points=time_points, accuracies=accuracies, baseline=accuracies_shu, accuracy_std=accuracy_std, baseline_std=accuracy_std_shu, save_path=os.path.join(fig_out_dir, "classification_accuracy.png"))

    # %% [markdown]
    # <a id="sec-3"></a>
    # ## 3. Spike Preprocessing and Network Construction
    # 
    # Load spike data and compute class-wise functional network metrics.
    # 

    # %%
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data(data_path)
    segments_spi, labels_spi, neuron_pos_spi = preprocess_spike_data(
            neuron_data,
            neuron_pos,
            start_edges,
            stimulus_data,
        )

    nx_result = compute_network_metrics_by_class(segments_spi, labels_spi, neuron_pos_spi, do_bootstrap=False)


    # %%

    # heatmap
    try:
        import seaborn as sns
        matrix = nx_result[1]['corr_matrix']
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, cmap='viridis', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Matrix Heatmap')
        plt.xlabel('Neurons')
        plt.ylabel('Neurons')
        plt.tight_layout()  
        plt.savefig(os.path.join(fig_out_dir, "correlation_matrix_heatmap.png"))
        plt.show()
    except Exception as e:
        print(f"Error occurred while plotting correlation matrix heatmap: {e}")

    # %% [markdown]
    # <a id="sec-31"></a>
    # ### 3.1 Network Metrics Visualization
    # 
    # Plot pairwise correlation distribution and graph-theoretic metrics.
    # 

    # %%
    plot_correlation_violin_optim(nx_result, fig_out_dir=fig_out_dir)
    plot_network_metrics_bars_optim(nx_result, metrics=['efficiency', 'modularity', 'local_efficiency', 'avg_clustering'], figsize=(6,2), fig_out_dir=fig_out_dir)


    # %% [markdown]
    # ## 3.2 Weak-Connection Graph Analysis
    # 
    # Compare graph-theoretic metrics between strongest and weakest connections using matched edge density.
    # 

    # %%

    import networkx as nx
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    def build_rank_graph_from_corr(corr_matrix, frac=0.05, mode="strong", weighted=False):
        """Build a graph by selecting strongest or weakest raw-correlation edges."""
        corr = np.asarray(corr_matrix, dtype=float)
        n = corr.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                v = float(corr[i, j])
                edges.append((v, i, j))

        edges.sort(key=lambda x: x[0], reverse=True)  # high -> low
        k = max(1, int(np.floor(frac * len(edges))))

        if mode == "strong":
            selected = edges[:k]
        elif mode == "weak":
            selected = edges[-k:]
        else:
            raise ValueError("mode must be 'strong' or 'weak'")

        for v, i, j in selected:
            if weighted:
                G.add_edge(i, j, weight=v)
            else:
                G.add_edge(i, j)

        return G


    def compute_graph_metrics(G):
        if G.number_of_edges() == 0:
            return {
                "efficiency": 0.0,
                "modularity": 0.0,
                "local_efficiency": 0.0,
                "avg_clustering": 0.0,
                "density": 0.0,
                "n_edges": 0,
            }

        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        modularity = nx.algorithms.community.modularity(G, communities) if communities else 0.0

        return {
            "efficiency": float(nx.global_efficiency(G)),
            "modularity": float(modularity),
            "local_efficiency": float(nx.local_efficiency(G)) if G.number_of_nodes() > 1 else 0.0,
            "avg_clustering": float(nx.average_clustering(G)),
            "density": float(nx.density(G)),
            "n_edges": int(G.number_of_edges()),
        }


    edge_frac = 0.05  # keep consistent with current strong-network setting
    rows = []
    for cls in sorted(nx_result):
        corr_matrix = np.asarray(nx_result[cls]["corr_matrix"], dtype=float)
        class_name = label_names.get(cls, str(cls)) if "label_names" in globals() else str(cls)

        for mode in ["strong", "weak"]:
            G = build_rank_graph_from_corr(corr_matrix, frac=edge_frac, mode=mode, weighted=False)
            metrics = compute_graph_metrics(G)
            rows.append({
                "Class_ID": cls,
                "Class_Name": class_name,
                "Network_Type": mode,
                "Edge_Fraction": edge_frac,
                **metrics,
            })


    df_topology_sw = pd.DataFrame(rows)

    # Save tabular results
    csv_path = os.path.join(data_out_dir, "network_metrics_strong_vs_weak.csv")
    df_topology_sw.to_csv(csv_path, index=False)
    print(f"[*] Strong-vs-weak graph metrics saved to: {csv_path}")

    # Plot line comparison for key graph metrics
    metrics_to_plot = ["efficiency", "modularity", "local_efficiency", "avg_clustering"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=180, sharex=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics_to_plot):
        sns.lineplot(
            data=df_topology_sw,
            x="Class_Name",
            y=metric,
            hue="Network_Type",
            style="Network_Type",
            markers=True,
            dashes=False,
            linewidth=2,
            palette={"strong": "#1F77B4", "weak": "#D55E00"},
            ax=ax,
        )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Class")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # keep one legend only
    for ax in axes[1:]:
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=labels, frameon=False, title="Network Type")

    fig.suptitle("Graph Metrics: Strongest vs Weakest Connections (Matched Density)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_out_dir, "network_metrics_strong_vs_weak_line.png"), dpi=300, bbox_inches="tight")
    plt.show()



    # %% [markdown]
    # ## 3.3 Threshold-based Graph Analysis
    # 
    # Build strong/weak graphs using fixed raw-correlation thresholds: strong >= 0.35, weak <= 0.10.
    # 

    # %%

    import networkx as nx
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    strong_threshold = 0.35
    weak_threshold = 0.10


    def build_threshold_graph_from_corr(corr_matrix, mode="strong", strong_thr=0.35, weak_thr=0.10, weighted=False):
        """Build graph using thresholded raw correlations."""
        corr = np.asarray(corr_matrix, dtype=float)
        n = corr.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))

        edge_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                v = float(corr[i, j])
                if mode == "strong" and v >= strong_thr:
                    edge_vals.append(v)
                    if weighted:
                        G.add_edge(i, j, weight=v)
                    else:
                        G.add_edge(i, j)
                elif mode == "weak" and v <= weak_thr:
                    edge_vals.append(v)
                    if weighted:
                        G.add_edge(i, j, weight=v)
                    else:
                        G.add_edge(i, j)

        return G, np.asarray(edge_vals, dtype=float)


    # Reuse metric helper if available; otherwise define a local one.
    if 'compute_graph_metrics' not in globals():
        def compute_graph_metrics(G):
            if G.number_of_edges() == 0:
                return {
                    'efficiency': 0.0,
                    'modularity': 0.0,
                    'local_efficiency': 0.0,
                    'avg_clustering': 0.0,
                    'density': 0.0,
                    'n_edges': 0,
                }

            communities = list(nx.algorithms.community.greedy_modularity_communities(G))
            modularity = nx.algorithms.community.modularity(G, communities) if communities else 0.0

            return {
                'efficiency': float(nx.global_efficiency(G)),
                'modularity': float(modularity),
                'local_efficiency': float(nx.local_efficiency(G)) if G.number_of_nodes() > 1 else 0.0,
                'avg_clustering': float(nx.average_clustering(G)),
                'density': float(nx.density(G)),
                'n_edges': int(G.number_of_edges()),
            }


    rows_thr = []
    for cls in sorted(nx_result):
        corr_matrix = np.asarray(nx_result[cls]['corr_matrix'], dtype=float)
        class_name = label_names.get(cls, str(cls)) if 'label_names' in globals() else str(cls)

        for mode in ['strong', 'weak']:
            G_thr, edge_vals = build_threshold_graph_from_corr(
                corr_matrix,
                mode=mode,
                strong_thr=strong_threshold,
                weak_thr=weak_threshold,
                weighted=False,
            )
            metrics = compute_graph_metrics(G_thr)
            rows_thr.append({
                'Class_ID': cls,
                'Class_Name': class_name,
                'Network_Type': f'{mode}_threshold',
                'Strong_Threshold': strong_threshold,
                'Weak_Threshold': weak_threshold,
                'Edge_Mean_Corr': float(np.mean(edge_vals)) if edge_vals.size > 0 else np.nan,
                'Edge_Min_Corr': float(np.min(edge_vals)) if edge_vals.size > 0 else np.nan,
                'Edge_Max_Corr': float(np.max(edge_vals)) if edge_vals.size > 0 else np.nan,
                **metrics,
            })


    df_topology_threshold = pd.DataFrame(rows_thr)

    csv_path_thr = os.path.join(data_out_dir, 'network_metrics_threshold_strong0.35_weak0.10.csv')
    df_topology_threshold.to_csv(csv_path_thr, index=False)
    print(f'[*] Threshold-based graph metrics saved to: {csv_path_thr}')

    # Plot threshold-based comparison
    metrics_to_plot = ['efficiency', 'modularity', 'local_efficiency', 'avg_clustering']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=180, sharex=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics_to_plot):
        sns.lineplot(
            data=df_topology_threshold,
            x='Class_Name',
            y=metric,
            hue='Network_Type',
            style='Network_Type',
            markers=True,
            dashes=False,
            linewidth=2,
            palette={'strong_threshold': '#1F77B4', 'weak_threshold': '#D55E00'},
            ax=ax,
        )
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Class')
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[1:]:
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=labels, frameon=False, title='Threshold Graph Type')

    fig.suptitle('Graph Metrics: Threshold-based Strong (>=0.35) vs Weak (<=0.10)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_out_dir, 'network_metrics_threshold_strong_vs_weak_line.png'), dpi=300, bbox_inches='tight')
    plt.show()



    # %% [markdown]
    # <a id="sec-4"></a>
    # ## 4. Trial Aggregation and Class Balancing
    # 
    # Aggregate activity in a key time window and balance class sample counts.
    # 

    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Trial 聚合：计算每个 Trial 的平均活动
    # 假设 X_trials 的形状是 (Trial总数, 帧数, 神经元数)
    X_trial_averaged = np.mean(segments_spi[:, :, 10:13], axis=2) # 得到 (n_trials, n_neurons)

    unique_labels = np.unique(labels_spi)
    min_count = min(np.sum(labels_spi == l) for l in unique_labels)

    # 2. 平衡数据集
    X_balanced_list = []
    labels_balanced = []

    for label in unique_labels:
        mask = (labels_spi == label)
        X_label = X_trial_averaged[mask]
        
        # 建议：随机采样而非固定取前几个
        indices = np.random.choice(len(X_label), min_count, replace=False)
        X_sampled = X_label[indices]
        
        X_balanced_list.append(X_sampled)
        labels_balanced.extend([label] * min_count)

    # 2. 关键修正：将列表转化为单一矩阵 [cite: 1]
    X_trial_averaged_balanced = np.vstack(X_balanced_list)
    labels_balanced = np.array(labels_balanced)


    # %%
    # 1. Trial 聚合：计算每个 Trial 的平均活动
    # 假设 X_trials 的形状是 (Trial总数, 帧数, 神经元数)
    X_trial_averaged = np.mean(segments_spi[:, :, 10:13], axis=2) # 得到 (n_trials, n_neurons)

    unique_labels = np.unique(labels_spi)
    min_count = min(np.sum(labels_spi == l) for l in unique_labels)

    # 2. 平衡数据集
    X_balanced_list = []
    labels_balanced = []

    for label in unique_labels:
        mask = (labels_spi == label)
        X_label = X_trial_averaged[mask]
        
        # 建议：随机采样而非固定取前几个
        indices = np.random.choice(len(X_label), min_count, replace=False)
        X_sampled = X_label[indices]
        
        X_balanced_list.append(X_sampled)
        labels_balanced.extend([label] * min_count)

    # 2. 关键修正：将列表转化为单一矩阵 [cite: 1]
    X_trial_averaged_balanced = np.vstack(X_balanced_list)
    labels_balanced = np.array(labels_balanced)


    # %% [markdown]
    # <a id="sec-5"></a>
    # ## 5. Neural Pattern Visualization
    # 
    # Show condition-related activity patterns by preference sorting and clustering.
    # 

    # %%
    from scipy.stats import zscore

    # 定义语义映射
    label_names = {1: 'Divergent', 2: 'Convergent', 3: 'Random', 0: 'Resting'}

    def plot_sorted_neural_patterns(X, labels, label_map):
        """
        可视化神经元活动热图，通过两种策略对神经元（X轴）进行排序，
        以揭示不同刺激条件（Y轴）下的Pattern差异。
        """
        
        # 1. 数据预处理
        # ------------------------------------------------
        # Z-score 归一化：消除神经元绝对发放率的差异，只看相对变化
        # axis=0 对每一列（神经元）做归一化
        X_norm = zscore(X, axis=0)
        
        # 替换 NaN (如果有些神经元完全不发放，std为0，zscore会产生nan)
        X_norm = np.nan_to_num(X_norm)
        
        unique_labels = np.unique(labels)
        sorted_labels = sorted(unique_labels) # 确保顺序: Divergent, Convergent, Random...
        
        # 获取标签的分界线位置，用于画横线
        label_counts = [np.sum(labels == l) for l in sorted_labels]
        boundaries = np.cumsum(label_counts)[:-1]
        
        # 重新排列 X 的行（Trial），确保它是按 Label 顺序排列的
        # (虽然通常已经是排好的，但为了保险起见)
        trial_sort_idx = np.argsort(labels)
        X_sorted_trials = X_norm[trial_sort_idx]
        labels_sorted = labels[trial_sort_idx]
        
        # ------------------------------------------------
        # 策略一：偏好性排序 (Preference Sorting)
        # ------------------------------------------------
        # 计算每个神经元在各条件下的平均活动
        n_neurons = X.shape[1]
        neuron_means = np.zeros((len(unique_labels), n_neurons))
        
        for i, l in enumerate(sorted_labels):
            # 找到该条件下的所有行
            mask = (labels == l)
            # 计算该条件下每个神经元的均值
            neuron_means[i, :] = np.mean(X_norm[mask], axis=0)
            
        # 找出每个神经元在哪个条件下最强 (Argmax)
        peak_condition_idx = np.argmax(neuron_means, axis=0)
        
        # 在同一个条件下，再按强度排个序，为了视觉更好看
        peak_magnitude = np.max(neuron_means, axis=0)
        
        # 综合排序键值: 先按条件ID排，再按强度排
        # 技巧: use lexsort. Primary key is last argument.
        sort_key = np.lexsort((-peak_magnitude, peak_condition_idx))
        
        X_pref_sorted = X_sorted_trials[:, sort_key]
        
        # ------------------------------------------------
        # 绘图 1: 偏好性排序热图
        # ------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(X_pref_sorted, ax=ax, cmap='viridis', vmin=-2, vmax=2, cbar_kws={'label': 'Z-scored Activity'})
        
        # 画水平线分隔不同条件
        for b in boundaries:
            ax.hlines(b, *ax.get_xlim(), colors='white', linestyles='--', linewidth=1)
            
        # 标注 Y 轴
        # 计算每个 Block 的中心位置
        y_centers = []
        current = 0
        for count in label_counts:
            y_centers.append(current + count/2)
            current += count
            
        ax.set_yticks(y_centers)
        ax.set_yticklabels([label_map[l] for l in sorted_labels], rotation=0, fontsize=12, fontweight='bold')
        
        ax.set_title("Neural Patterns Sorted by Tuning Preference\n(Neurons grouped by which stimulus drives them most)", fontsize=14)
        ax.set_xlabel("Neurons (Sorted by Preferred Condition)")
        ax.set_ylabel("Trials (Grouped by Condition)")
        
        plt.tight_layout()
        save_path = os.path.join(fig_out_dir, "neural_patterns_preference_sorted.png")
        fig.savefig(save_path, dpi=300)
        plt.show()

        # ------------------------------------------------
        # 策略二：层次聚类 (Hierarchical Clustering)
        # ------------------------------------------------
        # 这会展示神经元的自然功能团
        print("Generating Clustermap (Hierarchical Clustering)...")
        
        # 创建行颜色条
        label_colors = {l: sns.color_palette("tab10")[i] for i, l in enumerate(unique_labels)}
        row_colors = [label_colors[l] for l in labels_sorted]
        
        g = sns.clustermap(X_sorted_trials, 
                        method='ward', # 聚类算法
                        metric='euclidean',
                        col_cluster=True, # 对神经元聚类
                        row_cluster=False, # 不对 Trial 聚类 (保持条件顺序)
                        row_colors=row_colors,
                        cmap='magma', vmin=-2, vmax=2,
                        figsize=(12, 8),
                        dendrogram_ratio=(0.1, 0.2),
                        cbar_pos=(0.02, 0.8, 0.03, 0.15))
        
        # 添加水平线
        for b in boundaries:
            g.ax_heatmap.hlines(b, *g.ax_heatmap.get_xlim(), colors='white', linestyles='--', linewidth=1)
            
        g.fig.suptitle("Neural Patterns Sorted by Similarity (Hierarchical Clustering)", y=1.02, fontsize=14)
        
        # 手动添加 Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=label_colors[l], label=label_map[l]) for l in sorted_labels]
        plt.legend(handles=handles, title='Condition', bbox_to_anchor=(1.2, 1), loc='upper left')
        save_path = os.path.join(fig_out_dir, "neural_patterns_clustermap.png")
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # 运行代码
    # 确保 X_trial_averaged_balanced 是 (n_trials, n_neurons)
    plot_sorted_neural_patterns(X_trial_averaged_balanced, labels_balanced, label_names)

    # %% [markdown]
    # <a id="sec-6"></a>
    # ## 6. RSM and Entropy
    # 
    # Compute trial-level similarity distributions and Shannon entropy.
    # 

    # %%
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.stats import entropy

    def analyze_rsm_and_entropy(X, labels, label_map, n_bins=50):
        """
        计算不同刺激条件下的RSM熵值和相似度分布
        """
        results = []
        rsm_dict = {}
        similarity_values_dict = {}

        print(f"{'Stimulus':<15} | {'Entropy (Bits)':<15} | {'Mean Similarity':<15} | {'Std Similarity':<15}")
        print("-" * 70)

        # 遍历每种刺激标签
        unique_labels = np.unique(labels)
        for label in unique_labels:
            name = label_map[label]
            
            # 1. 提取该条件下的数据
            mask = (labels == label)
            X_sub = X[mask]
            
            # 2. 计算 RSM (Cosine Similarity)
            # 结果是 (n_trials, n_trials) 的矩阵，范围 [-1, 1]
            rsm = cosine_similarity(X_sub)
            rsm_dict[name] = rsm
            
            # 3. 提取上三角非对角线元素 (Off-diagonal upper triangle)
            # k=1 表示不包含对角线。因为我们只关心试次间的变异，不关心试次和自己的相似度
            triu_indices = np.triu_indices_from(rsm, k=1)
            sim_values = rsm[triu_indices]
            similarity_values_dict[name] = sim_values
            
            # 4. 计算香农熵 (Shannon Entropy)
            # 先将连续的相似度值通过直方图转为概率分布
            # range=(-1, 1) 覆盖余弦相似度的理论范围
            counts, _ = np.histogram(sim_values, bins=n_bins, range=(-1, 1), density=True)
            
            # 归一化为概率 (加上极小值防止 log(0) 错误)
            probs = counts / np.sum(counts)
            ent = entropy(probs, base=2)  # base=2 单位为 bits
            
            # 记录统计量
            results.append({
                'Stimulus': name,
                'Entropy': ent,
                'Mean_Sim': np.mean(sim_values),
                'Std_Sim': np.std(sim_values)
            })
            
            print(f"{name:<15} | {ent:.4f}           | {np.mean(sim_values):.4f}          | {np.std(sim_values):.4f}")

        return pd.DataFrame(results), rsm_dict, similarity_values_dict


    df_entropy, rsm_data, sim_values_data = analyze_rsm_and_entropy(
        X_trial_averaged_balanced, 
        labels_balanced, 
        label_names,
        n_bins=50 # 分箱数，可以根据数据量调整，通常 30-100 之间
    )

    # --- 可视化 1: 相似度分布直方图 (验证是否"变平"了) ---
    plt.figure(figsize=(12, 3))
    colors = {'Divergent': '#FF4B4B', 'Convergent': '#1C75BC', 'Random': '#7AC143', 'Resting': '#111111'}

    for name, values in sim_values_data.items():
        sns.kdeplot(values, label=f"{name} (H={df_entropy[df_entropy['Stimulus']==name]['Entropy'].values[0]:.2f})", 
                    color=colors[name], fill=True, alpha=0.3, linewidth=2)

    # plt.title("Distribution of Pairwise Cosine Similarities (RSM Off-diagonal)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.xlim(-0.5, 1.0) # 根据实际数据范围调整
    plt.legend(title="Stimulus Type")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_out_dir, "similarity_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    # plot heatmap of RSM
    for name, rsm in rsm_data.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(rsm, cmap='viridis', vmin=-1, vmax=1) #cbar_kws={'label': 'Cosine Similarity'})
        # 不要颜色条了，直接在标题里写熵值
        

        plt.title(f"RSM Heatmap - {name} (H={df_entropy[df_entropy['Stimulus']==name]['Entropy'].values[0]:.2f} bits)")
        plt.xlabel("Trials")
        plt.ylabel("Trials")
        plt.tight_layout()
        save_path = os.path.join(fig_out_dir, f"rsm_heatmap_{name}.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    # %% [markdown]
    # ## 6.1 Response Sparsity and Effective Dimensionality
    # 
    # Compute trial-level Gini and Participation Ratio, plus class-level covariance-spectrum effective dimensionality.
    # 

    # %%

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    def gini_coefficient(x):
        """Gini coefficient for a 1D response vector."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return np.nan

        # Shift to non-negative if needed (keeps inequality structure)
        min_x = np.min(x)
        if min_x < 0:
            x = x - min_x

        s = np.sum(x)
        if s <= 1e-12:
            return 0.0

        x_sorted = np.sort(x)
        n = x_sorted.size
        idx = np.arange(1, n + 1)
        g = (2.0 * np.sum(idx * x_sorted) / (n * s)) - (n + 1) / n
        return float(g)


    def participation_ratio_vector(x):
        """Participation ratio of one trial response vector."""
        x = np.asarray(x, dtype=float).ravel()
        num = np.sum(x ** 2) ** 2
        den = np.sum(x ** 4)
        if den <= 1e-12:
            return 0.0
        return float(num / den)


    def effective_dims_from_cov(X):
        """Return PR-dimension and entropy-rank dimension from covariance eigen-spectrum."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            return np.nan, np.nan, np.nan

        Xc = X - np.mean(X, axis=0, keepdims=True)
        cov = np.cov(Xc, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-12]
        if eigvals.size == 0:
            return 0.0, 0.0, 0.0

        # PR dimension
        pr_dim = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)

        # Entropy-rank (effective rank)
        p = eigvals / np.sum(eigvals)
        erank_dim = np.exp(-np.sum(p * np.log(p + 1e-12)))

        # 90% variance dimension
        eig_sorted = np.sort(eigvals)[::-1]
        cumsum = np.cumsum(eig_sorted) / np.sum(eig_sorted)
        dim90 = int(np.searchsorted(cumsum, 0.90) + 1)

        return float(pr_dim), float(erank_dim), float(dim90)


    # Use balanced trial-level matrix computed above: shape (n_trials, n_neurons)
    X_resp = np.asarray(X_trial_averaged_balanced, dtype=float)
    y_resp = np.asarray(labels_balanced)

    trial_rows = []
    for i in range(X_resp.shape[0]):
        cls = int(y_resp[i])
        v = X_resp[i]
        trial_rows.append({
            "Trial_Index": i,
            "Class_ID": cls,
            "Class_Name": label_names.get(cls, str(cls)) if "label_names" in globals() else str(cls),
            "Gini": gini_coefficient(v),
            "Participation_Ratio": participation_ratio_vector(v),
            "Participation_Ratio_Norm": participation_ratio_vector(v) / max(1, v.size),
        })

    df_trial_shape = pd.DataFrame(trial_rows)

    # Class-level effective dimensionality from covariance spectrum
    dim_rows = []
    for cls in sorted(np.unique(y_resp)):
        X_cls = X_resp[y_resp == cls]
        pr_dim, erank_dim, dim90 = effective_dims_from_cov(X_cls)
        dim_rows.append({
            "Class_ID": int(cls),
            "Class_Name": label_names.get(int(cls), str(int(cls))) if "label_names" in globals() else str(int(cls)),
            "N_Trials": int(X_cls.shape[0]),
            "N_Neurons": int(X_cls.shape[1]),
            "Effective_Dim_PR": pr_dim,
            "Effective_Dim_eRank": erank_dim,
            "Effective_Dim_90Var": dim90,
        })

    df_effective_dim = pd.DataFrame(dim_rows)

    # Summary stats for trial-level metrics
    agg = {
        "Gini": ["mean", "std"],
        "Participation_Ratio": ["mean", "std"],
        "Participation_Ratio_Norm": ["mean", "std"],
    }
    df_trial_shape_summary = df_trial_shape.groupby(["Class_ID", "Class_Name"]).agg(agg).reset_index()
    df_trial_shape_summary.columns = [
        "Class_ID", "Class_Name",
        "Gini_Mean", "Gini_STD",
        "PR_Mean", "PR_STD",
        "PR_Norm_Mean", "PR_Norm_STD",
    ]

    # Save tables
    trial_csv = os.path.join(data_out_dir, "trial_response_shape_metrics.csv")
    trial_summary_csv = os.path.join(data_out_dir, "trial_response_shape_summary.csv")
    dim_csv = os.path.join(data_out_dir, "effective_dimensionality_by_class.csv")

    df_trial_shape.to_csv(trial_csv, index=False)
    df_trial_shape_summary.to_csv(trial_summary_csv, index=False)
    df_effective_dim.to_csv(dim_csv, index=False)

    print(f"[*] Trial-level response shape metrics saved: {trial_csv}")
    print(f"[*] Trial-level summary saved: {trial_summary_csv}")
    print(f"[*] Effective dimensionality saved: {dim_csv}")

    # Plot A: Trial-level Gini / PR by class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    sns.boxplot(data=df_trial_shape, x="Class_Name", y="Gini", ax=axes[0], color="#9ecae1")
    axes[0].set_title("Trial-level Gini by Class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Gini")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    sns.boxplot(data=df_trial_shape, x="Class_Name", y="Participation_Ratio", ax=axes[1], color="#fdae6b")
    axes[1].set_title("Trial-level Participation Ratio by Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Participation Ratio")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_out_dir, "response_gini_pr_by_class.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Plot B: Effective dimensionality comparison
    df_dim_long = df_effective_dim.melt(
        id_vars=["Class_ID", "Class_Name"],
        value_vars=["Effective_Dim_PR", "Effective_Dim_eRank", "Effective_Dim_90Var"],
        var_name="Metric",
        value_name="Value",
    )

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    sns.barplot(data=df_dim_long, x="Class_Name", y="Value", hue="Metric", ax=ax2)
    ax2.set_title("Effective Dimensionality by Class")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Dimension")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_out_dir, "effective_dimensionality_by_class.png"), dpi=300, bbox_inches="tight")
    plt.show()



    # %% [markdown]
    # <a id="sec-7"></a>
    # ## 7. Class-wise RR Selection
    # 
    # Extract reliable-response neuron sets for each stimulus class.
    # 

    # %%
    # Experiment configuration
    rr_neurons_spi = rr_selection_class(segments_spi, labels_spi)


    # %%
    rr_neurons_spi.keys()
    rr_union = set.union(*rr_neurons_spi.values())
    # rr_union

    # %% [markdown]
    # ## 7.1 RR Set Export and Venn Visualization
    # 
    # Save class-wise RR neuron sets and overlap statistics, then visualize overlap with a Venn diagram.
    # 

    # %%

    import os
    import json
    import itertools
    import pandas as pd
    import matplotlib.pyplot as plt

    # Normalize class-wise RR sets
    rr_sets = {int(cls): set(map(int, neurons)) for cls, neurons in rr_neurons_spi.items()}
    class_ids = sorted(rr_sets.keys())

    if len(class_ids) == 0:
        raise ValueError("rr_neurons_spi is empty; cannot export RR set information.")

    # Basic counts
    rr_union = set.union(*rr_sets.values()) if rr_sets else set()
    rr_intersection_all = set.intersection(*rr_sets.values()) if rr_sets else set()

    # Build overlap summary rows
    overlap_rows = []
    for cls in class_ids:
        overlap_rows.append({
            "Subset": f"Class_{cls}",
            "Subset_Size": len(rr_sets[cls]),
        })

    for a, b in itertools.combinations(class_ids, 2):
        overlap_rows.append({
            "Subset": f"Class_{a}&Class_{b}",
            "Subset_Size": len(rr_sets[a] & rr_sets[b]),
        })

    if len(class_ids) >= 3:
        overlap_rows.append({
            "Subset": "All_Classes_Intersection",
            "Subset_Size": len(rr_intersection_all),
        })

    overlap_rows.append({"Subset": "Union_All", "Subset_Size": len(rr_union)})

    df_rr_overlap = pd.DataFrame(overlap_rows)

    # Build neuron membership table for downstream stats
    membership_rows = []
    for nid in sorted(rr_union):
        row = {"Neuron_ID": int(nid)}
        active = []
        for cls in class_ids:
            in_set = int(nid in rr_sets[cls])
            row[f"Class_{cls}"] = in_set
            if in_set:
                active.append(str(cls))
        row["Membership_Pattern"] = ",".join(active)
        row["Membership_Count"] = len(active)
        membership_rows.append(row)

    df_rr_membership = pd.DataFrame(membership_rows)

    # Save structured RR-set data
    rr_set_json_path = os.path.join(data_out_dir, "rr_sets_by_class.json")
    rr_overlap_csv_path = os.path.join(data_out_dir, "rr_overlap_summary.csv")
    rr_membership_csv_path = os.path.join(data_out_dir, "rr_neuron_membership.csv")

    rr_json = {
        "class_ids": class_ids,
        "rr_sets": {str(cls): sorted(list(rr_sets[cls])) for cls in class_ids},
        "union_all": sorted(list(rr_union)),
        "intersection_all": sorted(list(rr_intersection_all)),
    }

    with open(rr_set_json_path, "w", encoding="utf-8") as f:
        json.dump(rr_json, f, indent=2)

    df_rr_overlap.to_csv(rr_overlap_csv_path, index=False)
    df_rr_membership.to_csv(rr_membership_csv_path, index=False)

    print(f"[*] RR set JSON saved to: {rr_set_json_path}")
    print(f"[*] RR overlap summary CSV saved to: {rr_overlap_csv_path}")
    print(f"[*] RR membership CSV saved to: {rr_membership_csv_path}")

    # Plot venn diagram for exactly 3 classes if available
    if len(class_ids) == 3:
        try:
            from matplotlib_venn import venn3

            a, b, c = class_ids
            set_a, set_b, set_c = rr_sets[a], rr_sets[b], rr_sets[c]

            fig, ax = plt.subplots(figsize=(6.8, 6), dpi=180)
            venn3(
                subsets=(set_a, set_b, set_c),
                set_labels=(f"Class {a}", f"Class {b}", f"Class {c}"),
                ax=ax,
            )
            ax.set_title("RR Neuron Overlap Across Classes")
            plt.tight_layout()

            venn_path = os.path.join(fig_out_dir, "rr_sets_venn.png")
            fig.savefig(venn_path, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"[*] RR Venn figure saved to: {venn_path}")
        except ImportError:
            print("[!] matplotlib-venn is not installed; skipped Venn plot. Install via: pip install matplotlib-venn")
    else:
        print(f"[!] Venn plot skipped: expected exactly 3 classes, got {len(class_ids)} classes ({class_ids}).")



    # %% [markdown]
    # <a id="sec-8"></a>
    # ## 8. Class-RR vs Other-RR
    # 
    # Compare condition-specific RR neurons with other RR neurons in the RR pool.
    # 

    # %%
    # Compare class-specific RR neurons with the remaining RR neurons inside the RR pool.
    # This cell reuses rr_by_class / rr_union cached above and avoids recomputing RR sets.


    # 定义语义映射
    label_names = {1: 'Divergent', 2: 'Convergent', 3: 'Random'}

    class_colors = {1: '#1F77B4', 2: '#D55E00', 3: '#009E73'}

    participants = {}

    def class_curve_and_sem(segments_group, labels, cls):
        cls_segments = segments_group[labels == cls]
        pop_mean = np.nanmean(cls_segments, axis=(0, 1))
        trial_means = np.nanmean(cls_segments, axis=1)
        pop_sem = stats.sem(trial_means, axis=0, nan_policy="omit")
        return pop_mean, pop_sem

    def pop_curve_and_sem(trials_subset):
        # trials_subset: (n_trials, n_neurons_subset, n_time)
        m = np.nanmean(trials_subset, axis=(0, 1))
        trial_means = np.nanmean(trials_subset, axis=1)  # (n_trials, n_time)
        se = stats.sem(trial_means, axis=0, nan_policy='omit')
        return m, se


    time_axis = np.arange(-10,40) # Assuming segments_spi has 50 frames from -10 to +39
    fig, axes = plt.subplots(1, len(label_names), figsize=(5.2 * len(label_names), 4.5), dpi=180, sharey=True)
    if len(label_names) == 1:
        axes = [axes]
        
    for ax, cls in zip(axes, label_names.keys()):
        cls_trials = segments_spi[labels_spi == cls]
        rr_cls = rr_neurons_spi[cls]
        other_rr = rr_union - set(rr_cls)

        if len(rr_cls) == 0:
            ax.set_title(f'Class {cls}: no class-specific RR')
            ax.axis('off')
            continue

        m_rr, se_rr = pop_curve_and_sem(cls_trials[:, list(rr_cls), :])
        ax.plot(time_axis, m_rr, color=class_colors.get(cls, '#1F77B4'), lw=2.4, label=f'{label_names.get(cls, cls)}-RR')
        ax.fill_between(time_axis, m_rr - se_rr, m_rr + se_rr, color=class_colors.get(cls, '#1F77B4'), alpha=0.2, linewidth=0)

        if len(other_rr)     > 0:
            m_oth, se_oth = pop_curve_and_sem(cls_trials[:, list(other_rr), :])
            ax.plot(time_axis, m_oth, color='#444444', lw=2.2, label='Other-RR (within RR pool)')
            ax.fill_between(time_axis, m_oth - se_oth, m_oth + se_oth, color='#444444', alpha=0.15, linewidth=0)

        try:
            participants[cls] = np.mean(m_rr[10:13])/ np.mean(m_oth[10:13]) if len(other_rr) > 0 else np.mean(m_rr[10:13])
        except ZeroDivisionError:
            participants[cls] = 0
        ax.set_title(f'Condition {cls}: class-RR vs other-RR')
        ax.set_xlabel('Time (frames)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel('Mean response')
    fig.suptitle('Within RR pool: condition-specific RR vs other-RR', y=1.03)
    fig.tight_layout()
    plt.show()

    print("Participants (mean RR response at peak time / mean other-RR response at peak time):")
    for cls, value in participants.items():
        print(f"Condition {cls}: {value:.2f}")

    # %% [markdown]
    # <a id="sec-81"></a>
    # ## 8.1 RSM Geometry Analysis
    #
    # Quantify the geometric relationship between trial-cloud variability and condition mean axis.
    #

    # %%
    import itertools

    GEOM_EPS = 1e-12
    GEOM_N_BOOT = 500


    def _safe_unit(v, eps=GEOM_EPS):
        v = np.asarray(v, dtype=float).ravel()
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= eps:
            return np.zeros_like(v), 0.0
        return v / n, float(n)


    def _geometry_metrics_from_trials(X_cond):
        X = np.asarray(X_cond, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            return {
                "n_trials": int(X.shape[0]) if X.ndim == 2 else 0,
                "n_neurons": int(X.shape[1]) if X.ndim == 2 else 0,
                "mean_norm": np.nan,
                "angle_deg": np.nan,
                "var_parallel": np.nan,
                "var_orthogonal": np.nan,
                "orth_parallel_ratio": np.nan,
                "anisotropy_index": np.nan,
                "lambda1": np.nan,
                "lambda2": np.nan,
            }

        T, N = X.shape
        mu = np.mean(X, axis=0)
        mu_hat, mu_norm = _safe_unit(mu)
        Y = X - mu

        # PC1 of centered trial cloud
        try:
            _, _, vt = np.linalg.svd(Y, full_matrices=False)
            v1 = vt[0]
        except Exception:
            v1 = np.zeros(N, dtype=float)
            if N > 0:
                v1[0] = 1.0
        v1_u, _ = _safe_unit(v1)

        if np.linalg.norm(mu_hat) <= GEOM_EPS or np.linalg.norm(v1_u) <= GEOM_EPS:
            angle_deg = np.nan
        else:
            cosv = np.clip(np.abs(np.dot(mu_hat, v1_u)), 0.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cosv)))

        # Decompose centered cloud vectors (y_t = x_t - mu) wrt mean axis.
        # This follows "average component energy" rather than ellipse-only geometry.
        if np.linalg.norm(mu_hat) > GEOM_EPS:
            a = Y @ mu_hat
            # Mean energy of parallel component across points in the cloud.
            var_parallel = float(np.mean(a ** 2))
            r = Y - np.outer(a, mu_hat)
            # Mean energy of orthogonal residual across points in the cloud.
            var_orth = float(np.mean(np.sum(r ** 2, axis=1)))
            ratio = float(var_orth / (var_parallel + GEOM_EPS))
        else:
            # Mean axis undefined when ||mu|| is near zero.
            var_parallel = np.nan
            var_orth = np.nan
            ratio = np.nan

        # Spectrum-based anisotropy
        try:
            cov = np.cov(Y, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
        except Exception:
            eigvals = np.array([])
        lam1 = float(eigvals[0]) if eigvals.size >= 1 else np.nan
        lam2 = float(eigvals[1]) if eigvals.size >= 2 else np.nan
        anis = float(lam1 / (np.nansum(eigvals) + GEOM_EPS)) if eigvals.size > 0 else np.nan

        return {
            "n_trials": int(T),
            "n_neurons": int(N),
            "mean_norm": float(mu_norm),
            "angle_deg": angle_deg,
            "var_parallel": var_parallel,
            "var_orthogonal": var_orth,
            "orth_parallel_ratio": ratio,
            "anisotropy_index": anis,
            "lambda1": lam1,
            "lambda2": lam2,
        }


    def _bootstrap_geometry_distributions(X_cond, n_boot=GEOM_N_BOOT, seed=0):
        X = np.asarray(X_cond, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            return {}
        rng = np.random.default_rng(seed)
        T = X.shape[0]
        records = []
        for _ in range(int(n_boot)):
            idx_boot = rng.integers(0, T, size=T)
            met = _geometry_metrics_from_trials(X[idx_boot])
            records.append(met)
        dfb = pd.DataFrame(records)
        out = {}
        for col in ["angle_deg", "orth_parallel_ratio", "var_parallel", "var_orthogonal", "anisotropy_index"]:
            vals = dfb[col].astype(float).values if col in dfb.columns else np.array([])
            vals = vals[np.isfinite(vals)]
            out[col] = vals
        return out


    def _bootstrap_pairwise_from_distributions(dist_by_cond, metric, cond_order):
        rows = []
        for c1, c2 in itertools.combinations(cond_order, 2):
            a = np.asarray(dist_by_cond.get(c1, {}).get(metric, []), dtype=float)
            b = np.asarray(dist_by_cond.get(c2, {}).get(metric, []), dtype=float)
            n = min(a.size, b.size)
            if n == 0:
                rows.append(
                    {
                        "metric": metric,
                        "condition_1": c1,
                        "condition_2": c2,
                        "n_boot": 0,
                        "mean_diff_boot": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "p_boot_two_sided": np.nan,
                    }
                )
                continue
            d = a[:n] - b[:n]
            p_low = (np.sum(d <= 0) + 1.0) / (n + 1.0)
            p_high = (np.sum(d >= 0) + 1.0) / (n + 1.0)
            p_two = float(min(1.0, 2.0 * min(p_low, p_high)))
            rows.append(
                {
                    "metric": metric,
                    "condition_1": c1,
                    "condition_2": c2,
                    "n_boot": int(n),
                    "mean_diff_boot": float(np.mean(d)),
                    "ci95_low": float(np.quantile(d, 0.025)),
                    "ci95_high": float(np.quantile(d, 0.975)),
                    "p_boot_two_sided": p_two,
                }
            )
        return rows


    def _condition_order_from_labels(y, label_map):
        classes = sorted(np.unique(np.asarray(y).astype(int)).tolist())
        order = [c for c in [1, 2, 3] if c in classes]
        order.extend([c for c in classes if c not in order])
        return order, [label_map.get(int(c), str(c)) for c in order]


    ordered_cls, ordered_cond_names = _condition_order_from_labels(y_resp, label_names)
    geometry_rows = []
    geometry_boot = {}
    for cls in ordered_cls:
        X_cls = np.asarray(X_resp[y_resp == cls], dtype=float)
        cond_name = label_names.get(int(cls), str(cls))
        met = _geometry_metrics_from_trials(X_cls)
        geometry_rows.append(
            {
                "mouse": current_mouse_id,
                "Class_ID": int(cls),
                "Condition": cond_name,
                "n_trials": met["n_trials"],
                "n_neurons": met["n_neurons"],
                "mean_norm": met["mean_norm"],
                "angle_deg": met["angle_deg"],
                "var_parallel": met["var_parallel"],
                "var_orthogonal": met["var_orthogonal"],
                "orth_parallel_ratio": met["orth_parallel_ratio"],
                "anisotropy_index": met["anisotropy_index"],
                "lambda1": met["lambda1"],
                "lambda2": met["lambda2"],
            }
        )
        geometry_boot[cond_name] = _bootstrap_geometry_distributions(
            X_cls, n_boot=GEOM_N_BOOT, seed=20260329 + int(cls)
        )

    df_geometry = pd.DataFrame(geometry_rows)
    geometry_csv = os.path.join(data_out_dir, "geometry_condition_level_long.csv")
    df_geometry.to_csv(geometry_csv, index=False)
    print(f"[*] Geometry condition-level table saved: {geometry_csv}")

    # Pairwise bootstrap comparisons (single-mouse within-condition)
    pair_rows = []
    for m in ["angle_deg", "orth_parallel_ratio", "var_parallel", "var_orthogonal", "anisotropy_index"]:
        pair_rows.extend(_bootstrap_pairwise_from_distributions(geometry_boot, m, ordered_cond_names))
    df_geometry_pairwise = pd.DataFrame(pair_rows)
    geometry_pairwise_csv = os.path.join(data_out_dir, "geometry_condition_pairwise.csv")
    df_geometry_pairwise.to_csv(geometry_pairwise_csv, index=False)
    print(f"[*] Geometry pairwise bootstrap table saved: {geometry_pairwise_csv}")

    # Condition stats markdown
    geometry_stats_md = os.path.join(data_out_dir, "geometry_condition_stats.md")
    with open(geometry_stats_md, "w", encoding="utf-8") as f:
        f.write("# RSM Geometry Condition-level Summary\n\n")
        f.write("## Condition metrics\n\n")
        f.write(df_geometry.to_markdown(index=False) + "\n\n")
        f.write("## Pairwise bootstrap differences\n\n")
        f.write(df_geometry_pairwise.to_markdown(index=False) + "\n")
    print(f"[*] Geometry condition markdown saved: {geometry_stats_md}")

    # Merge with Mean RSM / participants / effective dimensionality for model proxies
    mean_rsm_map = {}
    if "df_entropy" in globals() and isinstance(df_entropy, pd.DataFrame) and "Stimulus" in df_entropy.columns:
        mean_rsm_map = {
            str(r["Stimulus"]): float(r["Mean_Sim"])
            for _, r in df_entropy.iterrows()
            if "Mean_Sim" in r and pd.notna(r["Mean_Sim"])
        }
    effdim_map = {}
    if "df_effective_dim" in globals() and isinstance(df_effective_dim, pd.DataFrame):
        if {"Class_Name", "Effective_Dim_PR"}.issubset(set(df_effective_dim.columns)):
            effdim_map = {
                str(r["Class_Name"]): float(r["Effective_Dim_PR"])
                for _, r in df_effective_dim.iterrows()
                if pd.notna(r["Effective_Dim_PR"])
            }
    participants_name_map = {}
    for cls, val in participants.items():
        cond_name = label_names.get(int(cls), str(cls))
        participants_name_map[cond_name] = float(val)

    df_geom_model = df_geometry.copy()
    df_geom_model["Mean_RSM_Sim"] = df_geom_model["Condition"].map(mean_rsm_map)
    df_geom_model["Participants_Ratio"] = df_geom_model["Condition"].map(participants_name_map)
    df_geom_model["Effective_Dim_PR"] = df_geom_model["Condition"].map(effdim_map)

    def _fit_ols_single_mouse(df, y, x, model_name):
        sub = df[[x, y]].dropna()
        if len(sub) < 3:
            return {
                "model_name": model_name,
                "formula": f"{y} ~ {x}",
                "n": int(len(sub)),
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "p_value": np.nan,
                "aic_like": np.nan,
                "bic_like": np.nan,
                "note": "N too small for stable fit",
            }
        lr = stats.linregress(sub[x].values, sub[y].values)
        yhat = lr.intercept + lr.slope * sub[x].values
        resid = sub[y].values - yhat
        n = len(sub)
        k = 2
        rss = float(np.sum(resid ** 2))
        sigma2 = max(rss / max(1, n), GEOM_EPS)
        loglik = float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1))
        aic = float(2 * k - 2 * loglik)
        bic = float(k * np.log(n) - 2 * loglik)
        return {
            "model_name": model_name,
            "formula": f"{y} ~ {x}",
            "n": int(n),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r_value": float(lr.rvalue),
            "p_value": float(lr.pvalue),
            "aic_like": aic,
            "bic_like": bic,
            "note": "Single-mouse OLS proxy; group-level mixed model in integration.py",
        }

    model_rows = []
    model_rows.append(_fit_ols_single_mouse(df_geom_model, "Mean_RSM_Sim", "angle_deg", "M1: MeanRSM~angle"))
    model_rows.append(_fit_ols_single_mouse(df_geom_model, "Mean_RSM_Sim", "orth_parallel_ratio", "M2: MeanRSM~ratio"))
    model_rows.append(_fit_ols_single_mouse(df_geom_model, "angle_deg", "Participants_Ratio", "A1: angle~participants"))
    model_rows.append(_fit_ols_single_mouse(df_geom_model, "orth_parallel_ratio", "Participants_Ratio", "A2: ratio~participants"))
    model_rows.append(_fit_ols_single_mouse(df_geom_model, "Mean_RSM_Sim", "Effective_Dim_PR", "D1: MeanRSM~effectiveDim"))
    df_geometry_model_compare = pd.DataFrame(model_rows)
    geom_model_compare_csv = os.path.join(data_out_dir, "geometry_rsm_model_compare.csv")
    df_geometry_model_compare.to_csv(geom_model_compare_csv, index=False)
    print(f"[*] Geometry model-compare table saved: {geom_model_compare_csv}")

    # Dedicated markdown summaries (single-mouse proxy)
    geom_rsm_md = os.path.join(data_out_dir, "geometry_rsm_lmm_summary.md")
    with open(geom_rsm_md, "w", encoding="utf-8") as f:
        f.write("# Geometry vs Mean RSM (Single-mouse proxy)\n\n")
        f.write(df_geometry_model_compare[df_geometry_model_compare["model_name"].str.startswith("M")].to_markdown(index=False) + "\n")
    geom_alloc_md = os.path.join(data_out_dir, "geometry_allocation_lmm_summary.md")
    with open(geom_alloc_md, "w", encoding="utf-8") as f:
        f.write("# Geometry vs Participants Ratio (Single-mouse proxy)\n\n")
        f.write(df_geometry_model_compare[df_geometry_model_compare["model_name"].str.startswith("A")].to_markdown(index=False) + "\n")
    geom_vs_dim_csv = os.path.join(data_out_dir, "geometry_vs_dimensionality_model_compare.csv")
    df_geometry_model_compare[df_geometry_model_compare["model_name"].str.startswith(("M", "D"))].to_csv(geom_vs_dim_csv, index=False)
    print(f"[*] Geometry markdown summaries saved: {geom_rsm_md}, {geom_alloc_md}")
    print(f"[*] Geometry-vs-dimensionality model table saved: {geom_vs_dim_csv}")

    # -------- Geometry figures --------
    from matplotlib.patches import Ellipse

    def _plot_cov_ellipse(ax, xy, color):
        if xy.shape[0] < 3:
            return
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        width, height = 2.0 * np.sqrt(np.maximum(vals[:2], 1e-12))
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = Ellipse(
            xy=(np.mean(xy[:, 0]), np.mean(xy[:, 1])),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
            lw=1.8,
        )
        ax.add_patch(ell)

    # G1: per-condition PC scatter with mean-axis and PC1 arrows
    fig, axes = plt.subplots(1, len(ordered_cls), figsize=(4.8 * len(ordered_cls), 4.2), dpi=180, sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, cls in zip(axes, ordered_cls):
        cond_name = label_names.get(int(cls), str(cls))
        X_cls = np.asarray(X_resp[y_resp == cls], dtype=float)
        mu = np.mean(X_cls, axis=0)
        Y = X_cls - mu
        try:
            _, _, vt = np.linalg.svd(Y, full_matrices=False)
            basis = vt[:2].T  # (N,2)
            z = Y @ basis
            mu_proj = mu @ basis
        except Exception:
            z = np.zeros((X_cls.shape[0], 2))
            mu_proj = np.zeros(2)
        ax.scatter(z[:, 0], z[:, 1], s=22, alpha=0.45, color=class_colors.get(int(cls), "#4c4c4c"), edgecolor="none")
        _plot_cov_ellipse(ax, z, class_colors.get(int(cls), "#4c4c4c"))
        # PC1 axis in local PC coordinates is x-axis
        scale = max(np.nanstd(z[:, 0]), np.nanstd(z[:, 1]), 1e-3)
        ax.arrow(0, 0, 1.0 * scale, 0, color="#111111", width=0.0, head_width=0.06 * scale, length_includes_head=True)
        ax.arrow(0, 0, mu_proj[0], mu_proj[1], color="#8C4A3E", width=0.0, head_width=0.06 * scale, length_includes_head=True)
        angle_val = float(df_geometry.loc[df_geometry["Condition"] == cond_name, "angle_deg"].iloc[0]) if (df_geometry["Condition"] == cond_name).any() else np.nan
        ax.set_title(f"{cond_name}\nangle={angle_val:.2f} deg")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.axvline(0, color="#cccccc", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    g1_png = os.path.join(fig_out_dir, "geometry_example_mouse_pc_scatter.png")
    g1_pdf = os.path.join(fig_out_dir, "geometry_example_mouse_pc_scatter.pdf")
    fig.savefig(g1_png, dpi=300, bbox_inches="tight")
    fig.savefig(g1_pdf, bbox_inches="tight")
    plt.show()

    # G2/G3: condition comparisons
    plot_order = [label_names.get(int(c), str(c)) for c in ordered_cls]

    def _plot_single_metric_condition(df, metric, ylabel, stem):
        figm, axm = plt.subplots(figsize=(5.2, 4.4), dpi=180)
        sub = df.copy()
        sub["Condition"] = pd.Categorical(sub["Condition"], categories=plot_order, ordered=True)
        sns.barplot(data=sub, x="Condition", y=metric, order=plot_order, palette=[class_colors.get(c, "#4c4c4c") for c in ordered_cls], ax=axm, alpha=0.85)
        sns.stripplot(data=sub, x="Condition", y=metric, order=plot_order, color="#222222", size=5, jitter=False, ax=axm)
        axm.set_xlabel("")
        axm.set_ylabel(ylabel)
        axm.spines["top"].set_visible(False)
        axm.spines["right"].set_visible(False)
        axm.grid(axis="y", linestyle="--", alpha=0.25)
        figm.tight_layout()
        out_png = os.path.join(fig_out_dir, f"{stem}.png")
        out_pdf = os.path.join(fig_out_dir, f"{stem}.pdf")
        figm.savefig(out_png, dpi=300, bbox_inches="tight")
        figm.savefig(out_pdf, bbox_inches="tight")
        plt.show()
        return out_png, out_pdf

    _plot_single_metric_condition(df_geometry, "angle_deg", "Angle (deg)", "geometry_angle_condition")
    _plot_single_metric_condition(df_geometry, "orth_parallel_ratio", "Orth/Parallel variance ratio", "geometry_orth_parallel_condition")

    # G4: geometry vs MeanRSM (single-mouse scatter)
    for x_metric, stem in [("angle_deg", "geometry_angle_vs_rsm"), ("orth_parallel_ratio", "geometry_ratio_vs_rsm")]:
        figm, axm = plt.subplots(figsize=(5.0, 4.4), dpi=180)
        sub = df_geom_model[[x_metric, "Mean_RSM_Sim", "Condition"]].dropna().copy()
        sns.regplot(data=sub, x=x_metric, y="Mean_RSM_Sim", scatter=False, color="#404040", ax=axm, line_kws={"lw": 2})
        sns.scatterplot(data=sub, x=x_metric, y="Mean_RSM_Sim", hue="Condition", hue_order=plot_order, palette={label_names.get(int(k), str(k)): v for k, v in class_colors.items()}, s=85, ax=axm)
        axm.set_xlabel(x_metric)
        axm.set_ylabel("Mean RSM similarity")
        axm.spines["top"].set_visible(False)
        axm.spines["right"].set_visible(False)
        axm.grid(axis="y", linestyle="--", alpha=0.25)
        axm.legend(frameon=False, title="")
        figm.tight_layout()
        out_png = os.path.join(fig_out_dir, f"{stem}.png")
        out_pdf = os.path.join(fig_out_dir, f"{stem}.pdf")
        figm.savefig(out_png, dpi=300, bbox_inches="tight")
        figm.savefig(out_pdf, bbox_inches="tight")
        plt.show()

    # %% [markdown]
    # <a id="sec-9"></a>
    # ## 9. Strong vs Weak Connectivity
    # 
    # Quantify weak/strong correlation tails and their gap by class.
    # 

    # %%

    def upper_triangle_values(matrix):
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        return matrix[mask]

    corr_rows = []
    decile_rows = []
    for cls in sorted(nx_result):
        corr_matrix = np.asarray(nx_result[cls]["corr_matrix"], dtype=float)
        offdiag = upper_triangle_values(corr_matrix)
        if offdiag.size == 0:
            continue

        # Split sorted correlation values into 10 equal-count bins (deciles)
        sorted_corr = np.sort(offdiag)
        decile_bins = np.array_split(sorted_corr, 10)

        decile_means = []
        for decile_idx, decile_vals in enumerate(decile_bins, start=1):
            q_low = (decile_idx - 1) / 10
            q_high = decile_idx / 10
            decile_mean = float(np.mean(decile_vals)) if decile_vals.size > 0 else np.nan

            decile_means.append(decile_mean)
            decile_rows.append({
                "Class_ID": cls,
                "Class_Name": label_names.get(cls, str(cls)),
                "Decile_Index": decile_idx,
                "Decile_Label": f"{int(q_low*100)}-{int(q_high*100)}%",
                "Lower_Quantile": q_low,
                "Upper_Quantile": q_high,
                "Pair_Count": int(decile_vals.size),
                "Mean_Correlation": decile_mean,
                "Min_Correlation": float(np.min(decile_vals)) if decile_vals.size > 0 else np.nan,
                "Max_Correlation": float(np.max(decile_vals)) if decile_vals.size > 0 else np.nan,
            })

        weak_mean = decile_means[0]
        strong_mean = decile_means[-1]
        corr_rows.append({
            "Class_ID": cls,
            "Class_Name": label_names.get(cls, str(cls)),
            "Mean_Correlation": float(np.mean(offdiag)),
            "Weak_Correlation_Mean": weak_mean,
            "Strong_Correlation_Mean": strong_mean,
            "Strong_Weak_Gap": strong_mean - weak_mean,
            "Pair_Count": int(offdiag.size),
        })

    df_corr_strength = pd.DataFrame(corr_rows)
    df_corr_deciles = pd.DataFrame(decile_rows)

    # Save decile-level stats for downstream group analysis
    decile_csv_path = os.path.join(data_out_dir, "correlation_deciles.csv")
    df_corr_deciles.to_csv(decile_csv_path, index=False)
    print(f"[*] Decile-level correlation stats saved to: {decile_csv_path}")

    # Figure 1: summary view
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    axes[0].bar(
        df_corr_strength["Class_Name"],
        df_corr_strength["Mean_Correlation"],
        color=[class_colors[c] for c in df_corr_strength["Class_ID"]],
        edgecolor="#333333",
    )
    axes[0].set_ylabel("Mean correlation")
    axes[0].set_title("Pairwise Correlation Strength")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    x = np.arange(len(df_corr_strength))
    w = 0.34
    axes[1].bar(x - w / 2, df_corr_strength["Weak_Correlation_Mean"], width=w, color="#B0BEC5", edgecolor="#333333", label="Weakest 10%")
    axes[1].bar(x + w / 2, df_corr_strength["Strong_Correlation_Mean"], width=w, color="#455A64", edgecolor="#333333", label="Strongest 10%")
    axes[1].plot(x, df_corr_strength["Strong_Weak_Gap"], color="#D55E00", lw=2.2, marker="o", label="Gap")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_corr_strength["Class_Name"])
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Weakest vs Strongest Decile")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_out_dir, "pairwise_correlation.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Figure 2: decile-wise line comparison (each class is one line)
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    for cls in sorted(df_corr_deciles["Class_ID"].unique()):
        sub = df_corr_deciles[df_corr_deciles["Class_ID"] == cls].sort_values("Decile_Index")
        ax2.plot(
            sub["Decile_Index"],
            sub["Mean_Correlation"],
            marker="o",
            lw=2,
            color=class_colors.get(cls, None),
            label=label_names.get(cls, str(cls)),
        )

    ax2.set_xticks(np.arange(1, 11))
    ax2.set_xticklabels([f"{(i-1)*10}-{i*10}%" for i in range(1, 11)], rotation=30, ha="right")
    ax2.set_xlabel("Connection-strength decile (low to high)")
    ax2.set_ylabel("Mean correlation")
    ax2.set_title("Decile-wise Correlation Comparison")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, ncol=2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_out_dir, "pairwise_correlation_deciles_line.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Figure 3: condition-wise stratified correlation change relative to the weakest decile
    df_corr_deciles_delta = df_corr_deciles.sort_values(["Class_ID", "Decile_Index"]).copy()
    df_corr_deciles_delta["Delta_vs_Decile1"] = df_corr_deciles_delta.groupby("Class_ID")["Mean_Correlation"].transform(lambda s: s - s.iloc[0])

    fig3, ax3 = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    for cls in sorted(df_corr_deciles_delta["Class_ID"].unique()):
        sub = df_corr_deciles_delta[df_corr_deciles_delta["Class_ID"] == cls].sort_values("Decile_Index")
        ax3.plot(
            sub["Decile_Index"],
            sub["Delta_vs_Decile1"],
            marker="o",
            lw=2,
            color=class_colors.get(cls, None),
            label=label_names.get(cls, str(cls)),
        )

    ax3.set_xticks(np.arange(1, 11))
    ax3.set_xticklabels([f"{(i-1)*10}-{i*10}%" for i in range(1, 11)], rotation=30, ha="right")
    ax3.set_xlabel("Connection-strength decile (low to high)")
    ax3.set_ylabel("Delta correlation (vs decile 1)")
    ax3.set_title("Decile-wise Correlation Change by Condition")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, ncol=2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_out_dir, "pairwise_correlation_deciles_delta.png"), dpi=300, bbox_inches="tight")
    plt.show()



    # %% [markdown]
    # ## 9.1 Signal vs Noise Correlation Coupling
    # 
    # Following standard definitions in Cohen & Kohn (2011, Nat Neurosci) and Averbeck et al. (2006, Nat Rev Neurosci), this section computes condition-wise signal/noise correlations and their coupling.
    # 

    # %%

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Literature-aligned definitions:
    # 1) Signal correlation: correlation between mean response features of neuron pairs.
    # 2) Noise correlation: correlation of trial-by-trial residual fluctuations under the same condition.
    # References: Cohen & Kohn (2011); Averbeck et al. (2006).


    def upper_triangle_values_local(matrix):
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        return matrix[mask]


    def robust_corrcoef(X, rowvar=False):
        """Safe Pearson correlation matrix with NaN handling."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if (rowvar and X.shape[1] < 2) or ((not rowvar) and X.shape[0] < 2):
            n = X.shape[0] if rowvar else X.shape[1]
            out = np.full((n, n), np.nan)
            np.fill_diagonal(out, 1.0)
            return out

        C = np.corrcoef(X, rowvar=rowvar)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
        return C


    # Use spike segments: shape (n_trials, n_neurons, n_time)
    X_trials = np.asarray(segments_spi, dtype=float)
    y_trials = np.asarray(labels_spi)
    classes = sorted(np.unique(y_trials).astype(int).tolist())

    # Noise-correlation window (same response window used elsewhere)
    noise_window = slice(10, 13)

    signal_corr_by_class = {}
    noise_corr_by_class = {}
    pair_rows = []
    summary_rows = []

    for cls in classes:
        trials_c = X_trials[y_trials == cls]
        if trials_c.shape[0] < 3:
            print(f"[!] Skip class {cls}: too few trials for noise correlation")
            continue

        # Condition-specific signal features:
        # each neuron's mean temporal response curve under this condition
        mean_time_profile = np.nanmean(trials_c, axis=0)  # (n_neurons, n_time)
        sig_corr = robust_corrcoef(mean_time_profile, rowvar=True)

        # Condition-specific noise features:
        # trial-wise scalar response residuals (same stimulus context)
        trial_resp = np.nanmean(trials_c[:, :, noise_window], axis=2)  # (n_trials, n_neurons)
        residual = trial_resp - np.nanmean(trial_resp, axis=0, keepdims=True)
        noi_corr = robust_corrcoef(residual, rowvar=False)

        signal_corr_by_class[cls] = sig_corr
        noise_corr_by_class[cls] = noi_corr

        sig_vals = upper_triangle_values_local(sig_corr)
        noi_vals = upper_triangle_values_local(noi_corr)

        # Pairwise table for coupling scatter
        n_pairs = min(sig_vals.size, noi_vals.size)
        class_name = label_names.get(cls, str(cls)) if 'label_names' in globals() else str(cls)
        for k in range(n_pairs):
            pair_rows.append({
                'Class_ID': cls,
                'Class_Name': class_name,
                'Signal_Corr': float(sig_vals[k]),
                'Noise_Corr': float(noi_vals[k]),
                'Abs_Signal_Corr': float(abs(sig_vals[k])),
                'Abs_Noise_Corr': float(abs(noi_vals[k])),
            })

        summary_rows.append({
            'Class_ID': cls,
            'Class_Name': class_name,
            'Mean_Signal_Corr': float(np.mean(sig_vals)),
            'Mean_Noise_Corr': float(np.mean(noi_vals)),
            'Mean_Abs_Signal_Corr': float(np.mean(np.abs(sig_vals))),
            'Mean_Abs_Noise_Corr': float(np.mean(np.abs(noi_vals))),
            'Signal_Noise_Coupling_r': float(np.corrcoef(sig_vals, noi_vals)[0, 1]) if sig_vals.size > 1 else np.nan,
        })


    df_sig_noise_pairs = pd.DataFrame(pair_rows)
    df_sig_noise_strength = pd.DataFrame(summary_rows)

    pair_csv = os.path.join(data_out_dir, 'sig_noise_pair_values_by_condition.csv')
    summary_csv = os.path.join(data_out_dir, 'sig_noise_strength_summary_by_condition.csv')
    df_sig_noise_pairs.to_csv(pair_csv, index=False)
    df_sig_noise_strength.to_csv(summary_csv, index=False)
    print(f"[*] Signal-noise pair table saved to: {pair_csv}")
    print(f"[*] Signal-noise summary saved to: {summary_csv}")

    # 1) Condition-wise coupling scatter: signal vs noise
    n_cls = len(df_sig_noise_pairs['Class_ID'].unique())
    fig, axes = plt.subplots(1, n_cls, figsize=(5.2 * n_cls, 4.6), dpi=180, sharex=True, sharey=True)
    if n_cls == 1:
        axes = [axes]

    for ax, cls in zip(axes, sorted(df_sig_noise_pairs['Class_ID'].unique())):
        sub = df_sig_noise_pairs[df_sig_noise_pairs['Class_ID'] == cls]
        cname = sub['Class_Name'].iloc[0]
        sns.regplot(
            data=sub,
            x='Signal_Corr',
            y='Noise_Corr',
            scatter_kws={'s': 10, 'alpha': 0.25, 'color': class_colors.get(cls, '#1F77B4')},
            line_kws={'color': '#222222', 'lw': 2},
            ax=ax,
        )
        r_val = np.corrcoef(sub['Signal_Corr'], sub['Noise_Corr'])[0, 1] if len(sub) > 1 else np.nan
        ax.set_title(f"{cname}\nr={r_val:.3f}")
        ax.set_xlabel('Signal correlation')
        ax.set_ylabel('Noise correlation')
        ax.grid(alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Signal-Noise Correlation Coupling by Condition', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_out_dir, 'signal_noise_coupling_scatter_by_condition.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2) Compare signal/noise strengths across conditions
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    plot_df_raw = df_sig_noise_strength.melt(
        id_vars=['Class_ID', 'Class_Name'],
        value_vars=['Mean_Signal_Corr', 'Mean_Noise_Corr'],
        var_name='Metric',
        value_name='Value',
    )
    plot_df_raw['Metric'] = plot_df_raw['Metric'].map({'Mean_Signal_Corr': 'Signal', 'Mean_Noise_Corr': 'Noise'})

    sns.barplot(data=plot_df_raw, x='Class_Name', y='Value', hue='Metric', ax=axes2[0])
    axes2[0].set_title('Condition-wise Mean Signal/Noise Correlation')
    axes2[0].set_xlabel('Condition')
    axes2[0].set_ylabel('Mean correlation')
    axes2[0].grid(axis='y', linestyle='--', alpha=0.3)
    axes2[0].spines['top'].set_visible(False)
    axes2[0].spines['right'].set_visible(False)

    plot_df_abs = df_sig_noise_strength.melt(
        id_vars=['Class_ID', 'Class_Name'],
        value_vars=['Mean_Abs_Signal_Corr', 'Mean_Abs_Noise_Corr'],
        var_name='Metric',
        value_name='Value',
    )
    plot_df_abs['Metric'] = plot_df_abs['Metric'].map({'Mean_Abs_Signal_Corr': 'Signal |r|', 'Mean_Abs_Noise_Corr': 'Noise |r|'})

    sns.barplot(data=plot_df_abs, x='Class_Name', y='Value', hue='Metric', ax=axes2[1])
    axes2[1].set_title('Condition-wise Signal/Noise Strength (|r|)')
    axes2[1].set_xlabel('Condition')
    axes2[1].set_ylabel('Mean |correlation|')
    axes2[1].grid(axis='y', linestyle='--', alpha=0.3)
    axes2[1].spines['top'].set_visible(False)
    axes2[1].spines['right'].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_out_dir, 'signal_noise_strength_by_condition.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 3) Couple condition-wise noise-correlation change with existing decile stratification
    noise_decile_rows = []
    for cls, Cn in noise_corr_by_class.items():
        vals = np.sort(upper_triangle_values_local(Cn))
        bins = np.array_split(vals, 10)
        for i, b in enumerate(bins, start=1):
            noise_decile_rows.append({
                'Class_ID': cls,
                'Class_Name': label_names.get(cls, str(cls)) if 'label_names' in globals() else str(cls),
                'Decile_Index': i,
                'Noise_Mean_Corr': float(np.mean(b)) if b.size > 0 else np.nan,
            })

    df_noise_deciles = pd.DataFrame(noise_decile_rows)

    # Merge with existing decile table from section 9
    df_decile_coupling = pd.merge(
        df_corr_deciles[['Class_ID', 'Class_Name', 'Decile_Index', 'Mean_Correlation']],
        df_noise_deciles,
        on=['Class_ID', 'Class_Name', 'Decile_Index'],
        how='inner',
    )

    # Add delta relative to decile1
    df_decile_coupling = df_decile_coupling.sort_values(['Class_ID', 'Decile_Index']).copy()
    df_decile_coupling['Corr_Delta_vs_D1'] = df_decile_coupling.groupby('Class_ID')['Mean_Correlation'].transform(lambda s: s - s.iloc[0])
    df_decile_coupling['Noise_Delta_vs_D1'] = df_decile_coupling.groupby('Class_ID')['Noise_Mean_Corr'].transform(lambda s: s - s.iloc[0])

    coupling_csv = os.path.join(data_out_dir, 'noise_corr_decile_coupling.csv')
    df_decile_coupling.to_csv(coupling_csv, index=False)
    print(f"[*] Noise-decile coupling table saved to: {coupling_csv}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)

    for cls in sorted(df_decile_coupling['Class_ID'].unique()):
        sub = df_decile_coupling[df_decile_coupling['Class_ID'] == cls].sort_values('Decile_Index')
        axes3[0].plot(
            sub['Decile_Index'],
            sub['Noise_Mean_Corr'],
            marker='o',
            lw=2,
            color=class_colors.get(cls, None),
            label=label_names.get(cls, str(cls)) if 'label_names' in globals() else str(cls),
        )

    axes3[0].set_xticks(np.arange(1, 11))
    axes3[0].set_title('Noise Correlation Decile Curve by Condition')
    axes3[0].set_xlabel('Decile')
    axes3[0].set_ylabel('Noise mean correlation')
    axes3[0].grid(axis='y', linestyle='--', alpha=0.3)
    axes3[0].legend(frameon=False, ncol=2)
    axes3[0].spines['top'].set_visible(False)
    axes3[0].spines['right'].set_visible(False)

    for cls in sorted(df_decile_coupling['Class_ID'].unique()):
        sub = df_decile_coupling[df_decile_coupling['Class_ID'] == cls].sort_values('Decile_Index')
        axes3[1].plot(
            sub['Corr_Delta_vs_D1'],
            sub['Noise_Delta_vs_D1'],
            marker='o',
            lw=2,
            color=class_colors.get(cls, None),
            label=label_names.get(cls, str(cls)) if 'label_names' in globals() else str(cls),
        )

    axes3[1].axhline(0, color='#999999', lw=1, ls='--')
    axes3[1].axvline(0, color='#999999', lw=1, ls='--')
    axes3[1].set_title('Coupling: Corr-decile change vs Noise-decile change')
    axes3[1].set_xlabel('Total correlation delta vs decile1')
    axes3[1].set_ylabel('Noise correlation delta vs decile1')
    axes3[1].grid(axis='both', linestyle='--', alpha=0.3)
    axes3[1].legend(frameon=False, ncol=1)
    axes3[1].spines['top'].set_visible(False)
    axes3[1].spines['right'].set_visible(False)

    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_out_dir, 'noise_corr_deciles_and_coupling.png'), dpi=300, bbox_inches='tight')
    plt.show()




    # %% [markdown]
    # <a id="sec-10"></a>
    # ## 10. Export and Report Generation
    # 
    # Export structured metrics to JSON and generate a Markdown report.
    # 

    # %%
    import json
    import os
    import pandas as pd
    from datetime import datetime

    def export_mouse_results(mouse_id, df_entropy, df_corr_strength, df_corr_deciles, participants_dict, save_dir, fig_dir):
        """
        Export integrated metrics for one mouse to JSON/CSV and a Markdown report.
        """
        os.makedirs(save_dir, exist_ok=True)

        entropy_records = df_entropy.to_dict(orient='records')
        corr_records = df_corr_strength.to_dict(orient='records')
        corr_decile_records = df_corr_deciles.to_dict(orient='records')

        mouse_data = {
            'mouse_id': mouse_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'entropy_metrics': entropy_records,
            'network_correlation': corr_records,
            'network_correlation_deciles': corr_decile_records,
            'rr_participants_ratio': participants_dict,
        }

        json_path = os.path.join(save_dir, f'{mouse_id}_statistics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mouse_data, f, indent=4)

        decile_csv_path = os.path.join(save_dir, f'{mouse_id}_correlation_deciles.csv')
        df_corr_deciles.to_csv(decile_csv_path, index=False)

        print(f'[*] Structured statistics saved to: {json_path}')
        print(f'[*] Decile-level correlation data saved to: {decile_csv_path}')

        md_path = os.path.join(save_dir, f'{mouse_id}_analysis_report.md')

        def dicts_to_md_table(dict_list):
            if not dict_list:
                return ''
            headers = list(dict_list[0].keys())
            header_row = '| ' + ' | '.join(headers) + ' |'
            sep_row = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
            rows = []
            for d in dict_list:
                formatted_vals = [f'{v:.4f}' if isinstance(v, float) else str(v) for v in d.values()]
                rows.append('| ' + ' | '.join(formatted_vals) + ' |')
            return '\n'.join([header_row, sep_row] + rows)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f'# Mouse Neural Activity Report - {mouse_id}\n\n')
            f.write(f'**Generated At**: {mouse_data["timestamp"]}\n\n')

            f.write('## 1. RSM and Shannon Entropy\n\n')
            f.write('Metrics for representation stability and variability across stimulus conditions.\n\n')
            f.write(dicts_to_md_table(entropy_records) + '\n\n')

            f.write('## 2. Pairwise Network Correlation Summary\n\n')
            f.write('Difference between lowest and highest correlation tails.\n\n')
            f.write(dicts_to_md_table(corr_records) + '\n\n')

            f.write('## 2.1 Decile-wise Correlation Strength (Every 10%)\n\n')
            f.write('Connectivity values are sorted by raw correlation and split into 10 equal bins.\n\n')
            f.write(dicts_to_md_table(corr_decile_records[:30]) + '\n\n')

            f.write('## 3. RR Participant Ratio\n\n')
            f.write('Response ratio of class-specific RR neurons relative to other RR neurons.\n\n')
            participants_rows = [{'Condition': k, 'Response_Ratio': v} for k, v in participants_dict.items()]
            f.write(dicts_to_md_table(participants_rows) + '\n\n')

            f.write('## 4. Figure Index\n\n')
            f.write('- Preference-sorted heatmap: `./figures/neural_patterns_preference_sorted.png`\n')
            f.write('- Hierarchical clustermap: `./figures/neural_patterns_clustermap.png`\n')
            f.write('- RSM similarity distribution: `./figures/similarity_distribution.png`\n')
            f.write('- Pairwise correlation summary: `./figures/pairwise_correlation.png`\n')
            f.write('- Decile-wise line chart: `./figures/pairwise_correlation_deciles_line.png`\n')

        print(f'[*] Markdown report generated: {md_path}')

    # --- Execute export after all analyses ---
    current_mouse_id = os.path.basename(os.path.normpath(data_path))

    export_mouse_results(
        mouse_id=current_mouse_id,
        df_entropy=df_entropy,
        df_corr_strength=df_corr_strength,
        df_corr_deciles=df_corr_deciles,
        participants_dict=participants,
        save_dir=data_out_dir,
        fig_dir=fig_out_dir,
    )



    # %% [markdown]
    # <a id="sec-11"></a>
    # ## 11. Decoder Robustness (Task 1-2)
    # 
    # Generate fixed-window decoder confusion matrix + shuffled baseline (Task1), and Top 10% neuron ablation with random-drop control (Task2).
    # 

    # %%
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC

    # ===== lightweight Task1-2 based on existing analysis =====
    n_splits = 3
    shuffle_repeats = 40
    random_drop_repeats = 20
    ablation_ratio = 0.10
    random_state = 42

    y_all = np.asarray(labels_flo)
    segments_all = np.asarray(segments_flo, dtype=float)

    # keep stimulus classes only (exclude label 0 if present)
    mask = y_all != 0
    y = y_all[mask].astype(int)
    segments_use = segments_all[mask]
    n_time = segments_use.shape[2]

    # choose decoder window from existing timepoint-accuracy result when available
    if 'accuracies' in globals() and len(np.asarray(accuracies).ravel()) == n_time:
        best_t = int(np.nanargmax(np.asarray(accuracies).ravel()))
    else:
        best_t = min(10, n_time - 1)
    win_start = max(0, best_t - 1)
    win_end = min(n_time, best_t + 2)
    if win_end <= win_start:
        win_start = max(0, best_t)
        win_end = min(n_time, win_start + 1)

    X = np.nanmean(segments_use[:, :, win_start:win_end], axis=2)

    def build_model():
        return Pipeline([
            ('scaler', RobustScaler()),
            ('svc', SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')),
        ])

    def cv_accuracy_with_pred(X_mat, y_vec, n_splits=3, random_state=42):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_pred = np.empty_like(y_vec)
        fold_acc = []
        for tr_idx, te_idx in cv.split(X_mat, y_vec):
            model = build_model()
            model.fit(X_mat[tr_idx], y_vec[tr_idx])
            pred = model.predict(X_mat[te_idx])
            y_pred[te_idx] = pred
            fold_acc.append(float((pred == y_vec[te_idx]).mean()))
        fold_acc = np.asarray(fold_acc, dtype=float)
        return float(fold_acc.mean()), float(fold_acc.std(ddof=1)), y_pred

    # ---------- Task1: decoder summary + confusion matrix ----------
    full_acc, full_std, y_pred = cv_accuracy_with_pred(X, y, n_splits=n_splits, random_state=random_state)

    rng = np.random.default_rng(random_state)
    shuffle_acc = []
    for rep in range(shuffle_repeats):
        y_shuf = rng.permutation(y)
        acc_shuf, _, _ = cv_accuracy_with_pred(
            X, y_shuf, n_splits=n_splits, random_state=random_state + 100 + rep
        )
        shuffle_acc.append(acc_shuf)
    shuffle_acc = np.asarray(shuffle_acc, dtype=float)

    classes = np.sort(np.unique(y))
    class_names = [label_names.get(int(c), str(c)) if 'label_names' in globals() else str(c) for c in classes]
    cm_norm = confusion_matrix(y, y_pred, labels=classes, normalize='true')
    cm_raw = confusion_matrix(y, y_pred, labels=classes, normalize=None)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=180)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='.2f')
    ax.set_title(
        f'Task1 Decoder Confusion Matrix (window {win_start}:{win_end})\\n'
        f'Acc={full_acc:.3f}±{full_std:.3f} | Shuffle={shuffle_acc.mean():.3f}±{shuffle_acc.std(ddof=1):.3f}'
    )
    fig.tight_layout()
    task1_fig = os.path.join(fig_out_dir, 'decoder_confusion_matrix.png')
    fig.savefig(task1_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    task1_summary = pd.DataFrame([
        {
            'window_start': int(win_start),
            'window_end': int(win_end),
            'n_trials': int(X.shape[0]),
            'n_neurons': int(X.shape[1]),
            'n_splits': int(n_splits),
            'accuracy_mean': float(full_acc),
            'accuracy_std': float(full_std),
            'shuffle_accuracy_mean': float(shuffle_acc.mean()),
            'shuffle_accuracy_std': float(shuffle_acc.std(ddof=1)),
            'accuracy_minus_shuffle': float(full_acc - shuffle_acc.mean()),
        }
    ])
    for idx, cname in enumerate(class_names):
        task1_summary[f'recall_{cname}'] = float(cm_norm[idx, idx])
    task1_csv = os.path.join(data_out_dir, 'decoder_summary.csv')
    task1_summary.to_csv(task1_csv, index=False)
    pd.DataFrame(cm_raw, index=class_names, columns=class_names).to_csv(
        os.path.join(data_out_dir, 'decoder_confusion_matrix.csv')
    )

    # ---------- Task2: Top10% ablation + random-drop control ----------
    n_neurons = X.shape[1]
    n_remove = max(1, int(np.ceil(n_neurons * ablation_ratio)))
    neuron_mean_resp = np.nanmean(X, axis=0)
    top_idx = np.argsort(neuron_mean_resp)[::-1][:n_remove]

    keep_mask = np.ones(n_neurons, dtype=bool)
    keep_mask[top_idx] = False
    X_ablate = X[:, keep_mask]
    ablate_acc, ablate_std, _ = cv_accuracy_with_pred(
        X_ablate, y, n_splits=n_splits, random_state=random_state + 500
    )

    rand_acc = []
    for rep in range(random_drop_repeats):
        drop_idx = rng.choice(n_neurons, size=n_remove, replace=False)
        keep_rand = np.ones(n_neurons, dtype=bool)
        keep_rand[drop_idx] = False
        acc_rand, _, _ = cv_accuracy_with_pred(
            X[:, keep_rand], y, n_splits=n_splits, random_state=random_state + 800 + rep
        )
        rand_acc.append(acc_rand)
    rand_acc = np.asarray(rand_acc, dtype=float)

    task2_summary = pd.DataFrame([
        {
            'window_start': int(win_start),
            'window_end': int(win_end),
            'n_trials': int(X.shape[0]),
            'n_neurons_total': int(n_neurons),
            'n_neurons_removed': int(n_remove),
            'removed_ratio': float(n_remove / n_neurons),
            'full_accuracy_mean': float(full_acc),
            'full_accuracy_std': float(full_std),
            'top10_ablation_accuracy_mean': float(ablate_acc),
            'top10_ablation_accuracy_std': float(ablate_std),
            'random_drop_mean_accuracy': float(rand_acc.mean()),
            'random_drop_std_accuracy': float(rand_acc.std(ddof=1)),
            'delta_full_minus_top10': float(full_acc - ablate_acc),
            'delta_top10_minus_random_mean': float(ablate_acc - rand_acc.mean()),
            'ablation_rank_in_random': float((rand_acc <= ablate_acc).mean()),
        }
    ])
    task2_csv = os.path.join(data_out_dir, 'decoder_ablation_summary.csv')
    task2_summary.to_csv(task2_csv, index=False)
    pd.DataFrame({'repeat_idx': np.arange(rand_acc.size), 'accuracy_mean': rand_acc}).to_csv(
        os.path.join(data_out_dir, 'decoder_random_drop_repeats.csv'),
        index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    labels_bar = ['Full model', 'Top10% ablation', 'Random drop\\n(mean±sd)']
    means = [full_acc, ablate_acc, float(rand_acc.mean())]
    errs = [full_std, ablate_std, float(rand_acc.std(ddof=1))]
    colors = ['#4C78A8', '#F58518', '#54A24B']
    x = np.arange(3)
    ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=4, edgecolor='#333333')
    jitter = rng.uniform(-0.14, 0.14, size=rand_acc.size)
    ax.scatter(np.full(rand_acc.size, x[2]) + jitter, rand_acc, s=18, alpha=0.55, color='#2E7D32')
    chance = 1.0 / classes.size
    ax.axhline(chance, color='#777777', linestyle='--', linewidth=1.2, label=f'Chance={chance:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Cross-validated accuracy')
    ax.set_title(
        f'Task2 Top10% Neuron Ablation (random repeats={random_drop_repeats})\\n'
        f'Δ(full-top10)={full_acc - ablate_acc:.3f} | Δ(top10-rand)={ablate_acc - rand_acc.mean():.3f}'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    task2_fig = os.path.join(fig_out_dir, 'decoder_ablation_top10.png')
    fig.savefig(task2_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    decoder_task12_outputs = {
        'decoder_summary_csv': task1_csv,
        'decoder_confusion_fig': task1_fig,
        'decoder_ablation_summary_csv': task2_csv,
        'decoder_ablation_fig': task2_fig,
    }
    print('[*] Task1 summary:', task1_csv)
    print('[*] Task1 confusion fig:', task1_fig)
    print('[*] Task2 summary:', task2_csv)
    print('[*] Task2 ablation fig:', task2_fig)
    decoder_task12_outputs


    # %% [markdown]
    # 
    # <a id="sec-12"></a>
    # ## 12. FC Decoder Chain (Tasks 3-6)
    # 
    # This section is organized as Task3 (FC decoder), Task4 (robust edge importance + projection), Task5 (decile/regime enrichment), and Task6 (decoder-important neurons linked to RR overlap/selectivity).
    # 

    # %% [markdown]
    # ### 12.1 Task3: FC Matrix Decoder
    # 

    # %%
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC, LinearSVC
    from sklearn.decomposition import TruncatedSVD
    from scipy.stats import hypergeom

    # ===== Task3 config (lightweight) =====
    fc_window = (10, 30)  # response window for FC construction
    fc_n_splits = 3
    fc_shuffle_repeats = 30
    fc_max_components = 40
    fc_random_state = 123
    fc_use_rr_union = True


    def _balanced_indices(y_vec, random_state=123):
        """Downsample to balanced class counts."""
        rng_local = np.random.default_rng(random_state)
        classes_local = np.sort(np.unique(y_vec))
        min_count = min(int((y_vec == c).sum()) for c in classes_local)
        keep = []
        for c in classes_local:
            idx_c = np.where(y_vec == c)[0]
            keep.extend(rng_local.choice(idx_c, size=min_count, replace=False).tolist())
        return np.asarray(sorted(keep), dtype=int)


    def _trial_fc_upper_triangle(trial_neuron_time):
        """trial_neuron_time: (n_neurons, n_time) -> upper-triangle FC vector."""
        C = np.corrcoef(trial_neuron_time)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
        tri = np.triu_indices(C.shape[0], k=1)
        return C[tri]


    def _fc_model(n_components, random_state=123):
        """SVD (dim reduction) + SVC classifier."""
        return Pipeline([
            ('svd', TruncatedSVD(n_components=n_components, random_state=random_state)),
            ('svc', SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')),
        ])


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


    # ---------- 1) Prepare balanced spike trials ----------
    segments_fc_all = np.asarray(segments_spi, dtype=float)
    labels_fc_all = np.asarray(labels_spi).astype(int)
    valid_mask = labels_fc_all != 0
    segments_fc_valid = segments_fc_all[valid_mask]
    labels_fc_valid = labels_fc_all[valid_mask]

    keep_idx = _balanced_indices(labels_fc_valid, random_state=fc_random_state)
    segments_fc = segments_fc_valid[keep_idx]
    y_fc = labels_fc_valid[keep_idx]

    # optional neuron subset: RR union (if available)
    if fc_use_rr_union and 'rr_union' in globals() and len(rr_union) > 1:
        rr_idx = np.asarray(sorted([i for i in rr_union if i < segments_fc.shape[1]]), dtype=int)
        if rr_idx.size >= 5:
            segments_fc = segments_fc[:, rr_idx, :]
            fc_neuron_mode = 'rr_union'
        else:
            rr_idx = np.arange(segments_fc.shape[1], dtype=int)
            fc_neuron_mode = 'all_neurons_fallback'
    else:
        rr_idx = np.arange(segments_fc.shape[1], dtype=int)
        fc_neuron_mode = 'all_neurons'

    n_time_fc = segments_fc.shape[2]
    fc_start = max(0, min(fc_window[0], n_time_fc - 1))
    fc_end = max(fc_start + 1, min(fc_window[1], n_time_fc))

    # ---------- 2) Build trial-level FC features ----------
    X_fc_list = []
    for t_idx in range(segments_fc.shape[0]):
        trial_nt = segments_fc[t_idx, :, fc_start:fc_end]
        X_fc_list.append(_trial_fc_upper_triangle(trial_nt))
    X_fc = np.vstack(X_fc_list)

    # Fisher-z transform to stabilize correlation feature distribution
    X_fc = np.clip(X_fc, -0.999999, 0.999999)
    X_fc = np.arctanh(X_fc)

    n_samples_fc, n_features_fc = X_fc.shape
    n_components_fc = int(min(fc_max_components, n_samples_fc - 1, n_features_fc - 1))
    if n_components_fc < 2:
        raise ValueError(f'n_components too small: {n_components_fc}. Check trial/neuron counts.')

    # ---------- 3) FC decoder + shuffled baseline ----------
    fc_acc, fc_std, y_fc_pred = _fc_cv_with_pred(
        X_fc,
        y_fc,
        n_components=n_components_fc,
        n_splits=fc_n_splits,
        random_state=fc_random_state,
    )

    rng_fc = np.random.default_rng(fc_random_state)
    fc_shuffle_acc = []
    for rep in range(fc_shuffle_repeats):
        y_fc_shuf = rng_fc.permutation(y_fc)
        rep_acc, _, _ = _fc_cv_with_pred(
            X_fc,
            y_fc_shuf,
            n_components=n_components_fc,
            n_splits=fc_n_splits,
            random_state=fc_random_state + 200 + rep,
        )
        fc_shuffle_acc.append(rep_acc)
    fc_shuffle_acc = np.asarray(fc_shuffle_acc, dtype=float)

    fc_classes = np.sort(np.unique(y_fc))
    fc_class_names = [label_names.get(int(c), str(c)) if 'label_names' in globals() else str(c) for c in fc_classes]
    fc_cm_norm = confusion_matrix(y_fc, y_fc_pred, labels=fc_classes, normalize='true')
    fc_cm_raw = confusion_matrix(y_fc, y_fc_pred, labels=fc_classes, normalize=None)

    fig, ax = plt.subplots(figsize=(6.2, 5.1), dpi=180)
    disp = ConfusionMatrixDisplay(confusion_matrix=fc_cm_norm, display_labels=fc_class_names)
    disp.plot(ax=ax, cmap='Purples', colorbar=True, values_format='.2f')
    ax.set_title(
        f'Task3 FC Decoder (SVD→SVC, n_comp={n_components_fc})\\n'
        f'Acc={fc_acc:.3f}±{fc_std:.3f} | Shuffle={fc_shuffle_acc.mean():.3f}±{fc_shuffle_acc.std(ddof=1):.3f}'
    )
    fig.tight_layout()
    fc_fig_path = os.path.join(fig_out_dir, 'fc_decoder_confusion_matrix.png')
    fig.savefig(fc_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------- 4) Export summary ----------
    activity_acc_ref = np.nan
    if 'task1_summary' in globals() and isinstance(task1_summary, pd.DataFrame) and len(task1_summary) > 0:
        activity_acc_ref = float(task1_summary['accuracy_mean'].iloc[0])

    fc_summary = pd.DataFrame([
        {
            'window_start': int(fc_start),
            'window_end': int(fc_end),
            'n_trials': int(n_samples_fc),
            'n_neurons_used': int(segments_fc.shape[1]),
            'neuron_mode': fc_neuron_mode,
            'n_features_fc': int(n_features_fc),
            'n_components_svd': int(n_components_fc),
            'n_splits': int(fc_n_splits),
            'accuracy_mean': float(fc_acc),
            'accuracy_std': float(fc_std),
            'shuffle_accuracy_mean': float(fc_shuffle_acc.mean()),
            'shuffle_accuracy_std': float(fc_shuffle_acc.std(ddof=1)),
            'accuracy_minus_shuffle': float(fc_acc - fc_shuffle_acc.mean()),
            'activity_decoder_accuracy_ref': float(activity_acc_ref),
            'fc_minus_activity_ref': float(fc_acc - activity_acc_ref) if np.isfinite(activity_acc_ref) else np.nan,
        }
    ])
    for idx, cname in enumerate(fc_class_names):
        fc_summary[f'recall_{cname}'] = float(fc_cm_norm[idx, idx])

    fc_summary_csv = os.path.join(data_out_dir, 'fc_decoder_summary.csv')
    fc_summary.to_csv(fc_summary_csv, index=False)
    pd.DataFrame(fc_cm_raw, index=fc_class_names, columns=fc_class_names).to_csv(
        os.path.join(data_out_dir, 'fc_decoder_confusion_matrix.csv')
    )
    pd.DataFrame({'repeat_idx': np.arange(fc_shuffle_acc.size), 'accuracy_mean': fc_shuffle_acc}).to_csv(
        os.path.join(data_out_dir, 'fc_decoder_shuffle_repeats.csv'),
        index=False
    )


    # %%
    from IPython.display import Markdown, display

    activity_delta = fc_acc - activity_acc_ref if np.isfinite(activity_acc_ref) else np.nan
    lines = [
        "### Task3 Result Snapshot",
        f"- FC decoder CV accuracy: **{fc_acc:.4f} +/- {fc_std:.4f}**",
        f"- Shuffle baseline: **{float(fc_shuffle_acc.mean()):.4f} +/- {float(fc_shuffle_acc.std(ddof=1)):.4f}**",
        f"- Delta (FC - shuffle): **{fc_acc - float(fc_shuffle_acc.mean()):.4f}**",
        f"- Delta (FC - activity reference): **{activity_delta:.4f}**",
        f"- Summary table: `{fc_summary_csv}`",
        f"- Confusion matrix figure: `{fc_fig_path}`",
    ]
    display(Markdown('\\n'.join(lines)))


    # %% [markdown]
    # ### 12.2 Task4: Robust Importance and Projection to Connection Levels
    # 

    # %%
    # ---------- 5) Task4: SVD-component stability + ablation ----------
    task4_stability_repeats = 25
    task4_subsample_ratio = 0.80
    task4_topk_components = min(10, n_components_fc)
    task4_ablation_topk = min(10, n_components_fc)

    svd_ref = TruncatedSVD(n_components=n_components_fc, random_state=fc_random_state + 999)
    Z_fc = svd_ref.fit_transform(X_fc)

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
                class_weight='balanced',
                dual='auto',
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

    rng_task4 = np.random.default_rng(fc_random_state + 4000)
    component_select_count = np.zeros(n_components_fc, dtype=int)
    component_abscoef_sum = np.zeros(n_components_fc, dtype=float)

    for rep in range(task4_stability_repeats):
        sub_idx = _stratified_subsample_idx(y_fc, task4_subsample_ratio, rng_task4)
        X_sub = Z_fc[sub_idx]
        y_sub = y_fc[sub_idx]

        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            dual='auto',
            max_iter=5000,
            random_state=fc_random_state + 5000 + rep,
        )
        model.fit(X_sub, y_sub)
        coef_abs = np.abs(model.coef_)
        if coef_abs.ndim == 1:
            comp_score = coef_abs
        else:
            comp_score = coef_abs.mean(axis=0)

        component_abscoef_sum += comp_score
        top_idx_rep = np.argsort(comp_score)[::-1][:task4_topk_components]
        component_select_count[top_idx_rep] += 1

    selection_freq = component_select_count / task4_stability_repeats
    mean_abs_coef = component_abscoef_sum / task4_stability_repeats

    stability_df = pd.DataFrame(
        {
            'component_idx': np.arange(n_components_fc, dtype=int),
            'selection_frequency': selection_freq,
            'mean_abs_coef': mean_abs_coef,
        }
    ).sort_values(['selection_frequency', 'mean_abs_coef'], ascending=False, ignore_index=True)
    stability_csv_path = os.path.join(data_out_dir, 'fc_component_stability_selection.csv')
    stability_df.to_csv(stability_csv_path, index=False)

    linear_base_acc, linear_base_std = _linear_cv_acc(
        Z_fc,
        y_fc,
        n_splits=fc_n_splits,
        random_state=fc_random_state + 6000,
        drop_components=None,
    )

    top_components = stability_df['component_idx'].to_numpy(dtype=int)[:task4_ablation_topk]
    ablation_records = []
    for comp_idx in top_components:
        drop_acc, drop_std = _linear_cv_acc(
            Z_fc,
            y_fc,
            n_splits=fc_n_splits,
            random_state=fc_random_state + 6000,
            drop_components=[int(comp_idx)],
        )
        freq_now = float(selection_freq[int(comp_idx)])
        coef_now = float(mean_abs_coef[int(comp_idx)])
        ablation_records.append(
            {
                'component_idx': int(comp_idx),
                'base_accuracy_mean': float(linear_base_acc),
                'base_accuracy_std': float(linear_base_std),
                'ablation_accuracy_mean': float(drop_acc),
                'ablation_accuracy_std': float(drop_std),
                'delta_vs_base': float(linear_base_acc - drop_acc),
                'selection_frequency': freq_now,
                'mean_abs_coef': coef_now,
            }
        )

    ablation_df = pd.DataFrame(ablation_records).sort_values(
        'delta_vs_base', ascending=False, ignore_index=True
    )
    ablation_csv_path = os.path.join(data_out_dir, 'fc_component_ablation_delta_acc.csv')
    ablation_df.to_csv(ablation_csv_path, index=False)

    top_freq_plot = stability_df.head(task4_topk_components).iloc[::-1]
    top_drop_plot = ablation_df.head(task4_ablation_topk).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)
    axes[0].barh(
        [f'PC{int(i)}' for i in top_freq_plot['component_idx']],
        top_freq_plot['selection_frequency'],
        color='#4C78A8',
        alpha=0.9,
    )
    axes[0].set_xlim(0, 1.0)
    axes[0].set_xlabel('Selection frequency')
    axes[0].set_title(f'Task4 Stability Top-{task4_topk_components}')
    axes[0].grid(axis='x', linestyle='--', alpha=0.25)

    axes[1].barh(
        [f'PC{int(i)}' for i in top_drop_plot['component_idx']],
        top_drop_plot['delta_vs_base'],
        color='#F58518',
        alpha=0.9,
    )
    axes[1].axvline(0.0, color='#666666', linewidth=1.0)
    axes[1].set_xlabel('Accuracy drop after single-component ablation')
    axes[1].set_title(f'Task4 Ablation Top-{task4_ablation_topk}')
    axes[1].grid(axis='x', linestyle='--', alpha=0.25)

    fig.tight_layout()
    task4_fig_path = os.path.join(fig_out_dir, 'fc_component_importance_task4.png')
    fig.savefig(task4_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


    # %%
    # ---------- 6) Task4 projection back to connection hierarchy ----------
    task4_projection_top_edges = min(500, n_features_fc)

    def _depth_tertile_labels(depth_vals):
        q_low, q_high = np.quantile(depth_vals, [1.0 / 3.0, 2.0 / 3.0])
        out = np.empty(depth_vals.shape[0], dtype=object)
        out[depth_vals <= q_low] = 'DepthLow'
        out[(depth_vals > q_low) & (depth_vals <= q_high)] = 'DepthMid'
        out[depth_vals > q_high] = 'DepthHigh'
        return out

    n_neurons_fc = int(segments_fc.shape[1])
    tri_i, tri_j = np.triu_indices(n_neurons_fc, k=1)

    proj_components = ablation_df['component_idx'].to_numpy(dtype=int)
    proj_weights = ablation_df['delta_vs_base'].to_numpy(dtype=float)
    if np.all(np.abs(proj_weights) < 1e-12):
        proj_weights = ablation_df['selection_frequency'].to_numpy(dtype=float)

    edge_importance_raw = np.zeros(n_features_fc, dtype=float)
    for comp_idx, comp_w in zip(proj_components, proj_weights):
        edge_importance_raw += float(abs(comp_w)) * np.abs(svd_ref.components_[int(comp_idx)])
    edge_importance = edge_importance_raw / (edge_importance_raw.sum() + 1e-12)

    edge_corr_mean = np.tanh(np.nanmean(X_fc, axis=0))
    edge_order = np.argsort(edge_corr_mean)
    strength_decile = np.empty(n_features_fc, dtype=int)
    strength_decile[edge_order] = (np.arange(n_features_fc, dtype=int) * 10 // n_features_fc) + 1

    if 'layer_labels_spi' in globals():
        layer_arr = np.asarray(layer_labels_spi)
        if layer_arr.ndim == 1 and layer_arr.shape[0] > int(rr_idx.max()):
            layer_labels_fc = layer_arr[rr_idx].astype(str)
            layer_source = 'layer_labels_spi'
        elif 'neuron_pos_spi' in globals():
            pos_arr = np.asarray(neuron_pos_spi, dtype=float)
            if pos_arr.ndim == 2 and pos_arr.shape[1] > int(rr_idx.max()):
                layer_labels_fc = _depth_tertile_labels(pos_arr[0, rr_idx])
                layer_source = 'depth_tertile_fallback_axis0'
            else:
                layer_labels_fc = np.array(['Unknown'] * n_neurons_fc, dtype=object)
                layer_source = 'unknown'
        else:
            layer_labels_fc = np.array(['Unknown'] * n_neurons_fc, dtype=object)
            layer_source = 'unknown'
    elif 'neuron_pos_spi' in globals():
        pos_arr = np.asarray(neuron_pos_spi, dtype=float)
        if pos_arr.ndim == 2 and pos_arr.shape[1] > int(rr_idx.max()):
            layer_labels_fc = _depth_tertile_labels(pos_arr[0, rr_idx])
            layer_source = 'depth_tertile_from_neuron_pos_spi_axis0'
        else:
            layer_labels_fc = np.array(['Unknown'] * n_neurons_fc, dtype=object)
            layer_source = 'unknown'
    else:
        layer_labels_fc = np.array(['Unknown'] * n_neurons_fc, dtype=object)
        layer_source = 'unknown'

    layer_labels_fc = np.asarray(layer_labels_fc, dtype=str)
    layer_names = pd.Index(layer_labels_fc).unique().tolist()
    layer_to_code = {name: idx for idx, name in enumerate(layer_names)}
    layer_codes = np.asarray([layer_to_code[name] for name in layer_labels_fc], dtype=int)

    li = layer_codes[tri_i]
    lj = layer_codes[tri_j]
    pair_a = np.minimum(li, lj)
    pair_b = np.maximum(li, lj)
    n_layers = len(layer_names)
    pair_code = pair_a * n_layers + pair_b

    decile_rows = []
    for d in range(1, 11):
        mask_d = strength_decile == d
        decile_rows.append(
            {
                'strength_decile': int(d),
                'n_edges': int(mask_d.sum()),
                'importance_sum': float(edge_importance[mask_d].sum()),
                'importance_mean': float(edge_importance[mask_d].mean()),
                'corr_mean': float(edge_corr_mean[mask_d].mean()),
            }
        )
    decile_df = pd.DataFrame(decile_rows)
    decile_csv_path = os.path.join(data_out_dir, 'fc_projection_by_strength_decile_task4.csv')
    decile_df.to_csv(decile_csv_path, index=False)

    strong_tail = decile_df.loc[decile_df['strength_decile'] == 10].iloc[0]
    weak_tail = decile_df.loc[decile_df['strength_decile'] == 1].iloc[0]
    strong_weak_match_df = pd.DataFrame(
        [
            {
                'importance_strong_tail_decile10': float(strong_tail['importance_sum']),
                'importance_weak_tail_decile1': float(weak_tail['importance_sum']),
                'importance_gap_d10_minus_d1': float(strong_tail['importance_sum'] - weak_tail['importance_sum']),
                'corr_mean_decile10': float(strong_tail['corr_mean']),
                'corr_mean_decile1': float(weak_tail['corr_mean']),
            }
        ]
    )
    strong_weak_csv_path = os.path.join(data_out_dir, 'fc_projection_strong_weak_match_task4.csv')
    strong_weak_match_df.to_csv(strong_weak_csv_path, index=False)

    pair_rows = []
    for code in np.unique(pair_code):
        mask_p = pair_code == code
        a = int(code // n_layers)
        b = int(code % n_layers)
        pair_rows.append(
            {
                'layer_pair': f'{layer_names[a]}--{layer_names[b]}',
                'layer_a': layer_names[a],
                'layer_b': layer_names[b],
                'n_edges': int(mask_p.sum()),
                'importance_sum': float(edge_importance[mask_p].sum()),
                'importance_mean': float(edge_importance[mask_p].mean()),
                'corr_mean': float(edge_corr_mean[mask_p].mean()),
                'layer_source': layer_source,
            }
        )
    depth_pair_df = pd.DataFrame(pair_rows).sort_values('importance_sum', ascending=False, ignore_index=True)
    depth_pair_csv_path = os.path.join(data_out_dir, 'fc_projection_by_layer_pair_task4.csv')
    depth_pair_df.to_csv(depth_pair_csv_path, index=False)

    top_edge_idx = np.argsort(edge_importance)[::-1][:task4_projection_top_edges]
    top_i_local = tri_i[top_edge_idx]
    top_j_local = tri_j[top_edge_idx]
    top_i_global = rr_idx[top_i_local]
    top_j_global = rr_idx[top_j_local]
    top_pair_labels = np.asarray(
        [
            '--'.join(sorted((layer_labels_fc[i], layer_labels_fc[j])))
            for i, j in zip(top_i_local, top_j_local)
        ],
        dtype=object,
    )

    top_edges_df = pd.DataFrame(
        {
            'rank': np.arange(1, top_edge_idx.size + 1, dtype=int),
            'edge_idx': top_edge_idx.astype(int),
            'neuron_i_local': top_i_local.astype(int),
            'neuron_j_local': top_j_local.astype(int),
            'neuron_i_global': top_i_global.astype(int),
            'neuron_j_global': top_j_global.astype(int),
            'layer_i': layer_labels_fc[top_i_local],
            'layer_j': layer_labels_fc[top_j_local],
            'layer_pair': top_pair_labels,
            'strength_decile': strength_decile[top_edge_idx].astype(int),
            'corr_mean': edge_corr_mean[top_edge_idx],
            'edge_importance': edge_importance[top_edge_idx],
            'edge_importance_raw': edge_importance_raw[top_edge_idx],
        }
    )
    top_edges_csv_path = os.path.join(data_out_dir, 'fc_projection_top_edges_task4.csv')
    top_edges_df.to_csv(top_edges_csv_path, index=False)

    depth_pair_plot = depth_pair_df.head(min(10, len(depth_pair_df))).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=180)

    axes[0].bar(decile_df['strength_decile'], decile_df['importance_sum'], color='#4C78A8', alpha=0.9)
    axes[0].set_xlabel('Correlation strength decile (1=weak,10=strong)')
    axes[0].set_ylabel('Projected importance sum')
    axes[0].set_title('Task4 Projection to Strength Levels')
    axes[0].set_xticks(np.arange(1, 11, 1))
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)

    axes[1].barh(depth_pair_plot['layer_pair'], depth_pair_plot['importance_sum'], color='#72B7B2', alpha=0.9)
    axes[1].set_xlabel('Projected importance sum')
    axes[1].set_title('Task4 Projection to Layer Pairs (Top)')
    axes[1].grid(axis='x', linestyle='--', alpha=0.25)

    fig.tight_layout()
    task4_proj_fig_path = os.path.join(fig_out_dir, 'fc_projection_levels_task4.png')
    fig.savefig(task4_proj_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------- 6.1) Task4 edge-level outputs aligned with task list ----------
    all_i_local = tri_i
    all_j_local = tri_j
    all_i_global = rr_idx[all_i_local]
    all_j_global = rr_idx[all_j_local]
    all_pair_labels = np.asarray(
        [
            '--'.join(sorted((layer_labels_fc[i], layer_labels_fc[j])))
            for i, j in zip(all_i_local, all_j_local)
        ],
        dtype=object,
    )

    edge_stability_df = pd.DataFrame(
        {
            'edge_idx': np.arange(n_features_fc, dtype=int),
            'neuron_i_local': all_i_local.astype(int),
            'neuron_j_local': all_j_local.astype(int),
            'neuron_i_global': all_i_global.astype(int),
            'neuron_j_global': all_j_global.astype(int),
            'layer_i': layer_labels_fc[all_i_local],
            'layer_j': layer_labels_fc[all_j_local],
            'layer_pair': all_pair_labels,
            'strength_decile': strength_decile.astype(int),
            'corr_mean': edge_corr_mean,
            'edge_importance': edge_importance,
            'edge_importance_raw': edge_importance_raw,
        }
    ).sort_values('edge_importance', ascending=False, ignore_index=True)
    edge_stability_df.insert(0, 'rank', np.arange(1, edge_stability_df.shape[0] + 1, dtype=int))

    edge_stability_csv_path = os.path.join(data_out_dir, 'fc_edge_importance_stability.csv')
    edge_stability_df.to_csv(edge_stability_csv_path, index=False)


    def _fc_cv_acc_edge_mask(X_mat, y_vec, drop_edge_idx=None, n_splits=3, random_state=123):
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
                    ('svd', TruncatedSVD(n_components=n_components_fc, random_state=random_state + 100 + fold_i)),
                    (
                        'clf',
                        LinearSVC(
                            C=1.0,
                            class_weight='balanced',
                            dual='auto',
                            max_iter=5000,
                            random_state=random_state + 200 + fold_i,
                        ),
                    ),
                ]
            )
            pipe.fit(X_tr, y_vec[tr_idx])
            pred = pipe.predict(X_te)
            fold_acc.append(float((pred == y_vec[te_idx]).mean()))

        fold_acc = np.asarray(fold_acc, dtype=float)
        return float(fold_acc.mean()), float(fold_acc.std(ddof=1))


    task4_edge_ablation_fracs = [0.01, 0.03, 0.05]
    task4_edge_ablation_random_repeats = 8
    rng_task4_edge = np.random.default_rng(fc_random_state + 8000)

    base_edge_acc, base_edge_std = _fc_cv_acc_edge_mask(
        X_fc,
        y_fc,
        drop_edge_idx=None,
        n_splits=fc_n_splits,
        random_state=fc_random_state + 9000,
    )

    edge_rank_desc = np.argsort(edge_importance)[::-1]
    edge_ablation_rows = []

    for frac in task4_edge_ablation_fracs:
        n_drop = int(max(1, np.ceil(n_features_fc * frac)))

        top_drop_idx = edge_rank_desc[:n_drop]
        top_acc, top_std = _fc_cv_acc_edge_mask(
            X_fc,
            y_fc,
            drop_edge_idx=top_drop_idx,
            n_splits=fc_n_splits,
            random_state=fc_random_state + 9000,
        )
        edge_ablation_rows.append(
            {
                'ablation_type': 'top',
                'drop_fraction': float(frac),
                'n_edges_dropped': int(n_drop),
                'repeat_idx': -1,
                'base_accuracy_mean': float(base_edge_acc),
                'base_accuracy_std': float(base_edge_std),
                'accuracy_mean': float(top_acc),
                'accuracy_std': float(top_std),
                'delta_vs_base': float(base_edge_acc - top_acc),
            }
        )

        for rep in range(task4_edge_ablation_random_repeats):
            rnd_idx = rng_task4_edge.choice(n_features_fc, size=n_drop, replace=False)
            rnd_acc, rnd_std = _fc_cv_acc_edge_mask(
                X_fc,
                y_fc,
                drop_edge_idx=rnd_idx,
                n_splits=fc_n_splits,
                random_state=fc_random_state + 9000 + 50 + rep,
            )
            edge_ablation_rows.append(
                {
                    'ablation_type': 'random',
                    'drop_fraction': float(frac),
                    'n_edges_dropped': int(n_drop),
                    'repeat_idx': int(rep),
                    'base_accuracy_mean': float(base_edge_acc),
                    'base_accuracy_std': float(base_edge_std),
                    'accuracy_mean': float(rnd_acc),
                    'accuracy_std': float(rnd_std),
                    'delta_vs_base': float(base_edge_acc - rnd_acc),
                }
            )

    edge_ablation_df = pd.DataFrame(edge_ablation_rows)
    edge_ablation_csv_path = os.path.join(data_out_dir, 'fc_edge_ablation_delta_acc.csv')
    edge_ablation_df.to_csv(edge_ablation_csv_path, index=False)

    edge_ablation_plot_df = (
        edge_ablation_df.groupby(['drop_fraction', 'n_edges_dropped', 'ablation_type'], as_index=False)
        .agg(
            mean_delta_vs_base=('delta_vs_base', 'mean'),
            sem_delta_vs_base=(
                'delta_vs_base',
                lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
            ),
        )
        .sort_values(['drop_fraction', 'ablation_type'], ignore_index=True)
    )

    plot_top_edge = edge_ablation_plot_df[edge_ablation_plot_df['ablation_type'] == 'top'].copy()
    plot_rand_edge = edge_ablation_plot_df[edge_ablation_plot_df['ablation_type'] == 'random'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), dpi=180)

    axes[0].barh(
        [f'PC{int(i)}' for i in top_freq_plot['component_idx']],
        top_freq_plot['selection_frequency'],
        color='#4C78A8',
        alpha=0.9,
    )
    axes[0].set_xlim(0, 1.0)
    axes[0].set_xlabel('Selection frequency')
    axes[0].set_title(f'Task4 Stability Top-{task4_topk_components}')
    axes[0].grid(axis='x', linestyle='--', alpha=0.25)

    axes[1].barh(
        [f'PC{int(i)}' for i in top_drop_plot['component_idx']],
        top_drop_plot['delta_vs_base'],
        color='#F58518',
        alpha=0.9,
    )
    axes[1].axvline(0.0, color='#666666', linewidth=1.0)
    axes[1].set_xlabel('Accuracy drop after single-component ablation')
    axes[1].set_title(f'Task4 Ablation Top-{task4_ablation_topk}')
    axes[1].grid(axis='x', linestyle='--', alpha=0.25)

    x3 = np.arange(plot_top_edge.shape[0])
    axes[2].plot(
        x3,
        plot_top_edge['mean_delta_vs_base'].to_numpy(dtype=float),
        marker='o',
        linewidth=2.0,
        color='#E45756',
        label='Top edges',
    )
    axes[2].errorbar(
        x3,
        plot_rand_edge['mean_delta_vs_base'].to_numpy(dtype=float),
        yerr=plot_rand_edge['sem_delta_vs_base'].to_numpy(dtype=float),
        fmt='s--',
        capsize=3,
        color='#54A24B',
        label='Random edges',
    )
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels([f"{int(100 * v)}%" for v in plot_top_edge['drop_fraction']])
    axes[2].set_xlabel('Dropped edge fraction')
    axes[2].set_ylabel('Accuracy drop vs baseline')
    axes[2].set_title('Task4 Edge Ablation')
    axes[2].grid(axis='y', linestyle='--', alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    task4_robust_fig_path = os.path.join(fig_out_dir, 'fc_importance_robustness.png')
    fig.savefig(task4_robust_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Keep backward-compatible component-level artifacts, but expose edge-level files
    # as the canonical Task4 outputs (matching task list naming).
    component_stability_csv_path = stability_csv_path
    component_ablation_csv_path = ablation_csv_path
    component_task4_fig_path = task4_fig_path

    stability_csv_path = edge_stability_csv_path
    ablation_csv_path = edge_ablation_csv_path
    task4_fig_path = task4_robust_fig_path


    # %%
    from IPython.display import Markdown, display

    weak_sum = float(decile_df.loc[decile_df['strength_decile'].isin([1, 2]), 'importance_sum'].sum())
    strong_sum = float(decile_df.loc[decile_df['strength_decile'].isin([9, 10]), 'importance_sum'].sum())
    top_pair = str(depth_pair_df.iloc[0]['layer_pair']) if len(depth_pair_df) > 0 else 'NA'

    top_drop_row = edge_ablation_df[edge_ablation_df['ablation_type'] == 'top'].sort_values('n_edges_dropped')
    if len(top_drop_row) > 0:
        top_drop_row = top_drop_row.iloc[-1]
        n_drop = int(top_drop_row['n_edges_dropped'])
        top_delta = float(top_drop_row['delta_vs_base'])
        rand_ref = edge_ablation_df[
            (edge_ablation_df['ablation_type'] == 'random')
            & (edge_ablation_df['n_edges_dropped'] == n_drop)
        ]
        rand_delta = float(rand_ref['delta_vs_base'].mean()) if len(rand_ref) > 0 else np.nan
    else:
        n_drop, top_delta, rand_delta = 0, np.nan, np.nan

    lines = [
        "### Task4 Result Snapshot",
        f"- Stable edge list: `{edge_stability_csv_path}` (n={len(edge_stability_df)})",
        f"- Top projected layer pair: **{top_pair}**",
        f"- Weak-tail importance (D1-D2): **{weak_sum:.4f}**; strong-tail (D9-D10): **{strong_sum:.4f}**",
        f"- Edge ablation ({n_drop} edges): top-drop Delta acc **{top_delta:.4f}**, random-drop mean Delta acc **{rand_delta:.4f}**",
        f"- Robustness figure: `{task4_robust_fig_path}`",
        f"- Projection figure: `{task4_proj_fig_path}`",
    ]
    display(Markdown('\\n'.join(lines)))


    # %% [markdown]
    # ### 12.3 Task5: Important-Edge Enrichment Across Deciles/Regimes
    # 

    # %%
    # ---------- 7) Task5: important-edge enrichment in deciles/regimes ----------
    task5_important_frac = 0.10
    task5_weak_deciles = [1, 2]
    task5_strong_deciles = [9, 10]

    n_total_edges = int(n_features_fc)
    n_important_edges = int(max(1, np.ceil(n_total_edges * task5_important_frac)))
    important_edge_idx = np.argsort(edge_importance)[::-1][:n_important_edges]
    important_decile = strength_decile[important_edge_idx]

    imp_i_local = tri_i[important_edge_idx]
    imp_j_local = tri_j[important_edge_idx]
    imp_i_global = rr_idx[imp_i_local]
    imp_j_global = rr_idx[imp_j_local]
    important_pair_labels = np.asarray(
        [
            '--'.join(sorted((layer_labels_fc[i], layer_labels_fc[j])))
            for i, j in zip(imp_i_local, imp_j_local)
        ],
        dtype=object,
    )
    important_edge_map_df = pd.DataFrame(
        {
            'rank': np.arange(1, n_important_edges + 1, dtype=int),
            'edge_idx': important_edge_idx.astype(int),
            'neuron_i_local': imp_i_local.astype(int),
            'neuron_j_local': imp_j_local.astype(int),
            'neuron_i_global': imp_i_global.astype(int),
            'neuron_j_global': imp_j_global.astype(int),
            'layer_i': layer_labels_fc[imp_i_local],
            'layer_j': layer_labels_fc[imp_j_local],
            'layer_pair': important_pair_labels,
            'strength_decile': strength_decile[important_edge_idx].astype(int),
            'corr_mean': edge_corr_mean[important_edge_idx],
            'edge_importance': edge_importance[important_edge_idx],
            'edge_importance_raw': edge_importance_raw[important_edge_idx],
        }
    )
    important_edge_map_csv_path = os.path.join(data_out_dir, 'fc_important_edge_decile_map_task5.csv')
    important_edge_map_df.to_csv(important_edge_map_csv_path, index=False)

    def _bh_fdr(pvals):
        pvals = np.asarray(pvals, dtype=float)
        n = int(pvals.size)
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj_ranked = np.empty(n, dtype=float)
        prev = 1.0
        for i in range(n - 1, -1, -1):
            rank = i + 1
            val = ranked[i] * n / rank
            prev = min(prev, val)
            adj_ranked[i] = min(prev, 1.0)
        adj = np.empty(n, dtype=float)
        adj[order] = adj_ranked
        return adj

    def _enrichment_row(level_type, level_name, level_mask):
        M = int(n_total_edges)
        N = int(n_important_edges)
        K = int(level_mask.sum())
        x = int(level_mask[important_edge_idx].sum())

        expected_count = float(N * K / M)
        observed_prop = float(x / N)
        expected_prop = float(K / M)
        enrichment_ratio = float(observed_prop / (expected_prop + 1e-12))
        log2_enrichment = float(np.log2((observed_prop + 1e-12) / (expected_prop + 1e-12)))

        p_over = float(hypergeom.sf(x - 1, M, K, N))
        p_under = float(hypergeom.cdf(x, M, K, N))
        p_two = float(min(1.0, 2.0 * min(p_over, p_under)))

        return {
            'level_type': level_type,
            'level': level_name,
            'observed_count': int(x),
            'expected_count': expected_count,
            'observed_prop': observed_prop,
            'expected_prop': expected_prop,
            'enrichment_ratio': enrichment_ratio,
            'log2_enrichment': log2_enrichment,
            'p_over': p_over,
            'p_under': p_under,
            'p_two_sided': p_two,
            'n_important_edges': int(N),
            'n_total_edges': int(M),
        }

    enrichment_rows = []
    for d in range(1, 11):
        mask_d = strength_decile == d
        enrichment_rows.append(_enrichment_row('decile', f'D{d}', mask_d))

    mask_weak = np.isin(strength_decile, task5_weak_deciles)
    mask_strong = np.isin(strength_decile, task5_strong_deciles)
    enrichment_rows.append(_enrichment_row('regime', 'WeakTail_D1D2', mask_weak))
    enrichment_rows.append(_enrichment_row('regime', 'StrongTail_D9D10', mask_strong))

    enrichment_df = pd.DataFrame(enrichment_rows)
    decile_only_mask = enrichment_df['level_type'] == 'decile'
    enrichment_df.loc[decile_only_mask, 'p_fdr_bh'] = _bh_fdr(
        enrichment_df.loc[decile_only_mask, 'p_two_sided'].to_numpy(dtype=float)
    )
    enrichment_df.loc[~decile_only_mask, 'p_fdr_bh'] = np.nan

    enrichment_csv_path = os.path.join(data_out_dir, 'fc_edge_decile_enrichment.csv')
    enrichment_df.to_csv(enrichment_csv_path, index=False)

    enrichment_plot_df = enrichment_df[enrichment_df['level_type'] == 'decile'].copy()
    enrichment_plot_df['decile_idx'] = enrichment_plot_df['level'].str.replace('D', '', regex=False).astype(int)
    enrichment_plot_df = enrichment_plot_df.sort_values('decile_idx', ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=180)

    x = enrichment_plot_df['decile_idx'].to_numpy(dtype=int)
    obs = enrichment_plot_df['observed_prop'].to_numpy(dtype=float)
    exp = enrichment_plot_df['expected_prop'].to_numpy(dtype=float)
    bar_w = 0.38

    axes[0].bar(x - bar_w / 2, obs, width=bar_w, color='#4C78A8', alpha=0.9, label='Observed')
    axes[0].bar(x + bar_w / 2, exp, width=bar_w, color='#B9CFE7', alpha=0.95, label='Expected')
    axes[0].set_xlabel('Correlation decile (1=weak, 10=strong)')
    axes[0].set_ylabel('Proportion in important-edge set')
    axes[0].set_title('Task5 Observed vs Expected')
    axes[0].set_xticks(np.arange(1, 11, 1))
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)
    axes[0].legend(frameon=False)

    log2_enr = enrichment_plot_df['log2_enrichment'].to_numpy(dtype=float)
    colors = np.where(log2_enr >= 0, '#72B7B2', '#E45756')
    axes[1].bar(x, log2_enr, color=colors, alpha=0.9)
    axes[1].axhline(0.0, color='#666666', linewidth=1.0)
    axes[1].set_xlabel('Correlation decile (1=weak, 10=strong)')
    axes[1].set_ylabel('log2 enrichment (Observed / Expected)')
    axes[1].set_title('Task5 Enrichment Effect Size')
    axes[1].set_xticks(np.arange(1, 11, 1))
    axes[1].grid(axis='y', linestyle='--', alpha=0.25)

    for _, row in enrichment_plot_df.iterrows():
        if float(row['p_fdr_bh']) < 0.05:
            y = float(row['log2_enrichment'])
            y_txt = y + (0.04 if y >= 0 else -0.08)
            axes[1].text(int(row['decile_idx']), y_txt, '*', ha='center', va='center', fontsize=11)

    fig.tight_layout()
    enrichment_fig_path = os.path.join(fig_out_dir, 'fc_edge_decile_enrichment.png')
    fig.savefig(enrichment_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


    # %%
    from IPython.display import Markdown, display

    weak_row = enrichment_df[
        (enrichment_df['level_type'] == 'regime') & (enrichment_df['level'] == 'WeakTail_D1D2')
    ]
    strong_row = enrichment_df[
        (enrichment_df['level_type'] == 'regime') & (enrichment_df['level'] == 'StrongTail_D9D10')
    ]
    weak_log2 = float(weak_row['log2_enrichment'].iloc[0]) if len(weak_row) > 0 else np.nan
    weak_p = float(weak_row['p_two_sided'].iloc[0]) if len(weak_row) > 0 else np.nan
    strong_log2 = float(strong_row['log2_enrichment'].iloc[0]) if len(strong_row) > 0 else np.nan

    decile_df_task5 = enrichment_df[enrichment_df['level_type'] == 'decile'].copy()
    top_decile = decile_df_task5.sort_values('log2_enrichment', ascending=False).iloc[0] if len(decile_df_task5) > 0 else None
    top_decile_txt = top_decile['level'] if top_decile is not None else 'NA'
    top_decile_enr = float(top_decile['log2_enrichment']) if top_decile is not None else np.nan

    lines = [
        "### Task5 Result Snapshot",
        f"- Important-edge map: `{important_edge_map_csv_path}`",
        f"- Weak-tail enrichment log2(O/E): **{weak_log2:.4f}** (p={weak_p:.3g})",
        f"- Strong-tail enrichment log2(O/E): **{strong_log2:.4f}**",
        f"- Most enriched decile: **{top_decile_txt}** (log2 enrichment={top_decile_enr:.4f})",
        f"- Enrichment table: `{enrichment_csv_path}`",
        f"- Enrichment figure: `{enrichment_fig_path}`",
    ]
    display(Markdown('\\n'.join(lines)))


    # %% [markdown]
    # ### 12.4 Task6: Decoder-Important Neurons Linked to RR Overlap/Selectivity
    # 

    # %%
    # ---------- 8) Task6: decoder-important neurons x RR overlap/selectivity ----------
    task6_important_neuron_frac = 0.20
    task6_selectivity_window = (10, 13)
    task6_ablation_eval_per_category = 30

    if 'rr_sets' in globals() and isinstance(rr_sets, dict) and len(rr_sets) > 0:
        rr_sets_task6 = {int(k): set(map(int, v)) for k, v in rr_sets.items()}
    elif 'rr_neurons_spi' in globals() and isinstance(rr_neurons_spi, dict) and len(rr_neurons_spi) > 0:
        rr_sets_task6 = {int(k): set(map(int, v)) for k, v in rr_neurons_spi.items()}
    else:
        rr_sets_task6 = {}

    class_ids_task6 = sorted(rr_sets_task6.keys())
    if len(class_ids_task6) == 0:
        class_ids_task6 = sorted(np.unique(y_fc).astype(int).tolist())
        rr_sets_task6 = {int(c): set() for c in class_ids_task6}

    n_neurons_fc = int(segments_fc.shape[1])
    neuron_ids_global = rr_idx.copy().astype(int)

    membership_mat = np.zeros((n_neurons_fc, len(class_ids_task6)), dtype=int)
    for ci, cls in enumerate(class_ids_task6):
        cls_set = rr_sets_task6.get(int(cls), set())
        membership_mat[:, ci] = np.isin(neuron_ids_global, list(cls_set)).astype(int)
    membership_count = membership_mat.sum(axis=1)

    coarse_category = np.empty(n_neurons_fc, dtype=object)
    detail_category = np.empty(n_neurons_fc, dtype=object)
    preferred_rr_label = np.empty(n_neurons_fc, dtype=object)

    for i in range(n_neurons_fc):
        if membership_count[i] >= 2:
            coarse_category[i] = 'Shared_Core'
            detail_category[i] = 'Shared_Core'
            preferred_rr_label[i] = 'Shared_Core'
        elif membership_count[i] == 1:
            ci = int(np.argmax(membership_mat[i]))
            cls = int(class_ids_task6[ci])
            cls_name = label_names.get(cls, f'Class{cls}') if 'label_names' in globals() else f'Class{cls}'
            coarse_category[i] = 'Condition_Biased'
            detail_category[i] = f'Biased_{cls_name}'
            preferred_rr_label[i] = cls_name
        else:
            coarse_category[i] = 'Non_RR'
            detail_category[i] = 'Non_RR'
            preferred_rr_label[i] = 'Non_RR'

    neuron_importance = (
        np.bincount(tri_i, weights=edge_importance, minlength=n_neurons_fc)
        + np.bincount(tri_j, weights=edge_importance, minlength=n_neurons_fc)
    )
    neuron_importance_norm = neuron_importance / (neuron_importance.sum() + 1e-12)

    n_important_neurons = int(max(1, np.ceil(n_neurons_fc * task6_important_neuron_frac)))
    important_neuron_idx = np.argsort(neuron_importance_norm)[::-1][:n_important_neurons]
    is_important_neuron = np.zeros(n_neurons_fc, dtype=bool)
    is_important_neuron[important_neuron_idx] = True

    sel_start = max(0, min(task6_selectivity_window[0], segments_spi.shape[2] - 1))
    sel_end = max(sel_start + 1, min(task6_selectivity_window[1], segments_spi.shape[2]))

    resp_by_class = np.full((n_neurons_fc, len(class_ids_task6)), np.nan, dtype=float)
    for ci, cls in enumerate(class_ids_task6):
        cls_mask = labels_spi == int(cls)
        if np.any(cls_mask):
            cls_resp_all = np.nanmean(segments_spi[cls_mask, :, sel_start:sel_end], axis=(0, 2))
            resp_by_class[:, ci] = cls_resp_all[neuron_ids_global]

    selectivity_index = np.zeros(n_neurons_fc, dtype=float)
    preferred_class = np.empty(n_neurons_fc, dtype=object)
    for i in range(n_neurons_fc):
        vals = resp_by_class[i]
        if np.all(~np.isfinite(vals)):
            selectivity_index[i] = 0.0
            preferred_class[i] = 'Unknown'
            continue
        pref_ci = int(np.nanargmax(vals))
        pref_cls = int(class_ids_task6[pref_ci])
        preferred_class[i] = label_names.get(pref_cls, f'Class{pref_cls}') if 'label_names' in globals() else f'Class{pref_cls}'
        top_val = float(vals[pref_ci])
        oth = np.delete(vals, pref_ci)
        oth_mean = float(np.nanmean(oth)) if np.any(np.isfinite(oth)) else 0.0
        den = abs(top_val) + abs(oth_mean) + 1e-12
        si = (top_val - oth_mean) / den
        selectivity_index[i] = float(si if np.isfinite(si) else 0.0)

    incident_edges = [[] for _ in range(n_neurons_fc)]
    for e_idx, (ii, jj) in enumerate(zip(tri_i, tri_j)):
        incident_edges[int(ii)].append(int(e_idx))
        incident_edges[int(jj)].append(int(e_idx))
    incident_edges = [np.asarray(v, dtype=int) for v in incident_edges]

    coarse_order = [c for c in ['Shared_Core', 'Condition_Biased', 'Non_RR'] if np.any(coarse_category == c)]
    ablation_eval_idx = []
    for cat in coarse_order:
        cand = np.where(coarse_category == cat)[0]
        cand_sorted = cand[np.argsort(neuron_importance_norm[cand])[::-1]]
        take_n = int(min(task6_ablation_eval_per_category, cand_sorted.size))
        ablation_eval_idx.extend(cand_sorted[:take_n].tolist())
    ablation_eval_idx = np.asarray(sorted(set(ablation_eval_idx)), dtype=int)

    ablation_drop_actual = np.full(n_neurons_fc, np.nan, dtype=float)
    for ni in ablation_eval_idx:
        inc = incident_edges[int(ni)]
        if inc.size == 0:
            ablation_drop_actual[int(ni)] = 0.0
            continue
        delta_z = X_fc[:, inc] @ svd_ref.components_[:, inc].T
        Z_abl = Z_fc - delta_z
        acc_abl, _ = _linear_cv_acc(
            Z_abl,
            y_fc,
            n_splits=fc_n_splits,
            random_state=fc_random_state + 6000,
            drop_components=None,
        )
        ablation_drop_actual[int(ni)] = float(linear_base_acc - acc_abl)

    neuron_detail_df = pd.DataFrame(
        {
            'neuron_local_idx': np.arange(n_neurons_fc, dtype=int),
            'neuron_global_idx': neuron_ids_global,
            'membership_count': membership_count.astype(int),
            'coarse_overlap': coarse_category,
            'detail_overlap': detail_category,
            'rr_label_single': preferred_rr_label,
            'preferred_class': preferred_class,
            'selectivity_index': selectivity_index,
            'decoder_importance': neuron_importance_norm,
            'ablation_effect_proxy': neuron_importance_norm,
            'ablation_drop_actual': ablation_drop_actual,
            'is_important_neuron': is_important_neuron.astype(int),
        }
    )
    for ci, cls in enumerate(class_ids_task6):
        cname = label_names.get(int(cls), f'Class{cls}') if 'label_names' in globals() else f'Class{cls}'
        safe = str(cname).replace(' ', '_')
        neuron_detail_df[f'resp_{safe}'] = resp_by_class[:, ci]

    neuron_detail_csv_path = os.path.join(data_out_dir, 'neuron_decoder_linking_detail.csv')
    neuron_detail_df.to_csv(neuron_detail_csv_path, index=False)

    def _mean_sem(arr):
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan
        m = float(vals.mean())
        if vals.size <= 1:
            return m, np.nan
        sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
        return m, sem

    overlap_rows = []
    M_n = int(n_neurons_fc)
    N_n = int(n_important_neurons)
    for cat in coarse_order:
        mask_cat = coarse_category == cat
        K_n = int(mask_cat.sum())
        x_n = int(np.logical_and(mask_cat, is_important_neuron).sum())
        expected_count = float(N_n * K_n / (M_n + 1e-12))
        observed_prop = float(x_n / (N_n + 1e-12))
        expected_prop = float(K_n / (M_n + 1e-12))
        enrichment_ratio = float(observed_prop / (expected_prop + 1e-12))
        log2_enrichment = float(np.log2((observed_prop + 1e-12) / (expected_prop + 1e-12)))
        p_over = float(hypergeom.sf(x_n - 1, M_n, K_n, N_n))
        p_under = float(hypergeom.cdf(x_n, M_n, K_n, N_n))
        p_two = float(min(1.0, 2.0 * min(p_over, p_under)))
        overlap_rows.append(
            {
                'overlap_category': cat,
                'n_neurons_category': K_n,
                'n_important_neurons_total': N_n,
                'observed_important_count': x_n,
                'expected_important_count': expected_count,
                'observed_important_fraction': observed_prop,
                'expected_important_fraction': expected_prop,
                'enrichment_ratio': enrichment_ratio,
                'log2_enrichment': log2_enrichment,
                'p_over': p_over,
                'p_under': p_under,
                'p_two_sided': p_two,
            }
        )

    overlap_enrichment_df = pd.DataFrame(overlap_rows)
    if len(overlap_enrichment_df) > 0:
        overlap_enrichment_df['p_fdr_bh'] = _bh_fdr(overlap_enrichment_df['p_two_sided'].to_numpy(dtype=float))
    else:
        overlap_enrichment_df['p_fdr_bh'] = []
    overlap_enrichment_csv_path = os.path.join(data_out_dir, 'neuron_overlap_enrichment.csv')
    overlap_enrichment_df.to_csv(overlap_enrichment_csv_path, index=False)

    selectivity_rows = []
    for level_type, arr_cat in [('coarse', coarse_category), ('detail', detail_category)]:
        for cat in sorted(pd.Index(arr_cat).unique().tolist()):
            mask = arr_cat == cat
            n_cat = int(mask.sum())
            n_imp_cat = int(np.logical_and(mask, is_important_neuron).sum())
            m_sel, sem_sel = _mean_sem(selectivity_index[mask])
            m_imp, sem_imp = _mean_sem(neuron_importance_norm[mask])
            m_proxy, sem_proxy = _mean_sem(neuron_importance_norm[mask])
            mask_act = np.logical_and(mask, np.isfinite(ablation_drop_actual))
            m_act, sem_act = _mean_sem(ablation_drop_actual[mask_act])
            selectivity_rows.append(
                {
                    'level_type': level_type,
                    'overlap_category': cat,
                    'n_neurons': n_cat,
                    'n_important_neurons': n_imp_cat,
                    'important_fraction': float(n_imp_cat / (n_cat + 1e-12)),
                    'mean_selectivity_index': m_sel,
                    'sem_selectivity_index': sem_sel,
                    'mean_decoder_importance': m_imp,
                    'sem_decoder_importance': sem_imp,
                    'mean_ablation_effect_proxy': m_proxy,
                    'sem_ablation_effect_proxy': sem_proxy,
                    'n_actual_ablation_eval': int(mask_act.sum()),
                    'mean_ablation_drop_actual': m_act,
                    'sem_ablation_drop_actual': sem_act,
                }
            )

    selectivity_by_overlap_df = pd.DataFrame(selectivity_rows)
    selectivity_by_overlap_csv_path = os.path.join(data_out_dir, 'neuron_selectivity_by_overlap.csv')
    selectivity_by_overlap_df.to_csv(selectivity_by_overlap_csv_path, index=False)

    plot_sel_df = selectivity_by_overlap_df[
        (selectivity_by_overlap_df['level_type'] == 'coarse')
    ].copy()
    plot_sel_df['overlap_category'] = pd.Categorical(plot_sel_df['overlap_category'], categories=coarse_order, ordered=True)
    plot_sel_df = plot_sel_df.sort_values('overlap_category', ignore_index=True)
    plot_enr_df = overlap_enrichment_df.copy()
    plot_enr_df['overlap_category'] = pd.Categorical(plot_enr_df['overlap_category'], categories=coarse_order, ordered=True)
    plot_enr_df = plot_enr_df.sort_values('overlap_category', ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=180)

    x = np.arange(plot_enr_df.shape[0])
    axes[0].bar(x, plot_enr_df['log2_enrichment'].to_numpy(dtype=float), color='#4C78A8', alpha=0.9)
    axes[0].axhline(0.0, color='#666666', linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_enr_df['overlap_category'].astype(str), rotation=20)
    axes[0].set_ylabel('log2 enrichment of decoder-important neurons')
    axes[0].set_title('Task6 Overlap Enrichment')
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)
    for i, row in plot_enr_df.iterrows():
        if np.isfinite(row['p_fdr_bh']) and float(row['p_fdr_bh']) < 0.05:
            y = float(row['log2_enrichment'])
            axes[0].text(i, y + (0.06 if y >= 0 else -0.10), '*', ha='center', va='center', fontsize=11)

    x2 = np.arange(plot_sel_df.shape[0])
    axes[1].bar(
        x2,
        plot_sel_df['mean_selectivity_index'].to_numpy(dtype=float),
        yerr=plot_sel_df['sem_selectivity_index'].to_numpy(dtype=float),
        color='#72B7B2',
        alpha=0.9,
        capsize=4,
    )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(plot_sel_df['overlap_category'].astype(str), rotation=20)
    axes[1].set_ylabel('Mean selectivity index')
    axes[1].set_title('Task6 Selectivity by Overlap')
    axes[1].grid(axis='y', linestyle='--', alpha=0.25)

    proxy_vals = plot_sel_df['mean_ablation_effect_proxy'].to_numpy(dtype=float)
    proxy_sem = plot_sel_df['sem_ablation_effect_proxy'].to_numpy(dtype=float)
    axes[2].bar(x2, proxy_vals, yerr=proxy_sem, color='#F58518', alpha=0.85, capsize=4, label='Ablation effect proxy')
    act_vals = plot_sel_df['mean_ablation_drop_actual'].to_numpy(dtype=float)
    act_sem = plot_sel_df['sem_ablation_drop_actual'].to_numpy(dtype=float)
    axes[2].errorbar(x2, act_vals, yerr=act_sem, fmt='o', color='#1A1A1A', capsize=3, label='Actual drop (subset eval)')
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(plot_sel_df['overlap_category'].astype(str), rotation=20)
    axes[2].set_ylabel('Mean ablation effect')
    axes[2].set_title('Task6 Ablation Link by Overlap')
    axes[2].grid(axis='y', linestyle='--', alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    task6_fig_path = os.path.join(fig_out_dir, 'neuron_decoder_linking_panel.png')
    fig.savefig(task6_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    task34_outputs = {
        'fc_decoder_summary_csv': fc_summary_csv,
        'fc_decoder_confusion_fig': fc_fig_path,
        'fc_stability_csv': edge_stability_csv_path,
        'fc_component_stability_csv': component_stability_csv_path,
        'fc_ablation_csv': edge_ablation_csv_path,
        'fc_component_ablation_csv': component_ablation_csv_path,
        'fc_task4_fig': task4_robust_fig_path,
        'fc_task4_component_fig': component_task4_fig_path,
        'fc_projection_top_edges_csv': top_edges_csv_path,
        'fc_projection_by_strength_decile_csv': decile_csv_path,
        'fc_projection_by_layer_pair_csv': depth_pair_csv_path,
        'fc_projection_strong_weak_match_csv': strong_weak_csv_path,
        'fc_projection_fig': task4_proj_fig_path,
        'fc_important_edge_decile_map_csv': important_edge_map_csv_path,
        'fc_edge_decile_enrichment_csv': enrichment_csv_path,
        'fc_edge_decile_enrichment_fig': enrichment_fig_path,
        'neuron_overlap_enrichment_csv': overlap_enrichment_csv_path,
        'neuron_selectivity_by_overlap_csv': selectivity_by_overlap_csv_path,
        'neuron_decoder_linking_detail_csv': neuron_detail_csv_path,
        'neuron_decoder_linking_panel_fig': task6_fig_path,
    }
    print('[*] Task3 summary:', fc_summary_csv)
    print('[*] Task3 confusion fig:', fc_fig_path)
    print('[*] Task4 stability csv:', edge_stability_csv_path)
    print('[*] Task4 component stability csv:', component_stability_csv_path)
    print('[*] Task4 ablation csv:', edge_ablation_csv_path)
    print('[*] Task4 component ablation csv:', component_ablation_csv_path)
    print('[*] Task4 figure:', task4_robust_fig_path)
    print('[*] Task4 component figure:', component_task4_fig_path)
    print('[*] Task4 projection top edges csv:', top_edges_csv_path)
    print('[*] Task4 projection by strength decile csv:', decile_csv_path)
    print('[*] Task4 projection by layer pair csv:', depth_pair_csv_path)
    print('[*] Task4 strong-weak match csv:', strong_weak_csv_path)
    print('[*] Task4 projection figure:', task4_proj_fig_path)
    print('[*] Task5 important edge decile map csv:', important_edge_map_csv_path)
    print('[*] Task5 decile enrichment csv:', enrichment_csv_path)
    print('[*] Task5 decile enrichment fig:', enrichment_fig_path)
    print('[*] Task6 overlap enrichment csv:', overlap_enrichment_csv_path)
    print('[*] Task6 selectivity by overlap csv:', selectivity_by_overlap_csv_path)
    print('[*] Task6 neuron linking detail csv:', neuron_detail_csv_path)
    print('[*] Task6 linking panel fig:', task6_fig_path)
    print('[*] Task4 layer source:', layer_source)
    task34_outputs


    # %%
    from IPython.display import Markdown, display

    if len(overlap_enrichment_df) > 0:
        best_overlap = overlap_enrichment_df.sort_values('log2_enrichment', ascending=False).iloc[0]
        best_overlap_txt = str(best_overlap['overlap_category'])
        best_overlap_enr = float(best_overlap['log2_enrichment'])
        best_overlap_p = float(best_overlap['p_two_sided'])
    else:
        best_overlap_txt = 'NA'
        best_overlap_enr = np.nan
        best_overlap_p = np.nan

    coarse_sel = selectivity_by_overlap_df[selectivity_by_overlap_df['level_type'] == 'coarse'].copy()
    if len(coarse_sel) > 0:
        best_sel = coarse_sel.sort_values('mean_selectivity_index', ascending=False).iloc[0]
        best_sel_txt = str(best_sel['overlap_category'])
        best_sel_val = float(best_sel['mean_selectivity_index'])
    else:
        best_sel_txt = 'NA'
        best_sel_val = np.nan

    lines = [
        "### Task6 Result Snapshot",
        f"- Most enriched overlap category: **{best_overlap_txt}** (log2 enrichment={best_overlap_enr:.4f}, p={best_overlap_p:.3g})",
        f"- Highest selectivity category (coarse): **{best_sel_txt}** (mean SI={best_sel_val:.4f})",
        f"- Overlap enrichment table: `{overlap_enrichment_csv_path}`",
        f"- Selectivity table: `{selectivity_by_overlap_csv_path}`",
        f"- Linking panel: `{task6_fig_path}`",
    ]
    display(Markdown('\\n'.join(lines)))


    # %% [markdown]
    # <a id="sec-13"></a>
    # ## 13. Population-Pattern Shuffle Dependence
    # 
    # Run neuron-wise trial permutation analysis and save shuffle outputs.
    # 
    # 

    # %%
    import importlib
    import population_shuffle_analysis as psa
    importlib.reload(psa)
    from population_shuffle_analysis import run_population_pattern_shuffle_analysis



    # %%
    shuffle_result = run_population_pattern_shuffle_analysis(
        segments_spi=segments_spi,
        labels_spi=labels_spi,
        data_out_dir=data_out_dir,
        fig_out_dir=fig_out_dir,
        mouse_id=current_mouse_id if 'current_mouse_id' in globals() else 'mouse_unknown',
        label_names=label_names if 'label_names' in globals() else None,
        class_colors=class_colors if 'class_colors' in globals() else None,
        rr_neurons_spi=rr_neurons_spi if 'rr_neurons_spi' in globals() else None,
        shuffle_repeats=200,
        shuffle_seed=20260328,
        shuffle_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        response_window=slice(10, 13),
        rsm_bins=50,
        show_live_plots=True,
        verbose=True,
    )
    shuffle_result['outputs']



    # %% [markdown]
    # ### Generated Outputs
    # 
    # - `population_pattern_shuffle_manifest.csv`
    # - `group_corr_shuffle_long.csv`
    # - `group_corr_decile_shuffle_long.csv`
    # - `group_rsm_shuffle_long.csv`
    # - `group_shuffle_delta_long.csv`
    # - `group_shuffle_dose_response_long.csv`
    # - `group_allocation_shuffle_long.csv` (includes `pr_mean/pr_std/pr_norm_mean`)
    # - `group_shuffle_effect_stats.csv`
    # - `group_shuffle_delta_stats.csv` (includes `Delta_PR_Mean`)
    # - `group_shuffle_condition_summary.csv`
    # - `group_shuffle_condition_stats.csv`
    # - `group_shuffle_sync_contribution.csv`
    # - `group_shuffle_sync_contribution_repeats.csv`
    # 
    # 

    # %% [markdown]
    # <a id="sec-14"></a>
    # ## 14. Shuffled Condition Differences
    # 
    # Focus on condition differences inside shuffled surrogates (including PR).
    # 
    # 

    # %%
    import pandas as pd
    from IPython.display import Image, display

    shuf_summary = shuffle_result['tables']['shuffled_condition_summary'].copy()
    shuf_stats = shuffle_result['tables']['shuffled_condition_stats'].copy()

    print('[*] Shuffled condition summary (mean ± sem)')
    display(shuf_summary.sort_values(['metric', 'condition']).reset_index(drop=True))

    print('[*] Shuffled condition tests (Friedman + pairwise Wilcoxon)')
    display(shuf_stats.sort_values(['metric', 'test', 'comparison']).reset_index(drop=True))

    fig_path = shuffle_result['figures'].get('shuffled_condition_diff', None)
    if fig_path is not None:
        display(Image(filename=fig_path))



    # %% [markdown]
    # <a id="sec-15"></a>
    # ## 15. Final Integrated Export
    # 
    # Write final JSON/CSV/Markdown report with both baseline analysis and shuffle analysis.
    # 
    # 

    # %%
    import json
    import os
    import pandas as pd
    from datetime import datetime


    def export_mouse_results_integrated(
        mouse_id,
        df_entropy,
        df_corr_strength,
        df_corr_deciles,
        participants_dict,
        save_dir,
        fig_dir,
        shuffle_res=None,
        geometry_res=None,
    ):
        os.makedirs(save_dir, exist_ok=True)

        entropy_records = df_entropy.to_dict(orient='records')
        corr_records = df_corr_strength.to_dict(orient='records')
        corr_decile_records = df_corr_deciles.to_dict(orient='records')

        shuffle_outputs = {}
        shuffle_figs = {}
        shuffle_effect_records = []
        shuffle_cond_summary_records = []
        shuffle_cond_stats_records = []
        shuffle_sync_contrib_records = []
        geometry_outputs = {}
        geometry_figs = {}
        geometry_condition_records = []
        geometry_pairwise_records = []
        geometry_model_compare_records = []
        if shuffle_res is not None:
            shuffle_outputs = shuffle_res.get('outputs', {})
            shuffle_figs = shuffle_res.get('figures', {})
            tables = shuffle_res.get('tables', {})
            if isinstance(tables.get('effect_stats', None), pd.DataFrame):
                shuffle_effect_records = tables['effect_stats'].to_dict(orient='records')
            if isinstance(tables.get('shuffled_condition_summary', None), pd.DataFrame):
                shuffle_cond_summary_records = tables['shuffled_condition_summary'].to_dict(orient='records')
            if isinstance(tables.get('shuffled_condition_stats', None), pd.DataFrame):
                shuffle_cond_stats_records = tables['shuffled_condition_stats'].to_dict(orient='records')
            if isinstance(tables.get('sync_contribution', None), pd.DataFrame):
                shuffle_sync_contrib_records = tables['sync_contribution'].to_dict(orient='records')
        if geometry_res is not None:
            geometry_outputs = geometry_res.get('outputs', {})
            geometry_figs = geometry_res.get('figures', {})
            tables = geometry_res.get('tables', {})
            if isinstance(tables.get('condition_level', None), pd.DataFrame):
                geometry_condition_records = tables['condition_level'].to_dict(orient='records')
            if isinstance(tables.get('condition_pairwise', None), pd.DataFrame):
                geometry_pairwise_records = tables['condition_pairwise'].to_dict(orient='records')
            if isinstance(tables.get('model_compare', None), pd.DataFrame):
                geometry_model_compare_records = tables['model_compare'].to_dict(orient='records')

        mouse_data = {
            'mouse_id': mouse_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'entropy_metrics': entropy_records,
            'network_correlation': corr_records,
            'network_correlation_deciles': corr_decile_records,
            'rr_participants_ratio': participants_dict,
            'shuffle_outputs': shuffle_outputs,
            'shuffle_effect_stats': shuffle_effect_records,
            'shuffle_condition_summary': shuffle_cond_summary_records,
            'shuffle_condition_stats': shuffle_cond_stats_records,
            'shuffle_sync_contribution': shuffle_sync_contrib_records,
            'geometry_outputs': geometry_outputs,
            'geometry_condition_level': geometry_condition_records,
            'geometry_condition_pairwise': geometry_pairwise_records,
            'geometry_model_compare': geometry_model_compare_records,
        }

        json_path = os.path.join(save_dir, f'{mouse_id}_statistics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mouse_data, f, indent=4)

        decile_csv_path = os.path.join(save_dir, f'{mouse_id}_correlation_deciles.csv')
        df_corr_deciles.to_csv(decile_csv_path, index=False)

        print(f'[*] Structured statistics saved to: {json_path}')
        print(f'[*] Decile-level correlation data saved to: {decile_csv_path}')

        md_path = os.path.join(save_dir, f'{mouse_id}_analysis_report.md')

        def dicts_to_md_table(dict_list):
            if not dict_list:
                return ''
            headers = list(dict_list[0].keys())
            header_row = '| ' + ' | '.join(headers) + ' |'
            sep_row = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
            rows = []
            for d in dict_list:
                formatted_vals = [f'{v:.4f}' if isinstance(v, float) else str(v) for v in d.values()]
                rows.append('| ' + ' | '.join(formatted_vals) + ' |')
            return '\n'.join([header_row, sep_row] + rows)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f'# Mouse Neural Activity Report - {mouse_id}\n\n')
            f.write(f'**Generated At**: {mouse_data["timestamp"]}\n\n')

            f.write('## 1. RSM and Shannon Entropy\n\n')
            f.write('Metrics for representation stability and variability across stimulus conditions.\n\n')
            f.write(dicts_to_md_table(entropy_records) + '\n\n')

            f.write('## 2. Pairwise Network Correlation Summary\n\n')
            f.write('Difference between lowest and highest correlation tails.\n\n')
            f.write(dicts_to_md_table(corr_records) + '\n\n')

            f.write('## 2.1 Decile-wise Correlation Strength (Every 10%)\n\n')
            f.write('Connectivity values are sorted by raw correlation and split into 10 equal bins.\n\n')
            f.write(dicts_to_md_table(corr_decile_records[:30]) + '\n\n')

            f.write('## 3. RR Participant Ratio\n\n')
            f.write('Response ratio of class-specific RR neurons relative to other RR neurons.\n\n')
            participants_rows = [{'Condition': k, 'Response_Ratio': v} for k, v in participants_dict.items()]
            f.write(dicts_to_md_table(participants_rows) + '\n\n')

            if shuffle_res is not None:
                f.write('## 4. Shuffle: Original vs Shuffled Effects\n\n')
                f.write(dicts_to_md_table(shuffle_effect_records) + '\n\n')

                f.write('## 5. Shuffle: Condition Differences within Surrogates\n\n')
                f.write('### 5.1 Summary (mean/sem)\n\n')
                f.write(dicts_to_md_table(shuffle_cond_summary_records) + '\n\n')
                f.write('### 5.2 Friedman + Wilcoxon\n\n')
                f.write(dicts_to_md_table(shuffle_cond_stats_records) + '\n\n')

                f.write('## 6. Shuffle: Synchrony Contribution (Random - Coherent)\n\n')
                f.write(dicts_to_md_table(shuffle_sync_contrib_records) + '\n\n')

                f.write('## 7. Shuffle Output Files\n\n')
                for k, v in shuffle_outputs.items():
                    f.write(f'- {k}: `{v}`\n')
                f.write('\n')

                f.write('## 8. Shuffle Figures\n\n')
                for k, v in shuffle_figs.items():
                    rel = os.path.basename(v)
                    f.write(f'- {k}: `./figures/{rel}`\n')
                f.write('\n')

            if geometry_res is not None:
                f.write('## 9. Geometry: Condition-level Metrics\n\n')
                f.write(dicts_to_md_table(geometry_condition_records) + '\n\n')
                f.write('## 10. Geometry: Pairwise Bootstrap Tests\n\n')
                f.write(dicts_to_md_table(geometry_pairwise_records) + '\n\n')
                f.write('## 11. Geometry: Model Compare\n\n')
                f.write(dicts_to_md_table(geometry_model_compare_records) + '\n\n')
                f.write('## 12. Geometry Output Files\n\n')
                for k, v in geometry_outputs.items():
                    f.write(f'- {k}: `{v}`\n')
                f.write('\n')
                f.write('## 13. Geometry Figures\n\n')
                for k, v in geometry_figs.items():
                    rel = os.path.basename(v)
                    f.write(f'- {k}: `./figures/{rel}`\n')
                f.write('\n')

            f.write('## 14. Figure Index\n\n')
            f.write('- Preference-sorted heatmap: `./figures/neural_patterns_preference_sorted.png`\n')
            f.write('- Hierarchical clustermap: `./figures/neural_patterns_clustermap.png`\n')
            f.write('- RSM similarity distribution: `./figures/similarity_distribution.png`\n')
            f.write('- Pairwise correlation summary: `./figures/pairwise_correlation.png`\n')
            f.write('- Decile-wise line chart: `./figures/pairwise_correlation_deciles_line.png`\n')

        print(f'[*] Markdown report generated: {md_path}')
        return {"json_path": json_path, "md_path": md_path}


    geometry_result = None
    if 'df_geometry' in globals() and isinstance(df_geometry, pd.DataFrame):
        geometry_outputs = {
            "condition_level_long": geometry_csv if 'geometry_csv' in globals() else "",
            "condition_pairwise": geometry_pairwise_csv if 'geometry_pairwise_csv' in globals() else "",
            "condition_stats_md": geometry_stats_md if 'geometry_stats_md' in globals() else "",
            "rsm_lmm_summary_md": geom_rsm_md if 'geom_rsm_md' in globals() else "",
            "model_compare_csv": geom_model_compare_csv if 'geom_model_compare_csv' in globals() else "",
            "allocation_lmm_summary_md": geom_alloc_md if 'geom_alloc_md' in globals() else "",
            "vs_dimensionality_csv": geom_vs_dim_csv if 'geom_vs_dim_csv' in globals() else "",
        }
        geometry_figs = {
            "example_pc_scatter": g1_png if 'g1_png' in globals() else "",
            "angle_condition": os.path.join(fig_out_dir, "geometry_angle_condition.png"),
            "orth_parallel_condition": os.path.join(fig_out_dir, "geometry_orth_parallel_condition.png"),
            "angle_vs_rsm": os.path.join(fig_out_dir, "geometry_angle_vs_rsm.png"),
            "ratio_vs_rsm": os.path.join(fig_out_dir, "geometry_ratio_vs_rsm.png"),
        }
        geometry_result = {
            "outputs": {k: v for k, v in geometry_outputs.items() if isinstance(v, str) and len(v) > 0},
            "figures": {k: v for k, v in geometry_figs.items() if isinstance(v, str) and len(v) > 0},
            "tables": {
                "condition_level": df_geometry if 'df_geometry' in globals() else pd.DataFrame(),
                "condition_pairwise": df_geometry_pairwise if 'df_geometry_pairwise' in globals() else pd.DataFrame(),
                "model_compare": df_geometry_model_compare if 'df_geometry_model_compare' in globals() else pd.DataFrame(),
            },
        }

    current_mouse_id = os.path.basename(os.path.normpath(data_path))
    final_export = export_mouse_results_integrated(
        mouse_id=current_mouse_id,
        df_entropy=df_entropy,
        df_corr_strength=df_corr_strength,
        df_corr_deciles=df_corr_deciles,
        participants_dict=participants,
        save_dir=data_out_dir,
        fig_dir=fig_out_dir,
        shuffle_res=shuffle_result if 'shuffle_result' in globals() else None,
        geometry_res=geometry_result,
    )
    final_export
