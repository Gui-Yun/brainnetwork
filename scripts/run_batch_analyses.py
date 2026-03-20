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


# %% [markdown]
# <a id="sec-1"></a>
# ## 1. Dataset and Output Paths
# 
# Select one mouse dataset and initialize output directories.
# 

# %%
base_dir = "/beegfs_hdd/data/nfs_share/users/guiyun/nishome/Micedata/"
data_paths = ["M21_1107", "M71_1024", "M73_1128", "M77_1031", "M77_1107", "M78_1017", "M79_1128", "M91_1017"]
for idx, path in enumerate(data_paths):
    data_path = base_dir + data_paths[idx] # 'M77_1031'
    print(f"Processing data from: {data_path}")
    save_dir = "./results/" + data_paths[idx] 

    data_out_dir = os.path.join(save_dir, "data")
    fig_out_dir = os.path.join(save_dir, "figures")

    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(fig_out_dir, exist_ok=True)

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
    current_mouse_id = data_paths[idx]

    export_mouse_results(
        mouse_id=current_mouse_id,
        df_entropy=df_entropy,
        df_corr_strength=df_corr_strength,
        df_corr_deciles=df_corr_deciles,
        participants_dict=participants,
        save_dir=data_out_dir,
        fig_dir=fig_out_dir,
    )




