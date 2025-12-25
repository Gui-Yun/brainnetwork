import numpy as np
import networkx as nx

def reshape_segments(segments, time_range=None):
    """Convert segments to shape (neurons, samples) for correlation analysis."""
    segments = np.asarray(segments)
    if segments.ndim != 3:
        raise ValueError("segments must be shaped (trials, neurons, timepoints)")

    if time_range is not None:
        start, end = time_range
        segments = segments[:, :, start:end]

    _, neurons, _ = segments.shape
    return segments.transpose(1, 0, 2).reshape(neurons, -1)
def _balance_trials(segments, labels, classes, random_state=0):
    """Down-sample trials so each class in `classes` has the same count."""
    labels = np.asarray(labels)
    segments = np.asarray(segments)
    classes = np.array(sorted(set(np.atleast_1d(classes))), dtype=labels.dtype)

    counts = {cls: int((labels == cls).sum()) for cls in classes}
    if any(n == 0 for n in counts.values()):
        return segments, labels  # skip balancing if any class missing

    min_n = min(counts.values())
    rng = np.random.default_rng(random_state)
    keep_indices = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size <= min_n:
            keep_indices.extend(cls_idx.tolist())
        else:
            keep_indices.extend(rng.choice(cls_idx, size=min_n, replace=False).tolist())

    keep_indices = sorted(keep_indices)
    return segments[keep_indices], labels[keep_indices]


# %% ========= core functions =========
# =========== 计算相关矩阵 ============
def compute_correlation_matrix(segments, labels=None, class_filter=None, time_range=None, zscore=True, balance=True, random_state=0):
    """Return neuron x neuron correlation matrix with optional trial/time selection."""
    segments = np.asarray(segments)

    if class_filter is not None:
        if labels is None:
            raise ValueError("labels are required when class_filter is set")
        labels = np.asarray(labels)
        if np.isscalar(class_filter):
            mask = labels == class_filter
        else:
            mask = np.isin(labels, class_filter)
        if mask.sum() == 0:
            raise ValueError(f"No trials found for class filter {class_filter}")
        segments = segments[mask]
        labels = labels[mask]
        if balance:
            segments, labels = _balance_trials(segments, labels, classes=np.unique(labels), random_state=random_state)

    elif balance and labels is not None:
        labels = np.asarray(labels)
        segments, labels = _balance_trials(segments, labels, classes=np.unique(labels), random_state=random_state)

    data = reshape_segments(segments, time_range=time_range)

    if zscore:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-9
        data = (data - mean) / std

    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
# ========= 构建相关网络 =========
def build_correlation_graph(corr_matrix, threshold=None, top_k=None, weighted=True, absolute=True, weight_attr="weight"):
    """Convert correlation matrix to a NetworkX graph."""
    corr = np.asarray(corr_matrix)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr_matrix must be square")

    n_nodes = corr.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            value = float(corr[i, j])
            score = abs(value) if absolute else value
            edges.append((score, i, j, value))

    if threshold is not None:
        edges = [edge for edge in edges if edge[0] >= threshold]

    edges.sort(key=lambda e: e[0], reverse=True)
    total_edges = len(edges)
    if top_k is not None:
        edges = edges[: int(top_k * total_edges)]

    for _, i, j, value in edges:
        if weighted:
            graph.add_edge(i, j, **{weight_attr: value})
        else:
            graph.add_edge(i, j)

    return graph
# ========= 相关网络统计 =========
def correlation_network_summary(graph):
    """Return basic statistics for the correlation graph."""
    if graph.number_of_nodes() == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "mean_degree": 0.0,
            "largest_component": 0,
            "avg_clustering": 0.0,
            "global_efficiency": 0.0,
            "local_efficiency": 0.0,
            "transitivity": 0.0,
        }

    degrees = [deg for _, deg in graph.degree()]
    components = [len(c) for c in nx.connected_components(graph)]
    largest_component = max(components) if components else 0
    avg_clustering = nx.average_clustering(graph, weight="weight") if graph.number_of_nodes() > 1 else 0.0
    global_eff = nx.global_efficiency(graph)
    local_eff = nx.local_efficiency(graph) if graph.number_of_nodes() > 1 else 0.0
    trans = nx.transitivity(graph)

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "largest_component": largest_component,
        "avg_clustering": float(avg_clustering),
        "global_efficiency": float(global_eff),
        "local_efficiency": float(local_eff),
        "transitivity": float(trans),
    }
# ========= 构建相关网络的流水线 =========
def construct_correlation_network(segments, labels=None, class_filter=None, time_range=None, zscore=True, threshold=0.5, top_k=None, weighted=True, absolute=True, balance=False, random_state=0):
    """Pipeline: compute correlation matrix, build graph, and summarise."""

    corr_matrix = compute_correlation_matrix(
        segments,
        labels=labels,
        class_filter=class_filter,
        time_range=time_range,
        zscore=zscore,
        balance=balance,
        random_state=random_state,
    )

    graph = build_correlation_graph(
        corr_matrix,
        threshold=threshold,
        top_k=top_k,
        weighted=weighted,
        absolute=absolute,
    )

    summary = correlation_network_summary(graph)
    return corr_matrix, graph, summary

def create_supernodes(segments, neuron_pos, n_supernodes=150):
    """Cluster neurons into supernodes using k-means based on spatial positions."""
    from sklearn.cluster import KMeans

    neuron_pos = np.asarray(neuron_pos)
    if neuron_pos.shape[0] != 2:
        raise ValueError("neuron_pos must be of shape (2, n_neurons)")

    n_neurons = neuron_pos.shape[1]
    if n_supernodes >= n_neurons:
        print("Warning: n_supernodes >= n_neurons, skipping supernode creation.")
        return segments, neuron_pos

    kmeans = KMeans(n_clusters=n_supernodes, random_state=42)
    labels = kmeans.fit_predict(neuron_pos.T)

    new_segments = np.zeros((segments.shape[0], n_supernodes, segments.shape[2]), dtype=segments.dtype)

    # 使用均值聚合每个超节点的信号
    for i in range(n_supernodes):
        cluster_indices = np.where(labels == i)[0]
        if cluster_indices.size > 0:
            new_segments[:, i, :] = segments[:, cluster_indices, :].mean(axis=1)

    new_neuron_pos = kmeans.cluster_centers_.T
    return new_segments, new_neuron_pos
# =========== 计算网络指标 =========
def compute_network_metrics(graph):
    """Compute degree, clustering coefficient, and eigenvector centrality for each node."""
    degrees = dict(graph.degree())
    clustering = nx.clustering(graph)
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    degree_values = np.array(list(degrees.values()))
    clustering_values = np.array(list(clustering.values()))
    eigenvector_values = np.array(list(eigenvector.values()))
    return degree_values, clustering_values, eigenvector_values


def compute_betweenness_stats(graph, weight="weight"):
    """Betweenness centrality distribution and summary stats."""
    if graph.number_of_nodes() == 0:
        return np.array([]), {"mean": 0.0, "median": 0.0, "std": 0.0}
    bc = np.array(list(nx.betweenness_centrality(graph, weight=weight).values()), dtype=float)
    return bc, {"mean": float(np.mean(bc)), "median": float(np.median(bc)), "std": float(np.std(bc))}
# =========== 计算差异网络 ===========
def compute_network_metrics_by_class(segments, labels, neuron_pos=None, n_bootstrap=100, do_bootstrap=0):
    """Compute network metrics for each class with bootstrap variance estimation."""

    nx_result = {}
    for cls in np.unique(labels):
        # ???????? trial
        cls_indices = np.where(labels == cls)[0]
        n_trials = len(cls_indices)

        # ????
        corr_matrix, corr_graph, summary = construct_correlation_network(
            segments,
            labels=labels,
            class_filter=cls,
            time_range=None,
            zscore=False,
            threshold=None,
            top_k=0.05,
            weighted=False,
            absolute=False,
            balance=True,
            random_state=0,
        )
        # ?????/????
        efficiency = nx.global_efficiency(corr_graph)
        modularity = nx.algorithms.community.modularity(
            corr_graph,
            nx.algorithms.community.greedy_modularity_communities(corr_graph),
        )
        # ?????
        bc_values, bc_stats = compute_betweenness_stats(corr_graph, weight='weight')
        # ??????
        # lr_stats = long_range_edge_stats(neuron_pos, corr_matrix, graph=corr_graph, threshold=500.0) if neuron_pos is not None else {
        #     'threshold': 500.0,
        #     'total_edges': 0,
        #     'long_edges': 0,
        #     'long_fraction': 0.0,
        #     'long_mean_weight': 0.0,
        #     'long_mean_distance': 0.0,
        # }

        # Bootstrap??
        bootstrap_metrics = {
            "largest_component": [],
            "avg_clustering": [],
            "global_efficiency": [],
            "transitivity": [],
            "efficiency": [],
            "modularity": []
        }

        if do_bootstrap:
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(cls_indices, size=n_trials, replace=True)
                boot_segments = segments[bootstrap_indices]
                boot_labels = np.zeros(n_trials, dtype=labels.dtype)

                try:
                    _, boot_graph, boot_summary = construct_correlation_network(
                        boot_segments,
                        labels=boot_labels,
                        class_filter=0,
                        time_range=None,
                        zscore=False,
                        threshold=None,
                        top_k=0.05,
                        weighted=False,
                        absolute=False,
                        balance=False,
                        use_cache=False,
                    )

                    boot_efficiency = nx.global_efficiency(boot_graph)
                    boot_modularity = nx.algorithms.community.modularity(
                        boot_graph,
                        nx.algorithms.community.greedy_modularity_communities(boot_graph),
                    )

                    bootstrap_metrics["largest_component"].append(boot_summary["largest_component"])
                    bootstrap_metrics["avg_clustering"].append(boot_summary["avg_clustering"])
                    bootstrap_metrics["global_efficiency"].append(boot_summary["global_efficiency"])
                    bootstrap_metrics["transitivity"].append(boot_summary["transitivity"])
                    bootstrap_metrics["efficiency"].append(boot_efficiency)
                    bootstrap_metrics["modularity"].append(boot_modularity)
                except Exception:
                    continue

        summary_std = {
            metric: np.std(values) if len(values) > 0 else 0.0
            for metric, values in bootstrap_metrics.items()
        }

        # ??
        nx_result[cls] = {
            "corr_matrix": corr_matrix,
            "corr_graph": corr_graph,
            "summary": summary,
            "summary_std": summary_std,  # ???
            "bootstrap_samples": bootstrap_metrics,  # ?? bootstrap ??
            "efficiency": efficiency,
            "modularity": modularity,
            "betweenness": bc_values,
            "betweenness_stats": bc_stats,
            # "long_range": lr_stats,
        }
    return nx_result