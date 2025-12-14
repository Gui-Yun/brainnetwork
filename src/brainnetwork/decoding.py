import numpy as np
from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# ========== parameters ===========
t_stimulus = 10
l_stimulus = 20
l_trials = 50
# ========== functions ===========
def _split_by_class(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples must match labels")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("Fisher information currently supports exactly two classes")
    class_a, class_b = classes
    group_a = X[y == class_a]
    group_b = X[y == class_b]
    if group_a.size == 0 or group_b.size == 0:
        raise ValueError("Each class must contain at least one sample")
    return group_a, group_b


def fisher_information_univariate(group_a, group_b, epsilon=1e-6):
    """计算单变量Fisher信息"""
    mean_diff = np.mean(group_a, axis=0) - np.mean(group_b, axis=0)
    var_sum = np.var(group_a, axis=0, ddof=1) + np.var(group_b, axis=0, ddof=1)
    return (mean_diff ** 2) / (var_sum + epsilon)


def fisher_information_multivariate(group_a, group_b, shrinkage=1e-3):
    """计算多变量Fisher信息"""
    def _cov(data):
        if data.shape[0] <= 1:
            return np.zeros((data.shape[1], data.shape[1]))
        return np.cov(data, rowvar=False, ddof=1)
    cov_a = _cov(group_a)
    cov_b = _cov(group_b)
    pooled = ((group_a.shape[0] - 1) * cov_a + (group_b.shape[0] - 1) * cov_b)
    denom = max(group_a.shape[0] + group_b.shape[0] - 2, 1)
    pooled = pooled / denom
    pooled += shrinkage * np.eye(pooled.shape[0])
    mean_diff = np.mean(group_a, axis=0) - np.mean(group_b, axis=0)
    try:
        inv_cov = np.linalg.inv(pooled)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(pooled)
    return float(mean_diff @ inv_cov @ mean_diff)


def Fisher_information(X, y, epsilon=1e-6, mode="univariate", shrinkage=1e-3):
    """
    计算单变量Fisher信息或多变量Fisher信息
    mode: 'univariate' 单变量
          'multivariate' 多变量
    """
    mode = (mode or "univariate").lower()
    if mode not in {"univariate", "multivariate"}:
        raise ValueError(f"无效的 mode: {mode}")
    group_a, group_b = _split_by_class(X, y)

    if mode == "univariate":
        return fisher_information_univariate(group_a, group_b, epsilon=epsilon)
    return fisher_information_multivariate(group_a, group_b, shrinkage=shrinkage)


# ========== 分类准确率 ===========
def classify_by_timepoints(segments, labels, window_size=5, step_size=1):
    """
    ????????????????????????
    """
    n_trials, _, n_timepoints = segments.shape
    assert len(labels) == n_trials, "??????????"
    time_points = []
    accuracies = []
    accuracy_std = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_folds = cv.get_n_splits()
    for t in range(n_timepoints):
        timepoint_data = segments[:, :, t]
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(timepoint_data)
        clf = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
        scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')
        accuracies.append(scores.mean())
        accuracy_std.append(scores.std(ddof=1))
        time_points.append((t - t_stimulus) / 4)
    return np.array(accuracies), np.array(time_points), np.array(accuracy_std), n_folds

# ========== Fisher信息 ===========
def FI_by_timepoints(segments, labels, reduction="mean", epsilon=1e-6, mode="univariate", shrinkage=1e-3):
    """
    计算每个时间点上的Fisher信息
    参数:
    segments: (trials, neurons, timepoints)
    labels: 每个trial的标签
    reduction: 'mean' / 'sum' / 'none'，默认为'mean'，仅在mode='univariate'时有效
    mode: 'univariate'，表示单变量分析，或'multivariate'，表示多变量分析
    fisher_dict: dict，包含以下键值对:
    time_points: 每个时间点的时间戳
    """
    segments = np.asarray(segments)
    labels = np.asarray(labels)
    mode = (mode or "univariate").lower()
    if segments.ndim != 3:
        raise ValueError("segments 必须是 (trials, neurons, timepoints) 形式的三维数组?")
    if mode not in {"univariate", "multivariate"}:
        raise ValueError(f"无效的 mode: {mode}")
    
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Fisher information currently supports exactly two classes")
    
    n_trials, n_neurons, n_timepoints = segments.shape
    time_points = np.array([(t - t_stimulus) / 4 for t in range(n_timepoints)])
    
    def _reduce(matrix):
        if reduction in (None, "none"):
            return matrix
        if reduction == "mean":
            return matrix.mean(axis=1)
        if reduction == "sum":
            return matrix.sum(axis=1)
        raise ValueError(f"无效的 reduction 方法: {reduction}")
   
    if mode == "multivariate" and reduction not in (None, "none"):
        raise ValueError("mode='multivariate' does not support reduction aggregation")
   
    fisher_dict = {}
    for pair in combinations(unique_labels, 2):
        mask = np.isin(labels, pair)
        if mask.sum() == 0:
            continue
        pair_segments = segments[mask]
        pair_labels = labels[mask]

        if mode == "univariate":
            fi_matrix = np.zeros((n_timepoints, n_neurons))
            for t in range(n_timepoints):
                fi_matrix[t] = Fisher_information(pair_segments[:, :, t], pair_labels, epsilon=epsilon, mode=mode)
            fisher_dict[pair] = _reduce(fi_matrix)
        else:
            fi_series = np.zeros(n_timepoints)
            
            # 定义PCA。基于当前pair的样本数
            n_samples_pair = pair_labels.shape[0]
            n_components_target = int(n_samples_pair / 2) - 1
            if n_components_target < 1:
                n_components_target = 1
            _pca = PCA(n_components=n_components_target)

            for t in range(n_timepoints):
                # 先从 (n_samples, n_neurons) 降维到 (n_samples, n_components)
                X_reduced = _pca.fit_transform(pair_segments[:, :, t])
                
                # 在降维后的数据上计算FI
                fi_series[t] = Fisher_information(
                    X_reduced, 
                    pair_labels, 
                    epsilon=epsilon, 
                    mode="multivariate", 
                    shrinkage=shrinkage
                )
            fisher_dict[pair] = fi_series
    return fisher_dict, time_points
# ========== Fisher信息 ===========
def FI_by_timepoints_v2(segments, labels, reduction="mean", epsilon=1e-6, 
                        mode="univariate", shrinkage=1e-3, 
                        balance_samples=True, random_state=42):
    """
    计算每个时间点上的Fisher信息
    v3版: 
    1. 修正了 multivariate 模式下的 PCA 降维
    2. 增加了 balance_samples 选项以通过下采样平衡类别
    """
    
    segments = np.asarray(segments)
    labels = np.asarray(labels)
    mode = (mode or "univariate").lower()
    
    # ... (所有前置检查代码保持不变) ...
    if segments.ndim != 3:
        raise ValueError("segments 必须是 (trials, neurons, timepoints) 形式的三维数组?")
    if mode not in {"univariate", "multivariate"}:
        raise ValueError(f"无效的 mode: {mode}")
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Fisher information currently supports exactly two classes")
        
    n_trials, n_neurons, n_timepoints = segments.shape
    time_points = np.array([(t - t_stimulus) / 4 for t in range(n_timepoints)])
    
    def _reduce(matrix):
        if reduction in (None, "none"):
            return matrix
        if reduction == "mean":
            return matrix.mean(axis=1)
        if reduction == "sum":
            return matrix.sum(axis=1)
        raise ValueError(f"无效的 reduction 方法: {reduction}")
        
    if mode == "multivariate" and reduction not in (None, "none"):
        raise ValueError("mode='multivariate' does not support reduction aggregation")
        
    # 初始化一个可复现的随机数生成器
    rng = np.random.default_rng(random_state)
    fisher_dict = {}
    
    for pair in combinations(unique_labels, 2):
        mask = np.isin(labels, pair)
        if mask.sum() == 0:
            continue
            
        pair_segments_all = segments[mask]
        pair_labels_all = labels[mask]
     
        # --- *** 新增：样本均衡逻辑 *** ---
        if balance_samples:
            class_a, class_b = pair
            # 1. 找到两个类别的索引 (相对于 'pair_labels_all')
            indices_a = np.where(pair_labels_all == class_a)[0]
            indices_b = np.where(pair_labels_all == class_b)[0]
            # 2. 找到最小的样本数
            n_min = min(len(indices_a), len(indices_b))
            if n_min == 0:
                continue # 如果有一个类没有样本，跳过
            # 3. 从每个类别中随机抽取 n_min 个样本
            balanced_indices_a = rng.choice(indices_a, size=n_min, replace=False)
            balanced_indices_b = rng.choice(indices_b, size=n_min, replace=False)
            
            # 4. 合并索引，得到均衡后的数据
            balanced_indices = np.concatenate([balanced_indices_a, balanced_indices_b])
            # 5. 更新用于后续计算的变量
            pair_segments = pair_segments_all[balanced_indices]
            pair_labels = pair_labels_all[balanced_indices]
        else:
            # 如果不均衡，就使用所有数据
            pair_segments = pair_segments_all
            pair_labels = pair_labels_all
        # --- *** 样本均衡结束 *** ---

        if mode == "univariate":
            # --- 单变量模式 (无需PCA) ---
            fi_matrix = np.zeros((n_timepoints, n_neurons))
            for t in range(n_timepoints):
                fi_matrix[t] = Fisher_information(
                    pair_segments[:, :, t], 
                    pair_labels, 
                    epsilon=epsilon, 
                    mode="univariate"
                )
            fisher_dict[pair] = _reduce(fi_matrix)
            
        else:
            # --- 多变量模式 (需要PCA) ---
            fi_series = np.zeros(n_timepoints)
            
            # 定义PCA。基于均衡后（或原始）的样本数
            n_samples_pair = pair_labels.shape[0]
            n_components_target = int(n_samples_pair / 2) - 1
            if n_components_target < 1:
                n_components_target = 1
                
            # 检查 n_features 是否会小于 n_components_target
            # (虽然在这里 n_neurons 很大，但以防万一)
            if n_neurons < n_components_target:
                n_components_target = n_neurons
                
            _pca = PCA(n_components=n_components_target)
            
            for t in range(n_timepoints):
                X_t = pair_segments[:, :, t]
                
                # 检查此时间点的方差是否为0 (在某些预处理中可能发生)
                if np.var(X_t) < 1e-10:
                    fi_series[t] = 0.0
                    continue
                
                X_reduced = _pca.fit_transform(X_t)
                
                fi_series[t] = Fisher_information(
                    X_reduced, 
                    pair_labels, 
                    epsilon=epsilon, 
                    mode="multivariate", 
                    shrinkage=shrinkage
                )
            fisher_dict[pair] = fi_series
            
    return fisher_dict, time_points

# ========== Fisher信息随着神经元数量增加的变化 ===========
def FI_by_neuron_count(segments, labels, n_neurons_step = 5):
    """计算Fisher信息随着神经元数量增加的变化"""
    neuron_counts = np.arange(n_neurons_step, segments.shape[1] + 1, n_neurons_step)
    fi_values = []
    for n_neurons in neuron_counts:
        selected_segments = segments[(labels == 1) | (labels == 2), :n_neurons, 10:14]
        selected_labels = labels[(labels == 1) | (labels == 2)]
        X = selected_segments.reshape(selected_segments.shape[0], -1)

        # 进行维度检查
        n_samples = selected_segments.shape[0]
        n_features = X.shape[1]
        if n_features <= n_samples/2 - 1:
            X_reduce = X
        else:
            _pca = PCA(n_components=int(n_samples/2 - 1))
            X_reduce = _pca.fit_transform(X)
        fi_values.append(Fisher_information(X_reduce, selected_labels, mode="multivariate"))
    fi_values = np.asarray(fi_values, dtype=float)    
    return neuron_counts, fi_values