import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
from pathlib import Path

# ======= parameters =======
# Experiment configuration
t_stimulus = 10
l_stimulus = 20
l_trials = 50
ipd = 5.0
isi = 5.0
trials_num = 180
# RR neuron selection parameters
reliability_threshold = 0.75
snr_threshold = 0.8


# ======= functions =======
def process_trigger(txt_file, IPD=ipd, ISI=isi, fre=None, min_sti_gap=4.0):
    """
    处理触发文件，修改自step1x_trigger_725right.m
    
    参数:
    txt_file: str, txt文件路径
    IPD: float, 刺激呈现时长(s)，默认2s
    ISI: float, 刺激间隔(s)，默认6s
    fre: float, 相机帧率Hz，None则从相机触发时间自动估计
    min_sti_gap: float, 相邻刺激"2"小于此间隔(s)视作同一次（用于去重合并），默认5s
    
    返回:
    dict: 包含start_edge, end_edge, stimuli_array的字典
    """
    # 读入文件
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    ch_str = parts[1]
                    abs_ts = float(parts[2]) if len(parts) >= 3 else None
                    data.append((time_val, ch_str, abs_ts))
                except ValueError:
                    continue
    assert len(data) > 0, "数据文件为空或格式不正确"
    # 解析数据
    times, channels, abs_timestamps = zip(*data)
    times = np.array(times)
    # 转换通道为数值，非数值的设为NaN
    ch_numeric = []
    valid_indices = []
    for i, ch_str in enumerate(channels):
        try:
            ch_val = float(ch_str)
            ch_numeric.append(ch_val)
            valid_indices.append(i)
        except ValueError:
            continue
    assert valid_indices, "未找到有效的数值通道数据"
    # 只保留有效数据
    t = times[valid_indices]
    ch = np.array(ch_numeric)
    
    # 相机帧与刺激起始时间
    cam_t_raw = t[ch == 1]
    # print(len(cam_t_raw))
    sti_t_raw = t[ch == 2]
    # print(len(sti_t_raw))
    assert len(cam_t_raw) > 0, "未检测到相机触发(值=1)"
    assert len(sti_t_raw) > 0, "未检测到刺激触发(值=2)"

    
    # 去重/合并：将时间靠得很近的"2"视作同一次刺激
    sti_t = np.sort(sti_t_raw)
    if len(sti_t) > 0:
        keep = np.ones(len(sti_t), dtype=bool)
        for i in range(1, len(sti_t)):
            if (sti_t[i] - sti_t[i-1]) < min_sti_gap:
                keep[i] = False  # 合并到前一个
        sti_t = sti_t[keep]
    
    # 帧率估计或使用给定值
    if fre is None:
        dt = np.diff(cam_t_raw)
        fre = 1 / np.median(dt)  # 用相机帧时间戳的中位间隔

    IPD_frames = max(1, round(IPD * fre))
    isi_frames = round((IPD + ISI) * fre)
    
    # 把每个刺激时间映射到最近的相机帧索引
    cam_t = cam_t_raw.copy()
    nFrames = len(cam_t)
    start_edge = np.zeros(len(sti_t), dtype=int)
    
    for k in range(len(sti_t)):
        idx = np.argmin(np.abs(cam_t - sti_t[k]))
        start_edge[k] = idx
    
    end_edge = start_edge + IPD_frames - 1
    
    # 边界裁剪，避免越界
    valid = (start_edge >= 0) & (end_edge < nFrames) & (start_edge <= end_edge)
    start_edge = start_edge[valid]
    end_edge = end_edge[valid]
    
    # 尾段完整性检查（与旧逻辑一致）
    if len(start_edge) >= 2:
        d = np.diff(start_edge)
        while len(d) > 0 and d[-1] not in [isi_frames-1, isi_frames, isi_frames+1, isi_frames+2]:
            # 丢掉最后一个可疑的刺激段
            start_edge = start_edge[:-1]
            end_edge = end_edge[:-1]
            if len(start_edge) >= 2:
                d = np.diff(start_edge)
            else:
                break
    
    return {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'camera_frames': len(cam_t),
        'stimuli_count': len(start_edge)
    }

def rr_selection(trials, labels, t_stimulus=t_stimulus, l=l_stimulus, alpha_fdr=0.05, alpha_level=0.05, reliability_threshold=reliability_threshold, snr_threshold=snr_threshold, effect_size_threshold=0.5, response_ratio_threshold=0.6, max_n = 2000):
    """
    快速RR神经元筛选
    优化策略:
    1. 向量化计算替代循环
    2. 简化统计检验（t检验替代Mann-Whitney U）
    3. 批量处理所有神经元
    """
    import time
    start_time = time.time()
    
    print("使用快速RR筛选算法...")
    
    # 过滤有效数据
    valid_mask = (labels == 1) | (labels == 2) | (labels == 0)
    valid_trials = trials[valid_mask]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_trials.shape
    
    # 定义时间窗口
    baseline_pre = np.arange(0, t_stimulus)
    baseline_post = np.arange(t_stimulus + l, n_timepoints)
    stimulus_window = np.arange(t_stimulus, t_stimulus + l)
    
    print(f"处理 {n_trials} 个试次, {n_neurons} 个神经元")
    
    # 1. 响应性检测 - 向量化计算
    # 计算基线和刺激期的平均值
    baseline_pre_mean = np.mean(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_mean = np.mean(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的平均
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2
    
    stimulus_mean = np.mean(valid_trials[:, :, stimulus_window], axis=2)  # (trials, neurons)
    
    # 简化的响应性检测：基于效应大小和标准误差
    baseline_pre_std = np.std(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_std = np.std(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的标准差
    baseline_std = (baseline_pre_std + baseline_post_std) / 2
    
    stimulus_std = np.std(valid_trials[:, :, stimulus_window], axis=2)
    
    # Cohen's d效应大小
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)
    
    # 响应性标准：平均效应大小 > 阈值 且 至少指定比例试次有响应
    response_ratio = np.mean(effect_size > effect_size_threshold, axis=0)
    enhanced_neurons = np.where((response_ratio > response_ratio_threshold) & 
                              (np.mean(stimulus_mean > baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()

    # 2. 可靠性检测 - 简化版本
    # 计算每个神经元在每个试次的信噪比
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    noise_level = baseline_std + 1e-8
    snr = signal_strength / noise_level
    
    # 可靠性：指定比例的试次信噪比 > 阈值
    reliability_ratio = np.mean(snr > snr_threshold, axis=0)
    reliable_neurons = np.where(reliability_ratio >= reliability_threshold)[0].tolist()
    
    # 3. 最终RR神经元，限制最大数量
    rr_neurons = list(set(enhanced_neurons) & set(reliable_neurons))
    if len(rr_neurons) > max_n:
        np.random.seed(101)
        np.random.shuffle(rr_neurons) 
        rr_neurons = rr_neurons[:max_n]   
    elapsed_time = time.time() - start_time
    print(f"快速RR筛选完成，耗时: {elapsed_time:.2f}秒")
    
    return rr_neurons

def load_data(data_path, end_idx=trials_num, data_type='spikes'):
    '''
    加载神经数据、位置数据、触发数据和刺激数据
    '''
    ######### 读取神经数据 #########
    # print("开始处理数据...")
    assert data_type in ['spikes', 'fluorescence'], ValueError("data_type 必须为 'spikes' 或 'fluorescence'")
    
    if data_type == 'spikes':
        mat_file = os.path.join(data_path, 'wholebrain_processed.mat')
    elif data_type == 'fluorescence':
        mat_file = os.path.join(data_path, 'wholebrain_output.mat')
    if not os.path.exists(mat_file):
        raise ValueError(f"未找到神经数据文件: {mat_file}")
    try:
        data = h5py.File(mat_file, 'r') 
    except Exception as e:
        # raise ValueError(f"无法读取mat文件: {mat_file}，错误信息: {e}")
        data = scipy.io.loadmat(mat_file)
    # print(data)
    # 检查关键数据集是否存在
    if 'whole_trace_ori' not in data or 'whole_center' not in data:
        raise ValueError("mat文件缺少必要的数据集（'whole_trace_ori' 或 'whole_center'）")

    # ==========神经数据================
    neuron_data = data['whole_trace_ori']
    # 转化成numpy数组
    neuron_data = np.array(neuron_data)
    print(f"原始神经数据形状: {neuron_data.shape}")
    
    # 只做基本的数据清理：移除NaN和Inf
    neuron_data = np.nan_to_num(neuron_data, nan=0.0, posinf=0.0, neginf=0.0)
    neuron_pos = data['whole_center']
    # 检查和处理neuron_pos维度
    if len(neuron_pos.shape) != 2:
        raise ValueError(f"neuron_pos 应为2D数组，实际为: {neuron_pos.shape}")
    
    # 灵活处理不同维度的neuron_pos
    if neuron_pos.shape[0] > 2:
        # 标准格式 (4, n)，提取前两维
        neuron_pos = neuron_pos[0:2, :]
    elif neuron_pos.shape[0] == 2:
        # 已经是2维，直接使用
        print(f"检测到2维neuron_pos格式: {neuron_pos.shape}")
    else:
        raise ValueError(f"不支持的neuron_pos维度: {neuron_pos.shape[0]}，期望为2、3或4维")

    trigger_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    trigger_data = process_trigger(trigger_files[0])
    
    # 刺激数据
    stimulus_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    stimulus_data = pd.read_csv(stimulus_files[0])
    # 转化成numpy数组
    stimulus_data = np.array(stimulus_data)
    
    # 保持指定试验数，去掉首尾 - 对触发数据和刺激数据同时处理
    start_edges = trigger_data['start_edge'][:end_idx]
    stimulus_data = stimulus_data[0:end_idx, :]

    return neuron_data, neuron_pos, start_edges, stimulus_data 

def segment_neuron_data(neuron_data, trigger_data, label, pre_frames=t_stimulus, post_frames=l_trials-t_stimulus):
    """
    改进的数据分割函数
    
    参数:
    pre_frames: 刺激前的帧数（用于基线）
    post_frames: 刺激后的帧数（用于反应）
    baseline_correct: 是否进行基线校正 (ΔF/F)
    """
    total_frames = pre_frames + post_frames
    segments = np.zeros((len(trigger_data), neuron_data.shape[1], total_frames))
    labels = []

    for i in range(len(trigger_data)): # 遍历每个触发事件
        start = trigger_data[i] - pre_frames
        end = trigger_data[i] + post_frames
        # 边界检查
        if start < 0 or end >= neuron_data.shape[0]:
            print(f"警告: 第{i}个刺激的时间窗口超出边界，跳过")
            continue
        segment = neuron_data[start:end, :]
        segments[i] = segment.T
        labels.append(label[i])
    labels = np.array(labels)
    return segments, labels

def preprocess_spike_data(neuron_data, neuron_pos, start_edge, stimulus_data):

    labels = stimulus_data[:, 0] 
    segments, labels = segment_neuron_data(neuron_data, start_edge, labels)
    
    # ============= 第四步 筛选rr神经元
    rr_neurons = rr_selection(segments, np.array(labels))
    segments = segments[:, rr_neurons, :]
    neuron_pos = neuron_pos[:, rr_neurons]

    return segments, labels, neuron_pos

def preprocess_data(neuron_data, neuron_pos, start_edge, stimulus_data, win_size=151):
    # =========== 第一步 提取仅有正值的神经元==================
    # 带负值的神经元索引
    mask = np.any(neuron_data <= 0, axis=0)   # 每列是否存在 <=0
    keep_idx = np.where(~mask)[0]

    # 如果 neuron_pos 与 neuron_data 的列对齐，则同步删除对应列
    if neuron_pos.shape[1] == neuron_data.shape[1]:
        # 从数据中删除这些列
        neuron_data = neuron_data[:, keep_idx]
        neuron_pos = neuron_pos[:, keep_idx]
    else:
        raise ValueError(f"警告: neuron_pos 列数({neuron_pos.shape[1]}) 与 neuron_data 列数({neuron_data.shape[1]}) 不匹配，未修改 neuron_pos")
    
    from scipy import ndimage
    # =========== 第二步 预处理 ===========================
    if win_size % 2 == 0:
        win_size += 1
    T, N = neuron_data.shape
    F0_dynamic = np.zeros((T, N), dtype=float)
    for i in range(N):
        # ndimage.percentile_filter 输出每帧的窗口百分位值
        F0_dynamic[:, i] = ndimage.percentile_filter(neuron_data[:, i], percentile=8, size=win_size, mode='reflect')
    # 计算 dF/F（逐帧）
    neuron_data = (neuron_data - F0_dynamic) / F0_dynamic
    # =========== 第三步 分割神经数据 =====================================

    labels = stimulus_data[:, 0] 
    segments, labels = segment_neuron_data(neuron_data, start_edge, labels)
    # ============= 第四步 筛选rr神经元
    rr_neurons = rr_selection(segments, np.array(labels))
    segments = segments[:, rr_neurons, :]
    neuron_pos = neuron_pos[:, rr_neurons]
    return segments, labels, neuron_pos
