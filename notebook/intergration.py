import os
import json
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# 全局设置
RESULTS_BASE_DIR = "./results/"
GROUP_OUT_DIR = os.path.join(RESULTS_BASE_DIR, "group_summary")
os.makedirs(GROUP_OUT_DIR, exist_ok=True)

# 颜色映射保持一致
COLORS = {'Divergent': '#FF4B4B', 'Convergent': '#1C75BC', 'Random': '#7AC143'}
CONDITIONS = ['Divergent', 'Convergent', 'Random']

def load_all_mice_data(base_dir):
    """加载所有小鼠的统计JSON数据"""
    # 匹配类似于 ../results/M77_1031/data/M77_1031_statistics.json 的文件
    json_pattern = os.path.join(base_dir, "*", "data", "*_statistics.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        raise FileNotFoundError(f"未在 {base_dir} 找到任何 JSON 统计文件。请检查路径。")
    
    print(f"[*] 找到 {len(json_files)} 只小鼠的数据文件。")
    
    all_data = []
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            all_data.append(data)
    return all_data

def build_master_dataframe(all_data):
    """将嵌套的JSON字典拍平为利于统计分析的 Pandas DataFrame"""
    rows = []
    # 反向映射，应对 participants 键值为数字的情况
    name_to_id = {'Divergent': '1', 'Convergent': '2', 'Random': '3'}
    
    for mouse in all_data:
        mid = mouse['mouse_id']
        
        entropy_dict = {item['Stimulus']: item for item in mouse['entropy_metrics']}
        corr_dict = {item['Class_Name']: item for item in mouse['network_correlation']}
        part_dict = mouse.get('rr_participants_ratio', {})
        
        for cond in CONDITIONS:
            part_val = part_dict.get(cond)
            if part_val is None:
                part_val = part_dict.get(name_to_id[cond], np.nan)
                
            row = {
                'mouse_id': mid,
                'Condition': cond,
                'Entropy': float(entropy_dict.get(cond, {}).get('Entropy', np.nan)),
                'Mean_RSM_Sim': float(entropy_dict.get(cond, {}).get('Mean_Sim', np.nan)),
                # 新增以下两行提取强、弱连接均值
                'Strong_Correlation': float(corr_dict.get(cond, {}).get('Strong_Abs_Correlation_Mean', np.nan)),
                'Weak_Correlation': float(corr_dict.get(cond, {}).get('Weak_Abs_Correlation_Mean', np.nan)),
                # 原有的 Gap
                'Strong_Weak_Gap': float(corr_dict.get(cond, {}).get('Strong_Weak_Gap', np.nan)),
                'Participants_Ratio': float(part_val) if pd.notna(part_val) else np.nan
            }
            rows.append(row)
            
    return pd.DataFrame(rows)
def perform_statistical_tests(df, metric):
    """
    对指定指标执行统计检验。
    由于是同一批小鼠在三种条件下的表现，使用非参数的 Friedman 检验。
    如果主效应显著，则进行 Wilcoxon 符号秩检验进行两两比较。
    """
    # 将数据透视，行是小鼠，列是条件
    pivot_df = df.pivot(index='mouse_id', columns='Condition', values=metric).dropna()
    
    if len(pivot_df) < 3:
        return {"main_effect": "样本量过小，无法进行稳健统计", "post_hoc": {}}

    # 1. Friedman 检验 (被试内非参数 ANOVA)
    stat, p_val = stats.friedmanchisquare(
        pivot_df['Divergent'], 
        pivot_df['Convergent'], 
        pivot_df['Random']
    )
    
    result = {
        "main_effect": f"Friedman $\chi^2$={stat:.3f}, $p$={p_val:.4e}",
        "p_main": p_val,
        "post_hoc": {}
    }
    
    # 2. 事后两两配对检验 (Wilcoxon signed-rank)
    for cond1, cond2 in combinations(CONDITIONS, 2):
        try:
            w_stat, p_pair = stats.wilcoxon(pivot_df[cond1], pivot_df[cond2])
            result["post_hoc"][f"{cond1} vs {cond2}"] = p_pair
        except Exception as e:
            result["post_hoc"][f"{cond1} vs {cond2}"] = np.nan
            
    return result

def get_star(p_val):
    """根据p值返回显著性星号"""
    if pd.isna(p_val): return "ns"
    if p_val < 0.001: return "***"
    elif p_val < 0.01: return "**"
    elif p_val < 0.05: return "*"
    else: return "ns"

def plot_group_metric(df, metric, ylabel, title, stat_res, save_name):
    """绘制组水平对比图（箱线图 + 单个数据点），并添加显著性标注"""
    # 批判性审查：数据如果全为空，直接终止绘制该图并抛出警告
    if df[metric].isna().all():
        print(f"[*] 警告: 指标 '{metric}' 的所有数据均为 NaN，跳过绘图。请检查 JSON 数据提取是否正确。")
        return None
        
    plt.figure(figsize=(6, 5))
    
    # 获取有效样本量
    n_mice = df['mouse_id'].nunique()
    
    if n_mice < 3:
        # 样本量过小时，强制不画箱线图，只画散点，避免误导性的统计表征
        print(f"[*] 提示: 指标 '{metric}' 样本量 N={n_mice} < 3，仅绘制散点。")
        sns.stripplot(data=df, x='Condition', y=metric, order=CONDITIONS, 
                      hue='Condition', palette=COLORS, size=8, alpha=0.8, jitter=False, legend=False)
    else:
        # Seaborn 0.13+ 规范：使用 palette 时建议同时指定 hue，并通过 legend=False 隐藏冗余图例
        sns.boxplot(data=df, x='Condition', y=metric, order=CONDITIONS, 
                    hue='Condition', palette=COLORS, width=0.5, showfliers=False, legend=False)
        sns.stripplot(data=df, x='Condition', y=metric, order=CONDITIONS, 
                      color='black', size=6, alpha=0.7, jitter=True)
    
    plt.ylabel(ylabel)
    plt.title(f"{title}\n{stat_res.get('main_effect', '')}")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    
    # 简单标注两两比较的星号
    if stat_res.get('p_main', 1) < 0.1 and n_mice >= 3:
        y_max = df[metric].max()
        y_range = y_max - df[metric].min()
        step = y_range * 0.08
        
        pairs = list(combinations(CONDITIONS, 2))
        for i, (c1, c2) in enumerate(pairs):
            p = stat_res['post_hoc'].get(f"{c1} vs {c2}", 1)
            star = get_star(p)
            if star != "ns":
                x1, x2 = CONDITIONS.index(c1), CONDITIONS.index(c2)
                y = y_max + step * (i + 1)
                plt.plot([x1, x1, x2, x2], [y, y+step*0.2, y+step*0.2, y], lw=1.2, c='k')
                plt.text((x1+x2)*.5, y+step*0.2, star, ha='center', va='bottom', color='k')
                
    plt.tight_layout()
    save_path = os.path.join(GROUP_OUT_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def generate_group_markdown(df, stat_results, image_paths):
    """生成最终的多鼠综合分析 Markdown 报告"""
    md_path = os.path.join(GROUP_OUT_DIR, "Group_Analysis_Report.md")
    
    # 1. 扩充纯数值列名单，包含新增的 Strong 和 Weak
    numeric_cols = [
        'Entropy', 'Mean_RSM_Sim', 
        'Strong_Correlation', 'Weak_Correlation', 'Strong_Weak_Gap', 
        'Participants_Ratio'
    ]
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    summary_df = df.groupby('Condition')[numeric_cols].agg(['mean', 'sem']).round(4)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 多小鼠综合显著性分析报告 (Group-level Analysis)\n\n")
        f.write(f"**总计包含小鼠数量**: {df['mouse_id'].nunique()} 只\n")
        f.write(f"**纳入数据集**: {', '.join(df['mouse_id'].unique())}\n\n")
        
        f.write("## 1. 组水平描述性统计 (Mean ± SEM)\n\n")
        desc_df = pd.DataFrame()
        for col in summary_df.columns.levels[0]:
            if col in numeric_cols:
                desc_df[col] = summary_df[col]['mean'].astype(str) + " ± " + summary_df[col]['sem'].astype(str)
        f.write(desc_df.reset_index().to_markdown(index=False) + "\n\n")
        
        f.write("## 2. 统计检验结果 (Friedman Test & Wilcoxon post-hoc)\n\n")
        f.write("| 评估指标 | 主效应 (Friedman) | Divergent vs Convergent | Divergent vs Random | Convergent vs Random |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        # 更新指标名称映射字典
        metrics_name_map = {
            'Entropy': '群体表征熵 (Entropy)',
            'Mean_RSM_Sim': '表征相似度 (RSM Mean)',
            'Strong_Correlation': '强连接均值 (Strong Correlation)',
            'Weak_Correlation': '弱连接均值 (Weak Correlation)',
            'Strong_Weak_Gap': '网络连接强度差 (Strong-Weak Gap)',
            'Participants_Ratio': '特异性响应比例 (Participants Ratio)'
        }
        
        for metric, name in metrics_name_map.items():
            res = stat_results.get(metric, {})
            ph = res.get('post_hoc', {})
            f.write(f"| **{name}** | {res.get('main_effect', 'N/A')} | "
                    f"p={ph.get('Divergent vs Convergent', np.nan):.4f} ({get_star(ph.get('Divergent vs Convergent', 1))}) | "
                    f"p={ph.get('Divergent vs Random', np.nan):.4f} ({get_star(ph.get('Divergent vs Random', 1))}) | "
                    f"p={ph.get('Convergent vs Random', np.nan):.4f} ({get_star(ph.get('Convergent vs Random', 1))}) |\n")
        f.write("\n*(注: ns = 不显著, * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$)*\n\n")
        
        f.write("## 3. 组间对比可视化\n\n")
        for img_name, img_path in image_paths.items():
            rel_path = os.path.basename(img_path)
            f.write(f"### {img_name}\n")
            f.write(f"![{img_name}](./{rel_path})\n\n")

    print(f"[*] 综合报告已生成至: {md_path}")

if __name__ == "__main__":
    # 1. 载入所有数据
    all_data = load_all_mice_data(RESULTS_BASE_DIR)
    master_df = build_master_dataframe(all_data)
    
    # 2. 扩充需要进行统计检验的指标列表
    metrics_to_test = [
        'Entropy', 'Mean_RSM_Sim', 
        'Strong_Correlation', 'Weak_Correlation', 'Strong_Weak_Gap', 
        'Participants_Ratio'
    ]
    
    stat_results = {}
    for m in metrics_to_test:
        stat_results[m] = perform_statistical_tests(master_df, m)
        
    # 3. 绘图保存（增加对 Strong 和 Weak 的独立绘图）
    image_paths = {}
    image_paths['群体表征熵'] = plot_group_metric(
        master_df, 'Entropy', 'Entropy (bits)', 'Population Representation Entropy', stat_results['Entropy'], 'group_entropy.png')
    
    image_paths['强连接均值 (Top 10%)'] = plot_group_metric(
        master_df, 'Strong_Correlation', '|Correlation|', 'Strong Connections (Top 10%)', stat_results['Strong_Correlation'], 'group_strong_corr.png')
        
    image_paths['弱连接均值 (Bottom 10%)'] = plot_group_metric(
        master_df, 'Weak_Correlation', '|Correlation|', 'Weak Connections (Bottom 10%)', stat_results['Weak_Correlation'], 'group_weak_corr.png')
    
    image_paths['网络强弱连接差'] = plot_group_metric(
        master_df, 'Strong_Weak_Gap', 'Gap (|Corr|)', 'Network Connectivity Gap', stat_results['Strong_Weak_Gap'], 'group_corr_gap.png')
    
    image_paths['RR神经元特异性响应比例'] = plot_group_metric(
        master_df, 'Participants_Ratio', 'Response Ratio', 'Specific Response Ratio of RR Neurons', stat_results['Participants_Ratio'], 'group_participants.png')

    # 4. 生成报告
    generate_group_markdown(master_df, stat_results, image_paths)
    print("====== 多鼠综合分析完成 ======")