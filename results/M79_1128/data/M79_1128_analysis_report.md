# 小鼠神经活动综合分析报告 - M79_1128

**生成时间**: 2026-03-13 19:56:03

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.6882 | 0.5698 | 0.1324 |
| Convergent | 3.5299 | 0.5659 | 0.1122 |
| Random | 3.8149 | 0.5514 | 0.1441 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.2395 | 0.2395 | 0.1100 | 0.4088 | 0.2988 | 66430 |
| 2 | Convergent | 0.2421 | 0.2421 | 0.1130 | 0.4128 | 0.2998 | 66430 |
| 3 | Random | 0.2295 | 0.2295 | 0.0974 | 0.4040 | 0.3066 | 66430 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 1.3439 |
| 2 | 1.4720 |
| 3 | 1.6073 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
