# 小鼠神经活动综合分析报告 - M73_1128

**生成时间**: 2026-03-13 19:43:06

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.4957 | 0.5497 | 0.1101 |
| Convergent | 3.3217 | 0.5882 | 0.0992 |
| Random | 3.4855 | 0.6647 | 0.1152 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.1715 | 0.1715 | 0.0441 | 0.3429 | 0.2988 | 131328 |
| 2 | Convergent | 0.1873 | 0.1873 | 0.0535 | 0.3605 | 0.3070 | 131328 |
| 3 | Random | 0.1508 | 0.1508 | 0.0129 | 0.3521 | 0.3392 | 131328 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 2.3480 |
| 2 | 2.0019 |
| 3 | 3.8856 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
