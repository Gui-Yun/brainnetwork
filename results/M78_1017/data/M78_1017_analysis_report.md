# 小鼠神经活动综合分析报告 - M78_1017

**生成时间**: 2026-03-13 19:53:22

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.7758 | 0.5057 | 0.1392 |
| Convergent | 3.7143 | 0.5423 | 0.1302 |
| Random | 3.2692 | 0.6579 | 0.0957 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.1868 | 0.1868 | 0.0632 | 0.3565 | 0.2933 | 263175 |
| 2 | Convergent | 0.1879 | 0.1879 | 0.0541 | 0.3570 | 0.3028 | 263175 |
| 3 | Random | 0.1955 | 0.1955 | 0.0633 | 0.3644 | 0.3011 | 263175 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 1.6630 |
| 2 | 1.8407 |
| 3 | 2.3661 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
