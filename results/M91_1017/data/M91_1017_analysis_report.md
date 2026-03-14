# 小鼠神经活动综合分析报告 - M91_1017

**生成时间**: 2026-03-13 20:03:55

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.4257 | 0.6816 | 0.1093 |
| Convergent | 3.4899 | 0.6288 | 0.1120 |
| Random | 3.5983 | 0.6917 | 0.1349 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.2615 | 0.2615 | 0.1175 | 0.4522 | 0.3348 | 27495 |
| 2 | Convergent | 0.2610 | 0.2610 | 0.1106 | 0.4498 | 0.3392 | 27495 |
| 3 | Random | 0.2541 | 0.2541 | 0.0946 | 0.4563 | 0.3617 | 27495 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 2.0712 |
| 2 | 1.4740 |
| 3 | 2.2128 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
