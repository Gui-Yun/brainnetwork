# 小鼠神经活动综合分析报告 - M71_1024

**生成时间**: 2026-03-13 19:39:10

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.8600 | 0.5657 | 0.1442 |
| Convergent | 4.0205 | 0.5372 | 0.1675 |
| Random | 3.7369 | 0.6122 | 0.1329 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.1778 | 0.1778 | 0.0455 | 0.3778 | 0.3324 | 25425 |
| 2 | Convergent | 0.1778 | 0.1778 | 0.0445 | 0.3806 | 0.3361 | 25425 |
| 3 | Random | 0.1566 | 0.1566 | 0.0326 | 0.3543 | 0.3217 | 25425 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 1.5856 |
| 2 | 1.3690 |
| 3 | 3.1884 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
