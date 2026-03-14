# 小鼠神经活动综合分析报告 - M21_1107

**生成时间**: 2026-03-13 19:36:02

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.4353 | 0.5502 | 0.1075 |
| Convergent | 3.6383 | 0.5915 | 0.1233 |
| Random | 3.5031 | 0.6674 | 0.1255 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.2217 | 0.2217 | 0.0826 | 0.4225 | 0.3399 | 18145 |
| 2 | Convergent | 0.2064 | 0.2064 | 0.0733 | 0.4030 | 0.3296 | 18145 |
| 3 | Random | 0.2458 | 0.2458 | 0.0661 | 0.4500 | 0.3839 | 18145 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 1.5199 |
| 2 | 2.1406 |
| 3 | 2.1299 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
