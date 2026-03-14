# 小鼠神经活动综合分析报告 - M77_1031

**生成时间**: 2026-03-13 19:48:12

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.3496 | 0.5091 | 0.1000 |
| Convergent | 3.2839 | 0.5583 | 0.0942 |
| Random | 3.5266 | 0.5700 | 0.1154 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.1884 | 0.1884 | 0.0606 | 0.3527 | 0.2921 | 252405 |
| 2 | Convergent | 0.1890 | 0.1890 | 0.0583 | 0.3577 | 0.2994 | 252405 |
| 3 | Random | 0.1682 | 0.1682 | 0.0335 | 0.3487 | 0.3152 | 252405 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 1.7740 |
| 2 | 1.6347 |
| 3 | 2.3148 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
