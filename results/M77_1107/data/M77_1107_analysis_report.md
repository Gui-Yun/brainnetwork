# 小鼠神经活动综合分析报告 - M77_1107

**生成时间**: 2026-03-13 19:32:09

## 1. 表征相似度 (RSM) 与香农熵

衡量不同刺激条件下，神经群体表征稳定性和变异度。

| Stimulus | Entropy | Mean_Sim | Std_Sim |
| --- | --- | --- | --- |
| Divergent | 3.9988 | 0.5171 | 0.1568 |
| Convergent | 3.9780 | 0.5489 | 0.1593 |
| Random | 3.8568 | 0.5962 | 0.1444 |

## 2. 神经元功能网络成对相关性

衡量网络强弱连接的差异（Strong vs Weak Gap）。

| Class_ID | Class_Name | Mean_Correlation | Mean_Abs_Correlation | Weak_Abs_Correlation_Mean | Strong_Abs_Correlation_Mean | Strong_Weak_Gap | Pair_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Divergent | 0.1933 | 0.1933 | 0.0591 | 0.4016 | 0.3425 | 14878 |
| 2 | Convergent | 0.2080 | 0.2080 | 0.0743 | 0.4002 | 0.3259 | 14878 |
| 3 | Random | 0.1873 | 0.1873 | 0.0365 | 0.3984 | 0.3620 | 14878 |

## 3. RR神经元响应比例 (Participants)

衡量特异性可靠响应神经元（Class-RR）相对于群体内其他RR神经元的响应强度倍数。

| Condition | Response_Ratio |
| --- | --- |
| 1 | 1.4862 |
| 2 | 1.3887 |
| 3 | 2.5187 |

## 4. 关键可视化图表索引

- **偏好性排序热图**: `![Neural Patterns](./figures/neural_patterns_preference_sorted.png)`
- **层次聚类图谱**: `![Clustermap](./figures/neural_patterns_clustermap.png)`
- **RSM 相似度分布**: `![Similarity Distribution](./figures/similarity_distribution.png)`
- **相关性强度对比**: `![Pairwise Correlation](./figures/pairwise_correlation.png)`
