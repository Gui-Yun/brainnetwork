# Modeling v3 Iteration Summary

更新时间：2026-04-16  
数据来源：`results/group_summary` 当前产物 + 本次迭代过程中的已记录关键结果

## 1. 背景与目标

当前 mechanistic scan 的目标是拟合经验数据中 `Coherent - Random` 的关键差值，核心关注：

- `participants_ratio_delta`（目标 `-0.833325`）
- `orth_parallel_ratio_delta`（目标 `-0.308074`）
- `strong_frac_delta`（目标 `-0.012236`）
- `mean_rsm_sim_delta`（目标 `-0.067825`）

## 2. 版本改动概览

### V0 -> V1（打分框架修订）

- 修复 `strong_quantile` 逻辑（不再硬编码）。
- 增加分组分数上限（`score_cap_fc/alloc/geom`）。
- 增加方向惩罚（directional penalty）和关键指标权重（尤其 participants / orth ratio）。
- 增加更完整的诊断输出列（`score_base`、各类 penalty 分解）。

### V1 -> V2（幅度约束修订）

- 增加双侧幅度惩罚（under/over band penalty）。
- 对 participants 与 orth ratio 引入最小/最大幅度区间控制。
- 观察到：方向性更稳定，但幅度仍普遍不足或在 orth 上过冲。

### V2 -> V3（allocation 口径对齐旧实验设计）

- `participants_ratio` 改为旧设计口径：`class-specific RR vs other-RR within RR union`。
- `gini` 改为 trial-level 均值口径（对齐 `Gini_Mean`）。
- 增加 `pr_mean` 诊断列（用于与历史 `PR_Mean` 对照）。
- 说明：V2 与 V3 的 `score_alloc` 可比性降低（定义发生变化），但核心行为趋势仍可比较。

## 3. 主要结果对比（关键轮次）

| 版本 | 运行时间 | 采样规模 | best score_total | participants_delta | orth_ratio_delta | strong_frac_delta | mean_rsm_delta | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| V2 | 2026-04-15 19:03 | 480 x 5 | 4.248 | 0.000 | -0.513 | +0.00251 | -0.00436 | participants 方向错误，幅度明显不足 |
| V3（中间轮） | 2026-04-15（480轮） | 480 x 5 | 4.282 | 0.000 | -0.513 | +0.00251 | -0.00436 | allocation 口径切换后，小轮结果仍未改善核心矛盾 |
| V3（当前最新） | 2026-04-15 21:21 | 1200 x 6 | 2.742 | -0.213 | -0.519 | -0.00139 | -0.00192 | 分数显著下降，但幅度拟合仍不足 |

注：当前最新 best 行来自 `sample_id=523`。

## 4. 当前最新结果解读（2026-04-15 21:21）

### 4.1 优点

- `penalty_directional = 0`，说明最优解在主要方向约束上已基本满足。
- 总分从 4.x 下降到 2.74，搜索过程较早期更稳定。

### 4.2 主要问题

- `participants_ratio_delta = -0.2126`，仅达到目标幅度约 `25.5%`。
- `mean_rsm_sim_delta = -0.00192`，仅达到目标幅度约 `2.8%`。
- `strong_frac_delta = -0.00139`，仅达到目标幅度约 `11.3%`。
- `orth_parallel_ratio_delta = -0.5195`，约为目标的 `1.69x`，仍有过冲。

### 4.3 Top50 诊断（最新轮）

- `participants` 达到目标 45% 幅度以上：`3 / 50`。
- `orth_ratio` 超过目标 2 倍：`35 / 50`。
- 结论：方向改进存在，但“幅度不够 + orth 过冲”仍是主矛盾。

## 5. 阶段性结论（是否可作为第一版解释模型）

可以作为 `v1 prototype`（机制雏形）使用，但不建议作为定稿解释模型。

- 可以支持的表述：模型已能在部分关键指标上给出正确方向和可重复趋势。
- 需要保留的限制：尚未完成关键幅度的定量拟合，尤其 participants 与 rsm。

## 6. 下一步建议（面向 v2 可发布版）

- 继续提高 participants 与 rsm 的幅度约束强度（目标是先把幅度拉到 >= 45% 区间）。
- 进一步压制 orth ratio 过冲（收窄上界或增强 over-penalty）。
- 固定 allocation 新口径后，至少跑 2 轮 `1200 x 6` 复现实验，比较 top50 稳定性后再定稿。

