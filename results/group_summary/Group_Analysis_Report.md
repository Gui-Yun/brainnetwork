# Group-level Multi-mouse Analysis Report

**Number of mice**: 8

**Mouse IDs**: M21_1107, M71_1024, M73_1128, M77_1031, M77_1107, M78_1017, M79_1128, M91_1017

## 1. Descriptive Statistics (Mean ± SEM)

| Condition   | Entropy         | Mean_RSM_Sim    | Mean_Correlation   | Strong_Correlation   | Weak_Correlation   | Strong_Weak_Gap   | Participants_Ratio   | Gini_Mean       | Gini_STD        | PR_Mean          | PR_STD           | PR_Norm_Mean    | PR_Norm_STD     | Effective_Dim_PR   | Effective_Dim_eRank   | Effective_Dim_90Var   | Sig_Mean_Corr   | Noise_Mean_Corr   | SigAbs_Mean_Corr   | NoiseAbs_Mean_Corr   | SigNoise_Coupling_r   |
|:------------|:----------------|:----------------|:-------------------|:---------------------|:-------------------|:------------------|:---------------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|:-------------------|:----------------------|:----------------------|:----------------|:------------------|:-------------------|:---------------------|:----------------------|
| Convergent  | 3.6272 ± 0.085  | 0.5712 ± 0.0109 | 0.2074 ± 0.0104    | 0.3902 ± 0.0115      | 0.0727 ± 0.0092    | 0.3175 ± 0.006    | 1.6652 ± 0.1043      | 0.6033 ± 0.0185 | 0.0756 ± 0.0056 | 42.1088 ± 6.5365 | 17.0589 ± 2.7404 | 0.119 ± 0.0145  | 0.0478 ± 0.0064 | 12.9609 ± 1.1693   | 20.1429 ± 1.3999      | 20.875 ± 1.3685       | 0.8041 ± 0.0105 | 0.1386 ± 0.016    | 0.8044 ± 0.0104    | 0.191 ± 0.0102       | 0.1692 ± 0.0228       |
| Divergent   | 3.6478 ± 0.0861 | 0.5638 ± 0.0236 | 0.2051 ± 0.0114    | 0.3894 ± 0.0136      | 0.0728 ± 0.0099    | 0.3166 ± 0.008    | 1.724 ± 0.1179       | 0.6022 ± 0.0193 | 0.0836 ± 0.006  | 43.6522 ± 6.8255 | 17.6853 ± 2.5414 | 0.1205 ± 0.0119 | 0.0499 ± 0.0044 | 12.007 ± 1.1759    | 19.3624 ± 1.5661      | 20.5 ± 1.5            | 0.8128 ± 0.0076 | 0.162 ± 0.0199    | 0.8129 ± 0.0075    | 0.2052 ± 0.0146      | 0.1702 ± 0.0169       |
| Random      | 3.5989 ± 0.0692 | 0.6264 ± 0.0181 | 0.1985 ± 0.0142    | 0.391 ± 0.0154       | 0.0546 ± 0.0109    | 0.3364 ± 0.0107   | 2.5279 ± 0.2484      | 0.6291 ± 0.0196 | 0.0783 ± 0.006  | 35.9484 ± 4.2529 | 16.5695 ± 2.2046 | 0.1074 ± 0.0149 | 0.048 ± 0.0052  | 11.4929 ± 0.8069   | 18.7792 ± 1.1476      | 20.25 ± 1.3059        | 0.7198 ± 0.0223 | 0.1384 ± 0.0185   | 0.7215 ± 0.0216    | 0.2044 ± 0.0113      | 0.1769 ± 0.0096       |

## 2. Friedman + Wilcoxon Tests

| Metric | Main Effect | Div vs Con | Div vs Rand | Con vs Rand |
| :--- | :--- | :--- | :--- | :--- |
| **Entropy** | Friedman $\chi^2$=0.75, $p$=6.873e-01 | p=0.7422 (ns) | p=0.5469 (ns) | p=0.7422 (ns) |
| **Mean_RSM_Sim** | Friedman $\chi^2$=4.75, $p$=9.301e-02 | p=0.8438 (ns) | p=0.0391 (*) | p=0.0234 (*) |
| **Mean_Correlation** | Friedman $\chi^2$=3.25, $p$=1.969e-01 | p=0.3125 (ns) | p=0.3828 (ns) | p=0.3125 (ns) |
| **Strong_Correlation** | Friedman $\chi^2$=0.75, $p$=6.873e-01 | p=0.5469 (ns) | p=0.6406 (ns) | p=0.5469 (ns) |
| **Weak_Correlation** | Friedman $\chi^2$=7.00, $p$=3.020e-02 | p=1.0000 (ns) | p=0.0156 (*) | p=0.0234 (*) |
| **Strong_Weak_Gap** | Friedman $\chi^2$=6.25, $p$=4.394e-02 | p=0.7422 (ns) | p=0.0391 (*) | p=0.0547 (ns) |
| **Participants_Ratio** | Friedman $\chi^2$=9.25, $p$=9.804e-03 | p=0.6406 (ns) | p=0.0078 (**) | p=0.0156 (*) |
| **Gini_Mean** | Friedman $\chi^2$=7.00, $p$=3.020e-02 | p=0.8438 (ns) | p=0.0391 (*) | p=0.1094 (ns) |
| **PR_Mean** | Friedman $\chi^2$=1.75, $p$=4.169e-01 | p=0.5469 (ns) | p=0.1094 (ns) | p=0.1953 (ns) |
| **Effective_Dim_PR** | Friedman $\chi^2$=2.25, $p$=3.247e-01 | p=0.0391 (*) | p=0.8438 (ns) | p=0.3125 (ns) |
| **Sig_Mean_Corr** | Friedman $\chi^2$=13.00, $p$=1.503e-03 | p=0.3125 (ns) | p=0.0078 (**) | p=0.0078 (**) |
| **Noise_Mean_Corr** | Friedman $\chi^2$=2.25, $p$=3.247e-01 | p=0.1484 (ns) | p=0.1953 (ns) | p=0.8438 (ns) |

## 3. RR Overlap Summary Across Mice

| Subset                   |   Mean_Size |   SEM_Size |
|:-------------------------|------------:|-----------:|
| All_Classes_Intersection |      69.75  |    15.2746 |
| Class_1                  |     208.25  |    46.0212 |
| Class_1&Class_2          |     111.375 |    21.9813 |
| Class_1&Class_3          |      99.75  |    20.0105 |
| Class_2                  |     219.125 |    41.3489 |
| Class_2&Class_3          |      93     |    20.7743 |
| Class_3                  |     199.5   |    41.0422 |
| Union_All                |     392.5   |    81.2021 |

## 4. Figures

### Combined Strong vs Weak
![Combined Strong vs Weak](./group_combined_strong_weak.png)

### RSM Mean Similarity
![RSM Mean Similarity](./group_mean_rsm_sim.png)

### Strong Connections (Top 10%)
![Strong Connections (Top 10%)](./group_strong_correlation.png)

### Weak Connections (Bottom 10%)
![Weak Connections (Bottom 10%)](./group_weak_correlation.png)

### Strong-Weak Correlation Gap
![Strong-Weak Correlation Gap](./group_strong_weak_gap.png)

### RR Participants Ratio
![RR Participants Ratio](./group_participants_ratio.png)

### Response Gini (Mean)
![Response Gini (Mean)](./group_gini_mean.png)

### Decile Correlation Curve
![Decile Correlation Curve](./group_corr_decile_curve.png)

### Noise Decile Curve
![Noise Decile Curve](./group_noise_corr_decile_curve.png)

### Cross-animal Binding
![Cross-animal Binding](./group_cross_animal_binding.png)

### Absolute State Binding
![Absolute State Binding](./group_absolute_state_binding.png)

### LMM State Binding
![LMM State Binding](./group_lmm_state_binding.png)

