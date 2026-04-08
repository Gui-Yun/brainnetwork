# Group-level Multi-mouse Analysis Report

**Number of mice**: 8

**Mouse IDs**: M21_1107, M71_1024, M73_1128, M77_1031, M77_1107, M78_1017, M79_1128, M91_1017

## 1. Descriptive Statistics (Mean 卤 SEM)

| Condition   | Entropy          | Mean_RSM_Sim     | Mean_Correlation   | Strong_Correlation   | Weak_Correlation   | Strong_Weak_Gap   | Participants_Ratio   | Gini_Mean        | Gini_STD         | PR_Mean           | PR_STD            | PR_Norm_Mean     | PR_Norm_STD      | Effective_Dim_PR   | Effective_Dim_eRank   | Effective_Dim_90Var   | Sig_Mean_Corr    | Noise_Mean_Corr   | SigAbs_Mean_Corr   | NoiseAbs_Mean_Corr   | SigNoise_Coupling_r   | Geom_MeanNorm    | Geom_AngleDeg     | Geom_VarParallel   | Geom_VarOrthogonal   | Geom_OrthParallelRatio   | Geom_Anisotropy   | Geom_Lambda1     | Geom_Lambda2    |
|:------------|:-----------------|:-----------------|:-------------------|:---------------------|:-------------------|:------------------|:---------------------|:-----------------|:-----------------|:------------------|:------------------|:-----------------|:-----------------|:-------------------|:----------------------|:----------------------|:-----------------|:------------------|:-------------------|:---------------------|:----------------------|:-----------------|:------------------|:-------------------|:---------------------|:-------------------------|:------------------|:-----------------|:----------------|
| Convergent  | 3.6421 卤 0.0986 | 0.5648 卤 0.0102 | 0.2074 卤 0.0104   | 0.3902 卤 0.0115     | 0.0727 卤 0.0092   | 0.3175 卤 0.006   | 1.6652 卤 0.1043     | 0.6076 卤 0.0195 | 0.0748 卤 0.0055 | 41.6568 卤 6.5279 | 17.1336 卤 2.6436 | 0.1185 卤 0.0159 | 0.0484 卤 0.0058 | 12.2963 卤 0.8627  | 19.5742 卤 1.1304     | 20.75 卤 1.2783       | 0.8041 卤 0.0105 | 0.1386 卤 0.016   | 0.8044 卤 0.0104   | 0.191 卤 0.0102      | 0.1692 卤 0.0228      | 1.5319 卤 0.1713 | 65.3476 卤 6.6733 | 0.212 卤 0.0382    | 1.9252 卤 0.4639     | 8.7254 卤 1.0563         | 0.1907 卤 0.0118  | 0.4216 卤 0.1058 | 0.2646 卤 0.049 |
| Divergent   | 3.7015 卤 0.0715 | 0.5524 卤 0.0228 | 0.2051 卤 0.0114   | 0.3894 卤 0.0136     | 0.0728 卤 0.0099   | 0.3166 卤 0.008   | 1.724 卤 0.1179      | 0.6039 卤 0.0185 | 0.0846 卤 0.0058 | 42.4225 卤 6.7804 | 17.4578 卤 2.5413 | 0.1169 卤 0.0122 | 0.0489 卤 0.004  | 12.0264 卤 1.2009  | 19.3343 卤 1.5283     | 20.375 卤 1.375       | 0.8128 卤 0.0076 | 0.162 卤 0.0199   | 0.8129 卤 0.0075   | 0.2052 卤 0.0146     | 0.1702 卤 0.0169      | 1.4762 卤 0.1501 | 55.5713 卤 8.5911 | 0.23 卤 0.0431     | 1.8503 卤 0.4496     | 7.8139 卤 1.0659         | 0.2045 卤 0.0164  | 0.396 卤 0.0713  | 0.267 卤 0.0559 |
| Random      | 3.5989 卤 0.0692 | 0.6264 卤 0.0181 | 0.1985 卤 0.0142   | 0.391 卤 0.0154      | 0.0546 卤 0.0109   | 0.3364 卤 0.0107  | 2.5279 卤 0.2484     | 0.6291 卤 0.0196 | 0.0783 卤 0.006  | 35.9484 卤 4.2529 | 16.5695 卤 2.2046 | 0.1074 卤 0.0149 | 0.048 卤 0.0052  | 11.4929 卤 0.8069  | 18.7792 卤 1.1476     | 20.25 卤 1.3059       | 0.7198 卤 0.0223 | 0.1384 卤 0.0185  | 0.7215 卤 0.0216   | 0.2044 卤 0.0113     | 0.1769 卤 0.0096      | 1.5721 卤 0.1615 | 63.7571 卤 6.821  | 0.1699 卤 0.0218   | 1.5099 卤 0.3225     | 8.5777 卤 1.4236         | 0.2023 卤 0.0132  | 0.3355 卤 0.0613 | 0.2249 卤 0.039 |

## 2. Friedman + Wilcoxon Tests

| Metric | Main Effect | Div vs Con | Div vs Rand | Con vs Rand |
| :--- | :--- | :--- | :--- | :--- |
| **Entropy** | Friedman $\chi^2$=0.75, $p$=6.873e-01 | p=0.5469 (ns) | p=0.1953 (ns) | p=0.5469 (ns) |
| **Mean_RSM_Sim** | Friedman $\chi^2$=6.75, $p$=3.422e-02 | p=0.6406 (ns) | p=0.0234 (*) | p=0.0156 (*) |
| **Mean_Correlation** | Friedman $\chi^2$=3.25, $p$=1.969e-01 | p=0.3125 (ns) | p=0.3828 (ns) | p=0.3125 (ns) |
| **Strong_Correlation** | Friedman $\chi^2$=0.75, $p$=6.873e-01 | p=0.5469 (ns) | p=0.6406 (ns) | p=0.5469 (ns) |
| **Weak_Correlation** | Friedman $\chi^2$=7.00, $p$=3.020e-02 | p=1.0000 (ns) | p=0.0156 (*) | p=0.0234 (*) |
| **Strong_Weak_Gap** | Friedman $\chi^2$=6.25, $p$=4.394e-02 | p=0.7422 (ns) | p=0.0391 (*) | p=0.0547 (ns) |
| **Participants_Ratio** | Friedman $\chi^2$=9.25, $p$=9.804e-03 | p=0.6406 (ns) | p=0.0078 (**) | p=0.0156 (*) |
| **Gini_Mean** | Friedman $\chi^2$=4.75, $p$=9.301e-02 | p=0.9453 (ns) | p=0.0547 (ns) | p=0.1484 (ns) |
| **PR_Mean** | Friedman $\chi^2$=1.75, $p$=4.169e-01 | p=0.9453 (ns) | p=0.1953 (ns) | p=0.1953 (ns) |
| **Effective_Dim_PR** | Friedman $\chi^2$=0.75, $p$=6.873e-01 | p=0.7422 (ns) | p=0.7422 (ns) | p=0.6406 (ns) |
| **Sig_Mean_Corr** | Friedman $\chi^2$=13.00, $p$=1.503e-03 | p=0.3125 (ns) | p=0.0078 (**) | p=0.0078 (**) |
| **Noise_Mean_Corr** | Friedman $\chi^2$=2.25, $p$=3.247e-01 | p=0.1484 (ns) | p=0.1953 (ns) | p=0.8438 (ns) |
| **Geom_AngleDeg** | Friedman $\chi^2$=0.25, $p$=8.825e-01 | p=0.3125 (ns) | p=0.5469 (ns) | p=1.0000 (ns) |
| **Geom_OrthParallelRatio** | Friedman $\chi^2$=0.75, $p$=6.873e-01 | p=0.4609 (ns) | p=0.5469 (ns) | p=0.9453 (ns) |
| **Geom_VarParallel** | Friedman $\chi^2$=1.75, $p$=4.169e-01 | p=0.8438 (ns) | p=0.2500 (ns) | p=0.1953 (ns) |
| **Geom_VarOrthogonal** | Friedman $\chi^2$=13.00, $p$=1.503e-03 | p=0.1953 (ns) | p=0.0078 (**) | p=0.0078 (**) |
| **Geom_Anisotropy** | Friedman $\chi^2$=0.25, $p$=8.825e-01 | p=0.4609 (ns) | p=0.8438 (ns) | p=0.8438 (ns) |

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

## 4. Shuffle: Original vs Shuffled (Across Mice)

| metric             | Condition   |   n_mice |   orig_mean |   shuffled_mean |   delta_orig_minus_shuffled |   wilcoxon_p |
|:-------------------|:------------|---------:|------------:|----------------:|----------------------------:|-------------:|
| mean_corr          | Divergent   |        8 |   0.205062  |       0.115292  |                 0.0897703   |    0.0078125 |
| mean_corr          | Convergent  |        8 |   0.207438  |       0.119134  |                 0.088304    |    0.0078125 |
| mean_corr          | Random      |        8 |   0.198456  |       0.112923  |                 0.0855336   |    0.0078125 |
| weak_corr          | Divergent   |        8 |   0.072833  |       0.039736  |                 0.033097    |    0.0078125 |
| weak_corr          | Convergent  |        8 |   0.0727206 |       0.0388838 |                 0.0338368   |    0.0078125 |
| weak_corr          | Random      |        8 |   0.0546315 |       0.0193185 |                 0.0353131   |    0.0078125 |
| strong_corr        | Divergent   |        8 |   0.389392  |       0.2021    |                 0.187291    |    0.0078125 |
| strong_corr        | Convergent  |        8 |   0.390203  |       0.208351  |                 0.181853    |    0.0078125 |
| strong_corr        | Random      |        8 |   0.391037  |       0.226134  |                 0.164903    |    0.0078125 |
| strong_weak_gap    | Divergent   |        8 |   0.316559  |       0.162364  |                 0.154194    |    0.0078125 |
| strong_weak_gap    | Convergent  |        8 |   0.317483  |       0.169467  |                 0.148016    |    0.0078125 |
| strong_weak_gap    | Random      |        8 |   0.336405  |       0.206815  |                 0.12959     |    0.0078125 |
| mean_rsm           | Divergent   |        8 |   0.553446  |       0.530193  |                 0.0232532   |    0.0078125 |
| mean_rsm           | Convergent  |        8 |   0.567306  |       0.547285  |                 0.0200208   |    0.0078125 |
| mean_rsm           | Random      |        8 |   0.626439  |       0.608634  |                 0.0178047   |    0.0078125 |
| pr_mean            | Divergent   |        8 |  42.7208    |      36.4905    |                 6.2303      |    0.0078125 |
| pr_mean            | Convergent  |        8 |  42.1036    |      37.0944    |                 5.00927     |    0.0078125 |
| pr_mean            | Random      |        8 |  35.9484    |      31.6507    |                 4.29772     |    0.0078125 |
| participants_ratio | Divergent   |        8 |   1.724     |       1.724     |                -5.55112e-17 |    0.75      |
| participants_ratio | Convergent  |        8 |   1.66522   |       1.66522   |                 1.94289e-16 |    0.0625    |
| participants_ratio | Random      |        8 |   2.52793   |       2.52793   |                -1.66533e-16 |    0.5       |
| gini_mean          | Divergent   |        8 |   0.60557   |       0.615792  |                -0.0102224   |    0.0078125 |
| gini_mean          | Convergent  |        8 |   0.604039  |       0.612553  |                -0.00851376  |    0.0078125 |
| gini_mean          | Random      |        8 |   0.629104  |       0.635276  |                -0.00617178  |    0.0078125 |

## 5. Shuffle: Condition Differences in Shuffled Surrogates

| metric             | data_type   | main_effect                            |     p_main |   Divergent_vs_Convergent |   Divergent_vs_Random |   Convergent_vs_Random |
|:-------------------|:------------|:---------------------------------------|-----------:|--------------------------:|----------------------:|-----------------------:|
| mean_corr          | shuffled    | Friedman $\chi^2$=2.25, $p$=3.247e-01  | 0.324652   |                  0.25     |             0.25      |              0.382812  |
| weak_corr          | shuffled    | Friedman $\chi^2$=4.75, $p$=9.301e-02  | 0.0930145  |                  0.84375  |             0.015625  |              0.0390625 |
| strong_corr        | shuffled    | Friedman $\chi^2$=7.00, $p$=3.020e-02  | 0.0301974  |                  0.382812 |             0.015625  |              0.0390625 |
| strong_weak_gap    | shuffled    | Friedman $\chi^2$=13.00, $p$=1.503e-03 | 0.00150344 |                  0.25     |             0.0078125 |              0.0078125 |
| mean_rsm           | shuffled    | Friedman $\chi^2$=7.00, $p$=3.020e-02  | 0.0301974  |                  0.3125   |             0.015625  |              0.015625  |
| pr_mean            | shuffled    | Friedman $\chi^2$=0.25, $p$=8.825e-01  | 0.882497   |                  0.84375  |             0.382812  |              0.382812  |
| participants_ratio | shuffled    | Friedman $\chi^2$=9.25, $p$=9.804e-03  | 0.00980366 |                  0.640625 |             0.0078125 |              0.015625  |
| gini_mean          | shuffled    | Friedman $\chi^2$=4.75, $p$=9.301e-02  | 0.0930145  |                  1        |             0.109375  |              0.109375  |

## 6. Shuffle: Synchrony Contribution (Random - Coherent)

| metric             |   n_mice |   mean_sync_contribution_abs |   sem_sync_contribution_abs |   median_sync_contribution_abs |   n_positive |   ratio_positive |   wilcoxon_stat |   wilcoxon_p_two_sided |   wilcoxon_p_one_sided_greater |   binom_p_one_sided_greater |
|:-------------------|---------:|-----------------------------:|----------------------------:|-------------------------------:|-------------:|-----------------:|----------------:|-----------------------:|-------------------------------:|----------------------------:|
| gini_mean          |        8 |                  0.00173658  |                 0.00130139  |                    0.00192192  |            7 |            0.875 |               7 |              0.148438  |                      0.0742188 |                   0.0351562 |
| mean_corr          |        8 |                  0.00234661  |                 0.00251644  |                    0.00397008  |            6 |            0.75  |              13 |              0.546875  |                      0.273438  |                   0.144531  |
| mean_rsm           |        8 |                 -0.00309472  |                 0.00296479  |                   -0.00064327  |            4 |            0.5   |              12 |              0.460938  |                      0.808594  |                   0.636719  |
| participants_ratio |        8 |                 -7.63278e-17 |                 6.02069e-17 |                   -1.11022e-16 |            1 |            0.125 |               6 |              0.4375    |                      0.796875  |                   0.996094  |
| pr_mean            |        8 |                  0.962743    |                 0.683622    |                    1.76795     |            5 |            0.625 |               9 |              0.25      |                      0.125     |                   0.363281  |
| strong_corr        |        8 |                 -0.0117515   |                 0.00427268  |                   -0.0137348   |            1 |            0.125 |               2 |              0.0234375 |                      0.992188  |                   0.996094  |
| strong_weak_gap    |        8 |                 -0.0183852   |                 0.0022885   |                   -0.0190119   |            0 |            0     |               0 |              0.0078125 |                      1         |                   1         |
| weak_corr          |        8 |                 -0.000888617 |                 0.00234059  |                   -0.00141121  |            3 |            0.375 |              15 |              0.742188  |                      0.679688  |                   0.855469  |

## 7. Group Shuffle Output Files

- group_shuffle_manifest: `./results/group_summary/group_shuffle_manifest.csv`
- group_shuffle_corr_long: `./results/group_summary/group_shuffle_corr_long.csv`
- group_shuffle_corr_decile_long: `./results/group_summary/group_shuffle_corr_decile_long.csv`
- group_shuffle_rsm_long: `./results/group_summary/group_shuffle_rsm_long.csv`
- group_shuffle_delta_long: `./results/group_summary/group_shuffle_delta_long.csv`
- group_shuffle_dose_long: `./results/group_summary/group_shuffle_dose_long.csv`
- group_shuffle_alloc_long: `./results/group_summary/group_shuffle_alloc_long.csv`
- group_shuffle_effect_stats_raw: `./results/group_summary/group_shuffle_effect_stats_raw.csv`
- group_shuffle_condition_summary_raw: `./results/group_summary/group_shuffle_condition_summary_raw.csv`
- group_shuffle_condition_stats_raw: `./results/group_summary/group_shuffle_condition_stats_raw.csv`
- group_shuffle_sync_contribution_raw: `./results/group_summary/group_shuffle_sync_contribution_raw.csv`
- group_shuffle_sync_contribution_repeats_raw: `./results/group_summary/group_shuffle_sync_contribution_repeats_raw.csv`
- group_shuffle_core_long: `./results/group_summary/group_shuffle_core_long.csv`
- group_shuffle_original_vs_shuffled_stats: `./results/group_summary/group_shuffle_original_vs_shuffled_stats.csv`
- group_shuffle_condition_stats: `./results/group_summary/group_shuffle_condition_stats.csv`
- group_shuffle_sync_contribution_stats: `./results/group_summary/group_shuffle_sync_contribution_stats.csv`

## Geometry Condition-level Table

| mouse    |   Class_ID | Condition   |   n_trials |   n_neurons |   mean_norm |   angle_deg |   var_parallel |   var_orthogonal |   orth_parallel_ratio |   anisotropy_index |   lambda1 |   lambda2 | mouse_id   |
|:---------|-----------:|:------------|-----------:|------------:|------------:|------------:|---------------:|-----------------:|----------------------:|-------------------:|----------:|----------:|:-----------|
| M21_1107 |          1 | Divergent   |         35 |         191 |    0.930663 |     29.2844 |      0.141885  |         0.647991 |               4.56703 |           0.222759 |  0.181127 | 0.0990992 | M21_1107   |
| M21_1107 |          2 | Convergent  |         35 |         191 |    0.975667 |     79.2438 |      0.0972435 |         0.685729 |               7.05167 |           0.188413 |  0.151861 | 0.125911  | M21_1107   |
| M21_1107 |          3 | Random      |         35 |         191 |    1.18398  |     27.6128 |      0.124594  |         0.609712 |               4.89359 |           0.206362 |  0.15599  | 0.106094  | M21_1107   |
| M71_1024 |          1 | Divergent   |         45 |         226 |    1.61159  |     84.0488 |      0.177297  |         1.72634  |               9.737   |           0.260629 |  0.507419 | 0.313839  | M71_1024   |
| M71_1024 |          2 | Convergent  |         45 |         226 |    1.54524  |     67.5173 |      0.284193  |         1.84035  |               6.47572 |           0.243076 |  0.528163 | 0.349052  | M71_1024   |
| M71_1024 |          3 | Random      |         45 |         226 |    1.6686   |     89.5073 |      0.21658   |         1.6927   |               7.81561 |           0.201928 |  0.3943   | 0.283752  | M71_1024   |
| M73_1128 |          1 | Divergent   |         35 |         513 |    1.60256  |     74.4984 |      0.161946  |         2.05918  |              12.7152  |           0.207193 |  0.473738 | 0.23095   | M73_1128   |
| M73_1128 |          2 | Convergent  |         35 |         513 |    1.79506  |     79.4989 |      0.1658    |         2.19963  |              13.2668  |           0.137996 |  0.33602  | 0.254993  | M73_1128   |
| M73_1128 |          3 | Random      |         35 |         513 |    1.82326  |     64.143  |      0.208132  |         1.55028  |               7.44855 |           0.277517 |  0.502343 | 0.223846  | M73_1128   |
| M77_1031 |          1 | Divergent   |         43 |         711 |    2.04162  |     54.9905 |      0.463784  |         3.89672  |               8.40202 |           0.146729 |  0.655046 | 0.493198  | M77_1031   |
| M77_1031 |          2 | Convergent  |         43 |         711 |    2.28143  |     73.8832 |      0.315015  |         4.19208  |              13.3076  |           0.177047 |  0.816965 | 0.406154  | M77_1031   |
| M77_1031 |          3 | Random      |         43 |         711 |    2.01525  |     80.4567 |      0.1677    |         2.99294  |              17.847   |           0.172635 |  0.558627 | 0.409007  | M77_1031   |
| M77_1107 |          1 | Divergent   |         55 |         173 |    0.896925 |     69.0118 |      0.1006    |         0.721384 |               7.17084 |           0.229903 |  0.192048 | 0.119293  | M77_1107   |
| M77_1107 |          2 | Convergent  |         55 |         173 |    1.03064  |     84.087  |      0.107333  |         0.768788 |               7.16264 |           0.187949 |  0.167342 | 0.127828  | M77_1107   |
| M77_1107 |          3 | Random      |         55 |         173 |    0.997432 |     63.0142 |      0.0819622 |         0.630736 |               7.69545 |           0.19569  |  0.141753 | 0.124015  | M77_1107   |
| M78_1017 |          1 | Divergent   |         42 |         726 |    1.97332  |     77.8526 |      0.3608    |         3.57311  |               9.90329 |           0.163662 |  0.658093 | 0.506221  | M78_1017   |
| M78_1017 |          2 | Convergent  |         42 |         726 |    2.04763  |     60.496  |      0.366348  |         3.44986  |               9.4169  |           0.231541 |  0.90309  | 0.486939  | M78_1017   |
| M78_1017 |          3 | Random      |         42 |         726 |    2.30662  |     64.4017 |      0.263672  |         2.59946  |               9.85869 |           0.170278 |  0.498324 | 0.333846  | M78_1017   |
| M79_1128 |          1 | Divergent   |         55 |         365 |    1.43924  |     29.9956 |      0.219071  |         1.42545  |               6.50679 |           0.148476 |  0.248092 | 0.176446  | M79_1128   |
| M79_1128 |          2 | Convergent  |         55 |         365 |    1.47514  |     27.5671 |      0.262623  |         1.56694  |               5.96649 |           0.169889 |  0.315752 | 0.23992   | M79_1128   |
| M79_1128 |          3 | Random      |         55 |         365 |    1.37124  |     73.3876 |      0.189168  |         1.39097  |               7.3531  |           0.164968 |  0.264921 | 0.197973  | M79_1128   |
| M91_1017 |          1 | Divergent   |         45 |         235 |    1.31369  |     24.8887 |      0.214378  |         0.752233 |               3.50892 |           0.256999 |  0.252812 | 0.197024  | M91_1017   |
| M91_1017 |          2 | Convergent  |         45 |         235 |    1.10461  |     50.4873 |      0.0975522 |         0.698063 |               7.15579 |           0.189442 |  0.153728 | 0.126293  | M91_1017   |
| M91_1017 |          3 | Random      |         45 |         235 |    1.21016  |     47.5334 |      0.107223  |         0.612237 |               5.70995 |           0.229261 |  0.168134 | 0.12052   | M91_1017   |

## Geometry Pairwise Bootstrap Table

| metric              | condition_1   | condition_2   |   n_boot |   mean_diff_boot |     ci95_low |   ci95_high |   p_boot_two_sided | mouse_id   |
|:--------------------|:--------------|:--------------|---------:|-----------------:|-------------:|------------:|-------------------:|:-----------|
| angle_deg           | Divergent     | Convergent    |      500 |    -22.0003      | -53.3333     |  27.3752    |         0.291417   | M21_1107   |
| angle_deg           | Divergent     | Random        |      500 |     -0.492716    | -46.0246     |  44.6538    |         0.974052   | M21_1107   |
| angle_deg           | Convergent    | Random        |      500 |     21.5076      | -27.9866     |  57.5263    |         0.355289   | M21_1107   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |     -2.03376     |  -7.2328     |   4.18802   |         0.45509    | M21_1107   |
| orth_parallel_ratio | Divergent     | Random        |      500 |     -0.274783    |  -5.89566    |   5.34625   |         0.906188   | M21_1107   |
| orth_parallel_ratio | Convergent    | Random        |      500 |      1.75898     |  -4.12806    |   7.00167   |         0.443114   | M21_1107   |
| var_parallel        | Divergent     | Convergent    |      500 |      0.0469029   |  -0.0525867  |   0.188916  |         0.566866   | M21_1107   |
| var_parallel        | Divergent     | Random        |      500 |      0.0210566   |  -0.131137   |   0.182908  |         0.830339   | M21_1107   |
| var_parallel        | Convergent    | Random        |      500 |     -0.0258463   |  -0.138586   |   0.0569661 |         0.586826   | M21_1107   |
| var_orthogonal      | Divergent     | Convergent    |      500 |     -0.043905    |  -0.22844    |   0.137537  |         0.678643   | M21_1107   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.0329952   |  -0.116409   |   0.190981  |         0.694611   | M21_1107   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.0769001   |  -0.0964636  |   0.245809  |         0.371257   | M21_1107   |
| anisotropy_index    | Divergent     | Convergent    |      500 |      0.0226348   |  -0.142615   |   0.195903  |         0.786427   | M21_1107   |
| anisotropy_index    | Divergent     | Random        |      500 |      0.0200782   |  -0.1255     |   0.189564  |         0.850299   | M21_1107   |
| anisotropy_index    | Convergent    | Random        |      500 |     -0.00255654  |  -0.138589   |   0.115431  |         0.998004   | M21_1107   |
| angle_deg           | Divergent     | Convergent    |      500 |     12.3375      | -15.2205     |  40.7777    |         0.467066   | M71_1024   |
| angle_deg           | Divergent     | Random        |      500 |      6.35164     | -20.5393     |  35.6395    |         0.714571   | M71_1024   |
| angle_deg           | Convergent    | Random        |      500 |     -5.9859      | -40.1058     |  30.3172    |         0.714571   | M71_1024   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |      2.88208     |  -2.18548    |   7.98358   |         0.239521   | M71_1024   |
| orth_parallel_ratio | Divergent     | Random        |      500 |      1.57428     |  -4.61378    |   7.40996   |         0.51497    | M71_1024   |
| orth_parallel_ratio | Convergent    | Random        |      500 |     -1.3078      |  -7.35826    |   4.65905   |         0.622754   | M71_1024   |
| var_parallel        | Divergent     | Convergent    |      500 |     -0.103632    |  -0.268532   |   0.0334904 |         0.147705   | M71_1024   |
| var_parallel        | Divergent     | Random        |      500 |     -0.0402205   |  -0.179339   |   0.0733011 |         0.562874   | M71_1024   |
| var_parallel        | Convergent    | Random        |      500 |      0.0634118   |  -0.126265   |   0.232341  |         0.479042   | M71_1024   |
| var_orthogonal      | Divergent     | Convergent    |      500 |     -0.117783    |  -0.485765   |   0.2125    |         0.479042   | M71_1024   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.0293206   |  -0.298471   |   0.348929  |         0.850299   | M71_1024   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.147103    |  -0.205702   |   0.529601  |         0.427146   | M71_1024   |
| anisotropy_index    | Divergent     | Convergent    |      500 |      0.00600092  |  -0.073185   |   0.087848  |         0.886228   | M71_1024   |
| anisotropy_index    | Divergent     | Random        |      500 |      0.0398645   |  -0.03487    |   0.115956  |         0.303393   | M71_1024   |
| anisotropy_index    | Convergent    | Random        |      500 |      0.0338636   |  -0.05883    |   0.118363  |         0.447106   | M71_1024   |
| angle_deg           | Divergent     | Convergent    |      500 |      0.70289     | -27.0063     |  31.6231    |         0.982036   | M73_1128   |
| angle_deg           | Divergent     | Random        |      500 |      9.98143     | -14.1344     |  35.3966    |         0.467066   | M73_1128   |
| angle_deg           | Convergent    | Random        |      500 |      9.27854     | -22.7282     |  35.2016    |         0.518962   | M73_1128   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |     -1.00305     | -13.4444     |   9.71501   |         0.882236   | M73_1128   |
| orth_parallel_ratio | Divergent     | Random        |      500 |      5.40745     |  -2.80225    |  15.3385    |         0.219561   | M73_1128   |
| orth_parallel_ratio | Convergent    | Random        |      500 |      6.4105      |  -1.87224    |  18.8739    |         0.167665   | M73_1128   |
| var_parallel        | Divergent     | Convergent    |      500 |     -0.00081459  |  -0.116395   |   0.112584  |         0.99002    | M73_1128   |
| var_parallel        | Divergent     | Random        |      500 |     -0.0464342   |  -0.186312   |   0.0836215 |         0.499002   | M73_1128   |
| var_parallel        | Convergent    | Random        |      500 |     -0.0456196   |  -0.198163   |   0.0976227 |         0.538922   | M73_1128   |
| var_orthogonal      | Divergent     | Convergent    |      500 |     -0.134505    |  -0.468529   |   0.228356  |         0.439122   | M73_1128   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.497343    |   0.143031   |   0.837297  |         0.00798403 | M73_1128   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.631848    |   0.301181   |   0.966085  |         0.00399202 | M73_1128   |
| anisotropy_index    | Divergent     | Convergent    |      500 |      0.0522637   |  -0.0156617  |   0.116705  |         0.143713   | M73_1128   |
| anisotropy_index    | Divergent     | Random        |      500 |     -0.0653078   |  -0.176081   |   0.0377015 |         0.211577   | M73_1128   |
| anisotropy_index    | Convergent    | Random        |      500 |     -0.117571    |  -0.21726    |  -0.0232343 |         0.00798403 | M73_1128   |
| angle_deg           | Divergent     | Convergent    |      500 |    -13.4492      | -49.8883     |  22.4432    |         0.499002   | M77_1031   |
| angle_deg           | Divergent     | Random        |      500 |    -19.0295      | -52.5355     |  12.8006    |         0.291417   | M77_1031   |
| angle_deg           | Convergent    | Random        |      500 |     -5.58029     | -28.6766     |  17.2482    |         0.662675   | M77_1031   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |     -4.60594     | -13.6816     |   5.03601   |         0.291417   | M77_1031   |
| orth_parallel_ratio | Divergent     | Random        |      500 |     -9.12115     | -18.0048     |  -0.243734  |         0.0558882  | M77_1031   |
| orth_parallel_ratio | Convergent    | Random        |      500 |     -4.51521     | -15.1311     |   5.26977   |         0.331337   | M77_1031   |
| var_parallel        | Divergent     | Convergent    |      500 |      0.146566    |  -0.138952   |   0.453404  |         0.371257   | M77_1031   |
| var_parallel        | Divergent     | Random        |      500 |      0.300124    |   0.070308   |   0.557784  |         0.011976   | M77_1031   |
| var_parallel        | Convergent    | Random        |      500 |      0.153558    |   0.00535173 |   0.356492  |         0.0479042  | M77_1031   |
| var_orthogonal      | Divergent     | Convergent    |      500 |     -0.28182     |  -1.13471    |   0.518743  |         0.502994   | M77_1031   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.866667    |   0.226511   |   1.50966   |         0.0159681  | M77_1031   |
| var_orthogonal      | Convergent    | Random        |      500 |      1.14849     |   0.443217   |   1.90526   |         0.00399202 | M77_1031   |
| anisotropy_index    | Divergent     | Convergent    |      500 |     -0.0191935   |  -0.119985   |   0.0645885 |         0.710579   | M77_1031   |
| anisotropy_index    | Divergent     | Random        |      500 |     -0.0207722   |  -0.0899838  |   0.0549361 |         0.578842   | M77_1031   |
| anisotropy_index    | Convergent    | Random        |      500 |     -0.00157875  |  -0.0893661  |   0.104737  |         0.906188   | M77_1031   |
| angle_deg           | Divergent     | Convergent    |      500 |     -1.49439     | -30.1068     |  37.4194    |         0.882236   | M77_1107   |
| angle_deg           | Divergent     | Random        |      500 |      0.346564    | -32.1722     |  30.9815    |         0.9501     | M77_1107   |
| angle_deg           | Convergent    | Random        |      500 |      1.84096     | -35.1704     |  36.6629    |         0.894212   | M77_1107   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |      0.0124164   |  -3.73075    |   4.30753   |         0.946108   | M77_1107   |
| orth_parallel_ratio | Divergent     | Random        |      500 |     -0.533238    |  -5.30845    |   4.21618   |         0.790419   | M77_1107   |
| orth_parallel_ratio | Convergent    | Random        |      500 |     -0.545655    |  -4.80244    |   3.43691   |         0.810379   | M77_1107   |
| var_parallel        | Divergent     | Convergent    |      500 |     -0.00476881  |  -0.0561787  |   0.0438166 |         0.866267   | M77_1107   |
| var_parallel        | Divergent     | Random        |      500 |      0.0192483   |  -0.0296761  |   0.0707501 |         0.487026   | M77_1107   |
| var_parallel        | Convergent    | Random        |      500 |      0.0240171   |  -0.0276403  |   0.0728466 |         0.311377   | M77_1107   |
| var_orthogonal      | Divergent     | Convergent    |      500 |     -0.0440339   |  -0.167842   |   0.0659321 |         0.443114   | M77_1107   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.0895772   |  -0.0342097  |   0.199638  |         0.143713   | M77_1107   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.133611    |   0.0294029  |   0.25055   |         0.0199601  | M77_1107   |
| anisotropy_index    | Divergent     | Convergent    |      500 |      0.0338286   |  -0.0445636  |   0.111276  |         0.403194   | M77_1107   |
| anisotropy_index    | Divergent     | Random        |      500 |      0.0208187   |  -0.0485479  |   0.0954669 |         0.59481    | M77_1107   |
| anisotropy_index    | Convergent    | Random        |      500 |     -0.0130098   |  -0.0748303  |   0.0535301 |         0.662675   | M77_1107   |
| angle_deg           | Divergent     | Convergent    |      500 |      6.83021     | -26.9825     |  32.7743    |         0.642715   | M78_1017   |
| angle_deg           | Divergent     | Random        |      500 |      4.31234     | -32.089      |  36.1088    |         0.798403   | M78_1017   |
| angle_deg           | Convergent    | Random        |      500 |     -2.51787     | -29.3147     |  24.3501    |         0.834331   | M78_1017   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |      0.647651    |  -7.82665    |  11.865     |         0.938124   | M78_1017   |
| orth_parallel_ratio | Divergent     | Random        |      500 |      0.135269    |  -7.93621    |  11.1948    |         0.902196   | M78_1017   |
| orth_parallel_ratio | Convergent    | Random        |      500 |     -0.512382    |  -8.20006    |   7.30697   |         0.894212   | M78_1017   |
| var_parallel        | Divergent     | Convergent    |      500 |     -0.00279358  |  -0.326498   |   0.291671  |         0.97006    | M78_1017   |
| var_parallel        | Divergent     | Random        |      500 |      0.110433    |  -0.143839   |   0.397686  |         0.439122   | M78_1017   |
| var_parallel        | Convergent    | Random        |      500 |      0.113226    |  -0.129208   |   0.410242  |         0.447106   | M78_1017   |
| var_orthogonal      | Divergent     | Convergent    |      500 |      0.143926    |  -0.642016   |   0.979761  |         0.730539   | M78_1017   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.940813    |   0.346863   |   1.55245   |         0.00399202 | M78_1017   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.796887    |   0.114194   |   1.554     |         0.0319361  | M78_1017   |
| anisotropy_index    | Divergent     | Convergent    |      500 |     -0.0590203   |  -0.172475   |   0.0380819 |         0.203593   | M78_1017   |
| anisotropy_index    | Divergent     | Random        |      500 |     -0.000342146 |  -0.0766505  |   0.0785131 |         0.966068   | M78_1017   |
| anisotropy_index    | Convergent    | Random        |      500 |      0.0586781   |  -0.0502171  |   0.187264  |         0.275449   | M78_1017   |
| angle_deg           | Divergent     | Convergent    |      500 |      2.26838     | -46.4415     |  50.1637    |         0.99002    | M79_1128   |
| angle_deg           | Divergent     | Random        |      500 |    -19.9779      | -62.1814     |  41.9151    |         0.43513    | M79_1128   |
| angle_deg           | Convergent    | Random        |      500 |    -22.2463      | -55.5023     |  27.0327    |         0.383234   | M79_1128   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |      0.642426    |  -4.32892    |   6.11327   |         0.854291   | M79_1128   |
| orth_parallel_ratio | Divergent     | Random        |      500 |     -0.734605    |  -6.4265     |   5.20731   |         0.766467   | M79_1128   |
| orth_parallel_ratio | Convergent    | Random        |      500 |     -1.37703     |  -6.96694    |   3.76142   |         0.578842   | M79_1128   |
| var_parallel        | Divergent     | Convergent    |      500 |     -0.0429186   |  -0.206975   |   0.107863  |         0.582834   | M79_1128   |
| var_parallel        | Divergent     | Random        |      500 |      0.0263018   |  -0.111012   |   0.145822  |         0.690619   | M79_1128   |
| var_parallel        | Convergent    | Random        |      500 |      0.0692204   |  -0.0779739  |   0.224012  |         0.347305   | M79_1128   |
| var_orthogonal      | Divergent     | Convergent    |      500 |     -0.140192    |  -0.421142   |   0.121251  |         0.323353   | M79_1128   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.0331127   |  -0.207655   |   0.265609  |         0.802395   | M79_1128   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.173305    |  -0.0690716  |   0.457755  |         0.187625   | M79_1128   |
| anisotropy_index    | Divergent     | Convergent    |      500 |     -0.0256019   |  -0.0945727  |   0.0359148 |         0.45509    | M79_1128   |
| anisotropy_index    | Divergent     | Random        |      500 |     -0.0196749   |  -0.0911633  |   0.0474241 |         0.542914   | M79_1128   |
| anisotropy_index    | Convergent    | Random        |      500 |      0.00592699  |  -0.0597105  |   0.0777585 |         0.89022    | M79_1128   |
| angle_deg           | Divergent     | Convergent    |      500 |    -16.0106      | -50.3639     |  31.791     |         0.359281   | M91_1017   |
| angle_deg           | Divergent     | Random        |      500 |    -13.1433      | -43.4064     |  26.9955    |         0.447106   | M91_1017   |
| angle_deg           | Convergent    | Random        |      500 |      2.86729     | -28.5414     |  38.8603    |         0.898204   | M91_1017   |
| orth_parallel_ratio | Divergent     | Convergent    |      500 |     -3.54837     |  -7.50137    |   0.159094  |         0.0718563  | M91_1017   |
| orth_parallel_ratio | Divergent     | Random        |      500 |     -2.26596     |  -5.71663    |   0.792753  |         0.167665   | M91_1017   |
| orth_parallel_ratio | Convergent    | Random        |      500 |      1.28241     |  -2.82402    |   5.64264   |         0.506986   | M91_1017   |
| var_parallel        | Divergent     | Convergent    |      500 |      0.11918     |  -0.00419875 |   0.263341  |         0.0638723  | M91_1017   |
| var_parallel        | Divergent     | Random        |      500 |      0.112309    |  -0.00217124 |   0.264543  |         0.0638723  | M91_1017   |
| var_parallel        | Convergent    | Random        |      500 |     -0.00687098  |  -0.0719919  |   0.0460565 |         0.898204   | M91_1017   |
| var_orthogonal      | Divergent     | Convergent    |      500 |      0.0528678   |  -0.108979   |   0.233228  |         0.558882   | M91_1017   |
| var_orthogonal      | Divergent     | Random        |      500 |      0.136664    |  -0.0551122  |   0.3282    |         0.167665   | M91_1017   |
| var_orthogonal      | Convergent    | Random        |      500 |      0.0837963   |  -0.0672538  |   0.226984  |         0.279441   | M91_1017   |
| anisotropy_index    | Divergent     | Convergent    |      500 |      0.077056    |  -0.0231853  |   0.202309  |         0.151697   | M91_1017   |
| anisotropy_index    | Divergent     | Random        |      500 |      0.0337465   |  -0.0766894  |   0.155027  |         0.598802   | M91_1017   |
| anisotropy_index    | Convergent    | Random        |      500 |     -0.0433095   |  -0.135257   |   0.0258205 |         0.275449   | M91_1017   |

## Geometry Mixed-model Summary Table

| model_name   | formula                                                    | term                   |      beta |    p_value |     aic |     bic |       llf |   n_obs |   n_mice | converged   | note                        |
|:-------------|:-----------------------------------------------------------|:-----------------------|----------:|-----------:|--------:|--------:|----------:|--------:|---------:|:------------|:----------------------------|
| M1           | Mean_RSM_Sim ~ Geom_AngleDeg                               | Geom_AngleDeg          | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| M2           | Mean_RSM_Sim ~ Geom_OrthParallelRatio                      | Geom_OrthParallelRatio | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| M3           | Mean_RSM_Sim ~ Participants_Ratio + Geom_AngleDeg          | Participants_Ratio     | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| M3           | Mean_RSM_Sim ~ Participants_Ratio + Geom_AngleDeg          | Geom_AngleDeg          | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| M4           | Mean_RSM_Sim ~ Participants_Ratio + Geom_OrthParallelRatio | Participants_Ratio     | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| M4           | Mean_RSM_Sim ~ Participants_Ratio + Geom_OrthParallelRatio | Geom_OrthParallelRatio | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| A1           | Geom_AngleDeg ~ Participants_Ratio                         | Participants_Ratio     |   4.38759 |   0.507846 | 217.823 | 222.535 | -104.911  |      24 |        8 | True        |                             |
| A2           | Geom_OrthParallelRatio ~ Participants_Ratio                | Participants_Ratio     |  -0.22313 |   0.806376 | 126.281 | 130.993 |  -59.1403 |      24 |        8 | True        |                             |
| D1           | Mean_RSM_Sim ~ Effective_Dim_PR                            | Effective_Dim_PR       | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| D2           | Mean_RSM_Sim ~ Geom_AngleDeg + Effective_Dim_PR            | Geom_AngleDeg          | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| D2           | Mean_RSM_Sim ~ Geom_AngleDeg + Effective_Dim_PR            | Effective_Dim_PR       | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| D3           | Mean_RSM_Sim ~ Geom_OrthParallelRatio + Effective_Dim_PR   | Geom_OrthParallelRatio | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |
| D3           | Mean_RSM_Sim ~ Geom_OrthParallelRatio + Effective_Dim_PR   | Effective_Dim_PR       | nan       | nan        | nan     | nan     |  nan      |      24 |        8 | False       | fit failed: Singular matrix |

## Geometry Output Files

- group_geometry_condition_level_long: `./results/group_summary/group_geometry_condition_level_long.csv`
- group_geometry_condition_pairwise_long: `./results/group_summary/group_geometry_condition_pairwise_long.csv`
- group_geometry_model_compare_long: `./results/group_summary/group_geometry_model_compare_long.csv`
- group_geometry_rsm_model_compare: `./results/group_summary/group_geometry_rsm_model_compare.csv`
- group_geometry_rsm_lmm_summary: `./results/group_summary/group_geometry_rsm_lmm_summary.md`
- group_geometry_allocation_lmm_summary: `./results/group_summary/group_geometry_allocation_lmm_summary.md`
- group_geometry_vs_dimensionality_model_compare: `./results/group_summary/group_geometry_vs_dimensionality_model_compare.csv`

## 8. Figures

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

### Geometry Angle Condition
![Geometry Angle Condition](./group_geometry_angle_condition.png)

### Geometry Orth/Parallel Ratio Condition
![Geometry Orth/Parallel Ratio Condition](./group_geometry_orth_parallel_condition.png)

### Geometry Angle vs Mean RSM
![Geometry Angle vs Mean RSM](./group_geometry_angle_vs_rsm.png)

### Geometry Ratio vs Mean RSM
![Geometry Ratio vs Mean RSM](./group_geometry_ratio_vs_rsm.png)

### Shuffle Orig-vs-Shuffled: weak_corr
![Shuffle Orig-vs-Shuffled: weak_corr](./group_shuffle_orig_vs_shuffled_weak_corr.png)

### Shuffle Condition Difference: weak_corr
![Shuffle Condition Difference: weak_corr](./group_shuffle_condition_shuffled_weak_corr.png)

### Shuffle Dose-response: weak_corr
![Shuffle Dose-response: weak_corr](./group_shuffle_dose_weak_corr.png)

### Shuffle Orig-vs-Shuffled: strong_weak_gap
![Shuffle Orig-vs-Shuffled: strong_weak_gap](./group_shuffle_orig_vs_shuffled_strong_weak_gap.png)

### Shuffle Condition Difference: strong_weak_gap
![Shuffle Condition Difference: strong_weak_gap](./group_shuffle_condition_shuffled_strong_weak_gap.png)

### Shuffle Dose-response: strong_weak_gap
![Shuffle Dose-response: strong_weak_gap](./group_shuffle_dose_strong_weak_gap.png)

### Shuffle Orig-vs-Shuffled: mean_rsm
![Shuffle Orig-vs-Shuffled: mean_rsm](./group_shuffle_orig_vs_shuffled_mean_rsm.png)

### Shuffle Condition Difference: mean_rsm
![Shuffle Condition Difference: mean_rsm](./group_shuffle_condition_shuffled_mean_rsm.png)

### Shuffle Dose-response: mean_rsm
![Shuffle Dose-response: mean_rsm](./group_shuffle_dose_mean_rsm.png)

### Shuffle Orig-vs-Shuffled: pr_mean
![Shuffle Orig-vs-Shuffled: pr_mean](./group_shuffle_orig_vs_shuffled_pr_mean.png)

### Shuffle Condition Difference: pr_mean
![Shuffle Condition Difference: pr_mean](./group_shuffle_condition_shuffled_pr_mean.png)

### Shuffle Dose-response: pr_mean
![Shuffle Dose-response: pr_mean](./group_shuffle_dose_pr_mean.png)

### Shuffle Delta by Condition
![Shuffle Delta by Condition](./group_shuffle_delta_by_condition.png)

### Shuffle Synchrony Contribution
![Shuffle Synchrony Contribution](./group_shuffle_sync_contribution_abs.png)

