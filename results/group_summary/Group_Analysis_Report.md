# Group-level Multi-mouse Analysis Report

## 1. Dataset Overview

- Number of mice: 8
- Mouse IDs: M21_1107, M71_1024, M73_1128, M77_1031, M77_1107, M78_1017, M79_1128, M91_1017
- Conditions: Divergent, Convergent, Random

## 2. Exported Data Tables

- Master metrics: `group_master_metrics.csv`
- Correlation deciles long: `group_corr_deciles_long.csv`
- Noise decile coupling long: `group_noise_corr_decile_coupling_long.csv`
- RR overlap long: `group_rr_overlap_long.csv`
- Task1 decoder summary long: `group_decoder_summary_long.csv`
- Task2 ablation summary long: `group_decoder_ablation_summary_long.csv`
- Task3 FC decoder summary long: `group_fc_decoder_summary_long.csv`
- Task4 edge stability long: `group_fc_edge_importance_stability_long.csv`
- Task4 edge stability mouse summary: `group_fc_edge_importance_mouse_summary.csv`
- Task4 edge ablation long: `group_fc_edge_ablation_long.csv`
- Task4 projection decile long: `group_fc_projection_by_strength_decile_long.csv`
- Task4 projection layer pair long: `group_fc_projection_by_layer_pair_long.csv`
- Task4 projection strong-weak match long: `group_fc_projection_strong_weak_match_long.csv`
- Task5 edge decile enrichment long: `group_fc_edge_decile_enrichment_long.csv`
- Task6 neuron overlap enrichment long: `group_neuron_overlap_enrichment_long.csv`
- Task6 neuron selectivity by overlap long: `group_neuron_selectivity_by_overlap_long.csv`
- Condition-level statistical tests: `group_statistical_tests_summary.csv`
- Task1-6 decoder chain summary: `group_decoder_chain_summary.csv`
- Task1-6 decoder chain statistical tests: `group_decoder_chain_stat_tests.csv`

## 3. Descriptive Statistics (Mean ± SEM)

| Condition   | Entropy         | Mean_RSM_Sim    | Mean_Correlation   | Strong_Correlation   | Weak_Correlation   | Strong_Weak_Gap   | Participants_Ratio   | Gini_Mean       | Gini_STD        | PR_Mean          | PR_STD           | PR_Norm_Mean    | PR_Norm_STD     | Effective_Dim_PR   | Effective_Dim_eRank   | Effective_Dim_90Var   | Sig_Mean_Corr   | Noise_Mean_Corr   | SigAbs_Mean_Corr   | NoiseAbs_Mean_Corr   | SigNoise_Coupling_r   |
|:------------|:----------------|:----------------|:-------------------|:---------------------|:-------------------|:------------------|:---------------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|:-------------------|:----------------------|:----------------------|:----------------|:------------------|:-------------------|:---------------------|:----------------------|
| Convergent  | 3.6493 ± 0.0826 | 0.5698 ± 0.0097 | 0.2074 ± 0.0104    | 0.3902 ± 0.0115      | 0.0727 ± 0.0092    | 0.3175 ± 0.0060   | 1.6652 ± 0.1043      | 0.6075 ± 0.0184 | 0.0770 ± 0.0048 | 41.4055 ± 6.3583 | 17.9318 ± 2.9929 | 0.1176 ± 0.0147 | 0.0499 ± 0.0067 | 12.8488 ± 0.8609   | 20.1096 ± 1.1499      | 20.8750 ± 1.3554      | 0.8041 ± 0.0105 | 0.1386 ± 0.0160   | 0.8044 ± 0.0104    | 0.1910 ± 0.0102      | 0.1692 ± 0.0228       |
| Divergent   | 3.6928 ± 0.0682 | 0.5532 ± 0.0224 | 0.2051 ± 0.0114    | 0.3894 ± 0.0136      | 0.0728 ± 0.0099    | 0.3166 ± 0.0080   | 1.7240 ± 0.1179      | 0.6039 ± 0.0176 | 0.0828 ± 0.0054 | 43.2997 ± 6.9780 | 18.2171 ± 2.6207 | 0.1184 ± 0.0114 | 0.0512 ± 0.0042 | 11.6710 ± 0.8746   | 18.8798 ± 1.2750      | 20.3750 ± 1.3488      | 0.8128 ± 0.0076 | 0.1620 ± 0.0199   | 0.8129 ± 0.0075    | 0.2052 ± 0.0146      | 0.1702 ± 0.0169       |
| Random      | 3.5989 ± 0.0692 | 0.6264 ± 0.0181 | 0.1985 ± 0.0142    | 0.3910 ± 0.0154      | 0.0546 ± 0.0109    | 0.3364 ± 0.0107   | 2.5279 ± 0.2484      | 0.6291 ± 0.0196 | 0.0783 ± 0.0060 | 35.9484 ± 4.2529 | 16.5695 ± 2.2046 | 0.1074 ± 0.0149 | 0.0480 ± 0.0052 | 11.4929 ± 0.8069   | 18.7792 ± 1.1476      | 20.2500 ± 1.3059      | 0.7198 ± 0.0223 | 0.1384 ± 0.0185   | 0.7215 ± 0.0216    | 0.2044 ± 0.0113      | 0.1769 ± 0.0096       |

## 4. Friedman + Wilcoxon Tests

| Metric             | Main_Effect                            |   p_main | Main_Star   |   Div_vs_Con |   Div_vs_Rand |   Con_vs_Rand |
|:-------------------|:---------------------------------------|---------:|:------------|-------------:|--------------:|--------------:|
| Sig_Mean_Corr      | Friedman $\chi^2$=13.00, $p$=1.503e-03 |   0.0015 | **          |       0.3125 |        0.0078 |        0.0078 |
| Participants_Ratio | Friedman $\chi^2$=9.25, $p$=9.804e-03  |   0.0098 | **          |       0.6406 |        0.0078 |        0.0156 |
| Mean_RSM_Sim       | Friedman $\chi^2$=7.75, $p$=2.075e-02  |   0.0208 | *           |       0.25   |        0.0234 |        0.0156 |
| Weak_Correlation   | Friedman $\chi^2$=7.00, $p$=3.020e-02  |   0.0302 | *           |       1      |        0.0156 |        0.0234 |
| Strong_Weak_Gap    | Friedman $\chi^2$=6.25, $p$=4.394e-02  |   0.0439 | *           |       0.7422 |        0.0391 |        0.0547 |
| Gini_Mean          | Friedman $\chi^2$=4.75, $p$=9.301e-02  |   0.093  | ns          |       0.7422 |        0.0391 |        0.1484 |
| Mean_Correlation   | Friedman $\chi^2$=3.25, $p$=1.969e-01  |   0.1969 | ns          |       0.3125 |        0.3828 |        0.3125 |
| Noise_Mean_Corr    | Friedman $\chi^2$=2.25, $p$=3.247e-01  |   0.3247 | ns          |       0.1484 |        0.1953 |        0.8438 |
| PR_Mean            | Friedman $\chi^2$=1.75, $p$=4.169e-01  |   0.4169 | ns          |       0.7422 |        0.1484 |        0.25   |
| Entropy            | Friedman $\chi^2$=1.75, $p$=4.169e-01  |   0.4169 | ns          |       0.6406 |        0.1094 |        0.8438 |
| Effective_Dim_PR   | Friedman $\chi^2$=1.00, $p$=6.065e-01  |   0.6065 | ns          |       0.25   |        0.8438 |        0.1953 |
| Strong_Correlation | Friedman $\chi^2$=0.75, $p$=6.873e-01  |   0.6873 | ns          |       0.5469 |        0.6406 |        0.5469 |

## 5. RR Overlap Summary Across Mice

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

## 6. Decoder Chain Summary (Tasks 1-6)

| Metric                                  |   N_mice |       Mean |        SEM |
|:----------------------------------------|---------:|-----------:|-----------:|
| Task1 activity decoder accuracy         |        8 |  0.940278  | 0.0410772  |
| Task1 activity decoder minus shuffle    |        8 |  0.584948  | 0.0392507  |
| Task2 full minus top10 ablation         |        8 |  0.0111111 | 0.00481125 |
| Task3 FC decoder accuracy               |        8 |  0.675986  | 0.0511639  |
| Task3 FC decoder minus shuffle          |        8 |  0.341069  | 0.0508065  |
| Task4 top-edge ablation delta (drop=1%) |        8 |  0.0111833 | 0.00680579 |
| Task4 top-edge ablation delta (drop=3%) |        8 |  0.0111592 | 0.00920215 |
| Task4 top-edge ablation delta (drop=5%) |        8 |  0.0202891 | 0.0100752  |
| Task5 weak-tail log2 enrichment         |        8 | -1.79803   | 0.243811   |
| Task6 Shared_Core log2 enrichment       |        8 | -0.0550111 | 0.131404   |

## 7. Decoder Chain Statistical Tests (Tasks 1-6)

| Analysis                                          |   N_mice |   Mean_Delta |   SEM_Delta |   p_value | Significance   |
|:--------------------------------------------------|---------:|-------------:|------------:|----------:|:---------------|
| Task1: activity decoder vs shuffle                |        8 |       0.5849 |      0.0393 |    0.0078 | **             |
| Task2: full vs top10 neuron ablation              |        8 |       0.0111 |      0.0048 |    0.0781 | ns             |
| Task2: top10 ablation vs random drop              |        8 |      -0.0007 |      0.0022 |    0.6406 | ns             |
| Task3: FC decoder vs shuffle                      |        8 |       0.3411 |      0.0508 |    0.0078 | **             |
| Task3 vs Task1: FC decoder vs activity decoder    |        8 |      -0.2643 |      0.0383 |    0.0078 | **             |
| Task4: top-edge vs random-edge ablation (drop=1%) |        8 |      -0.0051 |      0.0128 |    0.7422 | ns             |
| Task4: top-edge vs random-edge ablation (drop=3%) |        8 |      -0.0015 |      0.0109 |    0.7422 | ns             |
| Task4: top-edge vs random-edge ablation (drop=5%) |        8 |       0.0067 |      0.0093 |    0.6406 | ns             |
| Task5: weak-tail vs strong-tail log2 enrichment   |        8 |      -2.8824 |      0.344  |    0.0078 | **             |
| Task5: weak-tail enrichment vs 0                  |        8 |      -1.798  |      0.2438 |    0.0078 | **             |
| Task6: Shared_Core vs Condition_Biased enrichment |        8 |      -0.0657 |      0.193  |    0.8438 | ns             |
| Task6: Shared_Core enrichment vs 0                |        8 |      -0.055  |      0.1314 |    0.8438 | ns             |

## 8. Figures

### Core and Correlation Metrics

#### Combined Strong vs Weak
![Combined Strong vs Weak](./group_combined_strong_weak.png)

#### RSM Mean Similarity
![RSM Mean Similarity](./group_mean_rsm_sim.png)

#### Strong Connections (Top 10%)
![Strong Connections (Top 10%)](./group_strong_correlation.png)

#### Weak Connections (Bottom 10%)
![Weak Connections (Bottom 10%)](./group_weak_correlation.png)

#### Strong-Weak Correlation Gap
![Strong-Weak Correlation Gap](./group_strong_weak_gap.png)

#### RR Participants Ratio
![RR Participants Ratio](./group_participants_ratio.png)

#### Response Gini (Mean)
![Response Gini (Mean)](./group_gini_mean.png)

#### Decile Correlation Curve
![Decile Correlation Curve](./group_corr_decile_curve.png)

#### Noise Decile Curve
![Noise Decile Curve](./group_noise_corr_decile_curve.png)

### Binding Analyses

#### Cross-animal Binding
![Cross-animal Binding](./group_cross_animal_binding.png)

#### Absolute State Binding
![Absolute State Binding](./group_absolute_state_binding.png)

#### LMM State Binding
![LMM State Binding](./group_lmm_state_binding.png)

### Decoder Chain (Tasks 1-6)

#### Decoder Accuracy (Task1+Task3)
![Decoder Accuracy (Task1+Task3)](./group_decoder_chain_accuracy.png)

#### Decoder Ablation (Task2)
![Decoder Ablation (Task2)](./group_decoder_ablation_task2.png)

#### Edge Ablation Robustness (Task4)
![Edge Ablation Robustness (Task4)](./group_fc_edge_ablation_task4.png)

#### Edge Decile Enrichment (Task5)
![Edge Decile Enrichment (Task5)](./group_fc_edge_decile_enrichment_task5.png)

#### Neuron Linking (Task6)
![Neuron Linking (Task6)](./group_neuron_linking_task6.png)

