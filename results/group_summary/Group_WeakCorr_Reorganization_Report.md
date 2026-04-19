# Group Weak-Correlation Reorganization Report

- n_mice used: **8**

## Condition Means (Mouse-level)

```
 Condition  mean_noise_corr  neg_frac  weak_pos_frac  strong_frac  strong_mean
  Coherent         0.150286  0.188203       0.213203     0.056607     0.543314
Convergent         0.138571  0.249526       0.188719     0.078885     0.536010
 Divergent         0.162002  0.213360       0.178671     0.096664     0.529309
    Random         0.138407  0.279678       0.168814     0.100011     0.530016
```

## Three-condition Tests (D/C/R)

```
                   scope          metric                        comparison  n_mice             test  p_value   p_holm  mean_delta
         three_condition mean_noise_corr Divergent vs Convergent vs Random       8         Friedman 0.324652      NaN         NaN
three_condition_pairwise mean_noise_corr           Divergent vs Convergent       8 Wilcoxon(paired) 0.148438 0.445312    0.023431
three_condition_pairwise mean_noise_corr               Divergent vs Random       8 Wilcoxon(paired) 0.195312 0.445312    0.023596
three_condition_pairwise mean_noise_corr              Convergent vs Random       8 Wilcoxon(paired) 0.843750 0.843750    0.000164
         three_condition        neg_frac Divergent vs Convergent vs Random       8         Friedman 0.043937      NaN         NaN
three_condition_pairwise        neg_frac           Divergent vs Convergent       8 Wilcoxon(paired) 0.109375 0.218750   -0.036165
three_condition_pairwise        neg_frac               Divergent vs Random       8 Wilcoxon(paired) 0.039062 0.117188   -0.066317
three_condition_pairwise        neg_frac              Convergent vs Random       8 Wilcoxon(paired) 0.109375 0.218750   -0.030152
         three_condition   weak_pos_frac Divergent vs Convergent vs Random       8         Friedman 0.043937      NaN         NaN
three_condition_pairwise   weak_pos_frac           Divergent vs Convergent       8 Wilcoxon(paired) 0.250000 0.500000   -0.010048
three_condition_pairwise   weak_pos_frac               Divergent vs Random       8 Wilcoxon(paired) 0.312500 0.500000    0.009857
three_condition_pairwise   weak_pos_frac              Convergent vs Random       8 Wilcoxon(paired) 0.007812 0.023438    0.019905
         three_condition     strong_frac Divergent vs Convergent vs Random       8         Friedman 0.072440      NaN         NaN
three_condition_pairwise     strong_frac           Divergent vs Convergent       8 Wilcoxon(paired) 0.250000 0.500000    0.017779
three_condition_pairwise     strong_frac               Divergent vs Random       8 Wilcoxon(paired) 0.742188 0.742188   -0.003347
three_condition_pairwise     strong_frac              Convergent vs Random       8 Wilcoxon(paired) 0.015625 0.046875   -0.021126
         three_condition     strong_mean Divergent vs Convergent vs Random       8         Friedman 0.324652      NaN         NaN
three_condition_pairwise     strong_mean           Divergent vs Convergent       8 Wilcoxon(paired) 0.109375 0.328125   -0.006701
three_condition_pairwise     strong_mean               Divergent vs Random       8 Wilcoxon(paired) 0.742188 0.921875   -0.000706
three_condition_pairwise     strong_mean              Convergent vs Random       8 Wilcoxon(paired) 0.460938 0.921875    0.005995
```

## Coherent vs Random Tests

```
             scope          metric        comparison  n_mice                 test  p_value   p_holm  mean_delta
coherent_vs_random mean_noise_corr Coherent - Random       8 Wilcoxon(delta vs 0) 0.460938 0.460938    0.011880
coherent_vs_random        neg_frac Coherent - Random       8 Wilcoxon(delta vs 0) 0.007812 0.039062   -0.091475
coherent_vs_random   weak_pos_frac Coherent - Random       8 Wilcoxon(delta vs 0) 0.015625 0.046875    0.044388
coherent_vs_random     strong_frac Coherent - Random       8 Wilcoxon(delta vs 0) 0.007812 0.039062   -0.043404
coherent_vs_random     strong_mean Coherent - Random       8 Wilcoxon(delta vs 0) 0.195312 0.390625    0.013299
```

## Transition Key Tests

```
         scope                                     metric                                      comparison  n_mice                 test  p_value   p_holm  mean_delta
transition_key                          weak_expand_delta                          weak_expand_delta vs 0       8 Wilcoxon(delta vs 0) 0.015625 0.062500    0.044388
transition_key                           neg_shrink_delta                           neg_shrink_delta vs 0       8 Wilcoxon(delta vs 0) 0.007812 0.039062   -0.091475
transition_key                          strong_mean_delta                          strong_mean_delta vs 0       8 Wilcoxon(delta vs 0) 0.195312 0.390625    0.013299
transition_key                           mean_noise_delta                           mean_noise_delta vs 0       8 Wilcoxon(delta vs 0) 0.460938 0.460938    0.011880
transition_key asymmetry_index_negrelief_minus_strongloss asymmetry_index_negrelief_minus_strongloss vs 0       8 Wilcoxon(delta vs 0) 0.078125 0.234375   -0.077375
```

