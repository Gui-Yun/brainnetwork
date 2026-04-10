# Group Spatial Coverage + Distance-binned Weak-Correlation Report

## Spatial condition tests (Friedman + Wilcoxon, Holm adjusted)

```
                         metric  n_mice main_test   main_p              comparison  pairwise_p  pairwise_p_holm
               binary_hull_area       8  Friedman 0.196912 Divergent vs Convergent    0.148438         0.445312
               binary_hull_area       8  Friedman 0.196912     Divergent vs Random    0.250000         0.500000
               binary_hull_area       8  Friedman 0.196912    Convergent vs Random    0.742188         0.742188
  binary_mean_pairwise_distance       8  Friedman 0.223130 Divergent vs Convergent    0.843750         0.843750
  binary_mean_pairwise_distance       8  Friedman 0.223130     Divergent vs Random    0.195312         0.585938
  binary_mean_pairwise_distance       8  Friedman 0.223130    Convergent vs Random    0.250000         0.585938
        binary_nn_distance_mean       8  Friedman 0.072440 Divergent vs Convergent    0.109375         0.328125
        binary_nn_distance_mean       8  Friedman 0.072440     Divergent vs Random    0.460938         0.921875
        binary_nn_distance_mean       8  Friedman 0.072440    Convergent vs Random    0.945312         0.945312
       binary_bin_coverage_prop       8  Friedman 0.968257 Divergent vs Convergent    0.812500         1.000000
       binary_bin_coverage_prop       8  Friedman 0.968257     Divergent vs Random    0.945312         1.000000
       binary_bin_coverage_prop       8  Friedman 0.968257    Convergent vs Random    1.000000         1.000000
         binary_spatial_entropy       8  Friedman 0.882497 Divergent vs Convergent    0.640625         1.000000
         binary_spatial_entropy       8  Friedman 0.882497     Divergent vs Random    0.460938         1.000000
         binary_spatial_entropy       8  Friedman 0.882497    Convergent vs Random    0.843750         1.000000
weighted_mean_pairwise_distance       8  Friedman 0.030197 Divergent vs Convergent    0.843750         0.843750
weighted_mean_pairwise_distance       8  Friedman 0.030197     Divergent vs Random    0.023438         0.070312
weighted_mean_pairwise_distance       8  Friedman 0.030197    Convergent vs Random    0.023438         0.070312
      weighted_nn_distance_mean       8  Friedman 0.324652 Divergent vs Convergent    0.312500         0.937500
      weighted_nn_distance_mean       8  Friedman 0.324652     Divergent vs Random    0.742188         1.000000
      weighted_nn_distance_mean       8  Friedman 0.324652    Convergent vs Random    0.742188         1.000000
       weighted_spatial_entropy       8  Friedman 0.882497 Divergent vs Convergent    0.460938         1.000000
       weighted_spatial_entropy       8  Friedman 0.882497     Divergent vs Random    0.945312         1.000000
       weighted_spatial_entropy       8  Friedman 0.882497    Convergent vs Random    1.000000         1.000000
    weighted_effective_bin_prop       8  Friedman 0.882497 Divergent vs Convergent    0.742188         1.000000
    weighted_effective_bin_prop       8  Friedman 0.882497     Divergent vs Random    0.843750         1.000000
    weighted_effective_bin_prop       8  Friedman 0.882497    Convergent vs Random    0.742188         1.000000
```

## Distance-binned tests (Friedman + Wilcoxon, Holm adjusted)

```
         metric distance_bin  n_mice main_test   main_p              comparison  pairwise_p  pairwise_p_holm
    weak30_mean         0-80       8  Friedman 0.416862 Divergent vs Convergent    0.945312         0.945312
    weak30_mean         0-80       8  Friedman 0.416862     Divergent vs Random    0.109375         0.234375
    weak30_mean         0-80       8  Friedman 0.416862    Convergent vs Random    0.078125         0.234375
    weak30_mean      160-240       8  Friedman 0.093014 Divergent vs Convergent    0.742188         0.742188
    weak30_mean      160-240       8  Friedman 0.093014     Divergent vs Random    0.015625         0.046875
    weak30_mean      160-240       8  Friedman 0.093014    Convergent vs Random    0.039062         0.078125
    weak30_mean      240-320       8  Friedman 0.223130 Divergent vs Convergent    0.742188         0.742188
    weak30_mean      240-320       8  Friedman 0.223130     Divergent vs Random    0.054688         0.117188
    weak30_mean      240-320       8  Friedman 0.223130    Convergent vs Random    0.039062         0.117188
    weak30_mean      320-400       8  Friedman 0.223130 Divergent vs Convergent    0.640625         0.640625
    weak30_mean      320-400       8  Friedman 0.223130     Divergent vs Random    0.039062         0.117188
    weak30_mean      320-400       8  Friedman 0.223130    Convergent vs Random    0.039062         0.117188
    weak30_mean      400-600       8  Friedman 0.093014 Divergent vs Convergent    0.640625         0.640625
    weak30_mean      400-600       8  Friedman 0.093014     Divergent vs Random    0.023438         0.070312
    weak30_mean      400-600       8  Friedman 0.093014    Convergent vs Random    0.039062         0.078125
    weak30_mean      600-800       8  Friedman 0.093014 Divergent vs Convergent    0.742188         0.742188
    weak30_mean      600-800       8  Friedman 0.093014     Divergent vs Random    0.023438         0.070312
    weak30_mean      600-800       8  Friedman 0.093014    Convergent vs Random    0.109375         0.218750
    weak30_mean       80-160       8  Friedman 0.196912 Divergent vs Convergent    0.843750         0.843750
    weak30_mean       80-160       8  Friedman 0.196912     Divergent vs Random    0.039062         0.117188
    weak30_mean       80-160       8  Friedman 0.196912    Convergent vs Random    0.039062         0.117188
    weak30_mean         800+       8  Friedman 0.093014 Divergent vs Convergent    0.460938         0.460938
    weak30_mean         800+       8  Friedman 0.093014     Divergent vs Random    0.039062         0.117188
    weak30_mean         800+       8  Friedman 0.093014    Convergent vs Random    0.039062         0.117188
strong_weak_gap         0-80       8  Friedman 0.007635 Divergent vs Convergent    0.945312         0.945312
strong_weak_gap         0-80       8  Friedman 0.007635     Divergent vs Random    0.007812         0.023438
strong_weak_gap         0-80       8  Friedman 0.007635    Convergent vs Random    0.023438         0.046875
strong_weak_gap      160-240       8  Friedman 0.011109 Divergent vs Convergent    0.742188         0.742188
strong_weak_gap      160-240       8  Friedman 0.011109     Divergent vs Random    0.007812         0.023438
strong_weak_gap      160-240       8  Friedman 0.011109    Convergent vs Random    0.039062         0.078125
strong_weak_gap      240-320       8  Friedman 0.011109 Divergent vs Convergent    0.546875         0.546875
strong_weak_gap      240-320       8  Friedman 0.011109     Divergent vs Random    0.007812         0.023438
strong_weak_gap      240-320       8  Friedman 0.011109    Convergent vs Random    0.039062         0.078125
strong_weak_gap      320-400       8  Friedman 0.007635 Divergent vs Convergent    0.382812         0.382812
strong_weak_gap      320-400       8  Friedman 0.007635     Divergent vs Random    0.007812         0.023438
strong_weak_gap      320-400       8  Friedman 0.007635    Convergent vs Random    0.015625         0.031250
strong_weak_gap      400-600       8  Friedman 0.002479 Divergent vs Convergent    0.945312         0.945312
strong_weak_gap      400-600       8  Friedman 0.002479     Divergent vs Random    0.007812         0.023438
strong_weak_gap      400-600       8  Friedman 0.002479    Convergent vs Random    0.007812         0.023438
strong_weak_gap      600-800       8  Friedman 0.000805 Divergent vs Convergent    0.195312         0.195312
strong_weak_gap      600-800       8  Friedman 0.000805     Divergent vs Random    0.007812         0.023438
strong_weak_gap      600-800       8  Friedman 0.000805    Convergent vs Random    0.007812         0.023438
strong_weak_gap       80-160       8  Friedman 0.011109 Divergent vs Convergent    0.382812         0.382812
strong_weak_gap       80-160       8  Friedman 0.011109     Divergent vs Random    0.007812         0.023438
strong_weak_gap       80-160       8  Friedman 0.011109    Convergent vs Random    0.078125         0.156250
strong_weak_gap         800+       7  Friedman 0.066252 Divergent vs Convergent    1.000000         1.000000
strong_weak_gap         800+       7  Friedman 0.066252     Divergent vs Random    0.046875         0.140625
strong_weak_gap         800+       7  Friedman 0.066252    Convergent vs Random    0.078125         0.156250
```

