# Group Population-Size Scaling + Dropout Report

## K90 condition tests (Friedman + Wilcoxon, Holm adjusted)

```
         strategy            k90_metric metric  n_mice main_test   main_p              comparison  pairwise_p  pairwise_p_holm
           random       split_half_corr  value       8  Friedman 0.484742 Divergent vs Convergent    0.640625           1.0000
           random       split_half_corr  value       8  Friedman 0.484742     Divergent vs Random    0.734375           1.0000
           random       split_half_corr  value       8  Friedman 0.484742    Convergent vs Random    0.312500           0.9375
           random trial_template_cosine  value       8  Friedman 0.687289 Divergent vs Convergent    0.531250           1.0000
           random trial_template_cosine  value       8  Friedman 0.687289     Divergent vs Random    0.750000           1.0000
           random trial_template_cosine  value       8  Friedman 0.687289    Convergent vs Random    0.515625           1.0000
           random         pc1_alignment  value       8  Friedman 0.443747 Divergent vs Convergent    0.375000           1.0000
           random         pc1_alignment  value       8  Friedman 0.443747     Divergent vs Random    0.500000           1.0000
           random         pc1_alignment  value       8  Friedman 0.443747    Convergent vs Random    0.375000           1.0000
spatial_clustered       split_half_corr  value       8  Friedman 0.274812 Divergent vs Convergent    0.187500           0.3750
spatial_clustered       split_half_corr  value       8  Friedman 0.274812     Divergent vs Random    0.062500           0.1875
spatial_clustered       split_half_corr  value       8  Friedman 0.274812    Convergent vs Random    0.812500           0.8125
spatial_clustered trial_template_cosine  value       8  Friedman 0.737604 Divergent vs Convergent    0.812500           1.0000
spatial_clustered trial_template_cosine  value       8  Friedman 0.737604     Divergent vs Random    1.000000           1.0000
spatial_clustered trial_template_cosine  value       8  Friedman 0.737604    Convergent vs Random    0.843750           1.0000
spatial_clustered         pc1_alignment  value       8  Friedman 0.018316 Divergent vs Convergent    0.125000           0.3750
spatial_clustered         pc1_alignment  value       8  Friedman 0.018316     Divergent vs Random    0.125000           0.3750
spatial_clustered         pc1_alignment  value       8  Friedman 0.018316    Convergent vs Random    1.000000           1.0000
spatial_dispersed       split_half_corr  value       8  Friedman 0.727471 Divergent vs Convergent    0.625000           0.9375
spatial_dispersed       split_half_corr  value       8  Friedman 0.727471     Divergent vs Random    0.312500           0.9375
spatial_dispersed       split_half_corr  value       8  Friedman 0.727471    Convergent vs Random    0.437500           0.9375
spatial_dispersed trial_template_cosine  value       8  Friedman 0.246597 Divergent vs Convergent    0.250000           0.7500
spatial_dispersed trial_template_cosine  value       8  Friedman 0.246597     Divergent vs Random    0.250000           0.7500
spatial_dispersed trial_template_cosine  value       8  Friedman 0.246597    Convergent vs Random    0.625000           0.7500
spatial_dispersed         pc1_alignment  value       8  Friedman 0.304983 Divergent vs Convergent    0.250000           0.7500
spatial_dispersed         pc1_alignment  value       8  Friedman 0.304983     Divergent vs Random    1.000000           1.0000
spatial_dispersed         pc1_alignment  value       8  Friedman 0.304983    Convergent vs Random    0.625000           1.0000
```

## Dropout method tests on delta split-half corr

```
 Condition  drop_fraction                metric        main_test  main_n_mice   main_p                             comparison  n_mice_pair    mean_a    mean_b  mean_diff_a_minus_b  pairwise_p  pairwise_p_holm
Convergent           0.05 delta_split_half_corr Friedman(method)            8 0.001470 spatial_distributed vs spatial_cluster            8 -0.002430  0.002196            -0.004626    0.312500         0.750000
Convergent           0.05 delta_split_half_corr Friedman(method)            8 0.001470          spatial_distributed vs random            8 -0.002430 -0.000339            -0.002092    0.945312         0.945312
Convergent           0.05 delta_split_half_corr Friedman(method)            8 0.001470              spatial_cluster vs random            8  0.002196 -0.000339             0.002535    0.250000         0.750000
Convergent           0.05 delta_split_half_corr Friedman(method)            8 0.001470                 top_response vs random            8 -0.048294 -0.000339            -0.047956    0.007812         0.039062
Convergent           0.05 delta_split_half_corr Friedman(method)            8 0.001470    top_response vs spatial_distributed            8 -0.048294 -0.002430            -0.045864    0.007812         0.039062
Convergent           0.10 delta_split_half_corr Friedman(method)            8 0.002245 spatial_distributed vs spatial_cluster            8 -0.000523  0.001250            -0.001773    0.843750         1.000000
Convergent           0.10 delta_split_half_corr Friedman(method)            8 0.002245          spatial_distributed vs random            8 -0.000523  0.002378            -0.002901    0.640625         1.000000
Convergent           0.10 delta_split_half_corr Friedman(method)            8 0.002245              spatial_cluster vs random            8  0.001250  0.002378            -0.001128    0.843750         1.000000
Convergent           0.10 delta_split_half_corr Friedman(method)            8 0.002245                 top_response vs random            8 -0.071092  0.002378            -0.073469    0.007812         0.039062
Convergent           0.10 delta_split_half_corr Friedman(method)            8 0.002245    top_response vs spatial_distributed            8 -0.071092 -0.000523            -0.070568    0.007812         0.039062
Convergent           0.20 delta_split_half_corr Friedman(method)            8 0.000777 spatial_distributed vs spatial_cluster            8 -0.002650  0.003199            -0.005849    0.382812         0.765625
Convergent           0.20 delta_split_half_corr Friedman(method)            8 0.000777          spatial_distributed vs random            8 -0.002650 -0.002168            -0.000482    0.945312         0.945312
Convergent           0.20 delta_split_half_corr Friedman(method)            8 0.000777              spatial_cluster vs random            8  0.003199 -0.002168             0.005367    0.039062         0.117188
Convergent           0.20 delta_split_half_corr Friedman(method)            8 0.000777                 top_response vs random            8 -0.113723 -0.002168            -0.111555    0.007812         0.039062
Convergent           0.20 delta_split_half_corr Friedman(method)            8 0.000777    top_response vs spatial_distributed            8 -0.113723 -0.002650            -0.111074    0.007812         0.039062
Convergent           0.30 delta_split_half_corr Friedman(method)            8 0.001276 spatial_distributed vs spatial_cluster            8 -0.006117  0.002030            -0.008147    0.250000         0.585938
Convergent           0.30 delta_split_half_corr Friedman(method)            8 0.001276          spatial_distributed vs random            8 -0.006117  0.001430            -0.007546    0.195312         0.585938
Convergent           0.30 delta_split_half_corr Friedman(method)            8 0.001276              spatial_cluster vs random            8  0.002030  0.001430             0.000601    1.000000         1.000000
Convergent           0.30 delta_split_half_corr Friedman(method)            8 0.001276                 top_response vs random            8 -0.167087  0.001430            -0.168516    0.007812         0.039062
Convergent           0.30 delta_split_half_corr Friedman(method)            8 0.001276    top_response vs spatial_distributed            8 -0.167087 -0.006117            -0.160970    0.007812         0.039062
Convergent           0.40 delta_split_half_corr Friedman(method)            8 0.001470 spatial_distributed vs spatial_cluster            8 -0.005242  0.003010            -0.008252    0.109375         0.328125
Convergent           0.40 delta_split_half_corr Friedman(method)            8 0.001470          spatial_distributed vs random            8 -0.005242  0.001544            -0.006786    0.250000         0.500000
Convergent           0.40 delta_split_half_corr Friedman(method)            8 0.001470              spatial_cluster vs random            8  0.003010  0.001544             0.001466    0.382812         0.500000
Convergent           0.40 delta_split_half_corr Friedman(method)            8 0.001470                 top_response vs random            8 -0.229485  0.001544            -0.231029    0.007812         0.039062
Convergent           0.40 delta_split_half_corr Friedman(method)            8 0.001470    top_response vs spatial_distributed            8 -0.229485 -0.005242            -0.224243    0.007812         0.039062
Convergent           0.50 delta_split_half_corr Friedman(method)            8 0.000628 spatial_distributed vs spatial_cluster            8 -0.009187  0.005309            -0.014495    0.039062         0.117188
Convergent           0.50 delta_split_half_corr Friedman(method)            8 0.000628          spatial_distributed vs random            8 -0.009187 -0.002763            -0.006423    0.312500         0.312500
Convergent           0.50 delta_split_half_corr Friedman(method)            8 0.000628              spatial_cluster vs random            8  0.005309 -0.002763             0.008072    0.109375         0.218750
Convergent           0.50 delta_split_half_corr Friedman(method)            8 0.000628                 top_response vs random            8 -0.278463 -0.002763            -0.275700    0.007812         0.039062
Convergent           0.50 delta_split_half_corr Friedman(method)            8 0.000628    top_response vs spatial_distributed            8 -0.278463 -0.009187            -0.269277    0.007812         0.039062
Convergent           0.60 delta_split_half_corr Friedman(method)            8 0.000628 spatial_distributed vs spatial_cluster            8 -0.004035  0.004296            -0.008331    0.078125         0.234375
Convergent           0.60 delta_split_half_corr Friedman(method)            8 0.000628          spatial_distributed vs random            8 -0.004035 -0.000227            -0.003808    0.546875         0.546875
Convergent           0.60 delta_split_half_corr Friedman(method)            8 0.000628              spatial_cluster vs random            8  0.004296 -0.000227             0.004523    0.250000         0.500000
Convergent           0.60 delta_split_half_corr Friedman(method)            8 0.000628                 top_response vs random            8 -0.355111 -0.000227            -0.354884    0.007812         0.039062
Convergent           0.60 delta_split_half_corr Friedman(method)            8 0.000628    top_response vs spatial_distributed            8 -0.355111 -0.004035            -0.351076    0.007812         0.039062
 Divergent           0.05 delta_split_half_corr Friedman(method)            8 0.001032 spatial_distributed vs spatial_cluster            8 -0.001585  0.003331            -0.004916    0.148438         0.328125
 Divergent           0.05 delta_split_half_corr Friedman(method)            8 0.001032          spatial_distributed vs random            8 -0.001585  0.002544            -0.004129    0.109375         0.328125
 Divergent           0.05 delta_split_half_corr Friedman(method)            8 0.001032              spatial_cluster vs random            8  0.003331  0.002544             0.000787    0.843750         0.843750
 Divergent           0.05 delta_split_half_corr Friedman(method)            8 0.001032                 top_response vs random            8 -0.037051  0.002544            -0.039595    0.007812         0.039062
 Divergent           0.05 delta_split_half_corr Friedman(method)            8 0.001032    top_response vs spatial_distributed            8 -0.037051 -0.001585            -0.035466    0.007812         0.039062
 Divergent           0.10 delta_split_half_corr Friedman(method)            8 0.000628 spatial_distributed vs spatial_cluster            8 -0.005608  0.002201            -0.007809    0.039062         0.117188
 Divergent           0.10 delta_split_half_corr Friedman(method)            8 0.000628          spatial_distributed vs random            8 -0.005608  0.002619            -0.008227    0.039062         0.117188
 Divergent           0.10 delta_split_half_corr Friedman(method)            8 0.000628              spatial_cluster vs random            8  0.002201  0.002619            -0.000418    0.945312         0.945312
 Divergent           0.10 delta_split_half_corr Friedman(method)            8 0.000628                 top_response vs random            8 -0.065238  0.002619            -0.067857    0.007812         0.039062
 Divergent           0.10 delta_split_half_corr Friedman(method)            8 0.000628    top_response vs spatial_distributed            8 -0.065238 -0.005608            -0.059630    0.007812         0.039062
 Divergent           0.20 delta_split_half_corr Friedman(method)            8 0.000628 spatial_distributed vs spatial_cluster            8 -0.001047  0.000679            -0.001727    0.945312         0.945312
 Divergent           0.20 delta_split_half_corr Friedman(method)            8 0.000628          spatial_distributed vs random            8 -0.001047  0.003769            -0.004816    0.015625         0.046875
 Divergent           0.20 delta_split_half_corr Friedman(method)            8 0.000628              spatial_cluster vs random            8  0.000679  0.003769            -0.003089    0.148438         0.296875
 Divergent           0.20 delta_split_half_corr Friedman(method)            8 0.000628                 top_response vs random            8 -0.115311  0.003769            -0.119080    0.007812         0.039062
 Divergent           0.20 delta_split_half_corr Friedman(method)            8 0.000628    top_response vs spatial_distributed            8 -0.115311 -0.001047            -0.114264    0.007812         0.039062
 Divergent           0.30 delta_split_half_corr Friedman(method)            8 0.000628 spatial_distributed vs spatial_cluster            8 -0.004386  0.002202            -0.006587    0.250000         0.500000
 Divergent           0.30 delta_split_half_corr Friedman(method)            8 0.000628          spatial_distributed vs random            8 -0.004386 -0.002968            -0.001417    0.843750         0.843750
 Divergent           0.30 delta_split_half_corr Friedman(method)            8 0.000628              spatial_cluster vs random            8  0.002202 -0.002968             0.005170    0.015625         0.046875
 Divergent           0.30 delta_split_half_corr Friedman(method)            8 0.000628                 top_response vs random            8 -0.159432 -0.002968            -0.156464    0.007812         0.039062
 Divergent           0.30 delta_split_half_corr Friedman(method)            8 0.000628    top_response vs spatial_distributed            8 -0.159432 -0.004386            -0.155047    0.007812         0.039062
 Divergent           0.40 delta_split_half_corr Friedman(method)            8 0.001817 spatial_distributed vs spatial_cluster            8 -0.001084  0.001031            -0.002115    0.843750         1.000000
 Divergent           0.40 delta_split_half_corr Friedman(method)            8 0.001817          spatial_distributed vs random            8 -0.001084 -0.001093             0.000010    0.742188         1.000000
 Divergent           0.40 delta_split_half_corr Friedman(method)            8 0.001817              spatial_cluster vs random            8  0.001031 -0.001093             0.002124    0.742188         1.000000
 Divergent           0.40 delta_split_half_corr Friedman(method)            8 0.001817                 top_response vs random            8 -0.222689 -0.001093            -0.221596    0.007812         0.039062
 Divergent           0.40 delta_split_half_corr Friedman(method)            8 0.001817    top_response vs spatial_distributed            8 -0.222689 -0.001084            -0.221606    0.007812         0.039062
 Divergent           0.50 delta_split_half_corr Friedman(method)            8 0.001470 spatial_distributed vs spatial_cluster            8 -0.002877  0.001458            -0.004335    0.640625         1.000000
 Divergent           0.50 delta_split_half_corr Friedman(method)            8 0.001470          spatial_distributed vs random            8 -0.002877  0.000188            -0.003065    0.195312         0.585938
 Divergent           0.50 delta_split_half_corr Friedman(method)            8 0.001470              spatial_cluster vs random            8  0.001458  0.000188             0.001270    0.945312         1.000000
 Divergent           0.50 delta_split_half_corr Friedman(method)            8 0.001470                 top_response vs random            8 -0.300339  0.000188            -0.300527    0.007812         0.039062
 Divergent           0.50 delta_split_half_corr Friedman(method)            8 0.001470    top_response vs spatial_distributed            8 -0.300339 -0.002877            -0.297462    0.007812         0.039062
 Divergent           0.60 delta_split_half_corr Friedman(method)            8 0.000355 spatial_distributed vs spatial_cluster            8 -0.008102 -0.002026            -0.006076    0.640625         0.640625
 Divergent           0.60 delta_split_half_corr Friedman(method)            8 0.000355          spatial_distributed vs random            8 -0.008102  0.000779            -0.008880    0.109375         0.328125
 Divergent           0.60 delta_split_half_corr Friedman(method)            8 0.000355              spatial_cluster vs random            8 -0.002026  0.000779            -0.002805    0.195312         0.390625
 Divergent           0.60 delta_split_half_corr Friedman(method)            8 0.000355                 top_response vs random            8 -0.380653  0.000779            -0.381432    0.007812         0.039062
 Divergent           0.60 delta_split_half_corr Friedman(method)            8 0.000355    top_response vs spatial_distributed            8 -0.380653 -0.008102            -0.372552    0.007812         0.039062
    Random           0.05 delta_split_half_corr Friedman(method)            8 0.001949 spatial_distributed vs spatial_cluster            8 -0.006158 -0.004252            -0.001906    0.843750         1.000000
    Random           0.05 delta_split_half_corr Friedman(method)            8 0.001949          spatial_distributed vs random            8 -0.006158 -0.002689            -0.003469    0.742188         1.000000
    Random           0.05 delta_split_half_corr Friedman(method)            8 0.001949              spatial_cluster vs random            8 -0.004252 -0.002689            -0.001563    0.382812         1.000000
    Random           0.05 delta_split_half_corr Friedman(method)            8 0.001949                 top_response vs random            8 -0.046563 -0.002689            -0.043874    0.007812         0.039062
    Random           0.05 delta_split_half_corr Friedman(method)            8 0.001949    top_response vs spatial_distributed            8 -0.046563 -0.006158            -0.040405    0.007812         0.039062
    Random           0.10 delta_split_half_corr Friedman(method)            8 0.001470 spatial_distributed vs spatial_cluster            8 -0.007820 -0.002112            -0.005708    0.148438         0.445312
    Random           0.10 delta_split_half_corr Friedman(method)            8 0.001470          spatial_distributed vs random            8 -0.007820 -0.003123            -0.004697    0.312500         0.625000
    Random           0.10 delta_split_half_corr Friedman(method)            8 0.001470              spatial_cluster vs random            8 -0.002112 -0.003123             0.001011    0.843750         0.843750
    Random           0.10 delta_split_half_corr Friedman(method)            8 0.001470                 top_response vs random            8 -0.081504 -0.003123            -0.078381    0.007812         0.039062
    Random           0.10 delta_split_half_corr Friedman(method)            8 0.001470    top_response vs spatial_distributed            8 -0.081504 -0.007820            -0.073683    0.007812         0.039062
    Random           0.20 delta_split_half_corr Friedman(method)            8 0.001276 spatial_distributed vs spatial_cluster            8 -0.004994 -0.005961             0.000967    0.250000         0.750000
    Random           0.20 delta_split_half_corr Friedman(method)            8 0.001276          spatial_distributed vs random            8 -0.004994 -0.004831            -0.000163    0.546875         1.000000
    Random           0.20 delta_split_half_corr Friedman(method)            8 0.001276              spatial_cluster vs random            8 -0.005961 -0.004831            -0.001130    0.546875         1.000000
    Random           0.20 delta_split_half_corr Friedman(method)            8 0.001276                 top_response vs random            8 -0.135834 -0.004831            -0.131004    0.007812         0.039062
    Random           0.20 delta_split_half_corr Friedman(method)            8 0.001276    top_response vs spatial_distributed            8 -0.135834 -0.004994            -0.130841    0.007812         0.039062
    Random           0.30 delta_split_half_corr Friedman(method)            8 0.002245 spatial_distributed vs spatial_cluster            8 -0.006141 -0.003422            -0.002718    0.546875         1.000000
    Random           0.30 delta_split_half_corr Friedman(method)            8 0.002245          spatial_distributed vs random            8 -0.006141 -0.002699            -0.003442    0.742188         1.000000
    Random           0.30 delta_split_half_corr Friedman(method)            8 0.002245              spatial_cluster vs random            8 -0.003422 -0.002699            -0.000724    0.742188         1.000000
    Random           0.30 delta_split_half_corr Friedman(method)            8 0.002245                 top_response vs random            8 -0.201003 -0.002699            -0.198304    0.007812         0.039062
    Random           0.30 delta_split_half_corr Friedman(method)            8 0.002245    top_response vs spatial_distributed            8 -0.201003 -0.006141            -0.194862    0.007812         0.039062
    Random           0.40 delta_split_half_corr Friedman(method)            8 0.002245 spatial_distributed vs spatial_cluster            8 -0.002233 -0.000227            -0.002005    0.640625         1.000000
    Random           0.40 delta_split_half_corr Friedman(method)            8 0.002245          spatial_distributed vs random            8 -0.002233  0.001390            -0.003623    0.742188         1.000000
    Random           0.40 delta_split_half_corr Friedman(method)            8 0.002245              spatial_cluster vs random            8 -0.000227  0.001390            -0.001617    0.460938         1.000000
    Random           0.40 delta_split_half_corr Friedman(method)            8 0.002245                 top_response vs random            8 -0.266270  0.001390            -0.267660    0.007812         0.039062
    Random           0.40 delta_split_half_corr Friedman(method)            8 0.002245    top_response vs spatial_distributed            8 -0.266270 -0.002233            -0.264037    0.007812         0.039062
    Random           0.50 delta_split_half_corr Friedman(method)            8 0.002245 spatial_distributed vs spatial_cluster            8 -0.008091 -0.001837            -0.006254    0.546875         1.000000
    Random           0.50 delta_split_half_corr Friedman(method)            8 0.002245          spatial_distributed vs random            8 -0.008091 -0.004755            -0.003336    0.460938         1.000000
    Random           0.50 delta_split_half_corr Friedman(method)            8 0.002245              spatial_cluster vs random            8 -0.001837 -0.004755             0.002917    0.843750         1.000000
    Random           0.50 delta_split_half_corr Friedman(method)            8 0.002245                 top_response vs random            8 -0.307524 -0.004755            -0.302769    0.007812         0.039062
    Random           0.50 delta_split_half_corr Friedman(method)            8 0.002245    top_response vs spatial_distributed            8 -0.307524 -0.008091            -0.299433    0.007812         0.039062
    Random           0.60 delta_split_half_corr Friedman(method)            8 0.001817 spatial_distributed vs spatial_cluster            8 -0.005363 -0.003332            -0.002031    0.742188         1.000000
    Random           0.60 delta_split_half_corr Friedman(method)            8 0.001817          spatial_distributed vs random            8 -0.005363 -0.000954            -0.004409    0.250000         0.750000
    Random           0.60 delta_split_half_corr Friedman(method)            8 0.001817              spatial_cluster vs random            8 -0.003332 -0.000954            -0.002378    0.640625         1.000000
    Random           0.60 delta_split_half_corr Friedman(method)            8 0.001817                 top_response vs random            8 -0.387197 -0.000954            -0.386243    0.007812         0.039062
    Random           0.60 delta_split_half_corr Friedman(method)            8 0.001817    top_response vs spatial_distributed            8 -0.387197 -0.005363            -0.381833    0.007812         0.039062
```

