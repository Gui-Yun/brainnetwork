# Section 1 Summary - M77_1031

## 1.1 RR neuron ratio and coding accuracy
|   Class_ID | Class_Name   |   Trial_Count |   Total_Neurons |   RR_Count |   RR_Ratio |
|-----------:|:-------------|--------------:|----------------:|-----------:|-----------:|
|          1 | Convergent   |            81 |           35499 |        376 |     0.0106 |
|          2 | Divergent    |            57 |           35499 |        364 |     0.0103 |
|          3 | Random       |            42 |           35499 |        345 |     0.0097 |

| Metric        |   Value |
|:--------------|--------:|
| mean_accuracy |  0.8234 |
| peak_accuracy |  0.9944 |
| peak_time     |  2      |

## 1.2 Participation and coding space
|   Class_ID | Class_Name   |   Current_RR_Count |   Other_RR_Count |   Current_RR_Response_10_18 |   Other_RR_Response_10_18 |   Response_Ratio_Current_vs_Other |
|-----------:|:-------------|-------------------:|-----------------:|----------------------------:|--------------------------:|----------------------------------:|
|          1 | Convergent   |                376 |              335 |                      0.0127 |                    0.0126 |                            1.0075 |
|          2 | Divergent    |                364 |              347 |                      0.0146 |                    0.0122 |                            1.2009 |
|          3 | Random       |                345 |              366 |                      0.016  |                    0.0164 |                            0.9776 |

|            |   Convergent |   Divergent |   Random |
|:-----------|-------------:|------------:|---------:|
| Convergent |       1      |      0.9626 |   0.9508 |
| Divergent  |       0.9626 |      1      |   0.956  |
| Random     |       0.9508 |      0.956  |   1      |

## 1.3 Pairwise correlation
|   Class_ID | Class_Name   |   Mean_Correlation |   Mean_Abs_Correlation |   Weak_Abs_Correlation_Mean |   Strong_Abs_Correlation_Mean |   Strong_Weak_Gap |   Pair_Count |
|-----------:|:-------------|-------------------:|-----------------------:|----------------------------:|------------------------------:|------------------:|-------------:|
|          1 | Convergent   |             0.1884 |                 0.1884 |                      0.0606 |                        0.3527 |            0.2921 |       252405 |
|          2 | Divergent    |             0.189  |                 0.189  |                      0.0583 |                        0.3577 |            0.2994 |       252405 |
|          3 | Random       |             0.1682 |                 0.1682 |                      0.0335 |                        0.3487 |            0.3152 |       252405 |

## 1.4 Graph metrics
|   Class_ID | Class_Name   |   n_nodes |   n_edges |   density |   mean_degree |   largest_component |   avg_clustering |   global_efficiency |   local_efficiency |   transitivity |   efficiency |   modularity |   betweenness_mean |   mean_correlation |   abs_mean_correlation |   positive_edge_fraction |   negative_edge_fraction |   degree_assortativity |   avg_shortest_path_lcc |   diameter_lcc |   degree_centralization |
|-----------:|:-------------|----------:|----------:|----------:|--------------:|--------------------:|-----------------:|--------------------:|-------------------:|---------------:|-------------:|-------------:|-------------------:|-------------------:|-----------------------:|-------------------------:|-------------------------:|-----------------------:|------------------------:|---------------:|------------------------:|
|          1 | Convergent   |       711 |     12620 |      0.05 |       35.4993 |                 648 |           0.6318 |              0.3515 |             0.7746 |         0.4457 |       0.3515 |       0.2669 |             0.002  |             0.1884 |                 0.1884 |                   0.9994 |                   0.0006 |                 0.0645 |                  2.6897 |              7 |                  0.3114 |
|          2 | Divergent    |       711 |     12620 |      0.05 |       35.4993 |                 635 |           0.6615 |              0.3381 |             0.787  |         0.4391 |       0.3381 |       0.3091 |             0.0019 |             0.189  |                 0.189  |                   0.9977 |                   0.0023 |                 0.0302 |                  2.6985 |              7 |                  0.2818 |
|          3 | Random       |       711 |     12620 |      0.05 |       35.4993 |                 602 |           0.6397 |              0.312  |             0.7618 |         0.4467 |       0.312  |       0.2341 |             0.0016 |             0.1682 |                 0.1684 |                   0.9924 |                   0.0076 |                -0.0075 |                  2.632  |              7 |                  0.3326 |