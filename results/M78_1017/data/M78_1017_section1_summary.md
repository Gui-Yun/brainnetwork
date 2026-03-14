# Section 1 Summary - M78_1017

## 1.1 RR neuron ratio and coding accuracy
|   Class_ID | Class_Name   |   Trial_Count |   Total_Neurons |   RR_Count |   RR_Ratio |
|-----------:|:-------------|--------------:|----------------:|-----------:|-----------:|
|          1 | Convergent   |            81 |           35499 |        404 |     0.0114 |
|          2 | Divergent    |            57 |           35499 |        361 |     0.0102 |
|          3 | Random       |            42 |           35499 |        412 |     0.0116 |

| Metric        |   Value |
|:--------------|--------:|
| mean_accuracy |  0.7887 |
| peak_accuracy |  0.9889 |
| peak_time     |  2.25   |

## 1.2 Participation and coding space
|   Class_ID | Class_Name   |   Current_RR_Count |   Other_RR_Count |   Current_RR_Response_10_18 |   Other_RR_Response_10_18 |   Response_Ratio_Current_vs_Other |
|-----------:|:-------------|-------------------:|-----------------:|----------------------------:|--------------------------:|----------------------------------:|
|          1 | Convergent   |                404 |              322 |                      0.0132 |                    0.012  |                            1.1032 |
|          2 | Divergent    |                361 |              365 |                      0.0137 |                    0.0129 |                            1.0644 |
|          3 | Random       |                412 |              314 |                      0.0165 |                    0.0157 |                            1.0466 |

|            |   Convergent |   Divergent |   Random |
|:-----------|-------------:|------------:|---------:|
| Convergent |       1      |      0.9624 |   0.9502 |
| Divergent  |       0.9624 |      1      |   0.9554 |
| Random     |       0.9502 |      0.9554 |   1      |

## 1.3 Pairwise correlation
|   Class_ID | Class_Name   |   Mean_Correlation |   Mean_Abs_Correlation |   Weak_Abs_Correlation_Mean |   Strong_Abs_Correlation_Mean |   Strong_Weak_Gap |   Pair_Count |
|-----------:|:-------------|-------------------:|-----------------------:|----------------------------:|------------------------------:|------------------:|-------------:|
|          1 | Convergent   |             0.1868 |                 0.1868 |                      0.0632 |                        0.3565 |            0.2933 |       263175 |
|          2 | Divergent    |             0.1879 |                 0.1879 |                      0.0541 |                        0.357  |            0.3028 |       263175 |
|          3 | Random       |             0.1955 |                 0.1955 |                      0.0633 |                        0.3644 |            0.3011 |       263175 |

## 1.4 Graph metrics
|   Class_ID | Class_Name   |   n_nodes |   n_edges |   density |   mean_degree |   largest_component |   avg_clustering |   global_efficiency |   local_efficiency |   transitivity |   efficiency |   modularity |   betweenness_mean |   mean_correlation |   abs_mean_correlation |   positive_edge_fraction |   negative_edge_fraction |   degree_assortativity |   avg_shortest_path_lcc |   diameter_lcc |   degree_centralization |
|-----------:|:-------------|----------:|----------:|----------:|--------------:|--------------------:|-----------------:|--------------------:|-------------------:|---------------:|-------------:|-------------:|-------------------:|-------------------:|-----------------------:|-------------------------:|-------------------------:|-----------------------:|------------------------:|---------------:|------------------------:|
|          1 | Convergent   |       726 |     13158 |      0.05 |       36.2479 |                 633 |           0.6597 |              0.3233 |             0.779  |         0.4653 |       0.3233 |       0.2527 |             0.0018 |             0.1868 |                 0.1868 |                   0.9991 |                   0.0009 |                 0.0101 |                  2.6956 |              7 |                  0.3454 |
|          2 | Divergent    |       726 |     13158 |      0.05 |       36.2479 |                 652 |           0.6498 |              0.3498 |             0.7818 |         0.4082 |       0.3498 |       0.2737 |             0.0018 |             0.1879 |                 0.188  |                   0.9976 |                   0.0024 |                -0.0247 |                  2.598  |              7 |                  0.3537 |
|          3 | Random       |       726 |     13158 |      0.05 |       36.2479 |                 646 |           0.614  |              0.3436 |             0.7524 |         0.4115 |       0.3436 |       0.2879 |             0.0018 |             0.1955 |                 0.1955 |                   0.9989 |                   0.0011 |                -0.0027 |                  2.6185 |              8 |                  0.297  |