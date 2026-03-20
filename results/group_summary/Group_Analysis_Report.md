# Group-level Multi-mouse Analysis Report

**Number of mice**: 8

**Mouse IDs**: M21_1107, M71_1024, M73_1128, M77_1031, M77_1107, M78_1017, M79_1128, M91_1017

## 1. Descriptive Statistics (Mean ± SEM)

| Condition   | Entropy         | Mean_RSM_Sim    | Mean_Correlation   | Strong_Correlation   | Weak_Correlation   | Strong_Weak_Gap   | Participants_Ratio   | Gini_Mean       | Gini_STD        | PR_Mean          | PR_STD           | PR_Norm_Mean    | PR_Norm_STD     | Effective_Dim_PR   | Effective_Dim_eRank   | Effective_Dim_90Var   | GraphStrong_efficiency   | GraphWeak_efficiency   | GraphGap_efficiency   | GraphStrong_modularity   | GraphWeak_modularity   | GraphGap_modularity   | GraphStrong_local_efficiency   | GraphWeak_local_efficiency   | GraphGap_local_efficiency   | GraphStrong_avg_clustering   | GraphWeak_avg_clustering   | GraphGap_avg_clustering   |
|:------------|:----------------|:----------------|:-------------------|:---------------------|:-------------------|:------------------|:---------------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|:-------------------|:----------------------|:----------------------|:-------------------------|:-----------------------|:----------------------|:-------------------------|:-----------------------|:----------------------|:-------------------------------|:-----------------------------|:----------------------------|:-----------------------------|:---------------------------|:--------------------------|
| Convergent  | 3.6712 ± 0.0909 | 0.565 ± 0.0106  | 0.2074 ± 0.0104    | 0.3902 ± 0.0115      | 0.0727 ± 0.0092    | 0.3175 ± 0.006    | 1.6652 ± 0.1043      | 0.6021 ± 0.0186 | 0.0774 ± 0.0059 | 42.2758 ± 6.7798 | 17.8451 ± 3.085  | 0.119 ± 0.015   | 0.0506 ± 0.0067 | 11.9847 ± 0.8331   | 19.4477 ± 1.1767      | 21.0 ± 1.3496         | 0.278 ± 0.018            | 0.422 ± 0.027          | -0.1439 ± 0.0154      | 0.3699 ± 0.0261          | 0.2033 ± 0.0146        | 0.1666 ± 0.0151       | 0.7393 ± 0.0155                | 0.4151 ± 0.072               | 0.3242 ± 0.0635             | 0.6348 ± 0.0141              | 0.2585 ± 0.0495            | 0.3763 ± 0.0461           |
| Divergent   | 3.7087 ± 0.0819 | 0.55 ± 0.0225   | 0.2051 ± 0.0114    | 0.3894 ± 0.0136      | 0.0728 ± 0.0099    | 0.3166 ± 0.008    | 1.724 ± 0.1179       | 0.6122 ± 0.0176 | 0.0863 ± 0.0056 | 41.2837 ± 6.5373 | 17.2922 ± 2.3249 | 0.1141 ± 0.0118 | 0.0496 ± 0.0051 | 11.9059 ± 1.0991   | 19.0662 ± 1.4372      | 20.5 ± 1.4392         | 0.285 ± 0.0174           | 0.4114 ± 0.0261        | -0.1263 ± 0.0138      | 0.3761 ± 0.0354          | 0.1957 ± 0.0145        | 0.1805 ± 0.0247       | 0.7325 ± 0.0164                | 0.4057 ± 0.0618              | 0.3267 ± 0.0584             | 0.6252 ± 0.0119              | 0.2401 ± 0.0376            | 0.3851 ± 0.0388           |
| Random      | 3.5989 ± 0.0692 | 0.6264 ± 0.0181 | 0.1985 ± 0.0142    | 0.391 ± 0.0154       | 0.0546 ± 0.0109    | 0.3364 ± 0.0107   | 2.5279 ± 0.2484      | 0.6291 ± 0.0196 | 0.0783 ± 0.006  | 35.9484 ± 4.2529 | 16.5695 ± 2.2046 | 0.1074 ± 0.0149 | 0.048 ± 0.0052  | 11.4929 ± 0.8069   | 18.7792 ± 1.1476      | 20.25 ± 1.3059        | 0.2678 ± 0.0171          | 0.4513 ± 0.0202        | -0.1835 ± 0.0214      | 0.3634 ± 0.0425          | 0.1884 ± 0.0127        | 0.1751 ± 0.0317       | 0.7123 ± 0.0216                | 0.4467 ± 0.0627              | 0.2655 ± 0.0664             | 0.6091 ± 0.0177              | 0.2752 ± 0.0413            | 0.3339 ± 0.048            |

## 2. Friedman + Wilcoxon Tests

| Metric | Main Effect | Div vs Con | Div vs Rand | Con vs Rand |
| :--- | :--- | :--- | :--- | :--- |
| **Entropy** | Friedman chi2=2.250, p=3.2465e-01 | p=0.7422 (ns) | p=0.1953 (ns) | p=0.3828 (ns) |
| **Mean_RSM_Sim** | Friedman chi2=4.750, p=9.3014e-02 | p=0.4609 (ns) | p=0.0391 (*) | p=0.0234 (*) |
| **Mean_Correlation** | Friedman chi2=3.250, p=1.9691e-01 | p=0.3125 (ns) | p=0.3828 (ns) | p=0.3125 (ns) |
| **Strong_Correlation** | Friedman chi2=0.750, p=6.8729e-01 | p=0.5469 (ns) | p=0.6406 (ns) | p=0.5469 (ns) |
| **Weak_Correlation** | Friedman chi2=7.000, p=3.0197e-02 | p=1.0000 (ns) | p=0.0156 (*) | p=0.0234 (*) |
| **Strong_Weak_Gap** | Friedman chi2=6.250, p=4.3937e-02 | p=0.7422 (ns) | p=0.0391 (*) | p=0.0547 (ns) |
| **Participants_Ratio** | Friedman chi2=9.250, p=9.8037e-03 | p=0.6406 (ns) | p=0.0078 (**) | p=0.0156 (*) |
| **Gini_Mean** | Friedman chi2=5.250, p=7.2440e-02 | p=0.4609 (ns) | p=0.2500 (ns) | p=0.0781 (ns) |
| **PR_Mean** | Friedman chi2=1.750, p=4.1686e-01 | p=1.0000 (ns) | p=0.1953 (ns) | p=0.1953 (ns) |
| **PR_Norm_Mean** | Friedman chi2=1.750, p=4.1686e-01 | p=0.9453 (ns) | p=0.3125 (ns) | p=0.3125 (ns) |
| **Effective_Dim_PR** | Friedman chi2=0.250, p=8.8250e-01 | p=0.9453 (ns) | p=0.5469 (ns) | p=0.7422 (ns) |
| **Effective_Dim_eRank** | Friedman chi2=0.750, p=6.8729e-01 | p=0.7422 (ns) | p=0.7422 (ns) | p=0.4609 (ns) |
| **Effective_Dim_90Var** | Friedman chi2=2.667, p=2.6360e-01 | p=0.3125 (ns) | p=0.7500 (ns) | p=0.2500 (ns) |
| **GraphGap_efficiency** | Friedman chi2=5.250, p=7.2440e-02 | p=0.4609 (ns) | p=0.0391 (*) | p=0.2500 (ns) |
| **GraphGap_modularity** | Friedman chi2=0.750, p=6.8729e-01 | p=0.7422 (ns) | p=1.0000 (ns) | p=0.8438 (ns) |
| **GraphGap_local_efficiency** | Friedman chi2=0.250, p=8.8250e-01 | p=0.6406 (ns) | p=0.1953 (ns) | p=0.7422 (ns) |
| **GraphGap_avg_clustering** | Friedman chi2=0.250, p=8.8250e-01 | p=0.9453 (ns) | p=0.1953 (ns) | p=0.9453 (ns) |

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

### Entropy
![Entropy](./group_entropy.png)

### RSM Mean Similarity
![RSM Mean Similarity](./group_rsm_mean.png)

### Mean Correlation
![Mean Correlation](./group_mean_corr.png)

### Strong Correlation (Top 10%)
![Strong Correlation (Top 10%)](./group_strong_corr.png)

### Weak Correlation (Bottom 10%)
![Weak Correlation (Bottom 10%)](./group_weak_corr.png)

### Strong-Weak Gap
![Strong-Weak Gap](./group_corr_gap.png)

### RR Participants Ratio
![RR Participants Ratio](./group_participants.png)

### Gini (Mean)
![Gini (Mean)](./group_gini_mean.png)

### Participation Ratio (Mean)
![Participation Ratio (Mean)](./group_pr_mean.png)

### Effective Dim (PR)
![Effective Dim (PR)](./group_effdim_pr.png)

### Effective Dim (eRank)
![Effective Dim (eRank)](./group_effdim_erank.png)

### Decile Correlation Curve
![Decile Correlation Curve](./group_corr_decile_curve.png)

### Graph Strong vs Weak - efficiency
![Graph Strong vs Weak - efficiency](./group_graph_sw_efficiency.png)

### Graph Strong vs Weak - modularity
![Graph Strong vs Weak - modularity](./group_graph_sw_modularity.png)

### Graph Strong vs Weak - local_efficiency
![Graph Strong vs Weak - local_efficiency](./group_graph_sw_local_efficiency.png)

### Graph Strong vs Weak - avg_clustering
![Graph Strong vs Weak - avg_clustering](./group_graph_sw_avg_clustering.png)

