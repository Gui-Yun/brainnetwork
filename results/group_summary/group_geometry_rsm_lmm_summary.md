# Group Geometry vs Mean RSM (LMM)

| model_name   | formula                                                    | term                   |   beta |   p_value |   aic |   bic |   llf |   n_obs |   n_mice | converged   | note                        |
|:-------------|:-----------------------------------------------------------|:-----------------------|-------:|----------:|------:|------:|------:|--------:|---------:|:------------|:----------------------------|
| M1           | Mean_RSM_Sim ~ Geom_AngleDeg                               | Geom_AngleDeg          |    nan |       nan |   nan |   nan |   nan |      24 |        8 | False       | fit failed: Singular matrix |
| M2           | Mean_RSM_Sim ~ Geom_OrthParallelRatio                      | Geom_OrthParallelRatio |    nan |       nan |   nan |   nan |   nan |      24 |        8 | False       | fit failed: Singular matrix |
| M3           | Mean_RSM_Sim ~ Participants_Ratio + Geom_AngleDeg          | Participants_Ratio     |    nan |       nan |   nan |   nan |   nan |      24 |        8 | False       | fit failed: Singular matrix |
| M3           | Mean_RSM_Sim ~ Participants_Ratio + Geom_AngleDeg          | Geom_AngleDeg          |    nan |       nan |   nan |   nan |   nan |      24 |        8 | False       | fit failed: Singular matrix |
| M4           | Mean_RSM_Sim ~ Participants_Ratio + Geom_OrthParallelRatio | Participants_Ratio     |    nan |       nan |   nan |   nan |   nan |      24 |        8 | False       | fit failed: Singular matrix |
| M4           | Mean_RSM_Sim ~ Participants_Ratio + Geom_OrthParallelRatio | Geom_OrthParallelRatio |    nan |       nan |   nan |   nan |   nan |      24 |        8 | False       | fit failed: Singular matrix |
