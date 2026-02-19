# Report Artifacts (Auto-generated)

## Baseline performance

- **accuracy**: 0.7932559425096739
- **f1**: 0.6684397163120568
- **precision**: 0.538058991436727
- **recall**: 0.8822152886115444
- **roc_auc**: 0.906797593978548

## Mitigation performance summary

### equal_opportunity_sex
- **accuracy**: 0.7508138320741969
- **f1**: 0.6340759448002165
- **precision**: 0.48543018920038666
- **recall**: 0.9139365574622985
- **roc_auc**: 0.906797593978548

### demographic_parity_sex
- **accuracy**: 0.684785946809164
- **f1**: 0.5807874530305506
- **precision**: 0.42341591233920917
- **recall**: 0.9243369734789392
- **roc_auc**: 0.906797593978548

### equal_opportunity_race
- **accuracy**: 0.7768564584484983
- **f1**: 0.6527100659592773
- **precision**: 0.5160997732426303
- **recall**: 0.8876755070202809
- **roc_auc**: 0.906797593978548

### demographic_parity_race
- **accuracy**: 0.7709600147411093
- **f1**: 0.6467746518897414
- **precision**: 0.5087170317389361
- **recall**: 0.8876755070202809
- **roc_auc**: 0.906797593978548

## Fairness tables (top groups by size)

### Baseline: Sex

| group   |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:--------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| Male    | 10860 |    0.299816 |         0.513996 | 2975 | 2607 | 4997 |  281 | 0.913698 | 0.342846  | 0.0863022 | 0.532963 | 0.94676  | Male              |           1        | True                    |
| Female  |  5421 |    0.108836 |         0.133555 |  418 |  306 | 4525 |  172 | 0.708475 | 0.0633409 | 0.291525  | 0.577348 | 0.963381 | Male              |           0.259836 | False                   |

### Baseline: Race

| group              |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |      FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:-------------------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|---------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| White              | 13946 |    0.250251 |         0.411731 | 3087 | 2655 | 7801 |  403 | 0.884527 | 0.253921  | 0.115473 | 0.537618 | 0.950878 | White             |           1        | True                    |
| Black              |  1561 |    0.11467  |         0.188981 |  150 |  145 | 1237 |   29 | 0.837989 | 0.10492   | 0.162011 | 0.508475 | 0.977093 | White             |           0.458992 | False                   |
| Asian-Pac-Islander |   480 |    0.277083 |         0.410417 |  115 |   82 |  265 |   18 | 0.864662 | 0.236311  | 0.135338 | 0.583756 | 0.936396 | White             |           0.996808 | True                    |
| Amer-Indian-Eskimo |   159 |    0.119497 |         0.264151 |   19 |   23 |  117 |    0 | 1        | 0.164286  | 0        | 0.452381 | 1        | White             |           0.641562 | False                   |
| Other              |   135 |    0.185185 |         0.222222 |   22 |    8 |  102 |    3 | 0.88     | 0.0727273 | 0.12     | 0.733333 | 0.971429 | White             |           0.539727 | False                   |

### Baseline: Sex×Race (intersection)

| group                              |    n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group     |   disparate_impact | four_fifths_rule_pass   |
|:-----------------------------------|-----:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:--------------------|-------------------:|:------------------------|
| sex=Male|race=White                | 9561 |   0.311265  |        0.534881  | 2719 | 2395 | 4190 |  257 | 0.913642 | 0.363705  | 0.0863575 | 0.531678 | 0.942208 | sex=Male|race=White |           1        | True                    |
| sex=Female|race=White              | 4385 |   0.117218  |        0.143216  |  368 |  260 | 3611 |  146 | 0.715953 | 0.0671661 | 0.284047  | 0.585987 | 0.961139 | sex=Male|race=White |           0.267752 | False                   |
| sex=Male|race=Black                |  808 |   0.169554  |        0.300743  |  125 |  118 |  553 |   12 | 0.912409 | 0.175857  | 0.0875912 | 0.514403 | 0.978761 | sex=Male|race=White |           0.56226  | False                   |
| sex=Female|race=Black              |  753 |   0.0557769 |        0.0690571 |   25 |   27 |  684 |   17 | 0.595238 | 0.0379747 | 0.404762  | 0.480769 | 0.975749 | sex=Male|race=White |           0.129107 | False                   |
| sex=Male|race=Asian-Pac-Islander   |  309 |   0.346278  |        0.533981  |   97 |   68 |  134 |   10 | 0.906542 | 0.336634  | 0.0934579 | 0.587879 | 0.930556 | sex=Male|race=White |           0.998316 | True                    |
| sex=Female|race=Asian-Pac-Islander |  171 |   0.152047  |        0.187135  |   18 |   14 |  131 |    8 | 0.692308 | 0.0965517 | 0.307692  | 0.5625   | 0.942446 | sex=Male|race=White |           0.349862 | False                   |
| sex=Male|race=Amer-Indian-Eskimo   |   93 |   0.172043  |        0.376344  |   16 |   19 |   58 |    0 | 1        | 0.246753  | 0         | 0.457143 | 1        | sex=Male|race=White |           0.703603 | False                   |
| sex=Male|race=Other                |   89 |   0.224719  |        0.280899  |   18 |    7 |   62 |    2 | 0.9      | 0.101449  | 0.1       | 0.72     | 0.96875  | sex=Male|race=White |           0.525161 | False                   |
| sex=Female|race=Amer-Indian-Eskimo |   66 |   0.0454545 |        0.106061  |    3 |    4 |   59 |    0 | 1        | 0.0634921 | 0         | 0.428571 | 1        | sex=Male|race=White |           0.198288 | False                   |
| sex=Female|race=Other              |   46 |   0.108696  |        0.108696  |    4 |    1 |   40 |    1 | 0.8      | 0.0243902 | 0.2       | 0.8      | 0.97561  | sex=Male|race=White |           0.203215 | False                   |

## Suggested report bullets (edit into your own words)

- Baseline model may be procedurally reasonable (same threshold for all), yet distributively uneven outcomes can appear across protected groups.
- Disparate impact is assessed via the 4/5ths rule (DI < 0.8 indicates potential adverse impact).
- Post-processing group thresholds can reduce specific fairness gaps (e.g., TPR gaps / selection-rate gaps) but may trade off with global accuracy/precision.
- Intersectional evaluation (sex×race) is included to avoid hiding harms that do not appear in single-attribute analysis.
