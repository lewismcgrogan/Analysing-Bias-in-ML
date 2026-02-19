# Report Artifacts (Auto-generated)

## Baseline performance

- **accuracy**: 0.7990295436398256
- **f1**: 0.673778664007976
- **precision**: 0.5464100905562742
- **recall**: 0.8785751430057203
- **roc_auc**: 0.9080931922439744

## Mitigation performance summary

### equal_opportunity_sex
- **accuracy**: 0.7688102696394571
- **f1**: 0.6493385504005962
- **precision**: 0.5059523809523809
- **recall**: 0.906136245449818
- **roc_auc**: 0.9080931922439744

### demographic_parity_sex
- **accuracy**: 0.6943676678336711
- **f1**: 0.5870539419087137
- **precision**: 0.4311311555338859
- **recall**: 0.9196567862714509
- **roc_auc**: 0.9080931922439744

### equal_opportunity_race
- **accuracy**: 0.7830600085989804
- **f1**: 0.6585460170146945
- **precision**: 0.5241612803939674
- **recall**: 0.8855954238169527
- **roc_auc**: 0.9080931922439744

### demographic_parity_race
- **accuracy**: 0.7775935139119219
- **f1**: 0.6525952221049602
- **precision**: 0.5171050630986772
- **recall**: 0.8842953718148726
- **roc_auc**: 0.9080931922439744

## Fairness tables (top groups by size)

### Baseline: Sex

| group   |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:--------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| Male    | 10860 |    0.299816 |         0.50221  | 2958 | 2496 | 5108 |  298 | 0.908477 | 0.328248  | 0.0915233 | 0.542354 | 0.944876 | Male              |           1        | True                    |
| Female  |  5421 |    0.108836 |         0.134662 |  421 |  309 | 4522 |  169 | 0.713559 | 0.0639619 | 0.286441  | 0.576712 | 0.963974 | Male              |           0.268138 | False                   |

### Baseline: Race

| group              |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:-------------------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| White              | 13946 |    0.250251 |         0.403341 | 3072 | 2553 | 7903 |  418 | 0.880229 | 0.244166  | 0.119771  | 0.546133 | 0.949766 | White             |           1        | True                    |
| Black              |  1561 |    0.11467  |         0.18706  |  152 |  140 | 1242 |   27 | 0.849162 | 0.101302  | 0.150838  | 0.520548 | 0.978723 | White             |           0.463775 | False                   |
| Asian-Pac-Islander |   480 |    0.277083 |         0.410417 |  115 |   82 |  265 |   18 | 0.864662 | 0.236311  | 0.135338  | 0.583756 | 0.936396 | White             |           1.01754  | True                    |
| Amer-Indian-Eskimo |   159 |    0.119497 |         0.257862 |   18 |   23 |  117 |    1 | 0.947368 | 0.164286  | 0.0526316 | 0.439024 | 0.991525 | White             |           0.639313 | False                   |
| Other              |   135 |    0.185185 |         0.214815 |   22 |    7 |  103 |    3 | 0.88     | 0.0636364 | 0.12      | 0.758621 | 0.971698 | White             |           0.532588 | False                   |

### Baseline: Sex×Race (intersection)

| group                              |    n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group     |   disparate_impact | four_fifths_rule_pass   |
|:-----------------------------------|-----:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:--------------------|-------------------:|:------------------------|
| sex=Male|race=White                | 9561 |   0.311265  |        0.522539  | 2702 | 2294 | 4291 |  274 | 0.90793  | 0.348368  | 0.0920699 | 0.540833 | 0.939978 | sex=Male|race=White |           1        | True                    |
| sex=Female|race=White              | 4385 |   0.117218  |        0.143444  |  370 |  259 | 3612 |  144 | 0.719844 | 0.0669078 | 0.280156  | 0.588235 | 0.961661 | sex=Male|race=White |           0.274512 | False                   |
| sex=Male|race=Black                |  808 |   0.169554  |        0.294554  |  126 |  112 |  559 |   11 | 0.919708 | 0.166915  | 0.080292  | 0.529412 | 0.980702 | sex=Male|race=White |           0.563698 | False                   |
| sex=Female|race=Black              |  753 |   0.0557769 |        0.0717131 |   26 |   28 |  683 |   16 | 0.619048 | 0.0393812 | 0.380952  | 0.481481 | 0.97711  | sex=Male|race=White |           0.13724  | False                   |
| sex=Male|race=Asian-Pac-Islander   |  309 |   0.346278  |        0.530744  |   97 |   67 |  135 |   10 | 0.906542 | 0.331683  | 0.0934579 | 0.591463 | 0.931034 | sex=Male|race=White |           1.0157   | True                    |
| sex=Female|race=Asian-Pac-Islander |  171 |   0.152047  |        0.192982  |   18 |   15 |  130 |    8 | 0.692308 | 0.103448  | 0.307692  | 0.545455 | 0.942029 | sex=Male|race=White |           0.369317 | False                   |
| sex=Male|race=Amer-Indian-Eskimo   |   93 |   0.172043  |        0.344086  |   15 |   17 |   60 |    1 | 0.9375   | 0.220779  | 0.0625    | 0.46875  | 0.983607 | sex=Male|race=White |           0.658488 | False                   |
| sex=Male|race=Other                |   89 |   0.224719  |        0.269663  |   18 |    6 |   63 |    2 | 0.9      | 0.0869565 | 0.1       | 0.75     | 0.969231 | sex=Male|race=White |           0.516062 | False                   |
| sex=Female|race=Amer-Indian-Eskimo |   66 |   0.0454545 |        0.136364  |    3 |    6 |   57 |    0 | 1        | 0.0952381 | 0         | 0.333333 | 1        | sex=Male|race=White |           0.260963 | False                   |
| sex=Female|race=Other              |   46 |   0.108696  |        0.108696  |    4 |    1 |   40 |    1 | 0.8      | 0.0243902 | 0.2       | 0.8      | 0.97561  | sex=Male|race=White |           0.208014 | False                   |

## Suggested report bullets (edit into your own words)

- Baseline model may be procedurally reasonable (same threshold for all), yet distributively uneven outcomes can appear across protected groups.
- Disparate impact is assessed via the 4/5ths rule (DI < 0.8 indicates potential adverse impact).
- Post-processing group thresholds can reduce specific fairness gaps (e.g., TPR gaps / selection-rate gaps) but may trade off with global accuracy/precision.
- Intersectional evaluation (sex×race) is included to avoid hiding harms that do not appear in single-attribute analysis.
