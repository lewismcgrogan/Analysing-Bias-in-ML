# Report Artifacts (Auto-generated)

## Baseline performance

- **accuracy**: 0.7872980775136662
- **f1**: 0.6631650617644198
- **precision**: 0.5297591297591298
- **recall**: 0.8863754550182007
- **roc_auc**: 0.9048326388222396

## Mitigation performance summary

### equal_opportunity_sex
- **accuracy**: 0.7530864197530864
- **f1**: 0.6360014487504527
- **precision**: 0.48791330925257015
- **recall**: 0.9131565262610505
- **roc_auc**: 0.9048326388222396

### demographic_parity_sex
- **accuracy**: 0.6789509243903937
- **f1**: 0.5762464531820024
- **precision**: 0.41865944163034513
- **recall**: 0.9240769630785232
- **roc_auc**: 0.9048326388222396

### equal_opportunity_race
- **accuracy**: 0.7701615379890671
- **f1**: 0.647844908714474
- **precision**: 0.5076696165191741
- **recall**: 0.8949557982319293
- **roc_auc**: 0.9048326388222396

### demographic_parity_race
- **accuracy**: 0.767459001289847
- **f1**: 0.6441060349689791
- **precision**: 0.5044169611307421
- **recall**: 0.890795631825273
- **roc_auc**: 0.9048326388222396

## Fairness tables (top groups by size)

### Baseline: Sex

| group   |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:--------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| Male    | 10860 |    0.299816 |         0.518508 | 2972 | 2659 | 4945 |  284 | 0.912776 | 0.349684  | 0.0872236 | 0.527793 | 0.945688 | Male              |           1        | True                    |
| Female  |  5421 |    0.108836 |         0.148312 |  437 |  367 | 4464 |  153 | 0.740678 | 0.0759677 | 0.259322  | 0.543532 | 0.966862 | Male              |           0.286036 | False                   |

### Baseline: Race

| group              |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |      FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:-------------------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|---------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| White              | 13946 |    0.250251 |         0.418041 | 3099 | 2731 | 7725 |  391 | 0.887966 | 0.26119   | 0.112034 | 0.531561 | 0.951824 | White             |           1        | True                    |
| Black              |  1561 |    0.11467  |         0.207559 |  154 |  170 | 1212 |   25 | 0.860335 | 0.12301   | 0.139665 | 0.475309 | 0.97979  | White             |           0.496505 | False                   |
| Asian-Pac-Islander |   480 |    0.277083 |         0.425    |  114 |   90 |  257 |   19 | 0.857143 | 0.259366  | 0.142857 | 0.558824 | 0.931159 | White             |           1.01665  | True                    |
| Amer-Indian-Eskimo |   159 |    0.119497 |         0.289308 |   19 |   27 |  113 |    0 | 1        | 0.192857  | 0        | 0.413043 | 1        | White             |           0.692057 | False                   |
| Other              |   135 |    0.185185 |         0.22963  |   23 |    8 |  102 |    2 | 0.92     | 0.0727273 | 0.08     | 0.741935 | 0.980769 | White             |           0.549299 | False                   |

### Baseline: Sex×Race (intersection)

| group                              |    n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group     |   disparate_impact | four_fifths_rule_pass   |
|:-----------------------------------|-----:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:--------------------|-------------------:|:------------------------|
| sex=Male|race=White                | 9561 |   0.311265  |        0.537601  | 2715 | 2425 | 4160 |  261 | 0.912298 | 0.368261  | 0.0877016 | 0.52821  | 0.940964 | sex=Male|race=White |           1        | True                    |
| sex=Female|race=White              | 4385 |   0.117218  |        0.157355  |  384 |  306 | 3565 |  130 | 0.747082 | 0.0790493 | 0.252918  | 0.556522 | 0.964817 | sex=Male|race=White |           0.292698 | False                   |
| sex=Male|race=Black                |  808 |   0.169554  |        0.321782  |  127 |  133 |  538 |   10 | 0.927007 | 0.198212  | 0.0729927 | 0.488462 | 0.981752 | sex=Male|race=White |           0.598552 | False                   |
| sex=Female|race=Black              |  753 |   0.0557769 |        0.0849934 |   27 |   37 |  674 |   15 | 0.642857 | 0.0520394 | 0.357143  | 0.421875 | 0.978229 | sex=Male|race=White |           0.158098 | False                   |
| sex=Male|race=Asian-Pac-Islander   |  309 |   0.346278  |        0.543689  |   96 |   72 |  130 |   11 | 0.897196 | 0.356436  | 0.102804  | 0.571429 | 0.921986 | sex=Male|race=White |           1.01133  | True                    |
| sex=Female|race=Asian-Pac-Islander |  171 |   0.152047  |        0.210526  |   18 |   18 |  127 |    8 | 0.692308 | 0.124138  | 0.307692  | 0.5      | 0.940741 | sex=Male|race=White |           0.391604 | False                   |
| sex=Male|race=Amer-Indian-Eskimo   |   93 |   0.172043  |        0.408602  |   16 |   22 |   55 |    0 | 1        | 0.285714  | 0         | 0.421053 | 1        | sex=Male|race=White |           0.760048 | False                   |
| sex=Male|race=Other                |   89 |   0.224719  |        0.280899  |   18 |    7 |   62 |    2 | 0.9      | 0.101449  | 0.1       | 0.72     | 0.96875  | sex=Male|race=White |           0.522505 | False                   |
| sex=Female|race=Amer-Indian-Eskimo |   66 |   0.0454545 |        0.121212  |    3 |    5 |   58 |    0 | 1        | 0.0793651 | 0         | 0.375    | 1        | sex=Male|race=White |           0.225469 | False                   |
| sex=Female|race=Other              |   46 |   0.108696  |        0.130435  |    5 |    1 |   40 |    0 | 1        | 0.0243902 | 0         | 0.833333 | 1        | sex=Male|race=White |           0.242624 | False                   |

## Suggested report bullets (edit into your own words)

- Baseline model may be procedurally reasonable (same threshold for all), yet distributively uneven outcomes can appear across protected groups.
- Disparate impact is assessed via the 4/5ths rule (DI < 0.8 indicates potential adverse impact).
- Post-processing group thresholds can reduce specific fairness gaps (e.g., TPR gaps / selection-rate gaps) but may trade off with global accuracy/precision.
- Intersectional evaluation (sex×race) is included to avoid hiding harms that do not appear in single-attribute analysis.
