# Report Artifacts (Auto-generated)

## Baseline performance

- **accuracy**: 0.7422762729562066
- **f1**: 0.6197897789054005
- **precision**: 0.4756606397774687
- **recall**: 0.8892355694227769
- **roc_auc**: 0.8896847695379467

## Mitigation performance summary

### equal_opportunity_sex
- **accuracy**: 0.6910509182482648
- **f1**: 0.588716271463614
- **precision**: 0.42938931297709926
- **recall**: 0.9360374414976599
- **roc_auc**: 0.8896847695379467

### demographic_parity_sex
- **accuracy**: 0.6061052760886924
- **f1**: 0.5312477158102478
- **precision**: 0.36949669547534314
- **recall**: 0.9448777951118045
- **roc_auc**: 0.8896847695379467

### equal_opportunity_race
- **accuracy**: 0.7293163810576746
- **f1**: 0.6105858443050278
- **precision**: 0.4624548253245884
- **recall**: 0.8983359334373375
- **roc_auc**: 0.8896847695379467

### demographic_parity_race
- **accuracy**: 0.722682881886862
- **f1**: 0.604675597583399
- **precision**: 0.45584158415841586
- **recall**: 0.8978159126365055
- **roc_auc**: 0.8896847695379467

## Fairness tables (top groups by size)

### Baseline: Sex

| group   |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:--------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| Male    | 10860 |    0.299816 |         0.604512 | 3049 | 3516 | 4088 |  207 | 0.936425 | 0.462388  | 0.0635749 | 0.464433 | 0.951804 | Male              |            1       | True                    |
| Female  |  5421 |    0.108836 |         0.115292 |  371 |  254 | 4577 |  219 | 0.628814 | 0.0525771 | 0.371186  | 0.5936   | 0.954337 | Male              |            0.19072 | False                   |

### Baseline: Race

| group              |     n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |      FPR |      FNR |      PPV |      NPV | reference_group   |   disparate_impact | four_fifths_rule_pass   |
|:-------------------|------:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|---------:|---------:|---------:|---------:|:------------------|-------------------:|:------------------------|
| White              | 13946 |    0.250251 |         0.466442 | 3118 | 3387 | 7069 |  372 | 0.89341  | 0.323929 | 0.10659  | 0.479324 | 0.950007 | White             |           1        | True                    |
| Black              |  1561 |    0.11467  |         0.240871 |  151 |  225 | 1157 |   28 | 0.843575 | 0.162808 | 0.156425 | 0.401596 | 0.976371 | White             |           0.516401 | False                   |
| Asian-Pac-Islander |   480 |    0.277083 |         0.439583 |  111 |  100 |  247 |   22 | 0.834586 | 0.288184 | 0.165414 | 0.526066 | 0.918216 | White             |           0.942418 | True                    |
| Amer-Indian-Eskimo |   159 |    0.119497 |         0.314465 |   17 |   33 |  107 |    2 | 0.894737 | 0.235714 | 0.105263 | 0.34     | 0.981651 | White             |           0.674179 | False                   |
| Other              |   135 |    0.185185 |         0.355556 |   23 |   25 |   85 |    2 | 0.92     | 0.227273 | 0.08     | 0.479167 | 0.977011 | White             |           0.762272 | False                   |

### Baseline: Sex×Race (intersection)

| group                              |    n |   base_rate |   selection_rate |   TP |   FP |   TN |   FN |      TPR |       FPR |       FNR |      PPV |      NPV | reference_group     |   disparate_impact | four_fifths_rule_pass   |
|:-----------------------------------|-----:|------------:|-----------------:|-----:|-----:|-----:|-----:|---------:|----------:|----------:|---------:|---------:|:--------------------|-------------------:|:------------------------|
| sex=Male|race=White                | 9561 |   0.311265  |        0.623784  | 2791 | 3173 | 3412 |  185 | 0.937836 | 0.481853  | 0.062164  | 0.467975 | 0.948568 | sex=Male|race=White |           1        | True                    |
| sex=Female|race=White              | 4385 |   0.117218  |        0.123375  |  327 |  214 | 3657 |  187 | 0.636187 | 0.0552829 | 0.363813  | 0.604436 | 0.951353 | sex=Male|race=White |           0.197785 | False                   |
| sex=Male|race=Black                |  808 |   0.169554  |        0.413366  |  128 |  206 |  465 |    9 | 0.934307 | 0.307004  | 0.0656934 | 0.383234 | 0.981013 | sex=Male|race=White |           0.662675 | False                   |
| sex=Female|race=Black              |  753 |   0.0557769 |        0.0557769 |   23 |   19 |  692 |   19 | 0.547619 | 0.0267229 | 0.452381  | 0.547619 | 0.973277 | sex=Male|race=White |           0.089417 | False                   |
| sex=Male|race=Asian-Pac-Islander   |  309 |   0.346278  |        0.585761  |   96 |   85 |  117 |   11 | 0.897196 | 0.420792  | 0.102804  | 0.530387 | 0.914062 | sex=Male|race=White |           0.939044 | True                    |
| sex=Female|race=Asian-Pac-Islander |  171 |   0.152047  |        0.175439  |   15 |   15 |  130 |   11 | 0.576923 | 0.103448  | 0.423077  | 0.5      | 0.921986 | sex=Male|race=White |           0.281249 | False                   |
| sex=Male|race=Amer-Indian-Eskimo   |   93 |   0.172043  |        0.451613  |   14 |   28 |   49 |    2 | 0.875    | 0.363636  | 0.125     | 0.333333 | 0.960784 | sex=Male|race=White |           0.723989 | False                   |
| sex=Male|race=Other                |   89 |   0.224719  |        0.494382  |   20 |   24 |   45 |    0 | 1        | 0.347826  | 0         | 0.454545 | 1        | sex=Male|race=White |           0.792553 | False                   |
| sex=Female|race=Amer-Indian-Eskimo |   66 |   0.0454545 |        0.121212  |    3 |    5 |   58 |    0 | 1        | 0.0793651 | 0         | 0.375    | 1        | sex=Male|race=White |           0.194317 | False                   |
| sex=Female|race=Other              |   46 |   0.108696  |        0.0869565 |    3 |    1 |   40 |    2 | 0.6      | 0.0243902 | 0.4       | 0.75     | 0.952381 | sex=Male|race=White |           0.139402 | False                   |

## Suggested report bullets (edit into your own words)

- Baseline model may be procedurally reasonable (same threshold for all), yet distributively uneven outcomes can appear across protected groups.
- Disparate impact is assessed via the 4/5ths rule (DI < 0.8 indicates potential adverse impact).
- Post-processing group thresholds can reduce specific fairness gaps (e.g., TPR gaps / selection-rate gaps) but may trade off with global accuracy/precision.
- Intersectional evaluation (sex×race) is included to avoid hiding harms that do not appear in single-attribute analysis.
