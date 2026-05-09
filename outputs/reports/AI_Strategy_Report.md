# F1 Undercut Strategy Analysis Report

## Highest Probability Scenarios
| Race                 |   Driver |   Car_Ahead |    Gap |   Tyre_Advantage |   AI_Probability_% |   Success |
|:---------------------|---------:|------------:|-------:|-----------------:|-------------------:|----------:|
| Dutch Grand Prix     |       77 |          23 |  1.491 |                0 |              100   |         0 |
| Canadian Grand Prix  |       20 |          11 |  1.678 |                0 |              100   |         0 |
| Canadian Grand Prix  |       18 |          27 |  0.303 |                0 |               98.5 |         1 |
| Dutch Grand Prix     |       31 |          44 |  0.266 |                0 |               98   |         1 |
| Abu Dhabi Grand Prix |       30 |          77 | 13.472 |               17 |               97.5 |         1 |

## Real Successful Examples
| Race                 |   Driver |   Car_Ahead |    Gap |   Tyre_Advantage |   AI_Probability_% |   Success |
|:---------------------|---------:|------------:|-------:|-----------------:|-------------------:|----------:|
| Canadian Grand Prix  |       18 |          27 |  0.303 |                0 |               98.5 |         1 |
| Dutch Grand Prix     |       31 |          44 |  0.266 |                0 |               98   |         1 |
| Abu Dhabi Grand Prix |       30 |          77 | 13.472 |               17 |               97.5 |         1 |
| Monaco Grand Prix    |       44 |          55 |  3.079 |               -2 |               96   |         1 |
| Austrian Grand Prix  |       77 |           5 |  1.275 |              -12 |               92.5 |         1 |

## Feature Importance
| Feature         |   Importance (%) |
|:----------------|-----------------:|
| Gap             |          32.8715 |
| Tyre_Advantage  |          26.586  |
| Ahead_TyreLife  |          20.4665 |
| Driver_TyreLife |          20.076  |

## Strategic Findings
- Average successful undercut gap: 3.353 sec
- Average failed undercut gap: 4.851 sec
- Most important feature: Gap
- Scenarios above 40% AI probability: 28
- Model comment: Araclar arasindaki saniye farki undercut basarisini dogrudan etkiliyor.
