# F1 Undercut Strateji Analiz Raporu

## En Yüksek Olasılıklı Senaryolar
| Race                   |   Driver |   Car_Ahead |    Gap |   Tyre_Advantage |   AI_Probability_% |   Success |
|:-----------------------|---------:|------------:|-------:|-----------------:|-------------------:|----------:|
| Monaco Grand Prix      |       11 |          27 | 12.592 |               32 |                100 |         1 |
| Dutch Grand Prix       |       14 |           4 |  0.531 |                0 |                 99 |         1 |
| Monaco Grand Prix      |        4 |          22 |  5.53  |                0 |                 99 |         1 |
| Abu Dhabi Grand Prix   |       30 |          77 | 13.472 |               17 |                 98 |         1 |
| Mexico City Grand Prix |       31 |          77 |  3.286 |                1 |                 97 |         0 |

## Gerçek Başarılı Örnekler
| Race                |   Driver |   Car_Ahead |    Gap |   Tyre_Advantage |   AI_Probability_% |   Success |
|:--------------------|---------:|------------:|-------:|-----------------:|-------------------:|----------:|
| Monaco Grand Prix   |        1 |          55 |  3.845 |                0 |                 66 |         1 |
| Monaco Grand Prix   |       11 |          55 |  2.857 |                0 |                 75 |         1 |
| Austrian Grand Prix |       77 |           5 |  1.275 |              -12 |                  2 |         1 |
| Monaco Grand Prix   |       11 |          27 | 12.592 |               32 |                100 |         1 |
| Monaco Grand Prix   |       24 |          27 |  4.521 |                0 |                 85 |         1 |

## Feature Importance
| Feature         |   Importance (%) |
|:----------------|-----------------:|
| Tyre_Advantage  |          43.1386 |
| Ahead_TyreLife  |          20.9817 |
| Gap             |          18.6681 |
| Driver_TyreLife |          17.2115 |

## Stratejik Bulgular
- Başarılı undercut ortalama gap: 3.353 sn
- Başarısız undercut ortalama gap: 4.889 sn
- En önemli değişken: Tyre_Advantage
- Yorum: Undercut başarısında lastik avantajı belirleyici faktördür.
