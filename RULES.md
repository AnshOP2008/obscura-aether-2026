# Verified rules

Sources inspected 2026-11-19:
- https://www.kaggle.com/competitions/obscura/rules (authenticated browser)
- https://obscura.devpost.com/rules
- https://obscura.devpost.com/resources

## Requirements
- Eligible enrolled college/university students; age and platform eligibility apply. Do not invent attestations.
- One Kaggle account, one team, matching participants across platforms.
- Deadline displayed on Kaggle and Devpost: November 24, 2026, 20:30 IST.
- Kaggle allows two final selections. Daily rule says “five (10)”; conservatively cap at five pending live confirmation.
- Mandatory valid Kaggle predictions and Devpost entry with research PDF and code/notebook reference.
- Official score is masked mean column-wise RMSE, averaged over five target-specific RMSE values.
- Never recover locations/dates/facilities/hidden labels, exploit identifiers/file properties/order, hand-label test data, or probe leaderboard answers.
- No outside data for corresponding samples/facilities/dates.
- Free public pretrained models allowed if not trained on competition data/answers; disclose provenance.
- Kaggle explicitly allows test-feature normalization and own-prediction pseudo-labeling. Prefer fold-local preprocessing initially.
- Data must remain inaccessible to people who have not accepted rules. Never publish tables, images, targets, identifiers, or reconstructable derivatives.
- Code/models/summary metrics may be shared without reconstructing data. Public code must also be shared on associated Kaggle forum/notebook under OSI-approved license.
- Winning code must reproduce predictions and include training/inference/environment documentation.

## Conflicts and unresolved details
- Team maximum: Kaggle 5, Devpost 20. Use stricter 5 if relevant.
- Video: Kaggle requires it in several clauses; Devpost explicitly states no video. Do not silently assume the waiver supersedes Kaggle.
- README, exact feature definitions, dataset license, evaluation implementation, live upload allowance, and starter notebook still pending inspection.
