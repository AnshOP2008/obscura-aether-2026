# Aether - OBSCURA atmospheric regression

Reproducible prediction of five atmospheric z-score targets from the official
OBSCURA inputs. Research is in progress; no private-leaderboard result is known.

## Data access

Join the official competition and accept its rules before downloading data:
https://www.kaggle.com/competitions/obscura . Competition files must remain private.
This repository does not redistribute tables, chips, identifiers, or targets.

Use Python 3.12 and install `requirements.txt`. Authenticate with `kaggle auth login`.
Download using `kaggle competitions download -c obscura -p data`.
The archive contains train.csv, test.csv, sample_submission.csv and image chips.
The current working copy uses `train.csv` and `testnew.csv` (the official test.csv).

## Training and inference

Run `python -m unittest test_pipeline` for metric and submission contracts.

Baseline examples:

```powershell
python pipeline.py --train train.csv --model mean --run mean_01
python pipeline.py --train train.csv --model ridge --run ridge_01
python pipeline.py --train train.csv --model catboost --run catboost_01
python pipeline.py --train train.csv --model lightgbm --run lightgbm_01
```

Run IDs must be new to protect completed experiments. Frozen fold assignments
are saved in `artifacts/folds.csv`. Every row from a training site stays in one
fold; identifiers are excluded from predictors. Missing labels are excluded
separately for each target and are never imputed.

```powershell
python nested_blend.py --output artifacts/nested_blend_01
python infer_nested.py --run artifacts/nested_blend_01 --test testnew.csv --template data/sample_submission.csv --output artifacts/nested_blend_01/submission.csv
```

The nested procedure uses five outer site folds and three inner site folds.
It fits nonnegative, target-specific ridge/CatBoost blend weights with sum at
most one; the remaining mass is on the fixed zero z-score reference. Weights
are trained solely on inner out-of-fold predictions. Outer validation labels
never enter preprocessing, base fitting, or blend fitting. Final inference
averages the five saved outer models and their training-only blend weights.

Pixel summaries can be computed without extracting image directories:

```powershell
python image_features.py --archive data/obscura.zip --train train.csv --test testnew.csv --output artifacts/image_features.csv --workers 8
```

This reads all 14 channels in the documented order, uses SCL and cloud
probability to define a clear-pixel mask, and records reflectance summaries,
center contrasts, SWIR tails, and local edges. All-cloud regions remain missing.
The image branch has not yet been evaluated and is not part of the submitted
tabular baseline.

## Verified interim results

Lower masked mean column-wise RMSE is better. Each score averages five
target-specific RMSEs over observed labels; it is not pooled cell RMSE.

| Method | Primary site-grouped OOF MCRMSE |
|---|---:|
| Training-fold mean | 1.004654 |
| Ridge | 0.943131 |
| CatBoost | 0.952982 |
| LightGBM | 0.982933 |
| Nested calibrated ridge/CatBoost | 0.912548 |

The nested method scored 0.919404 on a second grouped fold assignment.
Primary nested per-target RMSEs: methane 1.002189, NO2 0.941650,
CO 0.988152, aerosol 0.627213, fire 1.003537.

Public Kaggle scores: ridge 1.24413 (submission 58552717); nested ensemble
1.15981 (58553016). These public scores do not establish private performance.
Public test facilities and time periods differ from training, and the private
test is later still. The considerable validation/public gap remains a limitation.

## Research record and sources

See STATUS.md, RULES.md, EXPERIMENTS.csv, SOURCES.md and
SUBMISSION_CHECKLIST.md. Models, predictions, data and private connection
configuration are excluded from Git. Do not publish notebook outputs or
reconstructable data derivatives.

Official starter methodology reference:
https://www.kaggle.com/code/zelza0/obscura-starter-baseline .
Contains modified Copernicus Sentinel data; source measurements include
Sentinel-2/5P, NASA FIRMS, ECMWF ERA5-Land and Copernicus DEM, accessed by
the competition organizers through Google Earth Engine.
