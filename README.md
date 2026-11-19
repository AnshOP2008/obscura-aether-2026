# Aether - OBSCURA atmospheric regression

Reproducible prediction of five supplied z-score targets, with site-held-out
validation and masked labels. No private score or final ranking is known.

## Results

Lower masked mean column-wise RMSE is better. It averages the five separately
masked target RMSEs; it does not pool observed cells.

| Method | Primary grouped OOF | Secondary grouped OOF |
|---|---:|---:|
| Training-fold mean | 1.004654 | - |
| Tabular ridge | 0.943131 | - |
| Tabular CatBoost | 0.952982 | - |
| Tabular LightGBM | 0.982933 | - |
| Nested tabular calibration | 0.912548 | 0.919404 |
| Nested image/tabular | 0.890592 | 0.896503 |
| Compact spatial/tabular CNN | 0.902282 | 0.911014 |
| Fixed 75/25 hybrid | **0.882626** | **0.886535** |

Averaging the two site-held-out OOF predictions scores 0.882445. Each row's site
is excluded from both models predicting it. Deployment averages ten models per
component, so this summary is not an exact unbiased estimate of ten-model
inference. Model families were selected during repeated research.

Public Kaggle scores: ridge 1.24413 (58552717), nested tabular 1.15981 (58553016),
nested image/tabular 1.08275 (58554926). The validation/public gap remains a
material limitation; these results do not establish private ranking.

## Data and environment

Accept https://www.kaggle.com/competitions/obscura/rules and obtain the official
archive separately. Never redistribute tables, chips, identifiers, labels,
feature caches, OOF rows or test predictions.

The final training environment is Linux x86-64, Python 3.12 and CUDA 12.8 on a
shared H100 MIG slice (10 GB). Use a separate environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-server.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m unittest test_pipeline
```

`requirements-server.txt` pins final core dependencies. The earlier local
baseline environment is in `requirements-baselines.txt`. Saved sklearn objects
require the final versions; older local versions produced small discrepancies.
Load only trusted model files. CPU inference works; CNN training requires CUDA.

The optional Kaggle CLI avoids browser download dialogs:

```sh
python -m pip install kaggle
kaggle auth login
kaggle competitions download -c obscura -p data
```

Place the archive at `data/obscura.zip`. Our unchanged official test.csv is named
`testnew.csv`. The table preparer refuses to overwrite different existing data.

## Full reproduction

Run from the repository root:

```sh
python reproduce.py features --archive data/obscura.zip
python reproduce.py train
python reproduce.py infer
```

The feature stage prepares tables, 123 clear-pixel summaries, 23 SWIR-tail
summaries, 2,048 frozen SeCo-Eco features and sixteen-channel 96x96 tensors.
`embeddings.py` downloads the pinned MIT checkpoint if absent and verifies its
full SHA256. All 4,439 chips were processed; every archive entry passed CRC.
Allow several GB for caches beyond the 8.35 GB archive. All-cloud statistics
remain missing and are handled with fold-local imputers.

Both site assignments are recreated exactly with seeds 20261119 and 20261120.
Every learned transformation stays inside training folds. `final_ensemble.py`
uses five outer / three inner site folds, target-specific nonnegative calibration
with sum at most one, and remaining weight on the zero reference.
`small_multimodal.py` selects epochs on inner sites, resets and refits the outer
training partition. Completed folds resume; do not mix source/configuration
changes with an existing training run.

Inference reloads all four saved runs, applies fixed nested/CNN weights .75/.25,
bounds only FIRMS below at its no-detection floor (-.439), then averages the two
assignments. The exact IDs, schema, order, uniqueness and finite values are
checked before saving `artifacts/hybrid_10fold/submission.csv` (1,221 rows).

A permitted model bundle, when attached to the repository release, restores
`artifacts/final_primary`, `final_secondary`, `cnn_primary` and `cnn_secondary`
when extracted into the repository root. Rebuild features, then run inference;
training can be skipped. The bundle excludes restricted data and predictions.

## Research and limitations

`EXPERIMENTS.csv` contains completed experiments. Private run directories retain
OOF predictions, configurations and model hashes. `build_report.py` builds the
research PDF using ReportLab, installed separately. `RULES.md`, `SOURCES.md` and
`SUBMISSION_CHECKLIST.md` record requirements, provenance and completion evidence.

Site grouping cannot reproduce all undisclosed future-time shifts. Sparse
methane labels and selection across many model families limit confidence.
The targets are coarse satellite quantities, not facility emission rates.
No location/date recovery, sample matching, external corresponding data,
identifier/ordering exploit or hidden-label probing is used.

## Attribution

Our source code is MIT-licensed. SeCo-Eco weights retain their MIT attribution;
exact provenance is in SOURCES.md. The official starter informed baseline
comparisons. Contains modified Copernicus Sentinel data and information from
NASA FIRMS, ECMWF ERA5-Land and Copernicus DEM supplied by organizers via Earth
Engine. AI assistance supported research, implementation and documentation.
