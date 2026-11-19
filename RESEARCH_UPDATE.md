# Additional research — 19 November 2026

A bounded random Fourier feature search evaluated six configurations on each
of the two existing five-fold site assignments (12 completed evaluations).
All preprocessing was fitted within training folds. No test labels, site
identifiers, reconstructed locations or dates entered the models.

The full-target 20% blends did not improve the retained hybrid. A targeted
variant uses 20% random-feature ridge predictions for CH4, NO2, CO and AAI,
with the existing hybrid providing the remaining 80%. FIRMS is unchanged.
CH4 uses table plus image summaries; the other three use table features.
Both branches use 1,024 random Fourier features, gamma = 1 / feature count,
ridge alpha 10 and random seed 20261119. CH4 and feature choices were selected
from exploratory primary-split results; the secondary assignment reuses the
same underlying sites and is not an independent dataset.

| Evaluation | Existing hybrid | Additional candidate |
|---|---:|---:|
| Primary site assignment | 0.882626 | 0.879302 |
| Secondary site assignment | 0.886535 | 0.882818 |
| Average of two held-out predictions | 0.882445 | 0.879244 |
| Public Kaggle score | 1.08642 | 1.08615 |

Submission **58566190** completed successfully. It does not outperform the
best submitted nested ensemble, **58554926**, whose public score is **1.08275**.
No rank-1 result has been achieved or established. The previously verified
final selections, **58554926 and 58555559**, remain unchanged; this additional
candidate is available in submission history but was not selected.

Five entries have been submitted on 19 November. The project follows the
conservative five-per-day interpretation recorded in RULES.md, despite a
larger live platform allowance. No further submission is planned today.

Reproduce the additional experiment after generating the documented caches
and original model predictions:

```sh
python random_feature_search.py
python random_feature_candidate.py
python random_feature_candidate.py --infer-only
```

The candidate ran with NumPy 2.2.6 and scikit-learn 1.8.0. Reloaded saved-model
inference matched its CSV within 4.45e-16. This is a saved-model reload check,
not a new isolated-environment verification. The existing final release and
submitted PDF continue to describe the two unchanged final selections.

Paired site-bootstrap improvement intervals were [0.00110, 0.00577] on the
primary assignment and [0.00153, 0.00617] on the secondary assignment. These
are conditional on the selected models and existing predictions and exclude
selection uncertainty and future site/time distribution shift. The small
public gain does not support a claim of a substantial ranking improvement.

All row-level predictions, fitted artifacts and receipts remain in the private
local artifacts directory. The public release is unchanged.

## 20 November: fire intensity and equal-site weighting

Eighteen additional held-out evaluations completed: three fire-intensity
variants and six site-weighted variants, each on both frozen assignments.
No Kaggle submission was made in this batch.

Variable positive-intensity ridge combined with the previous fire-detection
classifier improved a fixed 25% blend to 0.882088 / 0.883716 overall.
Its paired site-bootstrap improvement intervals included zero on both splits
([-0.00294, 0.00389] and [-0.00036, 0.00612]); it is not upload-ready.

Unbounded linear image models were unstable because some held-out measurements
were far beyond training-fold variability. For example, a valid-data fraction
had approximately 11,931 training-fold standard deviations in one CH4 fold.
Clipping standardized predictors to [-5, 5] corrected that numerical
extrapolation problem. This clipping uses only training-fitted scales and a
fixed bound; it does not use validation labels or test data for preprocessing.

A ridge model with inverse observed-site-count training weights and bounded
table/image predictors improved the fixed 25% blend to **0.879563 / 0.879716**.
However, paired site-bootstrap improvement intervals still included zero:
[-0.00444, 0.01151] and [-0.00101, 0.01588]. This candidate is retained for
further validation rather than submitted. All intervals are conditional on
fitted predictions and exclude model-selection and future distribution-shift
uncertainty. The original final submissions and released models remain intact.

Reproduction: `python fire_intensity_search.py` and
`python site_balance_search.py`. Scripts resume completed configurations from
private local artifacts. Source parameters and aggregate results are recorded
in the experiment ledger. No rank improvement is claimed from these tests.
