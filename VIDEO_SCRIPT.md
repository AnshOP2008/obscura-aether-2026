# Video recording guide (about three minutes)

Open the research PDF full screen. Record your screen and microphone. You may
keep your camera visible if you prefer. Read the narration below naturally and
point to the indicated area. Do not show private tables, chips, IDs, credentials
or server screens. The current Devpost rules waive video; this guide also gives
you a presentation ready to record if desired.

## Page 1 - problem and honest evaluation (0:00-0:40)

Point to the title, then the observed-label table and evaluation paragraph.

"Hello, I am Anshu Gupta. This is Aether, my OBSCURA research entry. It predicts
five atmospheric targets using the supplied satellite and environmental inputs.
The main challenge is generalization: the test facilities and time periods are
different from training. Our validation keeps every facility entirely within one
fold. Missing target measurements are excluded from the loss and metric rather
than replaced with zero. We evaluate the official average of five target RMSEs."

## Page 2 - how the model works (0:40-1:30)

Point to the three image representations, model menu and prediction rule.

"I compared simple regularized models with boosted trees before adding images.
The image branch uses clear-pixel spectral summaries, spatial contrasts and SWIR
tail statistics. A frozen, publicly available SeCo-Eco encoder adds multispectral
features, with its license and checkpoint disclosed. A small image-plus-tabular
network provides another representation. The main ensemble chooses target-specific
weights using an inner layer of site-grouped validation. The final hybrid uses
fixed weights of seventy-five percent nested ensemble and twenty-five percent
neural model, averaged over two site assignments."

## Page 3 - actual results and limitations (1:30-2:20)

Point first to grouped results, then per-target results, then public scores.

"The hybrid scored about point eight eight three and point eight eight seven on
the two grouped validation assignments. Aerosol and fire provide the largest
gains, while methane and carbon monoxide remain difficult. These are validation
results, not a private leaderboard score. The best public submission scored one
point zero eight two seven five. The hybrid scored slightly worse publicly,
despite better grouped validation, so both candidates are retained. This gap is
an important finding: site validation cannot fully simulate future-time shift.
I am not claiming first place or an unavailable private score."

## Page 4 - reproducibility and conclusion (2:20-3:00)

Point to the repository link, smoke test and sources.

"The entry includes source code, saved models, dependency versions and commands
for feature extraction, training and inference. A fresh isolated environment
reloaded all four saved runs and reproduced every test prediction with zero
difference using the cached representations. Restricted competition data and
prediction rows are not published. The report distinguishes completed experiments
from claims and credits the data providers, pretrained encoder and software.
AI assistance supported implementation and documentation. Thank you."

Before recording, use the latest report and update any score if a later verified
submission changes it. Upload only a finished recording you have reviewed.
