# Sources and attribution

Source pipeline implemented locally. Frozen SeCo-Eco embeddings were evaluated and are included in the final nested CO candidate.

Official starter notebook downloaded privately and inspected for methodology: https://www.kaggle.com/code/zelza0/obscura-starter-baseline . It motivates shallow regularized models, image statistics, and shrinkage. Its same-OOF shrinkage evaluation will not be treated as unbiased.

- Competition overview: https://obscura.devpost.com/
- Binding platform rules: https://www.kaggle.com/competitions/obscura/rules
- Participation rules: https://obscura.devpost.com/rules
- Data/report resources: https://obscura.devpost.com/resources

## Frozen encoder (evaluated and incorporated)

- SeCo-Eco / SSL4Eco: Plekhanova et al., CVPR EarthVision 2025, https://arxiv.org/abs/2504.18256 . Official repository https://github.com/PlekhanovaElena/ssl4eco , MIT.
- TorchGeo model card https://huggingface.co/torchgeo/seco-eco explicitly lists MIT and the exported checkpoint architecture.
- Checkpoint pinned at revision 3ede08c40f70534eb1132057ee338b99e54319f0, file resnet50_sentinel2_all_seco_eco-a273eb78.pth. Full SHA256 verified against Hugging Face LFS metadata: a273eb78ab8e99b1d51d9a8ddfa58ab455b00734a0532dfe506ece2a71e30f13 (94,467,878 bytes). The model-card export example mentions MD5, but the actual published filename uses the SHA256 prefix; the implementation checks the full authoritative LFS hash.
- Official TorchGeo configuration: https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/resnet.py . Required channels match competition's 12 reflectance bands exactly: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12. Published transform resizes to 224 and divides by 10000. SCL/CLDPRB stay separate.
- No pretraining datasets downloaded, no sample identification or spatial matching. Public ecological imagery pretraining is distinct from OBSCURA labels. Use weights only under competition's public-pretrained-model permission; disclose if selected.
- Prithvi considered but not chosen initially: available 6-band HLS models require additional preprocessing and have differing metadata conventions. TorchGeo's 12-band SeCo-Eco is a direct semantic channel match.

Record package versions, licenses, checkpoint provenance, and reused code before use.

## Additional research references

- Grinsztajn et al., small tabular-data model comparisons: https://arxiv.org/abs/2207.08815 . Motivates strong-tree baselines, not a winning claim.
- CatBoost missing values: https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing .
- Public participant notebook inspected: https://www.kaggle.com/code/avikdas567/multimodal-atmospheric-radiometry-plume-dynamics . No code or claimed results incorporated; license not independently established. Several summaries overlap our already implemented pixel and SWIR features. Our scores come from our own runs.
