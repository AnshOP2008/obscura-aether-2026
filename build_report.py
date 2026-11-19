"""Build the concise research report from persisted aggregate results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.pagesizes import A4

OUT=Path('output/pdf');OUT.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyA',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#223347')))
styles.add(ParagraphStyle(name='LabelA',fontName='Helvetica-Bold',fontSize=10,leading=14,spaceAfter=7,textColor=colors.HexColor('#147A83')))
styles.add(ParagraphStyle(name='CellA',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=0,textColor=colors.HexColor('#223347')))
styles['Title'].fontSize=26;styles['Title'].leading=30;styles['Title'].textColor=colors.HexColor('#122C48')
styles['Heading2'].textColor=colors.HexColor('#122C48');styles['Heading2'].spaceBefore=13
story=[]
def p(text,style='BodyA'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'Heading2')
def table(rows,widths):
    data=[[Paragraph(escape(str(x)),styles['CellA']) for x in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F1F4')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#147A83')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D4DEE5'))]))
    story.append(t)
def new(label):story.append(PageBreak());p(label,'LabelA')
def score(name):return json.loads((Path('artifacts')/name/'result.json').read_text())['score']['overall']

p('OBSCURA / RESEARCH ENTRY / 19 NOVEMBER 2026','LabelA')
p('Aether: Generalization<br/>Beyond Familiar Sites','Title')
p('Anshu Gupta | Kaggle: anshop2008 | Devpost: anshurgupta1work')
h('Abstract')
p('Aether predicts five supplied atmospheric z-scores from environmental features and multispectral satellite chips. We emphasize transfer to unseen facilities through frozen site-grouped validation. A target-specific, nested calibrated ensemble is combined with a compact spatial/tabular network using fixed 75/25 weights. The hybrid scores 0.88263 and 0.88653 masked mean column-wise RMSE on two grouped assignments; averaging their held-out predictions gives 0.88244. These are research validation results, not private leaderboard results. The public validation gap remains a material limitation.')
h('Data and integrity')
p('The official tables contain 3,218 training rows from 81 sites, 1,221 test rows and 29 numeric predictors. IDs are unique and disjoint. The full 8,353,857,457-byte archive passed a sequential CRC audit of all 4,443 entries. All 4,439 chips were processed successfully. No site identity, location, date reconstruction, external sample matching or metadata predictor was used.')
table([['Target','Observed training labels','Meaning (supplied z-score)'],['CH4','1,113','Methane enhancement'],['NO2','3,130','Nitrogen dioxide enhancement'],['CO','3,188','Carbon monoxide enhancement'],['AAI','3,218','Absorbing aerosol index'],['FIRMS','3,218','Fire/flare thermal intensity']],[65,125,292])
h('Metric and evaluation design')
p('For each target j, RMSE is computed only over observed truths; the five column RMSEs are averaged equally. Missing labels are masked before arithmetic and are never imputed. Every site remains in one outer fold. Two deterministic five-fold group assignments use seeds 20261119 and 20261120. Every fold has observed labels for all targets. Identifiers serve only joining and grouping.')
p('All learned imputers, scalers, reductions and regressors are fit on training sites. Three inner site folds produce calibration predictions for each outer training partition. Calibration weights are nonnegative with sum at most one, with a fixed 0.03 regularization penalty; remaining weight belongs to the zero z-score reference. The research program selected model families using repeated validation, so these estimates retain selection uncertainty.')

new('METHOD / FEATURES AND MODELS')
h('Three complementary image representations')
p('<b>Pixel summaries:</b> 123 clear-pixel reflectance, quantile, spatial contrast, texture and quality summaries, plus 23 supplemental SWIR-tail summaries. The 12 reflectance bands use the documented 1e-4 scale. SCL classes 0, 1, 3, 8, 9, 10 and 11, or cloud probability at least 40, exclude pixels from the clear branch. All-cloud regions remain missing (11 chips); the separate SWIR-tail branch retains saturation information. No undisclosed physical chip extent is inferred.')
p('<b>Frozen encoder:</b> MIT-licensed SeCo-Eco ResNet50 from TorchGeo, pinned to Hugging Face revision 3ede08c40f70534eb1132057ee338b99e54319f0. The exact 12 reflectance channels match semantically; antialiased bilinear resize to 224 and division by 10000 follow the published transform. A 2,048-dimensional frozen feature vector is unit-normalized. No corresponding external samples or competition labels are used in pretraining.')
p('<b>Compact spatial branch:</b> sixteen 96 x 96 channels preserve twelve 4 x 4 mean reflectances, two SWIR maxima, cloud probability and clear fraction. Reflectances use log1p(5x). A three-stage convolutional encoder (16/32/64 channels) with GroupNorm and SiLU, global mean/max pooling and a 29-feature tabular branch produces five outputs. The model is trained from scratch with equally weighted masked target losses, AdamW (learning rate 0.001, decay 0.01), gradient clipping and dropout. No rotation or spectral color augmentation breaks wind relationships.')
h('Target-specific calibrated ensemble')
table([['Target','Candidate base models inside nested calibration'],['CH4','Tabular ridge; tabular CatBoost'],['NO2','Tabular ridge; combined pixel/tabular CatBoost'],['CO','Tabular ridge; normalized embedding ridge; tabular CatBoost'],['AAI','Tabular ridge; combined pixel/tabular CatBoost'],['FIRMS','Quantile-transformed SWIR ridge; logistic hurdle; combined CatBoost']],[65,417])
p('CatBoost uses depth 4, at most 1,000 trees, learning rate 0.025, L2 50 and feature fraction 0.6. An inner site holdout selects iterations before refitting each training partition. Fire ridge uses alpha 100; its logistic hurdle uses C=0.1 and training-only conditional means. Embedding ridge uses alpha 1000 without per-coordinate standardization. The neural branch similarly selects epochs on an inner 20% site holdout (maximum 30, patience 6), then resets and refits the outer training sites.')
h('Final prediction rule')
p('For each assignment, prediction = 0.75 x nested ensemble + 0.25 x compact CNN. Only FIRMS is bounded below at the documented no-detection training floor (-0.439); gas and aerosol predictions are not clipped. The two assignments are averaged, giving ten outer models per component. The fixed hybrid weights were checked on both assignments, without optimizing weights on the reported outer predictions.')

new('RESULTS / CONTROLLED COMPARISONS')
table([['Method','Primary grouped MCRMSE'],['Training-fold mean',f'{score("mean_01"):.6f}'],['Tabular ridge',f'{score("ridge_01"):.6f}'],['Tabular CatBoost',f'{score("catboost_01"):.6f}'],['Tabular LightGBM',f'{score("lightgbm_01"):.6f}'],['Nested tabular calibration','0.912548'],['Combined pixel/SWIR + CatBoost','0.900630'],['Nested target-specific image/tabular','0.890592'],['Compact spatial/tabular CNN','0.902282'],['Fixed 75/25 hybrid','0.882626']],[335,147])
p('Controlled comparisons reuse the frozen primary folds. The image/tabular nested ensemble also scores 0.896503 on the second assignment; the compact CNN scores 0.911014 there. The fixed hybrid improves both to 0.882626 / 0.886535. Rejected branches include unrestrained standardized embeddings (unstable held-out extrapolation), combined linear pixel regression and heavier tree variants without stable gains. The experiment ledger records completed runs and configurations.')
h('Hybrid per-target RMSE')
hy=json.loads(Path('artifacts/hybrid_10fold/result.json').read_text())
table([['Target','Primary','Secondary','Two-OOF average']]+[[t.replace('t_','').upper(),f'{hy["primary_score"]["targets"][t]:.6f}',f'{hy["secondary_score"]["targets"][t]:.6f}',f'{hy["score"]["targets"][t]:.6f}'] for t in ['t_ch4','t_no2','t_co','t_aai','t_firms']]+[['MCRMSE',f'{hy["primary_score"]["overall"]:.6f}',f'{hy["secondary_score"]["overall"]:.6f}',f'{hy["score"]["overall"]:.6f}']],[95,118,118,151])
p('Each OOF row averages two predictions from models that excluded that row\'s entire site. Test inference instead averages ten models; the two-OOF result is a stability summary rather than an exact unbiased evaluation of ten-model deployment. Methane and CO remain close to the zero-reference error; aerosol and fire account for most predictive gains.')
h('Public leaderboard evidence')
rows=[['Submission','Method','Public MCRMSE'],['58552717','Tabular ridge','1.24413'],['58553016','Nested tabular','1.15981'],['58554926','Nested image/tabular','1.08275']]
receipt=Path('artifacts/hybrid_10fold/receipt.json')
if receipt.exists():
    rr=json.loads(receipt.read_text());rows.append([rr['id'],'Fixed hybrid',rr['public_score']])
table(rows,[95,265,122])
p('The entries above were evaluated successfully. Public and private test sites and time ranges differ from training. No private score is available. A public rank cannot establish a private result; the organizer also requires private MCRMSE at most 0.80 for cash-prize eligibility.')

new('REPRODUCTION / LIMITATIONS AND SOURCES')
h('Reproduction and artifacts')
p('Code: <link href="https://github.com/AnshOP2008/obscura-aether-2026" color="#147A83">github.com/AnshOP2008/obscura-aether-2026</link>. The README specifies the final Python 3.12 Linux environment and all commands. Accept the competition rules and obtain the official archive separately. Run reproduce.py features, then reproduce.py train and reproduce.py infer. Saved permitted models and preprocessing artifacts can replace the training stage. The pipeline validates exact text IDs, target order, row count, uniqueness and finite predictions before writing the 1,221-row CSV.')
smoke=Path('artifacts/hybrid_10fold/smoke.json')
if smoke.exists():
    sr=json.loads(smoke.read_text());p('A fresh isolated environment reloaded all four component runs and regenerated every test prediction from cached pixel/embedding representations. Maximum absolute difference from the training environment: '+str(sr['max_absolute_difference'])+'. This is a cached-feature inference smoke test, not an independent raw-data retraining audit.')
else:p('Clean-environment inference verification is pending; this report will be updated with its measured result before final submission.')
p('Training used shared H100 MIG resources with a 10 GB device slice, bounded CPU threads and private random-named directories. The full archive remains private. Model/configuration hashes and dependency versions are retained. No tables, chips, identifiers, targets, feature caches or prediction files are redistributed in the public code repository.')
h('Limits and responsible interpretation')
p('Site grouping cannot reproduce the undisclosed future-time shift. Repeated model research creates selection bias even with nested calibration. Two grouped assignments are correlated and do not constitute an independent hidden test. Methane labels are sparse; targets describe coarse satellite columns, not measured facility emission rates. Reflectance thresholds are statistical cues, not proof of combustion. Learned predictions should not be used to attribute emissions to identified facilities.')
h('Sources, licenses and assistance')
refs=[
('Official competition, rules and data','https://www.kaggle.com/competitions/obscura'),
('Official baseline (methodology reference)','https://www.kaggle.com/code/zelza0/obscura-starter-baseline'),
('Devpost requirements and deadline','https://obscura.devpost.com/rules'),
('Organizer private-score prize threshold','https://www.kaggle.com/competitions/obscura/discussion/753176'),
('SeCo-Eco model card and MIT weights','https://huggingface.co/torchgeo/seco-eco'),
('SSL4Eco / Plekhanova et al., EarthVision 2025','https://arxiv.org/abs/2504.18256'),
('Small tabular-model research / Grinsztajn et al.','https://arxiv.org/abs/2207.08815')]
for label,url in refs:p(escape(label)+': <link href="'+url+'" color="#147A83">'+escape(url)+'</link>')
p('Implementation uses scikit-learn, CatBoost, LightGBM, PyTorch, timm and Rasterio. Our code is MIT-licensed; third-party weights retain their MIT attribution. Contains modified Copernicus Sentinel data and information from NASA FIRMS, ECMWF ERA5-Land and Copernicus DEM supplied by the organizers via Earth Engine. AI assistance supported research, coding and writing; reported results come from executed experiments. Public participant notebooks were inspected as research references; their unverified claims and code were not incorporated.')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D4DEE5'));canvas.line(45,42,550,42)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52677B'))
    canvas.drawString(45,29,'AETHER / OBSCURA 2026 / REPRODUCIBLE ATMOSPHERIC REGRESSION')
    canvas.drawRightString(550,29,str(doc.page))
doc=SimpleDocTemplate(str(OUT/'aether-research.pdf'),pagesize=A4,rightMargin=56,leftMargin=56,topMargin=42,bottomMargin=54,title='Aether: Generalization Beyond Familiar Sites',author='Anshu Gupta')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Research report generated')
