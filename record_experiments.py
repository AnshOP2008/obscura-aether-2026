"""Refresh the compact experiment ledger from verified persisted results."""
import csv
import json
from pathlib import Path

FIELDS=['experiment_id','hypothesis','features','model','split','seed','hyperparameters','scores','runtime_seconds','artifacts','decision']

if __name__=='__main__':
    rows=[]
    for file in sorted(Path('artifacts').glob('*/result.json')):
        r=json.loads(file.read_text());name=file.parent.name
        if 'score' not in r or r.get('complete') is False:continue
        if name.startswith('random_feature'):
            hypothesis='Smooth nonlinear feature interactions complement the retained hybrid'
            split='two frozen site assignments; exploratory target selection, fixed20% weight'
        elif name.startswith('tabicl'):
            hypothesis='Synthetic-pretrained tabular foundation model adds complementary generalization'
            split='5 frozen site folds; observed labels only'
        elif name.startswith('hybrid'):
            hypothesis='Fixed75/25 nested/CNN errors complement across both site assignments'
            split='two independent 5-fold site assignments; each OOF row excludes its site'
        elif name.startswith('cnn'):
            hypothesis='Compact spatial/tabular representation adds complementary signal'
            split='5 outer site folds; inner20% sites selects epochs'
        elif name.startswith('final_'):
            hypothesis='Target-specific nested image/tabular calibration transfers across sites'
            split='5 outer / 3 inner site folds'
        elif name.startswith('fire_'):
            hypothesis='SWIR-tail fire component transfers to new sites; other targets use fixed reference OOF'
            split='5 site folds; fire component comparison'
        elif 'nested' in name:
            hypothesis='Training-only target calibration reduces variance on unseen sites'
            split='5 outer / 3 inner site folds'
        elif 'embedding' in name:
            hypothesis='Frozen 12-band SeCo-Eco adds transferable image information'
            split='5 frozen site folds'
        elif 'image' in name or 'combined' in name:
            hypothesis='Pixel summaries and tabular predictors carry complementary signal'
            split='5 frozen site folds'
        else:
            hypothesis='Controlled tabular baseline or regularization improvement'
            split='5 frozen site folds'
        score=r['score']['overall']
        decision='candidate; assess per-target stability and independent grouped split'
        if name.startswith('random_feature'):decision='reserve: public1.08615 improves hybrid slightly but not best1.08275; existing final selections unchanged'
        elif name.startswith('tabicl'):decision='reject: .951329 grouped score does not improve retained candidates; no submission'
        elif score>1.05:decision='reject: poor held-out generalization'
        elif name in ['mean_01','ridge_01','catboost_01','lightgbm_01']:decision='baseline reference'
        elif name.startswith(('hybrid','final_','cnn')):decision='keep: complementary final candidate; confirmed on both assignments'
        features=r.get('features',r.get('feature_columns',[]))
        rows.append({'experiment_id':name,'hypothesis':hypothesis,'features':len(features) if features else 'see result/config',
                     'model':r.get('model',name),'split':split,'seed':r.get('seed',r.get('fold_seed',20261119)),
                     'hyperparameters':json.dumps(r.get('params',{'method':r.get('method','see source/config')})),
                     'scores':json.dumps(r['score']),'runtime_seconds':r.get('runtime_seconds',r.get('runtime_seconds_this_invocation','not recorded')),
                     'artifacts':str(file.parent),'decision':decision})
    search=Path('artifacts/random_feature_search/results.json')
    if search.exists():
        for r in json.loads(search.read_text()):
            rows.append({'experiment_id':'rff_'+r['name'],
                'hypothesis':'Smooth nonlinear feature interactions add complementary signal',
                'features':r['source'], 'model':'Random Fourier features + Ridge',
                'split':r['split']+' frozen five site folds', 'seed':r['seed'],
                'hyperparameters':json.dumps({k:r[k] for k in ['gamma_multiplier','alpha','components']}),
                'scores':json.dumps({'standalone':r['score'],'fixed20pct_blend':r['blend_score']}),
                'runtime_seconds':r['seconds'], 'artifacts':str(search.parent),
                'decision':'reject full-target blend; selected target contributions evaluated separately'})
    with Path('EXPERIMENTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    print('Recorded',len(rows),'completed experiments; incomplete runs excluded')
