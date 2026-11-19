"""Test equal-site training weights; site identifiers never enter predictors."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from pipeline import TARGETS, features, attach_images, metric


def main():
    d = pd.read_csv('train.csv', dtype={'sample_id': str, 'site_key': str})
    inputs = {'table':features(d), 'combined':features(attach_images(d,'artifacts/image_hot_features.csv'))}
    y = d[TARGETS].to_numpy(float)
    out = Path('artifacts/site_balance_search'); out.mkdir(exist_ok=True)
    results = []
    for split, foldfile in [('primary','folds.csv'),('secondary','folds_secondary.csv')]:
        folds = pd.read_csv('artifacts/'+foldfile,dtype={'sample_id':str})
        assert folds.sample_id.tolist() == d.sample_id.tolist()
        a = pd.read_csv('artifacts/final_'+split+'/oof.csv',dtype={'sample_id':str})
        b = pd.read_csv('artifacts/cnn_'+split+'/oof.csv',dtype={'sample_id':str})
        assert a.sample_id.tolist() == b.sample_id.tolist() == d.sample_id.tolist()
        base = .75*a[TARGETS].to_numpy(float)+.25*b[TARGETS].to_numpy(float)
        base[:,4] = np.maximum(base[:,4],-.439)
        for source, kind in [('table','ridge'),('combined','ridge'),('table','cat'),('combined','cat'),
                             ('table','ridge_clip'),('combined','ridge_clip')]:
            name = split+'_'+source+'_'+kind
            path = out/(name+'.json')
            if path.exists():
                results.append(json.loads(path.read_text())); continue
            started = time.monotonic(); pred = np.full_like(y,np.nan)
            for fold in sorted(folds.fold.unique()):
                train = np.flatnonzero(folds.fold.to_numpy()!=fold)
                va = np.flatnonzero(folds.fold.to_numpy()==fold)
                assert not set(d.site_key.iloc[train]) & set(d.site_key.iloc[va])
                for j in range(5):
                    tr = train[np.isfinite(y[train,j])]
                    groups = d.site_key.iloc[tr]
                    weights = 1/groups.map(groups.value_counts()).to_numpy(float)
                    weights /= weights.mean()
                    imp = SimpleImputer(keep_empty_features=True)
                    xt = imp.fit_transform(inputs[source].iloc[tr]); xv = imp.transform(inputs[source].iloc[va])
                    if kind.startswith('ridge'):
                        scaler = StandardScaler().fit(xt,sample_weight=weights)
                        xt = scaler.transform(xt); xv = scaler.transform(xv)
                        if kind == 'ridge_clip':
                            xt = np.clip(xt,-5,5); xv = np.clip(xv,-5,5)
                        model = Ridge(alpha=100.)
                    else:
                        model = CatBoostRegressor(iterations=400,depth=3,learning_rate=.025,l2_leaf_reg=50,
                            random_seed=20261120,thread_count=4,verbose=False,allow_writing_files=False)
                    model.fit(xt,y[tr,j],sample_weight=weights)
                    pred[va,j] = model.predict(xv)
            pred[:,4] = np.maximum(pred[:,4],-.439)
            blend = .75*base+.25*pred
            result = {'name':name,'score':metric(y,pred),'blend_score':metric(y,blend),
                      'reference_score':metric(y,base),'seconds':time.monotonic()-started,
                      'method':'Inverse observed training count per site and target; mean-normalized weights'}
            pd.DataFrame(pred,columns=TARGETS).assign(sample_id=d.sample_id).to_csv(out/(name+'.csv'),index=False)
            path.write_text(json.dumps(result,indent=2));results.append(result)
            print(json.dumps(result),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
