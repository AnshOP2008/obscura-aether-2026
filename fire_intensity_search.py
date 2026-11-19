"""Evaluate variable positive fire intensity on frozen held-out sites.

The earlier hurdle assumes constant intensity given a detection. This search
tests fold-fitted positive-intensity models and a nonnegative Poisson model.
Only permitted numeric measurements and pixel summaries are predictors.
"""
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fire_hurdle import signed_log, make_classifier
from pipeline import TARGETS, features, attach_images, metric


def main():
    d = pd.read_csv('train.csv', dtype={'sample_id': str, 'site_key': str})
    joined = attach_images(d, 'artifacts/hot_features.csv')
    hot = signed_log(joined[[c for c in joined if c.startswith('im_hot_')]])
    x = pd.concat([features(d), hot], axis=1)
    y = d.t_firms.to_numpy(float)
    out = Path('artifacts/fire_intensity_search'); out.mkdir(exist_ok=True)
    results = []
    for split, foldfile in [('primary', 'folds.csv'), ('secondary', 'folds_secondary.csv')]:
        f = pd.read_csv('artifacts/'+foldfile, dtype={'sample_id': str})
        assert f.sample_id.tolist() == d.sample_id.tolist()
        a = pd.read_csv('artifacts/final_'+split+'/oof.csv', dtype={'sample_id': str})
        b = pd.read_csv('artifacts/cnn_'+split+'/oof.csv', dtype={'sample_id': str})
        assert a.sample_id.tolist() == b.sample_id.tolist() == d.sample_id.tolist()
        reference = .75*a[TARGETS]+.25*b[TARGETS]
        reference.t_firms = reference.t_firms.clip(lower=-.439)
        for name in ('positive_ridge', 'positive_cat', 'poisson_cat'):
            path = out/(split+'_'+name+'.json')
            if path.exists():
                results.append(json.loads(path.read_text())); continue
            started = time.monotonic(); pred = np.zeros(len(d))
            for fold in sorted(f.fold.unique()):
                tr = np.flatnonzero(f.fold.to_numpy()!=fold)
                va = np.flatnonzero(f.fold.to_numpy()==fold)
                assert not set(d.site_key.iloc[tr]) & set(d.site_key.iloc[va])
                floor = float(y[tr].min()); positive = y[tr]>floor+1e-6
                imp = SimpleImputer(keep_empty_features=True)
                xt = imp.fit_transform(x.iloc[tr]); xv = imp.transform(x.iloc[va])
                params = dict(iterations=300, depth=2, learning_rate=.025,
                              l2_leaf_reg=50, thread_count=4, verbose=False,
                              allow_writing_files=False, random_seed=20261120)
                if name == 'poisson_cat':
                    model = CatBoostRegressor(loss_function='Poisson', **params)
                    model.fit(xt, y[tr]-floor)
                    pred[va] = floor+model.predict(xv, prediction_type='Exponent')
                else:
                    classifier = make_classifier('logistic_0.1')
                    classifier.fit(hot.iloc[tr], positive.astype(int))
                    probability = classifier.predict_proba(hot.iloc[va])[:,1]
                    if name == 'positive_ridge':
                        model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
                    else:
                        model = CatBoostRegressor(loss_function='RMSE', **params)
                    model.fit(xt[positive], y[tr[positive]]-floor)
                    intensity = np.maximum(model.predict(xv), 0.)
                    pred[va] = floor+probability*intensity
            whole = reference.copy(); whole.t_firms = pred
            blended = reference.copy(); blended.t_firms = .75*reference.t_firms+.25*pred
            result = {'name':split+'_'+name, 'score':metric(d[TARGETS],whole),
                      'blend_score':metric(d[TARGETS],blended),
                      'reference_score':metric(d[TARGETS],reference),
                      'seconds':time.monotonic()-started,
                      'limitation':'Exploratory comparisons on reused sites; other targets unchanged'}
            whole['sample_id'] = d.sample_id; whole.to_csv(out/(split+'_'+name+'.csv'),index=False)
            path.write_text(json.dumps(result,indent=2)); results.append(result)
            print(json.dumps(result), flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
