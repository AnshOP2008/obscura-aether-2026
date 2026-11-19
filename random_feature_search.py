"""Bounded random Fourier feature test with two frozen site assignments.

All imputation/scaling is fit on the outer training fold. No identifiers enter
the model. Fixed 20% candidate blends are diagnostic, not nested estimates of
the performance of selecting the best candidate from this search.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from pipeline import TARGETS, features, attach_images, metric


def main():
    d = pd.read_csv('train.csv', dtype={'sample_id': str, 'site_key': str})
    y = d[TARGETS].to_numpy(float)
    inputs = {'table': features(d), 'combined': features(attach_images(d, 'artifacts/image_hot_features.csv'))}
    out = Path('artifacts/random_feature_search'); out.mkdir(exist_ok=True)
    results = []
    for split, foldfile in [('primary', 'folds.csv'), ('secondary', 'folds_secondary.csv')]:
        folds = pd.read_csv('artifacts/' + foldfile, dtype={'sample_id': str})
        assert folds.sample_id.tolist() == d.sample_id.tolist()
        refs = []
        for name in ('final_', 'cnn_'):
            ref = pd.read_csv('artifacts/' + name + split + '/oof.csv', dtype={'sample_id': str})
            assert ref.sample_id.tolist() == d.sample_id.tolist()
            refs.append(ref[TARGETS].to_numpy(float))
        base = .75 * refs[0] + .25 * refs[1]
        base[:, 4] = np.maximum(base[:, 4], -.439)
        for source, x in inputs.items():
            for scale in (.25, 1., 4.):
                name = f'{split}_{source}_{scale}'
                path = out / (name + '.json')
                if path.exists():
                    results.append(json.loads(path.read_text())); continue
                started = time.monotonic()
                pred = np.full_like(y, np.nan)
                for fold in sorted(folds.fold.unique()):
                    tr = np.flatnonzero(folds.fold.to_numpy() != fold)
                    va = np.flatnonzero(folds.fold.to_numpy() == fold)
                    assert not set(d.site_key.iloc[tr]) & set(d.site_key.iloc[va])
                    imp = SimpleImputer(keep_empty_features=True)
                    scaler = StandardScaler()
                    xt = np.clip(scaler.fit_transform(imp.fit_transform(x.iloc[tr])), -8, 8)
                    xv = np.clip(scaler.transform(imp.transform(x.iloc[va])), -8, 8)
                    rff = RBFSampler(gamma=scale/x.shape[1], n_components=1024, random_state=20261119)
                    zt = rff.fit_transform(xt); zv = rff.transform(xv)
                    for j in range(len(TARGETS)):
                        keep = np.isfinite(y[tr, j])
                        model = Ridge(alpha=10.).fit(zt[keep], y[tr[keep], j])
                        pred[va, j] = model.predict(zv)
                blended = .8 * base + .2 * pred
                blended[:, 4] = np.maximum(blended[:, 4], -.439)
                result = {'name': name, 'split': split, 'source': source, 'gamma_multiplier': scale,
                          'alpha': 10., 'components': 1024, 'seed': 20261119,
                          'score': metric(y, pred), 'blend_score': metric(y, blended),
                          'reference_score': metric(y, base), 'seconds': time.monotonic()-started}
                frame = pd.DataFrame(pred, columns=TARGETS); frame['sample_id'] = d.sample_id
                frame.to_csv(out / (name + '.csv'), index=False)
                path.write_text(json.dumps(result, indent=2)); results.append(result)
                print(json.dumps(result), flush=True)
    (out/'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
