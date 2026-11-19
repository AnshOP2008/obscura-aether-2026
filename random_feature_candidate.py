"""Reproduce the exploratory 20% random-feature / 80% hybrid candidate.

CH4 uses image summaries plus table; NO2, CO, AAI use table only. FIRMS is
unchanged. These choices followed primary-split exploration and confirmation
on a second assignment of the same sites, not an independent unseen dataset.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from pipeline import TARGETS, features, attach_images, metric, validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer-only', action='store_true')
    args = parser.parse_args()
    out = Path('artifacts/random_feature_candidate'); out.mkdir(exist_ok=True)
    d = pd.read_csv('train.csv', dtype={'sample_id': str, 'site_key': str})
    test = pd.read_csv('testnew.csv', dtype={'sample_id': str})
    inputs = {}
    for name, frame in [('train', d), ('test', test)]:
        inputs[name] = {'table': features(frame),
                        'combined': features(attach_images(frame, 'artifacts/image_hot_features.csv'))}
    y = d[TARGETS].to_numpy(float)
    tests, oofs = [], []
    for split, foldname in [('primary', 'folds.csv'), ('secondary', 'folds_secondary.csv')]:
        folds = pd.read_csv('artifacts/'+foldname, dtype={'sample_id': str})
        assert folds.sample_id.tolist() == d.sample_id.tolist()
        pred = np.full_like(y, np.nan); tp = np.zeros((len(test), 5))
        for fold in sorted(folds.fold.unique()):
            tr = np.flatnonzero(folds.fold.to_numpy() != fold)
            va = np.flatnonzero(folds.fold.to_numpy() == fold)
            assert not set(d.site_key.iloc[tr]) & set(d.site_key.iloc[va])
            for source, targets in [('combined', [0]), ('table', [1, 2, 3])]:
                path = out/f'{split}_{fold}_{source}.joblib'
                x = inputs['train'][source]; xtest = inputs['test'][source]
                assert list(x) == list(xtest)
                if args.infer_only:
                    bundle = joblib.load(path)
                    transform, models = bundle['transform'], bundle['models']
                else:
                    transform = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                        FunctionTransformer(np.clip, kw_args={'a_min': -8, 'a_max': 8}),
                        RBFSampler(gamma=1./x.shape[1], n_components=1024, random_state=20261119))
                    z = transform.fit_transform(x.iloc[tr]); models = {}
                    for j in targets:
                        keep = np.isfinite(y[tr, j])
                        models[j] = Ridge(alpha=10.).fit(z[keep], y[tr[keep], j])
                    joblib.dump({'transform': transform, 'models': models, 'columns': list(x)}, path)
                zv = transform.transform(x.iloc[va]); ztest = transform.transform(xtest)
                for j in targets:
                    pred[va, j] = models[j].predict(zv)
                    tp[:, j] += models[j].predict(ztest)/5
        def reference(filename, ids):
            a = pd.read_csv('artifacts/final_'+split+'/'+filename, dtype={'sample_id': str})
            b = pd.read_csv('artifacts/cnn_'+split+'/'+filename, dtype={'sample_id': str})
            assert a.sample_id.tolist() == b.sample_id.tolist() == ids
            p = .75*a[TARGETS].to_numpy(float)+.25*b[TARGETS].to_numpy(float)
            p[:, 4] = np.maximum(p[:, 4], -.439)
            return p
        base = reference('oof.csv', d.sample_id.tolist())
        bt = reference('submission.csv', test.sample_id.tolist())
        base[:, :4] = .8*base[:, :4]+.2*pred[:, :4]
        bt[:, :4] = .8*bt[:, :4]+.2*tp[:, :4]
        oofs.append(base); tests.append(bt)
    submission = pd.DataFrame(np.mean(tests, axis=0), columns=TARGETS)
    submission['sample_id'] = test.sample_id
    template = pd.read_csv('data/sample_submission.csv', dtype={'sample_id': str})
    submission = submission[template.columns]
    validate_submission(submission, test, template)
    if args.infer_only:
        previous = pd.read_csv(out/'submission.csv', dtype={'sample_id': str})
        delta = float(np.abs(previous[TARGETS].to_numpy()-submission[TARGETS].to_numpy()).max())
        assert delta < 1e-12
        (out/'reload_check.json').write_text(json.dumps({'max_abs_difference': delta, 'passed': True}))
        print('Saved-model inference verified:', delta)
    else:
        submission.to_csv(out/'submission.csv', index=False)
        oof = pd.DataFrame(np.mean(oofs, axis=0), columns=TARGETS); oof['sample_id'] = d.sample_id
        oof.to_csv(out/'oof.csv', index=False)
        result = {'model': '20% random Fourier ridge on first four targets, 80% fixed hybrid; FIRMS unchanged',
                  'score': metric(y, oof[TARGETS]), 'primary_score': metric(y, oofs[0]),
                  'secondary_score': metric(y, oofs[1]), 'sklearn_version': sklearn.__version__,
                  'numpy_version': np.__version__, 'selection_limitation': __doc__}
        (out/'result.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result))


if __name__ == '__main__':
    main()
