"""Nested grouped evaluation of nonnegative target-specific shrinkage/blends.

Outer sites never enter base-model fitting, preprocessing, or weight fitting.
Zero is a fixed z-score reference, not an imputed target.
"""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from pipeline import TARGETS, features, metric, model_for, validate_submission


def weights(y, p):
    mask = np.isfinite(y)
    yy, pp = y[mask], p[mask]
    # The sum bound leaves the remaining mass on the fixed zero reference.
    result = minimize(lambda w: np.mean((yy - pp @ w) ** 2) + .03 * np.sum(w ** 2),
                      np.array([.25, .25]), method='SLSQP', bounds=[(0, 1)] * 2,
                      constraints=[{'type': 'ineq', 'fun': lambda w: 1 - w.sum()}],
                      options={'ftol': 1e-10, 'maxiter': 100})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def fit_predict(x, y, train_idx, predict_idx, name, seed):
    selected = train_idx[np.isfinite(y[train_idx])]
    model = model_for(name, seed, {})
    model.fit(x.iloc[selected], y[selected])
    return model, model.predict(x.iloc[predict_idx])


def main(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    d = pd.read_csv('train.csv', dtype={'sample_id': str, 'site_key': str})
    x, y = features(d), d[TARGETS].to_numpy(float)
    fold_path = Path(args.folds)
    if not fold_path.exists():
        unique = d.site_key.drop_duplicates().to_numpy(copy=True)
        np.random.default_rng(args.fold_seed).shuffle(unique)
        codes = d.site_key.map({g:i for i,g in enumerate(unique)})
        assignment = np.full(len(d),-1)
        for f,(_,va) in enumerate(GroupKFold(5).split(x,groups=codes)):
            assignment[va] = f
        pd.DataFrame({'sample_id':d.sample_id,'fold':assignment}).to_csv(fold_path,index=False)
    folds = pd.read_csv(fold_path, dtype={'sample_id': str})
    if folds.sample_id.tolist() != d.sample_id.tolist():
        raise ValueError('Fold alignment mismatch')
    groups = d.site_key.to_numpy()
    outer = folds.fold.to_numpy()
    oof = np.zeros_like(y)
    all_weights = []
    start = time.monotonic()
    names = ['ridge', 'catboost']
    for f in range(5):
        tr, va = np.flatnonzero(outer != f), np.flatnonzero(outer == f)
        if set(groups[tr]) & set(groups[va]):
            raise ValueError('Outer leakage')
        inner = np.full((len(d), 5, 2), np.nan)
        for it, iv in GroupKFold(3).split(x.iloc[tr], groups=groups[tr]):
            fit_idx, check_idx = tr[it], tr[iv]
            if set(groups[fit_idx]) & set(groups[check_idx]):
                raise ValueError('Inner leakage')
            for j in range(5):
                for k, name in enumerate(names):
                    _, pred = fit_predict(x, y[:, j], fit_idx, check_idx, name, 20261119)
                    inner[check_idx, j, k] = pred
        fw = np.zeros((5, 2))
        for j, target in enumerate(TARGETS):
            fw[j] = weights(y[tr, j], inner[tr, j])
            base = np.zeros((len(va), 2))
            for k, name in enumerate(names):
                model, base[:, k] = fit_predict(x, y[:, j], tr, va, name, 20261119)
                joblib.dump(model, out / f'{f}_{target}_{name}.joblib')
            oof[va, j] = base @ fw[j]
        all_weights.append(fw.tolist())
        print(json.dumps({'outer_fold': f, 'score': metric(y[va], oof[va]), 'weights': fw.tolist()}), flush=True)
        np.save(out / 'oof_partial.npy', oof)
    result = {'score': metric(y, oof), 'outer_weights': all_weights,
              'runtime_seconds': time.monotonic() - start, 'fold_seed':args.fold_seed,
              'method': 'Nested 5 outer / 3 inner site-grouped; nonnegative ridge/CatBoost weights with sum <= 1; penalty .03'}
    pd.DataFrame(oof, columns=TARGETS).assign(sample_id=d.sample_id).to_csv(out / 'oof.csv', index=False)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='artifacts/nested_blend_01')
    p.add_argument('--folds',default='artifacts/folds.csv')
    p.add_argument('--fold-seed',type=int,default=20261119)
    main(p.parse_args())
