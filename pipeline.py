"""Reproducible site-held-out regression. IDs are used only for alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ['t_ch4', 't_no2', 't_co', 't_aai', 't_firms']


def metric(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 5:
        raise ValueError('Expected matching (n, 5) arrays')
    if not np.isfinite(p).all():
        raise ValueError('Predictions must all be finite')
    scores = []
    for j in range(5):
        mask = np.isfinite(y[:, j])
        if not mask.any():
            raise ValueError(f'No observed labels for {TARGETS[j]}')
        scores.append(float(np.sqrt(np.mean((y[mask, j] - p[mask, j]) ** 2))))
    return {'overall': float(np.mean(scores)), 'targets': dict(zip(TARGETS, scores))}


def features(d, engineered=False):
    x = d.drop(columns=['sample_id', 'site_key', *TARGETS], errors='ignore').copy()
    if not all(pd.api.types.is_numeric_dtype(x[c]) for c in x):
        raise ValueError('Unexpected nonnumeric predictor')
    if engineered:
        angle = np.deg2rad(x['wind_direction'])
        x['wind_sin'], x['wind_cos'] = np.sin(angle), np.cos(angle)
        x['wind_u'] = x['wind_speed'] * np.sin(angle)
        x['wind_v'] = x['wind_speed'] * np.cos(angle)
        x['rain_log'] = np.log1p(x['precipitation'].clip(lower=0) * 1000)
    return x.replace([np.inf, -np.inf], np.nan)


def model_for(name, seed, params):
    if name == 'mean':
        return make_pipeline(SimpleImputer(keep_empty_features=True), DummyRegressor())
    if name == 'ridge':
        return make_pipeline(SimpleImputer(add_indicator=True, keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=params.get('alpha', 100.)))
    if name == 'catboost':
        from catboost import CatBoostRegressor
        config = dict(iterations=1000, depth=5, learning_rate=.025,
                      l2_leaf_reg=10, loss_function='RMSE', random_seed=seed,
                      thread_count=8, verbose=False, allow_writing_files=False)
        return CatBoostRegressor(**(config | params))
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        config = dict(n_estimators=700, learning_rate=.025, num_leaves=15,
                      min_child_samples=40, reg_lambda=10, verbosity=-1,
                      n_jobs=8, random_state=seed)
        return LGBMRegressor(**(config | params))
    raise ValueError(name)


def validate_submission(sub, test, template):
    if list(sub.columns) != list(template.columns):
        raise ValueError('Schema/order differs from sample submission')
    if sub.sample_id.isna().any() or sub.sample_id.duplicated().any():
        raise ValueError('Missing/duplicate IDs')
    if sub.sample_id.tolist() != template.sample_id.tolist():
        raise ValueError('Submission order differs from template')
    if set(sub.sample_id) != set(test.sample_id) or len(sub) != len(test):
        raise ValueError('Test-ID coverage mismatch')
    if not np.isfinite(sub[TARGETS].to_numpy(float)).all():
        raise ValueError('Nonfinite predictions')


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.train, dtype={'sample_id': str, 'site_key': str})
    if data.sample_id.isna().any() or data.sample_id.duplicated().any() or data.site_key.isna().any():
        raise ValueError('Invalid identifiers/groups')
    x = features(data, args.engineered)
    y = data[TARGETS].to_numpy(float)
    fold_file = out / 'folds.csv'
    if fold_file.exists():
        saved = pd.read_csv(fold_file, dtype={'sample_id': str})
        if saved.sample_id.tolist() != data.sample_id.tolist():
            raise ValueError('Saved folds do not match data')
        folds = saved.fold.to_numpy()
    else:
        # Shuffle group labels independently of identifiers, then balance by row count.
        groups = data.site_key.drop_duplicates().to_numpy(copy=True)
        np.random.default_rng(args.seed).shuffle(groups)
        group_index = {g: i for i, g in enumerate(groups)}
        codes = data.site_key.map(group_index)
        folds = np.full(len(data), -1)
        for f, (_, va) in enumerate(GroupKFold(5).split(x, groups=codes)):
            folds[va] = f
        pd.DataFrame({'sample_id': data.sample_id, 'fold': folds}).to_csv(fold_file, index=False)
    test = pd.read_csv(args.test, dtype={'sample_id': str}) if args.test else None
    xt = features(test, args.engineered) if test is not None else None
    if xt is not None and list(xt) != list(x):
        raise ValueError('Train/test predictor mismatch')
    params = json.loads(args.params)
    run = out / args.run
    run.mkdir(exist_ok=False)
    start = time.monotonic()
    oof = np.full_like(y, np.nan)
    pred = np.zeros((len(test), 5)) if test is not None else None
    fold_scores = []
    for f in sorted(np.unique(folds)):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        if set(data.site_key.iloc[tr]) & set(data.site_key.iloc[va]):
            raise ValueError('Group leakage')
        for j, target in enumerate(TARGETS):
            observed = tr[np.isfinite(y[tr, j])]
            if not len(observed) or not np.isfinite(y[va, j]).any():
                raise ValueError('Insufficient observed labels in fold')
            model = model_for(args.model, args.seed, params)
            # CatBoost accepts NaN, but not infinity; LightGBM handles NaN natively.
            model.fit(x.iloc[observed], y[observed, j])
            oof[va, j] = model.predict(x.iloc[va])
            if pred is not None:
                pred[:, j] += model.predict(xt) / len(np.unique(folds))
            joblib.dump(model, run / f'{f}_{target}.joblib')
        score = metric(y[va], oof[va])
        fold_scores.append(score)
        print(json.dumps({'fold': int(f), **score}), flush=True)
    result = {'model': args.model, 'seed': args.seed, 'params': params,
              'engineered': args.engineered, 'features': list(x), 'score': metric(y, oof),
              'fold_scores': fold_scores, 'runtime_seconds': time.monotonic() - start,
              'train_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest()}
    pd.DataFrame(oof, columns=TARGETS).assign(sample_id=data.sample_id).to_csv(run / 'oof.csv', index=False)
    if pred is not None:
        if not args.template:
            raise ValueError('Template required for submission')
        template = pd.read_csv(args.template, dtype={'sample_id': str})
        sub = template[['sample_id']].merge(pd.DataFrame(pred, columns=TARGETS).assign(sample_id=test.sample_id),
                                           on='sample_id', how='left', validate='one_to_one')
        sub = sub[template.columns]
        validate_submission(sub, test, template)
        sub.to_csv(run / 'submission.csv', index=False)
    (run / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='train.csv')
    p.add_argument('--test')
    p.add_argument('--template')
    p.add_argument('--output', default='artifacts')
    p.add_argument('--run', required=True)
    p.add_argument('--model', choices=['mean', 'ridge', 'catboost', 'lightgbm'], required=True)
    p.add_argument('--seed', type=int, default=20261119)
    p.add_argument('--params', default='{}')
    p.add_argument('--engineered', action='store_true')
    train(p.parse_args())
