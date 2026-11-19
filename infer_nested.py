import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pipeline import features, TARGETS, validate_submission


def predict(run, test):
    run = Path(run)
    info = json.loads((run / 'result.json').read_text())
    x = features(test)
    result = np.zeros((len(test), 5))
    ws = np.asarray(info['outer_weights'])
    if ws.shape != (5, 5, 2) or (ws < -1e-6).any() or (ws.sum(axis=2) > 1.000001).any():
        raise ValueError('Invalid blend weights')
    for f in range(5):
        for j, target in enumerate(TARGETS):
            for k, name in enumerate(['ridge', 'catboost']):
                model = joblib.load(run / f'{f}_{target}_{name}.joblib')
                result[:, j] += ws[f, j, k] * model.predict(x) / 5
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='artifacts/nested_blend_01')
    p.add_argument('--test', default='testnew.csv')
    p.add_argument('--template', default='data/sample_submission.csv')
    p.add_argument('--output', default='artifacts/nested_blend_01/submission.csv')
    a = p.parse_args()
    test = pd.read_csv(a.test, dtype={'sample_id': str})
    template = pd.read_csv(a.template, dtype={'sample_id': str})
    pred = pd.DataFrame(predict(a.run, test), columns=TARGETS).assign(sample_id=test.sample_id)
    sub = template[['sample_id']].merge(pred, on='sample_id', how='left', validate='one_to_one')[template.columns]
    validate_submission(sub, test, template)
    sub.to_csv(a.output, index=False)
    print(f'Validated {len(sub)} predictions using nested fold ensembles')
