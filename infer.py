"""Generate predictions using saved fold models, without retraining."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pipeline import attach_images, features, validate_submission, TARGETS


def predict(run, test):
    run = Path(run)
    info = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    x = features(test, info['engineered'])
    x = x[info['features']]
    result = np.zeros((len(test), 5))
    for j, target in enumerate(TARGETS):
        models = sorted(run.glob(f'*_{target}.joblib'))
        if len(models) != 5:
            raise ValueError('Expected five fold models')
        for path in models:
            result[:, j] += joblib.load(path).predict(x) / len(models)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--template', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--image-features')
    args = p.parse_args()
    test = attach_images(pd.read_csv(args.test, dtype={'sample_id': str}), args.image_features)
    template = pd.read_csv(args.template, dtype={'sample_id': str})
    pred = pd.DataFrame(predict(args.run, test), columns=TARGETS)
    pred['sample_id'] = test.sample_id
    sub = template[['sample_id']].merge(pred, on='sample_id', how='left', validate='one_to_one')
    sub = sub[template.columns]
    validate_submission(sub, test, template)
    sub.to_csv(args.output, index=False)
    print(f'Validated {len(sub)} predictions')
