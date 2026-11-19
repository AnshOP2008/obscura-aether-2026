"""Predeclared fixed ensemble of independently trained grouped procedures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import TARGETS,metric,validate_submission

WEIGHT_CNN=.25


def blend(a,b):
    if a.sample_id.tolist()!=b.sample_id.tolist():raise ValueError('Prediction alignment mismatch')
    result=a[['sample_id']].copy()
    result[TARGETS]=(1-WEIGHT_CNN)*a[TARGETS]+WEIGHT_CNN*b[TARGETS]
    # The no-detection FIRMS floor is documented and observed in every training fold.
    result['t_firms']=result['t_firms'].clip(lower=-.439)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true')
    p.add_argument('--nested-primary',default='artifacts/final_primary')
    p.add_argument('--nested-secondary',default='artifacts/final_secondary')
    p.add_argument('--cnn-primary',default='artifacts/cnn_primary')
    p.add_argument('--cnn-secondary',default='artifacts/cnn_secondary')
    p.add_argument('--output',default='artifacts/hybrid_10fold');a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True)
    filename='oof.csv' if a.evaluate else 'submission.csv'
    load=lambda path:pd.read_csv(Path(path)/filename,dtype={'sample_id':str})
    first=blend(load(a.nested_primary),load(a.cnn_primary))
    second=blend(load(a.nested_secondary),load(a.cnn_secondary))
    if first.sample_id.tolist()!=second.sample_id.tolist():raise ValueError('Fold ensemble alignment')
    final=first.copy();final[TARGETS]=(first[TARGETS]+second[TARGETS])/2
    if a.evaluate:
        y=pd.read_csv('train.csv',dtype={'sample_id':str})
        if final.sample_id.tolist()!=y.sample_id.tolist():raise ValueError('Truth alignment')
        report={'model':'fixed 75% nested image/tabular + 25% compact CNN; two grouped assignments',
                'primary_score':metric(y[TARGETS],first[TARGETS]),'secondary_score':metric(y[TARGETS],second[TARGETS]),
                'score':metric(y[TARGETS],final[TARGETS]),'cnn_weight':WEIGHT_CNN,
                'floor':{'t_firms':-.439},
                'limitation':'Each OOF row averages two site-held-out predictions; test inference averages ten folds per component. Model families were selected during the research program.'}
        final.to_csv(out/'oof.csv',index=False);(out/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
    else:
        template=pd.read_csv('data/sample_submission.csv',dtype={'sample_id':str})
        test=pd.read_csv('testnew.csv',dtype={'sample_id':str});final=final[template.columns]
        validate_submission(final,test,template);final.to_csv(out/'submission.csv',index=False)
        print(json.dumps({'validated_rows':len(final),'cnn_weight':WEIGHT_CNN,'finite':True}))
