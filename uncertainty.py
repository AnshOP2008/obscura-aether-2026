"""Paired site-bootstrap uncertainty for aligned OOF candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import TARGETS, metric


def group_errors(data,pred):
    y=data[TARGETS].to_numpy(float)
    p=pred[TARGETS].to_numpy(float)
    if pred.sample_id.tolist()!=data.sample_id.tolist():raise ValueError('OOF alignment mismatch')
    if not np.isfinite(p).all():raise ValueError('Invalid OOF predictions')
    groups=pd.factorize(data.site_key,sort=True)[0]
    total=np.zeros((groups.max()+1,5));count=np.zeros_like(total)
    for j in range(5):
        good=np.isfinite(y[:,j])
        np.add.at(total[:,j],groups[good],(y[good,j]-p[good,j])**2)
        np.add.at(count[:,j],groups[good],1)
    return total,count


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True);p.add_argument('--reference',default='artifacts/ridge_01/oof.csv')
    p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=2000)
    a=p.parse_args()
    d=pd.read_csv('train.csv',dtype={'sample_id':str,'site_key':str})
    candidate=pd.read_csv(a.candidate,dtype={'sample_id':str})
    reference=pd.read_csv(a.reference,dtype={'sample_id':str})
    cs,counts=group_errors(d,candidate);rs,_=group_errors(d,reference)
    draw=np.random.default_rng(20261119).integers(0,len(cs),size=(a.draws,len(cs)))
    cc=counts[draw].sum(axis=1)
    scores=np.sqrt(cs[draw].sum(axis=1)/cc).mean(axis=1)
    refs=np.sqrt(rs[draw].sum(axis=1)/cc).mean(axis=1)
    result={'candidate_score':metric(d[TARGETS],candidate[TARGETS]),
            'score_95pct_site_bootstrap_interval':np.quantile(scores,[.025,.975]).tolist(),
            'improvement_over_reference_95pct_interval':np.quantile(refs-scores,[.025,.975]).tolist(),
            'sites':len(cs),'draws':a.draws,
            'limitation':'Conditional on existing fitted OOF models and chosen folds; does not capture all model-selection or future-time shift uncertainty'}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
