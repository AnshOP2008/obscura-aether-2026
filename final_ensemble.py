"""Target-specific image/tabular ensemble with nested site-only calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from pipeline import TARGETS,features,attach_images,fit_model,validate_submission,metric
from fire_hurdle import signed_log,make_classifier

MENU={
 't_ch4':['ridge_table','cat_table'],
 't_no2':['ridge_table','cat_combined'],
 't_co':['ridge_table','ridge_embedding','cat_table'],
 't_aai':['ridge_table','cat_combined'],
 't_firms':['fire_direct','fire_hurdle','cat_combined'],
}
CAT={'depth':4,'iterations':1000,'l2_leaf_reg':50,'learning_rate':.025,'rsm':.6}


def load_inputs(d,args):
    hot=attach_images(d,args.hot)
    return {'table':features(d),'combined':features(attach_images(d,args.images)),
            'embedding':features(attach_images(d,args.embeddings)),
            'hot':signed_log(hot[[c for c in hot if c.startswith('im_hot_')]])}


def fit_base(name,inputs,y,indices,groups,seed):
    idx=indices[np.isfinite(y[indices])]
    if name.startswith('fire_'):
        x=inputs['hot'].iloc[idx];yy=y[idx];floor=float(yy.min())
        if name=='fire_direct':
            model=make_pipeline(SimpleImputer(keep_empty_features=True),
                QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=seed),Ridge(alpha=100.))
            model.fit(x,yy)
            return {'source':'hot','kind':'fire_direct','model':model,'floor':floor,'columns':list(x)}
        present=yy>floor+1e-6
        if not present.any():return {'source':'hot','kind':'constant','value':floor,'columns':list(x)}
        model=make_classifier('logistic_0.1');model.fit(x,present.astype(int))
        return {'source':'hot','kind':'fire_hurdle','model':model,'floor':floor,
                'positive_mean':float(yy[present].mean()),'columns':list(x)}
    source=name.split('_',1)[1]
    x=inputs[source].iloc[idx]
    if name=='ridge_embedding':
        params={'alpha':1000.,'embedding_normalize':True,'embedding_standardize':False,'embedding_multiplier':32.}
    elif name.startswith('ridge'):params={'alpha':100.}
    else:params=CAT
    model,iterations=fit_model('ridge' if name.startswith('ridge') else 'catboost',seed,params,
                               x,y[idx],groups[idx],early_stop=name.startswith('cat'))
    return {'source':source,'kind':'regression','model':model,'columns':list(x),'stopping_iterations':iterations}


def predict_base(bundle,inputs):
    x=inputs[bundle['source']][bundle['columns']]
    if bundle['kind']=='constant':return np.full(len(x),bundle['value'])
    if bundle['kind']=='fire_hurdle':
        classes=np.asarray(bundle['model'].classes_)
        if 1 not in classes:return np.full(len(x),bundle['floor'])
        p=bundle['model'].predict_proba(x)[:,int(np.flatnonzero(classes==1)[0])]
        return bundle['floor']+p*(bundle['positive_mean']-bundle['floor'])
    value=bundle['model'].predict(x)
    if bundle['kind']=='fire_direct':value=np.maximum(value,bundle['floor'])
    return np.asarray(value)


def calibration(y,p):
    keep=np.isfinite(y);yy=y[keep];pp=p[keep]
    k=pp.shape[1]
    result=minimize(lambda w:np.mean((yy-pp@w)**2)+.03*np.sum(w*w),np.full(k,.5/k),
                    method='SLSQP',bounds=[(0.,1.)]*k,
                    constraints=[{'type':'ineq','fun':lambda w:1-w.sum()}],
                    options={'ftol':1e-10,'maxiter':200})
    if not result.success:raise RuntimeError('Calibration failed: '+result.message)
    return result.x


def train(args):
    d=pd.read_csv(args.train,dtype={'sample_id':str,'site_key':str})
    folds=pd.read_csv(args.folds,dtype={'sample_id':str})
    if folds.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Fold alignment error')
    y=d[TARGETS].to_numpy(float);groups=d.site_key.to_numpy();outer=folds.fold.to_numpy()
    inputs=load_inputs(d,args)
    out=Path(args.output);out.mkdir(exist_ok=True)
    config={'menu':MENU,'catboost':CAT,'seed':args.seed,'calibration_penalty':.03,'inner_folds':3,
            'hashes':{key:hashlib.sha256(Path(getattr(args,key)).read_bytes()).hexdigest()
                      for key in ['train','folds','images','embeddings','hot']},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=config:raise ValueError('Resume configuration changed')
    else:(out/'config.json').write_text(json.dumps(config,indent=2))
    if (out/'oof.csv').exists():
        saved=pd.read_csv(out/'oof.csv',dtype={'sample_id':str})
        if saved.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Resume alignment error')
        oof=saved[TARGETS].to_numpy(float)
    else:oof=np.full_like(y,np.nan)
    started=time.monotonic()
    for f in sorted(np.unique(outer)):
        if (out/f'{f}.json').exists():continue
        tr=np.flatnonzero(outer!=f);va=np.flatnonzero(outer==f)
        if set(groups[tr])&set(groups[va]):raise ValueError('Outer group leakage')
        inner={t:np.full((len(tr),len(MENU[t])),np.nan) for t in TARGETS}
        for it,iv in GroupKFold(3).split(tr,groups=groups[tr]):
            fit,check=tr[it],tr[iv]
            if set(groups[fit])&set(groups[check]):raise ValueError('Inner group leakage')
            check_inputs={key:value.iloc[check] for key,value in inputs.items()}
            for j,target in enumerate(TARGETS):
                for k,name in enumerate(MENU[target]):
                    bundle=fit_base(name,inputs,y[:,j],fit,groups,args.seed)
                    inner[target][iv,k]=predict_base(bundle,check_inputs)
        fw={};va_inputs={key:value.iloc[va] for key,value in inputs.items()}
        for j,target in enumerate(TARGETS):
            fw[target]=calibration(y[tr,j],inner[target]).tolist()
            raw=np.zeros((len(va),len(MENU[target])))
            for k,name in enumerate(MENU[target]):
                bundle=fit_base(name,inputs,y[:,j],tr,groups,args.seed)
                raw[:,k]=predict_base(bundle,va_inputs)
                joblib.dump(bundle,out/f'{f}_{target}_{name}.joblib')
            oof[va,j]=raw@np.asarray(fw[target])
        pd.DataFrame(oof,columns=TARGETS).assign(sample_id=d.sample_id).to_csv(out/'oof.csv',index=False)
        info={'fold':int(f),'weights':fw,'score':metric(y[va],oof[va])}
        (out/f'{f}.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(info),flush=True)
    if not np.isfinite(oof).all():raise RuntimeError('Incomplete outer predictions')
    result={'score':metric(y,oof),'model':'target-specific nested image/tabular ensemble',
            'menu':MENU,'runtime_seconds_this_invocation':time.monotonic()-started,
            'method':'5 outer / 3 inner site folds; inner calibration and group-only stopping; zero reference receives remaining weight'}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def infer(args):
    run=Path(args.run)
    if not (run/'result.json').exists():raise ValueError('No completed training result')
    test=pd.read_csv(args.test,dtype={'sample_id':str})
    template=pd.read_csv(args.template,dtype={'sample_id':str})
    inputs=load_inputs(test,args);pred=np.zeros((len(test),5))
    for f in range(5):
        meta=json.loads((run/f'{f}.json').read_text())
        for j,target in enumerate(TARGETS):
            ws=np.asarray(meta['weights'][target])
            if (ws<-1e-6).any() or ws.sum()>1.000001:raise ValueError('Invalid saved weights')
            for k,name in enumerate(MENU[target]):
                bundle=joblib.load(run/f'{f}_{target}_{name}.joblib')
                pred[:,j]+=predict_base(bundle,inputs)*ws[k]/5
    frame=pd.DataFrame(pred,columns=TARGETS).assign(sample_id=test.sample_id)
    sub=template[['sample_id']].merge(frame,on='sample_id',how='left',validate='one_to_one')[template.columns]
    validate_submission(sub,test,template)
    sub.to_csv(args.output,index=False)
    print(json.dumps({'validated_rows':len(sub),'finite':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer'])
    for n in ['images','embeddings','hot','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--train',default='train.csv');p.add_argument('--folds',default='artifacts/folds.csv')
    p.add_argument('--seed',type=int,default=20261119);p.add_argument('--test',default='testnew.csv')
    p.add_argument('--template',default='data/sample_submission.csv');p.add_argument('--run')
    a=p.parse_args()
    train(a) if a.mode=='train' else infer(a)
