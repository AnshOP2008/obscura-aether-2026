"""Bounded site-held-out evaluation of public synthetic-pretrained TabICLv2."""
import argparse,gc,hashlib,json,time,urllib.request
from pathlib import Path
import numpy as np,pandas as pd,torch
from tabicl import TabICLRegressor
from importlib.metadata import version
from pipeline import TARGETS,features,metric,validate_submission

REVISION='4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
SHA256='0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'
URL=f'https://huggingface.co/jingang/TabICL/resolve/{REVISION}/tabicl-regressor-v2-20260212.ckpt'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['train','test','template','folds','checkpoint','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True)
    checkpoint=Path(a.checkpoint)
    if not checkpoint.exists():urllib.request.urlretrieve(URL,checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SHA256:raise ValueError('Checkpoint checksum')
    d=pd.read_csv(a.train,dtype={'sample_id':str,'site_key':str});test=pd.read_csv(a.test,dtype={'sample_id':str})
    template=pd.read_csv(a.template,dtype={'sample_id':str});folds=pd.read_csv(a.folds,dtype={'sample_id':str})
    if folds.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Fold alignment')
    x=features(d);xt=features(test);y=d[TARGETS].to_numpy(float);oof=np.full_like(y,np.nan);pred=np.zeros((len(test),5))
    started=time.monotonic();torch.set_num_threads(4)
    for f in range(5):
        tr=np.flatnonzero(folds.fold.to_numpy()!=f);va=np.flatnonzero(folds.fold.to_numpy()==f)
        if set(d.site_key.iloc[tr])&set(d.site_key.iloc[va]):raise ValueError('Group leakage')
        for j,target in enumerate(TARGETS):
            path=out/f'{f}_{target}.npz'
            if path.exists():
                saved=np.load(path);oof[va,j]=saved['valid'];pred[:,j]+=saved['test']/5;continue
            obs=tr[np.isfinite(y[tr,j])]
            model=TabICLRegressor(n_estimators=8,batch_size=1,model_path=str(checkpoint),allow_auto_download=False,device='cuda',use_fa3=False,offload_mode='cpu',random_state=20261119,n_jobs=4)
            model.fit(x.iloc[obs],y[obs,j])
            valid=model.predict(x.iloc[va]);tp=model.predict(xt)
            if not np.isfinite(valid).all() or not np.isfinite(tp).all():raise ValueError('Nonfinite prediction')
            np.savez_compressed(path,valid=valid,test=tp);oof[va,j]=valid;pred[:,j]+=tp/5
            print(json.dumps({'fold':f,'target':target,'rmse':float(np.sqrt(np.nanmean((y[va,j]-valid)**2)))}),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    result={'model':'TabICLv2 regression; table only','score':metric(y,oof),'runtime_seconds':time.monotonic()-started,
            'params':{'n_estimators':8,'batch_size':1,'seed':20261119},'tabicl_version':version('tabicl'),
            'checkpoint_revision':REVISION,'checkpoint_sha256':SHA256,'checkpoint_license':'BSD-3-Clause',
            'provenance':'Synthetic pretraining, no corresponding sample data; no supervised model/context serialized publicly.'}
    pd.DataFrame(oof,columns=TARGETS).assign(sample_id=d.sample_id).to_csv(out/'oof.csv',index=False)
    sub=pd.DataFrame(pred,columns=TARGETS).assign(sample_id=test.sample_id)[template.columns]
    validate_submission(sub,test,template);sub.to_csv(out/'submission.csv',index=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
