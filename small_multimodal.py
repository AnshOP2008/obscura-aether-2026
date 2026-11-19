"""Compact spatial/tabular multitask regression with nested stopping by site."""
import argparse
import json
from pathlib import Path
import random
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from pipeline import TARGETS,features,metric


class SmallModel(nn.Module):
    def __init__(self,tab_dim):
        super().__init__()
        layers=[]
        for cin,cout in [(16,16),(16,32),(32,64)]:
            layers.extend([nn.Conv2d(cin,cout,3,stride=2,padding=1),nn.GroupNorm(4,cout),nn.SiLU()])
        self.image=nn.Sequential(*layers)
        self.table=nn.Sequential(nn.Linear(tab_dim,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(160,64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,5))
    def forward(self,image,table):
        z=self.image(image)
        return self.head(torch.cat([z.mean(dim=(2,3)),z.amax(dim=(2,3)),self.table(table)],dim=1))


def loss(pred,truth):
    parts=[]
    for j in range(5):
        keep=torch.isfinite(truth[:,j])
        if keep.any():parts.append(torch.mean((pred[keep,j]-truth[keep,j])**2))
    return torch.stack(parts).mean()


def setup(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def fit_stage(images,table,y,train,check,epochs,device,seed,stopping):
    setup(seed)
    transform=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler())
    transform.fit(table.iloc[train])
    tab=transform.transform(table).astype(np.float32)
    # Accumulate only training-image channel moments, without materializing a giant float copy.
    sums=np.zeros(16);squares=np.zeros(16);count=0
    for batch in np.array_split(train,max(1,len(train)//64)):
        values=np.asarray(images[batch],dtype=np.float32)
        sums+=values.sum(axis=(0,2,3),dtype=np.float64)
        squares+=(values.astype(np.float64)**2).sum(axis=(0,2,3))
        count+=len(batch)*96*96
    mean=(sums/count).astype(np.float32)
    std=np.sqrt(np.maximum(squares/count-mean.astype(np.float64)**2,.0025)).astype(np.float32)
    model=SmallModel(tab.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    y_tensor=torch.from_numpy(y.astype(np.float32)).to(device)
    mean_t=torch.tensor(mean,device=device)[None,:,None,None]
    std_t=torch.tensor(std,device=device)[None,:,None,None]
    best=float('inf');best_epoch=1;stale=0
    def predict(indices):
        result=[];model.eval()
        with torch.inference_mode():
            for block in np.array_split(indices,max(1,int(np.ceil(len(indices)/64)))):
                im=torch.from_numpy(np.asarray(images[block],dtype=np.float32)).to(device)
                xx=torch.from_numpy(tab[block]).to(device)
                result.append(model((im-mean_t)/std_t,xx).cpu().numpy())
        return np.concatenate(result)
    for epoch in range(1,epochs+1):
        model.train()
        order=np.random.default_rng(seed+epoch).permutation(train)
        for block in np.array_split(order,max(1,int(np.ceil(len(order)/32)))):
            im=torch.from_numpy(np.asarray(images[block],dtype=np.float32)).to(device)
            xx=torch.from_numpy(tab[block]).to(device)
            pred=model((im-mean_t)/std_t,xx)
            objective=loss(pred,y_tensor[block])
            optimizer.zero_grad(set_to_none=True);objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if stopping:
            score=metric(y[check],predict(check))['overall']
            if score<best-1e-4:best=score;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=6:break
    return model,transform,mean,std,(best_epoch if stopping else epochs),predict


def main(a):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda')
    d=pd.read_csv(a.train,dtype={'sample_id':str,'site_key':str})
    folds=pd.read_csv(a.folds,dtype={'sample_id':str})
    if d.sample_id.tolist()!=folds.sample_id.tolist():raise ValueError('Fold alignment')
    meta=json.loads(Path(a.tensor_metadata).read_text())
    if not meta['complete'] or meta['ids'][:len(d)]!=d.sample_id.tolist():raise ValueError('Tensor alignment')
    images=np.load(a.tensor,mmap_mode='r')
    x=features(d);y=d[TARGETS].to_numpy(float);groups=d.site_key.to_numpy()
    out=Path(a.output);out.mkdir(exist_ok=True)
    if (out/'oof.csv').exists():
        saved=pd.read_csv(out/'oof.csv',dtype={'sample_id':str})
        if saved.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Resume alignment')
        oof=saved[TARGETS].to_numpy(float)
    else:oof=np.full_like(y,np.nan)
    details=[]
    for f in map(int,a.run_folds.split(',')):
        if (out/f'{f}.json').exists():continue
        tr=np.flatnonzero(folds.fold.to_numpy()!=f);va=np.flatnonzero(folds.fold.to_numpy()==f)
        if set(groups[tr])&set(groups[va]):raise ValueError('Outer site leakage')
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261119+f).split(x.iloc[tr],groups=groups[tr]))
        _,_,_,_,best_epochs,_=fit_stage(images,x,y,tr[it],tr[iv],a.epochs,device,20261119+f,True)
        torch.cuda.empty_cache()
        model,transform,mean,std,_,predict=fit_stage(images,x,y,tr,va,best_epochs,device,20261119+f,False)
        oof[va]=predict(va)
        torch.save(model.state_dict(),out/f'{f}.pt')
        joblib.dump({'transform':transform,'channel_mean':mean,'channel_std':std,'tab_dim':model.table[0].in_features,
                     'features':list(x)},out/f'{f}.joblib')
        info={'fold':f,'epochs':best_epochs,'score':metric(y[va],oof[va])}
        (out/f'{f}.json').write_text(json.dumps(info,indent=2))
        pd.DataFrame(oof,columns=TARGETS).assign(sample_id=d.sample_id).to_csv(out/'oof.csv',index=False)
        print(json.dumps(info),flush=True)
        del model;torch.cuda.empty_cache()
    done=np.isfinite(oof).all(axis=1)
    result={'model':'compact_16channel_cnn_tabular','completed_rows':int(done.sum()),'total_rows':len(d),
            'score_on_completed_outer_rows':metric(y[done],oof[done]),'complete':bool(done.all()),
            'early_stopping':'inner 20% site split; refit outer train for selected epochs',
            'epochs_limit':a.epochs,'feature_columns':list(x),'torch_version':torch.__version__}
    if done.all():result['score']=result['score_on_completed_outer_rows']
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='feature_columns'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['train','folds','tensor','tensor-metadata','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--run-folds',default='0,1');p.add_argument('--epochs',type=int,default=30)
    main(p.parse_args())
