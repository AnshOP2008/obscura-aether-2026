"""Supplementary SWIR-tail summaries retaining potentially saturated hot pixels.

These are pixel-content features. Cloudy/saturated candidates remain separate
from clear-pixel summaries so their utility can be evaluated by ablation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import warnings
import zipfile
import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning


def summary(a):
    if a.shape!=(14,384,384) or a.dtype!=np.uint16:raise ValueError('Invalid chip')
    # SCL class 1 can mark saturation: retain it here as a separate experiment.
    mask=(a[12]!=0)&np.any(a[:12]>0,axis=0)
    b11,b12=a[10].astype(np.float32)*1e-4,a[11].astype(np.float32)*1e-4
    out={'im_hot_saturated_fraction':float((a[12]==1).mean())}
    for j,b in [(11,b11),(12,b12)]:
        v=b[mask]
        for count in [1,8,64,512]:
            k=min(count,len(v))
            out[f'im_hot_B{j}_top{count}']=float(np.partition(v,len(v)-k)[-k:].mean()) if k else np.nan
        for threshold in [.4,.8,1.2]:
            out[f'im_hot_B{j}_above{threshold}']=float((v>threshold).mean()) if len(v) else np.nan
    excess=np.maximum(b12-b11,0)
    ratio=(b12-b11)/(b12+b11+1e-4)
    for label,values in [('excess',excess),('contrast',ratio)]:
        v=values[mask&(b11>.05)]
        for q in [.99,.999,1.]:
            out[f'im_hot_{label}_q{q}']=float(np.quantile(v,q)) if len(v) else np.nan
    hot=mask&(b12>.4)&(b12>b11*1.5)
    out['im_hot_joint_fraction']=float(hot.mean())
    out['im_hot_center_fraction']=float(hot[96:288,96:288].mean())
    return out


def main(a):
    ids=pd.read_csv(a.train,dtype={'sample_id':str}).sample_id.tolist()+pd.read_csv(a.test,dtype={'sample_id':str}).sample_id.tolist()
    output=Path(a.output)
    old=pd.read_csv(output,dtype={'sample_id':str}) if output.exists() else pd.DataFrame()
    rows=old.to_dict('records');present=set(old.sample_id) if len(old) else set()
    local=threading.local();handles=[];lock=threading.Lock()
    warnings.filterwarnings('ignore',category=NotGeoreferencedWarning)
    with zipfile.ZipFile(a.archive) as z:
        prefixes=[n[:-9] for n in z.namelist() if n.endswith('train.csv')]
    if len(prefixes)!=1:raise ValueError('Ambiguous archive')
    def one(sid):
        if not hasattr(local,'archive'):
            local.archive=zipfile.ZipFile(a.archive)
            with lock:handles.append(local.archive)
        with MemoryFile(local.archive.read(prefixes[0]+'images/'+sid+'.tif')) as mem:
            with mem.open() as ds:image=ds.read()
        return {'sample_id':sid,**summary(image)}
    with ThreadPoolExecutor(a.workers) as pool:
        for i,row in enumerate(pool.map(one,[sid for sid in ids if sid not in present]),1):
            rows.append(row)
            if i%500==0:
                pd.DataFrame(rows).to_csv(output,index=False)
                print(json.dumps({'completed':len(rows),'total':len(ids)}),flush=True)
    for h in handles:h.close()
    pd.DataFrame(rows).set_index('sample_id').loc[ids].reset_index().to_csv(output,index=False)
    print(json.dumps({'complete':True,'rows':len(rows),'features':len(rows[0])-1}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['archive','train','test','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=8)
    main(p.parse_args())
