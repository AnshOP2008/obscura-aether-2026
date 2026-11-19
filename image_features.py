"""Deterministic pixel-only summaries from the official archive.

Reads TIFF arrays from ZIP memory. Does not inspect geolocation, timestamps,
file sizes, naming patterns, or other metadata as predictive features.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import warnings
import zipfile
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning

BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']


def summary(a):
    if a.shape != (14,384,384) or a.dtype != np.uint16:
        raise ValueError('Unexpected chip shape or dtype')
    scl, cloud = a[12], a[13]
    has_data = ~np.isin(scl, [0,1]) & np.any(a[:12] > 0, axis=0)
    valid = has_data & ~np.isin(scl,[3,8,9,10,11]) & (cloud < 40)
    out = {'im_clear_fraction': float(valid.mean()), 'im_data_fraction': float(has_data.mean()),
           'im_cloud_mean': float(cloud.mean())}
    for cls in range(2,12):
        out[f'im_scl_{cls}'] = float((scl == cls).mean())
    for j, band in enumerate(BANDS):
        v = a[j][valid].astype(np.float32) * 1e-4
        stats = [float(v.mean()), float(v.std()), *np.quantile(v,[.1,.5,.9,.99]).tolist()] if len(v) else [np.nan]*6
        for name, value in zip(['mean','std','q10','q50','q90','q99'],stats):
            out[f'im_{band}_{name}'] = value
        for scale, sl in [('center',slice(144,240)),('middle',slice(96,288))]:
            center = a[j,sl,sl][valid[sl,sl]].astype(np.float32) * 1e-4
            out[f'im_{band}_{scale}_contrast'] = float(center.mean()-v.mean()) if len(center) and len(v) else np.nan
    for j in [10,11]:
        v = a[j][valid].astype(np.float32)*1e-4
        out[f'im_{BANDS[j]}_max'] = float(v.max()) if len(v) else np.nan
        out[f'im_{BANDS[j]}_q999'] = float(np.quantile(v,.999)) if len(v) else np.nan
        for threshold in [.2,.4,.8]:
            out[f'im_{BANDS[j]}_above_{threshold}'] = float((v>threshold).mean()) if len(v) else np.nan
    # No absolute ground-size assumption; these are pixel-space local contrasts.
    for j in [3,7,10,11]:
        b=a[j].astype(np.float32)*1e-4
        hm=valid[:,1:] & valid[:,:-1]
        vm=valid[1:,:] & valid[:-1,:]
        edges=np.concatenate([np.abs(np.diff(b,axis=1))[hm],np.abs(np.diff(b,axis=0))[vm]])
        out[f'im_{BANDS[j]}_edge_mean'] = float(edges.mean()) if len(edges) else np.nan
    return out


def main(args):
    train=pd.read_csv(args.train,dtype={'sample_id':str,'site_key':str})
    test=pd.read_csv(args.test,dtype={'sample_id':str})
    ids=train.sample_id.tolist()+test.sample_id.tolist()
    output=Path(args.output)
    done=pd.read_csv(output,dtype={'sample_id':str}) if output.exists() else pd.DataFrame()
    present=set(done.sample_id) if len(done) else set()
    if not present.issubset(ids): raise ValueError('Cache contains unexpected IDs')
    rows=done.to_dict('records')
    started=time.monotonic()
    warnings.filterwarnings('ignore',category=NotGeoreferencedWarning)
    with zipfile.ZipFile(args.archive) as archive:
        names=archive.namelist()
        candidates=[n[:-len('train.csv')] for n in names if n.endswith('train.csv')]
        if len(candidates)!=1: raise ValueError('Ambiguous archive layout')
        prefix=candidates[0]
        def one(sid):
            with MemoryFile(archive.read(prefix+'images/'+sid+'.tif')) as mem:
                with mem.open() as ds:
                    a=ds.read()
            return {'sample_id':sid,**summary(a)}
        pending=[sid for sid in ids if sid not in present]
        with ThreadPoolExecutor(args.workers) as pool:
            for i,row in enumerate(pool.map(one,pending),1):
                rows.append(row)
                if i%100==0:
                    pd.DataFrame(rows).to_csv(output,index=False)
                    print(json.dumps({'completed':len(rows),'total':len(ids),'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
    frame=pd.DataFrame(rows).set_index('sample_id').loc[ids].reset_index()
    frame.to_csv(output,index=False)
    print(json.dumps({'complete':True,'rows':len(frame),'predictors':len(frame.columns)-1}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['archive','train','test','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=8)
    main(p.parse_args())
