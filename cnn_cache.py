"""Prepare small pixel tensors, retaining subpixel SWIR maxima separately."""
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


def main(a):
    ids=pd.read_csv(a.train,dtype={'sample_id':str}).sample_id.tolist()+pd.read_csv(a.test,dtype={'sample_id':str}).sample_id.tolist()
    output=Path(a.output)
    if output.exists() and Path(a.metadata).exists():
        m=json.loads(Path(a.metadata).read_text())
        if m.get('complete') and m['ids']==ids:
            print('Complete tensor cache already exists');return
    tensor=np.lib.format.open_memmap(output,mode='w+',dtype=np.float16,shape=(len(ids),16,96,96))
    local=threading.local();handles=[];lock=threading.Lock()
    warnings.filterwarnings('ignore',category=NotGeoreferencedWarning)
    with zipfile.ZipFile(a.archive) as z:prefixes=[n[:-9] for n in z.namelist() if n.endswith('train.csv')]
    if len(prefixes)!=1:raise ValueError('Ambiguous archive')
    def one(item):
        i,sid=item
        if not hasattr(local,'archive'):
            local.archive=zipfile.ZipFile(a.archive)
            with lock:handles.append(local.archive)
        with MemoryFile(local.archive.read(prefixes[0]+'images/'+sid+'.tif')) as mem:
            with mem.open() as ds:im=ds.read()
        if im.shape!=(14,384,384) or im.dtype!=np.uint16:raise ValueError('Invalid chip')
        reflectance=im[:12].astype(np.float32)*1e-4
        mean=reflectance.reshape(12,96,4,96,4).mean(axis=(2,4))
        maximum=reflectance[10:12].reshape(2,96,4,96,4).max(axis=(2,4))
        cloud=im[13].astype(np.float32).reshape(96,4,96,4).mean(axis=(1,3))/100.
        valid=((im[13]<40)&~np.isin(im[12],[0,1,3,8,9,10,11])).astype(np.float32).reshape(96,4,96,4).mean(axis=(1,3))
        result=np.concatenate([np.log1p(mean*5),np.log1p(maximum*5),cloud[None],valid[None]])
        if not np.isfinite(result).all():raise ValueError('Invalid tensor')
        return i,result
    with ThreadPoolExecutor(a.workers) as pool:
        for n,(i,result) in enumerate(pool.map(one,enumerate(ids)),1):
            tensor[i]=result
            if n%500==0:print(json.dumps({'completed':n,'total':len(ids)}),flush=True)
    tensor.flush()
    for h in handles:h.close()
    Path(a.metadata).write_text(json.dumps({'complete':True,'ids':ids,'shape':list(tensor.shape),
        'channels':'12 log-reflectance means; 2 log-SWIR maxima; cloud probability; clear fraction',
        'spatial_transform':'nonoverlapping 4x4 mean/max pooling, no geospatial resolution inference'},indent=2))
    print(json.dumps({'complete':True,'rows':len(ids),'channels':16,'size':96}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['archive','train','test','output','metadata']:p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=8)
    main(p.parse_args())
