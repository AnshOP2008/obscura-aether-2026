"""Frozen SeCo-Eco embeddings, using semantically matched 12-band input.

Architecture/checkpoint provenance: TorchGeo ResNet50 SENTINEL2_ALL_SECO_ECO.
Weights: MIT, SSL4Eco (Plekhanova et al., CVPR EarthVision 2025).
No geographic coordinates, dates, metadata, or target labels are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize
import timm

URL='https://huggingface.co/torchgeo/seco-eco/resolve/3ede08c40f70534eb1132057ee338b99e54319f0/resnet50_sentinel2_all_seco_eco-a273eb78.pth'
SHA256='a273eb78ab8e99b1d51d9a8ddfa58ab455b00734a0532dfe506ece2a71e30f13'
BANDS=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']


class Chips(Dataset):
    def __init__(self, archive, ids):
        self.archive, self.ids, self.handle = archive, ids, None
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        if self.handle is None:
            self.handle=zipfile.ZipFile(self.archive)
            prefixes=[n[:-len('train.csv')] for n in self.handle.namelist() if n.endswith('train.csv')]
            if len(prefixes)!=1:raise ValueError('Ambiguous archive layout')
            self.prefix=prefixes[0]
        with MemoryFile(self.handle.read(self.prefix+'images/'+self.ids[index]+'.tif')) as mem:
            with mem.open() as ds:a=ds.read()
        if a.shape!=(14,384,384) or a.dtype!=np.uint16:raise ValueError('Invalid chip')
        # Published preprocessing: resize to 224 and divide raw reflectance by 10000.
        # SCL and cloud probability are excluded from the encoder and retained in
        # separately computed pixel summaries.
        image=resize(torch.from_numpy(a[:12].astype(np.float32)),[224,224],antialias=True)/10000.
        return image,index


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261119)
    warnings.filterwarnings('ignore',category=NotGeoreferencedWarning)
    checkpoint=Path(args.checkpoint)
    if not checkpoint.exists():urllib.request.urlretrieve(URL,checkpoint)
    raw=checkpoint.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA256:raise ValueError('Checkpoint integrity mismatch')
    model=timm.create_model('resnet50',pretrained=False,in_chans=12,num_classes=0)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model.to(device)
    train=pd.read_csv(args.train,dtype={'sample_id':str})
    test=pd.read_csv(args.test,dtype={'sample_id':str})
    ids=train.sample_id.tolist()+test.sample_id.tolist()
    output=Path(args.output)
    if output.exists():
        saved=np.load(output,allow_pickle=False)
        if saved['ids'].tolist()!=ids:raise ValueError('Cache alignment mismatch')
        emb,done=saved['embedding'],saved['done']
    else:
        emb=np.zeros((len(ids),2048),dtype=np.float32)
        done=np.zeros(len(ids),dtype=bool)
    remaining=np.flatnonzero(~done)
    loader=DataLoader(Chips(args.archive,[ids[i] for i in remaining]),batch_size=args.batch_size,
                      num_workers=args.workers,shuffle=False,pin_memory=device.type=='cuda')
    with torch.inference_mode():
        for number,(image,idx) in enumerate(loader):
            value=model(image.to(device,non_blocking=True)).float().cpu().numpy()
            if value.shape[1]!=2048 or not np.isfinite(value).all():raise ValueError('Invalid embedding')
            rows=remaining[idx.numpy()]
            emb[rows]=value;done[rows]=True
            if number%16==0:
                np.savez_compressed(output,ids=np.asarray(ids),embedding=emb,done=done)
                print(json.dumps({'completed':int(done.sum()),'total':len(ids)}),flush=True)
    if not done.all():raise RuntimeError('Incomplete embedding cache')
    np.savez_compressed(output,ids=np.asarray(ids),embedding=emb,done=done)
    table=pd.DataFrame(emb,columns=[f'im_emb_{i:04d}' for i in range(2048)])
    table.insert(0,'sample_id',ids)
    table.to_csv(args.csv,index=False)
    meta={'checkpoint_url':URL,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
          'checkpoint_md5':hashlib.md5(raw).hexdigest(),'license':'MIT','bands':BANDS,
          'preprocessing':'bilinear antialiased resize 224x224; raw uint16 reflectance divided by 10000',
          'rows':len(ids),'features':2048,'frozen':True,'torch':torch.__version__,'timm':timm.__version__}
    Path(args.metadata).write_text(json.dumps(meta,indent=2))
    print(json.dumps({'complete':True,'rows':len(ids),'features':2048}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['archive','train','test','checkpoint','output','csv','metadata']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    main(p.parse_args())
