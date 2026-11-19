"""Inference from saved compact CNN folds and a deterministic tensor cache."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from small_multimodal import SmallModel
from pipeline import TARGETS,features,validate_submission

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['run','test','template','tensor','tensor-metadata','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();run=Path(a.run)
    result=json.loads((run/'result.json').read_text())
    if not result.get('complete'):raise ValueError('CNN folds are incomplete')
    test=pd.read_csv(a.test,dtype={'sample_id':str});template=pd.read_csv(a.template,dtype={'sample_id':str})
    metadata=json.loads(Path(a.tensor_metadata).read_text());lookup={sid:i for i,sid in enumerate(metadata['ids'])}
    if not metadata['complete'] or any(sid not in lookup for sid in test.sample_id):raise ValueError('Tensor coverage')
    positions=np.asarray([lookup[sid] for sid in test.sample_id]);images=np.load(a.tensor,mmap_mode='r')
    x=features(test);prediction=np.zeros((len(test),5));torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode():
        for fold in range(5):
            config=joblib.load(run/f'{fold}.joblib')
            model=SmallModel(config['tab_dim']).to(device)
            model.load_state_dict(torch.load(run/f'{fold}.pt',map_location=device,weights_only=True),strict=True);model.eval()
            tab=config['transform'].transform(x[config['features']]).astype(np.float32)
            mean=torch.tensor(config['channel_mean'],device=device)[None,:,None,None]
            std=torch.tensor(config['channel_std'],device=device)[None,:,None,None]
            for ids in np.array_split(np.arange(len(test)),max(1,int(np.ceil(len(test)/64)))):
                im=torch.from_numpy(np.asarray(images[positions[ids]],dtype=np.float32)).to(device)
                xx=torch.from_numpy(tab[ids]).to(device)
                prediction[ids]+=model((im-mean)/std,xx).cpu().numpy()/5
            del model
    pred=pd.DataFrame(prediction,columns=TARGETS).assign(sample_id=test.sample_id)
    sub=template[['sample_id']].merge(pred,on='sample_id',how='left',validate='one_to_one')[template.columns]
    validate_submission(sub,test,template);sub.to_csv(a.output,index=False)
    print(json.dumps({'validated_rows':len(sub),'device':device.type,'finite':True}),flush=True)
