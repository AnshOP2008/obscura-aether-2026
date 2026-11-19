"""Run the documented final pipeline with authorized local competition inputs."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pipeline import features

def run(script,*args):
    subprocess.run([sys.executable,script,*map(str,args)],check=True)

def folds():
    d=pd.read_csv('train.csv',dtype={'sample_id':str,'site_key':str})
    for seed,name in [(20261119,'folds.csv'),(20261120,'folds_secondary.csv')]:
        unique=d.site_key.drop_duplicates().to_numpy(copy=True)
        np.random.default_rng(seed).shuffle(unique)
        codes=d.site_key.map({g:i for i,g in enumerate(unique)})
        assignment=np.full(len(d),-1)
        for f,(_,va) in enumerate(GroupKFold(5).split(features(d),groups=codes)):assignment[va]=f
        frame=pd.DataFrame({'sample_id':d.sample_id,'fold':assignment});path=Path('artifacts')/name
        if path.exists():
            saved=pd.read_csv(path,dtype={'sample_id':str})
            if not frame.equals(saved):raise ValueError('Existing folds differ from documented seed')
        else:frame.to_csv(path,index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','train','infer'])
    p.add_argument('--archive',default='data/obscura.zip');p.add_argument('--checkpoint',default='artifacts/encoder.pth')
    a=p.parse_args();Path('artifacts').mkdir(exist_ok=True)
    shared=['--train','train.csv','--test','testnew.csv']
    image_args=['--images','artifacts/image_hot_features.csv','--embeddings','artifacts/embeddings.csv','--hot','artifacts/hot_features.csv']
    tensor_args=['--tensor','artifacts/tensor.npy','--tensor-metadata','artifacts/tensor.json']
    if a.stage=='features':
        run('prepare_tables.py','--archive',a.archive)
        run('image_features.py','--archive',a.archive,*shared,'--output','artifacts/image_features.csv')
        run('hot_features.py','--archive',a.archive,*shared,'--output','artifacts/hot_features.csv')
        first=pd.read_csv('artifacts/image_features.csv',dtype={'sample_id':str})
        second=pd.read_csv('artifacts/hot_features.csv',dtype={'sample_id':str})
        first.merge(second,on='sample_id',validate='one_to_one',sort=False).to_csv('artifacts/image_hot_features.csv',index=False)
        run('embeddings.py','--archive',a.archive,*shared,'--checkpoint',a.checkpoint,'--output','artifacts/embeddings.npz','--csv','artifacts/embeddings.csv','--metadata','artifacts/encoder_metadata.json')
        run('cnn_cache.py','--archive',a.archive,*shared,'--output','artifacts/tensor.npy','--metadata','artifacts/tensor.json')
        folds()
    elif a.stage=='train':
        folds()
        for label,fold in [('primary','folds.csv'),('secondary','folds_secondary.csv')]:
            run('final_ensemble.py','train',*image_args,'--folds','artifacts/'+fold,'--output','artifacts/final_'+label)
            run('small_multimodal.py','--train','train.csv','--folds','artifacts/'+fold,*tensor_args,'--output','artifacts/cnn_'+label,'--run-folds','0,1,2,3,4','--epochs','30')
        run('hybrid.py','--evaluate')
    else:
        for label in ['primary','secondary']:
            run('final_ensemble.py','infer',*image_args,'--run','artifacts/final_'+label,'--output','artifacts/final_'+label+'/submission.csv')
            run('infer_cnn.py','--run','artifacts/cnn_'+label,'--test','testnew.csv','--template','data/sample_submission.csv',*tensor_args,'--output','artifacts/cnn_'+label+'/submission.csv')
        run('hybrid.py')
