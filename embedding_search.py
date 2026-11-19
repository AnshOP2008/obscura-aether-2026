"""Frozen-embedding regressions with fold-local preprocessing/reduction."""
import json
from pathlib import Path
import subprocess
import sys

CANDIDATES=[
 ('embedding_ridge_32','ridge','image',{'alpha':1000.,'embedding_pca':32},False),
 ('embedding_table_ridge_32','ridge','combined',{'alpha':1000.,'embedding_pca':32},False),
 ('embedding_table_ridge_128','ridge','combined',{'alpha':1000.,'embedding_pca':128},False),
 ('embedding_table_cat_01','catboost','combined',{'depth':3,'iterations':500,'l2_leaf_reg':50,'learning_rate':.03,'rsm':.25},True),
]
if __name__=='__main__':
 for name,model,feature_set,params,stopping in CANDIDATES:
    if (Path('artifacts')/name/'result.json').exists():continue
    command=[sys.executable,'pipeline.py','--model',model,'--run',name,'--image-features','artifacts/embeddings.csv',
             '--feature-set',feature_set,'--params',json.dumps(params)]
    if stopping:command.append('--early-stop')
    subprocess.run(command,check=True)
