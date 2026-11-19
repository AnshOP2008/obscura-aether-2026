"""Bounded image-statistic ablations on the original frozen site folds."""
import json
from pathlib import Path
import subprocess
import sys

CANDIDATES=[
 ('image_cat_01','catboost','image',{'depth':3,'iterations':800,'l2_leaf_reg':30,'learning_rate':.025,'rsm':.7},True),
 ('combined_cat_01','catboost','combined',{'depth':3,'iterations':800,'l2_leaf_reg':30,'learning_rate':.025,'rsm':.7},True),
 ('combined_cat_02','catboost','combined',{'depth':4,'iterations':1000,'l2_leaf_reg':50,'learning_rate':.025,'rsm':.6},True),
 ('combined_ridge_01','ridge','combined',{'alpha':1000.},False),
]
if __name__=='__main__':
 for name,model,feature_set,params,stopping in CANDIDATES:
    if (Path('artifacts')/name/'result.json').exists():continue
    command=[sys.executable,'pipeline.py','--model',model,'--run',name,'--image-features','artifacts/image_features.csv',
             '--feature-set',feature_set,'--params',json.dumps(params)]
    if stopping:command.append('--early-stop')
    subprocess.run(command,check=True)
