"""Small predeclared regularization search; saves all results on fixed folds."""
import json
from pathlib import Path
import subprocess
import sys

CANDIDATES=[
    ('cat_shallow_01',{'depth':2,'iterations':600,'l2_leaf_reg':30,'learning_rate':.03,'rsm':.8}),
    ('cat_shallow_02',{'depth':3,'iterations':800,'l2_leaf_reg':20,'learning_rate':.025,'rsm':.7}),
    ('cat_regularized_01',{'depth':4,'iterations':1000,'l2_leaf_reg':50,'learning_rate':.025,'rsm':.6}),
]
for name,params in CANDIDATES:
    if (Path('artifacts')/name/'result.json').exists():continue
    subprocess.run([sys.executable,'pipeline.py','--model','catboost','--run',name,
                    '--params',json.dumps(params),'--early-stop'],check=True)
