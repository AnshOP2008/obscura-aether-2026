"""Correct a diagnosed rare-coordinate scaling failure; bounded follow-up."""
import json
from pathlib import Path
import subprocess
import sys

for name,extra in [
 ('embedding_unit_ridge',{'embedding_standardize':False,'embedding_multiplier':32.}),
 ('embedding_unit_pca32',{'embedding_standardize':False,'embedding_multiplier':32.,'embedding_pca':32}),
 ('embedding_clip_pca32',{'embedding_clip':5.,'embedding_pca':32})]:
 if not (Path('artifacts')/name/'result.json').exists():
    params={'alpha':1000.,'embedding_normalize':True,**extra}
    subprocess.run([sys.executable,'pipeline.py','--model','ridge','--run',name,
                    '--image-features','artifacts/embeddings.csv','--feature-set','combined',
                    '--params',json.dumps(params)],check=True)
