import json
from pathlib import Path
import subprocess
import sys

for name,params in [
 ('embedding_norm_ridge_32',{'alpha':3000.,'embedding_pca':32,'embedding_normalize':True}),
 ('embedding_norm_ridge_full',{'alpha':10000.,'embedding_normalize':True})]:
 if not (Path('artifacts')/name/'result.json').exists():
    subprocess.run([sys.executable,'pipeline.py','--model','ridge','--run',name,
                    '--image-features','artifacts/embeddings.csv','--feature-set','combined',
                    '--params',json.dumps(params)],check=True)
