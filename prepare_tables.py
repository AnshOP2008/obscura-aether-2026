"""Extract only the official tables; leave image chips inside the archive."""
import argparse
from pathlib import Path
import zipfile

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',default='data/obscura.zip');a=p.parse_args()
    with zipfile.ZipFile(a.archive) as z:
        prefixes=[n[:-9] for n in z.namelist() if n.endswith('train.csv')]
        if len(prefixes)!=1:raise ValueError('Ambiguous archive layout')
        for source,target in [('train.csv','train.csv'),('test.csv','testnew.csv'),
                              ('sample_submission.csv','data/sample_submission.csv'),('README.md','data/README.md')]:
            path=Path(target);path.parent.mkdir(parents=True,exist_ok=True)
            content=z.read(prefixes[0]+source)
            if path.exists() and path.read_bytes()!=content:raise ValueError('Refusing to overwrite a different existing table')
            path.write_bytes(content)
    print('Official tables prepared; testnew.csv is an unchanged alias of test.csv')
