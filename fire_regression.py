"""Direct fire regressions as a controlled comparison with the hurdle model."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from fire_hurdle import signed_log
from pipeline import TARGETS,metric


def main():
    d=pd.read_csv('train.csv',dtype={'sample_id':str,'site_key':str})
    d=d.merge(pd.read_csv('artifacts/hot_features.csv',dtype={'sample_id':str}),on='sample_id',validate='one_to_one')
    cols=[c for c in d if c.startswith('im_hot_')];x=signed_log(d[cols]);y=d.t_firms.to_numpy()
    folds=pd.read_csv('artifacts/folds.csv',dtype={'sample_id':str})
    if folds.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Alignment error')
    reference=pd.read_csv('artifacts/nested_blend_01/oof.csv',dtype={'sample_id':str})
    for name in ['ridge_10','ridge_100','ridge_1000','extra_trees']:
        out=Path('artifacts')/('fire_direct_'+name)
        if (out/'result.json').exists():continue
        out.mkdir(exist_ok=False);oof=np.zeros(len(d))
        for f in range(5):
            tr=np.flatnonzero(folds.fold.to_numpy()!=f);va=np.flatnonzero(folds.fold.to_numpy()==f)
            if set(d.site_key.iloc[tr])&set(d.site_key.iloc[va]):raise ValueError('Site leakage')
            if name.startswith('ridge'):
                model=make_pipeline(SimpleImputer(keep_empty_features=True),
                  QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=20261119),
                  Ridge(alpha=float(name.split('_')[1])))
            else:model=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=20,max_features=.8,n_jobs=4,random_state=20261119)
            model.fit(x.iloc[tr],y[tr]);floor=float(y[tr].min())
            oof[va]=np.maximum(model.predict(x.iloc[va]),floor)
            joblib.dump({'model':model,'floor':floor,'columns':cols},out/f'{f}.joblib')
        pred=reference[TARGETS].copy();pred['t_firms']=oof
        result={'model':'fire_direct_'+name,'score':metric(d[TARGETS],pred),
                'fire_rmse':float(np.sqrt(np.mean((y-oof)**2))),
                'method':'Fold-local quantile ridge or ExtraTrees on log SWIR tails; prediction floor from training no-detection value'}
        pred.assign(sample_id=d.sample_id).to_csv(out/'oof.csv',index=False)
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'run':out.name,'fire_rmse':result['fire_rmse'],'overall':result['score']['overall']}),flush=True)


if __name__=='__main__':main()
