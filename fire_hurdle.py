"""Site-held-out fire presence/intensity models on pixel-derived SWIR tails."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from catboost import CatBoostClassifier
from pipeline import TARGETS, metric


def signed_log(x):
    # Fixed transformations, no target-dependent selection or fitted constants.
    result=x.copy()
    for c in result:
        factor=147456. if 'fraction' in c else 10.
        result[c]=np.sign(result[c])*np.log1p(np.abs(result[c])*factor)
    return result


def make_classifier(name):
    if name.startswith('logistic'):
        strength=float(name.split('_')[1])
        return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),
            FunctionTransformer(np.clip,kw_args={'a_min':-5.,'a_max':5.}),
            LogisticRegression(C=strength,max_iter=2000))
    if name=='histogram':
        return HistGradientBoostingClassifier(max_iter=250,max_depth=2,min_samples_leaf=40,
                                              l2_regularization=10,early_stopping=False,random_state=20261119)
    if name=='catboost':
        return CatBoostClassifier(iterations=500,depth=2,learning_rate=.03,l2_leaf_reg=30,
                                  loss_function='Logloss',thread_count=8,verbose=False,
                                  allow_writing_files=False,random_seed=20261119)
    raise ValueError(name)


def run(fold_path,output_prefix,names,reference_run='nested_blend_01'):
    d=pd.read_csv('train.csv',dtype={'sample_id':str,'site_key':str})
    image=pd.read_csv('artifacts/hot_features.csv',dtype={'sample_id':str})
    d=d.merge(image,on='sample_id',validate='one_to_one',sort=False)
    columns=[c for c in d if c.startswith('im_hot_')]
    x=signed_log(d[columns])
    folds=pd.read_csv(fold_path,dtype={'sample_id':str})
    if folds.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Fold alignment mismatch')
    y=d.t_firms.to_numpy()
    reference=pd.read_csv('artifacts/'+reference_run+'/oof.csv',dtype={'sample_id':str})
    if reference.sample_id.tolist()!=d.sample_id.tolist():raise ValueError('Reference alignment mismatch')
    for name in names:
        out=Path('artifacts')/(output_prefix+name)
        if (out/'result.json').exists():continue
        out.mkdir(exist_ok=False)
        oof=np.zeros(len(d));details=[]
        for f in sorted(folds.fold.unique()):
            tr=np.flatnonzero(folds.fold.to_numpy()!=f);va=np.flatnonzero(folds.fold.to_numpy()==f)
            if set(d.site_key.iloc[tr])&set(d.site_key.iloc[va]):raise ValueError('Site leakage')
            floor=float(y[tr].min())
            present=y[tr]>floor+1e-6
            positive_mean=float(y[tr][present].mean())
            model=make_classifier(name)
            model.fit(x.iloc[tr],present.astype(int))
            probability=model.predict_proba(x.iloc[va])[:,1]
            oof[va]=floor+probability*(positive_mean-floor)
            joblib.dump({'model':model,'floor':floor,'positive_mean':positive_mean,'columns':columns},out/f'{f}.joblib')
            details.append({'fold':int(f),'floor':floor,'positive_mean':positive_mean,
                            'positive_training_rows':int(present.sum()),'rmse':float(np.sqrt(np.mean((y[va]-oof[va])**2)))})
        prediction=reference[TARGETS].copy();prediction['t_firms']=oof
        result={'model':'fire_hurdle_'+name,'features':columns,'score':metric(d[TARGETS],prediction),
                'fire_rmse':float(np.sqrt(np.mean((y-oof)**2))), 'folds':details,
                'method':'Observed-training minimum as no-fire floor; classifier probability times training positive mean enhancement',
                'limitation':'Overall score uses unchanged primary nested OOF predictions for other targets; fire score is the relevant comparison',
                'fold_file':str(fold_path)}
        prediction.assign(sample_id=d.sample_id).to_csv(out/'oof.csv',index=False)
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'run':out.name,'fire_rmse':result['fire_rmse'],'overall':result['score']['overall']}),flush=True)


if __name__=='__main__':
    run('artifacts/folds.csv','fire_hurdle_', ['logistic_0.01','logistic_0.1','logistic_1','histogram','catboost'])
