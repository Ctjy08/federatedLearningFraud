from pathlib import Path
import json,sys
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossbank.data.pipeline import load_features
from crossbank.models.mlp import FraudMLP,train_model,scores
from crossbank.models.metrics import evaluate

torch.manual_seed(42)
base=Path('data/processed')
train_path=base/'all_train_temp.csv'
frames=[pd.read_csv(base/f'bank_{c}.csv') for c in 'abc']
train=pd.concat(frames,ignore_index=True)
train.to_csv(train_path,index=False)
try:
    _,x,y=load_features(train_path)
finally:
    train_path.unlink(missing_ok=True)
_,xt,yt=load_features(base/'test.csv')
scaler=StandardScaler().fit(x)
xn=scaler.transform(x); xtn=scaler.transform(xt)
if len(set(y))<2: raise SystemExit('Training data must contain both classes.')
logreg=LogisticRegression(max_iter=500,class_weight='balanced').fit(xn,y)
mlp=train_model(FraudMLP(xn.shape[1]),xn,y,epochs=5)
results={'logistic_regression':evaluate(yt,logreg.predict_proba(xtn)[:,1]),'centralized_mlp':evaluate(yt,scores(mlp,xtn))}
Path('results').mkdir(exist_ok=True)
Path('results/baselines.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
