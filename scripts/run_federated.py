"""In-process simulation (not Flower/networked FL)."""
from pathlib import Path
import sys,json
from copy import deepcopy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossbank.data.pipeline import load_features
from crossbank.models.mlp import FraudMLP,train_model,scores
from crossbank.models.metrics import evaluate
from crossbank.fl.fedavg import fedavg

torch.manual_seed(42)
base=Path('data/processed')
clients=[load_features(base/f'bank_{c}.csv') for c in 'abc']
_,xt,yt=load_features(base/'test.csv')
# Demo only: train preprocessing globally by merging bank statistics; this must later be distributed.
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(np.concatenate([x.to_numpy() for _,x,_ in clients]))
clients=[(scaler.transform(x.to_numpy()),y.to_numpy()) for _,x,y in clients]
xt=scaler.transform(xt.to_numpy())
global_model=FraudMLP(len(xt[0]))
results=[]
for round_num in range(1,11):
    states=[];sizes=[]
    for X,y in clients:
        local=deepcopy(global_model)
        train_model(local,X,y,epochs=1)
        states.append(local.state_dict());sizes.append(len(y))
    global_model.load_state_dict(fedavg(states,sizes))
    row={'round':round_num, **evaluate(yt,scores(global_model,xt))}
    results.append(row)
    print(f'round={round_num} f1={row["f1"]:.3f} pr_auc={row["pr_auc"]}')
Path('results').mkdir(exist_ok=True)
Path('results/federated_simulation.json').write_text(json.dumps(results,indent=2))
