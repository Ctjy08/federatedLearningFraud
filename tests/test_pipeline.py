import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from crossbank.data.pipeline import partition_non_iid,chrono_split

def test_disjoint_bank_splits():
    df=pd.DataFrame({'amount':range(200),'step':range(200),'isFraud':[0]*200})
    parts=partition_non_iid(df)
    assert sum(len(p) for p in parts)==len(df)
    assert len(set().union(*(set(p.index) for p in parts)))==len(df)

def test_chronological_split():
    df=pd.DataFrame({'step':list(range(100)),'amount':range(100)})
    a,b=chrono_split(df)
    assert a.step.max()<b.step.min()
