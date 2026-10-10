"""Educational in-process FedAvg: no networking or fault tolerance yet."""
from copy import deepcopy
import torch

def fedavg(states, sizes):
    if not states or len(states)!=len(sizes) or sum(sizes)<=0:
        raise ValueError('Need client weights and sample counts')
    result=deepcopy(states[0])
    total=sum(sizes)
    with torch.no_grad():
        for key in result:
            result[key]=sum(state[key]*(n/total) for state,n in zip(states,sizes))
    return result
