"""Batched bounds must equal per-domain bounds.

Batching is a performance change, never a semantic one.  If these disagree
beyond float tolerance the batched path is wrong and must not be used.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from crab.crown import compute_bounds, unstable_mask, ACTIVE, INACTIVE
from crab.batched import compute_bounds_batch
from crab.network import MLP

def random_net(rng, widths):
    W=[rng.normal(0,1.0/np.sqrt(widths[i]),(widths[i+1],widths[i])) for i in range(len(widths)-1)]
    b=[rng.normal(0,0.1,widths[i+1]) for i in range(len(widths)-1)]
    return MLP(W,b)

def run(seed=0):
    rng=np.random.default_rng(seed); worst=0.0; n=0
    for trial in range(15):
        depth=int(rng.integers(2,6)); w=int(rng.integers(12,30))
        widths=[5]+[w]*depth+[3]
        net=random_net(rng,widths)
        C=rng.normal(size=(2,3))
        B=int(rng.integers(4,24))
        doms=[]
        for _ in range(B):
            x0=rng.normal(size=5); eps=float(rng.uniform(.1,.5))
            st=[None]+[np.zeros(ww,dtype=np.int8) for ww in widths[1:-1]]
            r=compute_bounds(net,x0-eps,x0+eps,st,C)
            for i in range(1,net.L):
                idx=np.flatnonzero(unstable_mask(r.pre_lb[i],r.pre_ub[i],st[i]))
                if idx.size:
                    p=rng.choice(idx,size=max(1,idx.size//3),replace=False)
                    st[i][p]=rng.choice([ACTIVE,INACTIVE],size=len(p))
            doms.append((x0-eps,x0+eps,st))
        # per-domain reference
        ref=np.stack([compute_bounds(net,a,b,s,C).spec_lb for a,b,s in doms])
        # batched
        x_lb=np.stack([d[0] for d in doms]); x_ub=np.stack([d[1] for d in doms])
        status=[None]+[np.stack([d[2][i] for d in doms]) for i in range(1,net.L)]
        _,_,spec,_,_=compute_bounds_batch(net,x_lb,x_ub,status,C)
        d=np.abs(spec-ref).max(); worst=max(worst,float(d)); n+=B
    print(f"compared {n} domains across 15 networks (depth 2-5)")
    print(f"max |batched - per-domain| = {worst:.3e}")
    ok = worst < 1e-9
    print("PASS: batched bounds match per-domain" if ok else "FAIL: batched path is wrong")
    return ok

if __name__=="__main__":
    ok=run()
    # throughput comparison on the shared-work axis
    rng=np.random.default_rng(1); widths=[8]+[32]*5+[4]
    net=random_net(rng,widths); C=rng.normal(size=(3,4))
    for B in (1,64,512):
        x0=rng.normal(size=(B,8)); eps=0.3
        st=[None]+[np.zeros((B,32),dtype=np.int8) for _ in range(5)]
        t=time.time(); compute_bounds_batch(net,x0-eps,x0+eps,st,C); bt=time.time()-t
        print(f"  batch={B:4d}  {bt*1000:8.1f} ms  ({bt/B*1e6:7.1f} us/domain)")
    sys.exit(0 if ok else 1)
