"""Validate the torch backend on YOUR hardware before trusting it.

Order matters: correctness first, then precision, then speed.  A fast wrong
verifier is worse than no verifier.

  1. device detection
  2. float64-on-CPU torch vs NumPy reference   -> is the port correct at all?
  3. float32-on-MPS   vs NumPy reference       -> how far does precision drift?
  4. slack calibration                          -> what outward rounding is needed?
  5. throughput vs batch size                   -> is the GPU actually worth it?
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch not installed.\n  pip install torch")
    sys.exit(1)

from crab.network import MLP
from crab.batched import compute_bounds_batch
from crab.torch_backend import TorchBounds, pick_device, calibrate_slack

def make_net(rng, widths):
    W=[rng.normal(0,1.0/np.sqrt(widths[i]),(widths[i+1],widths[i])) for i in range(len(widths)-1)]
    b=[rng.normal(0,0.1,widths[i+1]) for i in range(len(widths)-1)]
    return MLP(W,b)

def compare(net, tb, B=32, seed=0):
    rng=np.random.default_rng(seed)
    x0=rng.normal(size=(B,net.n_in)); eps=rng.uniform(.1,.5,(B,1))
    st=[None]+[np.zeros((B,w),dtype=np.int8) for w in net.hidden_sizes()]
    C=(np.eye(net.n_out)[:1]-np.eye(net.n_out)[1:2])
    ref=compute_bounds_batch(net,(x0-eps).astype(np.float64),(x0+eps).astype(np.float64),st,C)[2]
    a,b,s,c=tb.to_device(x0-eps,x0+eps,st,C)
    got=tb.compute(a,b,s,c)[2].cpu().numpy().astype(np.float64)
    diff=got-ref
    return float(np.abs(diff).max()), float(diff.max())

def main():
    rng=np.random.default_rng(0)
    net=make_net(rng,[8]+[32]*5+[4])

    print("=== 1. device ===")
    dev,dt,note=pick_device()
    print(f"  selected: {dev}  dtype={dt}")
    print(f"  {note}")
    print(f"  mps available: {torch.backends.mps.is_available()}")

    print("\n=== 2. correctness: torch float64 on CPU vs NumPy ===")
    tb=TorchBounds(net,device=torch.device('cpu'),dtype=torch.float64)
    mx,_=compare(net,tb)
    ok64 = mx < 1e-9
    print(f"  max |torch - numpy| = {mx:.3e}   {'PASS' if ok64 else 'FAIL — port is wrong'}")
    if not ok64:
        print("  Stop here. Do not use the torch backend."); sys.exit(1)

    if dev.type=='cpu':
        print("\nNo GPU detected; nothing further to validate."); return

    print(f"\n=== 3. precision: float32 on {dev} vs NumPy float64 ===")
    tb32=TorchBounds(net,dtype=torch.float32)
    mx,worst_high=compare(net,tb32)
    print(f"  max |drift|              = {mx:.3e}")
    print(f"  worst OVERESTIMATE       = {worst_high:.3e}"
          f"   {'(unsound without slack)' if worst_high>0 else ''}")

    print("\n=== 4. slack calibration on this machine ===")
    t=time.time(); slack=calibrate_slack(net,n_trials=60)
    print(f"  recommended slack = {slack:.3e}   ({time.time()-t:.1f}s)")
    print("  subtract this from every float32 bound before comparing to zero")

    print("\n=== 5. throughput vs batch size ===")
    print(f"  {'batch':>6} {'CPU f64':>12} {'GPU f32':>12} {'speedup':>9}")
    for B in (1,64,256,1024,4096):
        x0=rng.normal(size=(B,8)); eps=0.3
        st=[None]+[np.zeros((B,32),dtype=np.int8) for _ in range(5)]
        C=(np.eye(4)[:1]-np.eye(4)[1:2])
        t=time.time(); compute_bounds_batch(net,x0-eps,x0+eps,st,C); ct=time.time()-t
        a,b,s,c=tb32.to_device(x0-eps,x0+eps,st,C)
        tb32.compute(a,b,s,c)                      # warm up kernels
        torch.mps.synchronize() if dev.type=='mps' else None
        t=time.time(); tb32.compute(a,b,s,c)
        torch.mps.synchronize() if dev.type=='mps' else None
        gt=time.time()-t
        print(f"  {B:>6} {ct*1e3:>10.1f}ms {gt*1e3:>10.1f}ms {ct/max(gt,1e-9):>8.1f}x")

    print("\nGuidance: use the GPU tier only where the batch is large. Escalate any")
    print("domain whose float32 bound lies within the slack of zero to float64 CPU.")

if __name__=="__main__":
    main()
