"""Private validation-only solver sensitivity; not a final method comparison."""
import argparse,json,math,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from src.physics.radon import CTOperator
from src.eval.metrics import psnr,ssim
from src.utils import load_ema
from experiments.budget_pilot.methods_checked import sample_dc,pixel_weighted_dc,measured_wls_weights
from experiments.budget_pilot.run_validation import select_validation,sha,PRIOR_SHA256,atomic_json,require_finite

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--processed',type=Path,required=True);ap.add_argument('--ckpt',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);start=time.monotonic()
    if sha(a.ckpt)!=PRIOR_SHA256:raise ValueError('Unverified prior')
    selected,n=select_validation(a.processed,4,1,20261031);arr=np.load(a.processed/'slices_val.npy',mmap_mode='r')
    if len(arr)!=n or tuple(arr.shape[1:])!=(256,256):raise ValueError('Data shape mismatch')
    atomic_json(a.out/'manifest_private.json',dict(selected=selected,nfe=20,lambdas=[.3,3.],cg_iters=[5,12,30],seed=0,prior_sha256=PRIOR_SHA256,
        val_sha256=sha(a.processed/'slices_val.npy'),scope='CG sensitivity on four validation patients; no test; not final performance'))
    model,_=load_ema(a.ckpt,'cuda');op=CTOperator(256,180,device='cuda');rows=[]
    for entry in selected:
        idx=entry['array_index'];x=torch.from_numpy(np.array(arr[idx],dtype=np.float32))[None,None].cuda()
        y=op.add_noise(op(x),i0=1e5,mu_scale=.06,generator=torch.Generator(device='cuda').manual_seed(1000003+idx))
        w=measured_wls_weights(y,1e5);fbp=op.fbp(y)
        latent=torch.randn(x.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(2000003+idx))
        with torch.no_grad():clean=latent+model(latent,torch.zeros(1,device='cuda'))
        for method in ['dc','wls']:
            weights=torch.ones_like(y) if method=='dc' else w
            for lam in [.3,3.]:
                absolute=lam*op.ata_scale;rhs=op.A_lin_T(weights*(y-op.offset))+absolute*clean
                def mv(v):return op.A_lin_T(weights*op.A_lin(v))+absolute*v
                initial_res=float((mv(clean)-rhs).norm()/rhs.norm())
                for cg in [5,12,30]:
                    if time.monotonic()-start>160:raise TimeoutError('Diagnostic internal160s budget')
                    prox=pixel_weighted_dc(op,clean,y,weights,absolute,cg)
                    require_finite(a.out,dict(method=method,cg=cg),prox=prox)
                    normal_res=float((mv(prox)-rhs).norm()/rhs.norm())
                    objective=float((weights*(op(prox)-y).square()).sum()+absolute*(prox-clean).square().sum())
                    torch.cuda.synchronize();t=time.monotonic()
                    pred=sample_dc(model,op,y,x.shape,nfe=20,lam_relative=lam,n_iter=cg,seed=2000003+idx,weights=None if method=='dc' else w)
                    torch.cuda.synchronize();elapsed=time.monotonic()-t
                    require_finite(a.out,dict(method=method,cg=cg),pred=pred)
                    r=dict(patient_id=entry['patient_id'],array_index=idx,method=method,lambda_relative=lam,cg_iters=cg,
                        psnr=float(psnr(pred,x,op.mask)[0]),ssim=float(ssim(pred,x,op.mask)[0]),seconds=elapsed,
                        initial_prox_relative_residual=initial_res,first_prox_relative_residual=normal_res,first_prox_objective=objective,
                        weighted_data_rms=float(((op(pred)-y).square()*w).mean().sqrt()),fbp_psnr=float(psnr(fbp,x,op.mask)[0]),
                        weight_min=float(w.min()),weight_p01=float(torch.quantile(w,.01)),weight_p99=float(torch.quantile(w,.99)),weight_max=float(w.max()))
                    require_finite(a.out,dict(method=method,cg=cg),**{k:v for k,v in r.items() if isinstance(v,(int,float))})
                    rows.append(r)
                    with open(a.out/'per_image_private.jsonl','a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                    groups=defaultdict(list)
                    for q in rows:groups[(q['method'],q['lambda_relative'],q['cg_iters'])].append(q)
                    summary=[]
                    for (m,l,c),rs in groups.items():
                        complete=len(rs)==len(selected)
                        summary.append(dict(method=m,lambda_relative=l,cg_iters=c,n_patients=len(rs),complete=complete,
                            means={k:float(np.mean([v[k] for v in rs])) for k in r if k not in ['patient_id','array_index','method','lambda_relative','cg_iters']} if complete else {},
                            warning='' if complete else 'Incomplete cohort: do not compare'))
                    atomic_json(a.out/'summary_deidentified.json',summary)
    atomic_json(a.out/'status.json',dict(complete=len(rows)==48,wall_seconds=time.monotonic()-start,scope='solver sensitivity, validation only'))

if __name__=='__main__':
    import sys,traceback
    try:main()
    except Exception as exc:
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
            atomic_json(out/'diagnostic_failure.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
