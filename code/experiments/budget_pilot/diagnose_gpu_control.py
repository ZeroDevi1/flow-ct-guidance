"""Diagnostic only: no tolerance relaxation, patient data or training.
Run before deciding whether a failed unit-weight numerical control is acceptable.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from src.physics.radon import CTOperator
from src.data.dataset import make_phantom
from experiments.budget_pilot.methods import pixel_weighted_dc

def difference(a,b):
    d=(a-b).double()
    return dict(max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),
        relative_l2=float(d.norm()/b.double().norm().clamp_min(1e-12)),
        original_allclose=bool(torch.allclose(a,b,atol=1e-5)),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13)
    x=make_phantom(a.size,torch.Generator().manual_seed(13))[None,None].to(a.device)
    x0=torch.randn_like(x)*.1
    report=dict(scope='Failure diagnosis only; not pass/fail replacement',device=a.device,size=a.size,
        torch=torch.__version__,cuda=torch.version.cuda,rows=[])
    start=time.monotonic()
    def save():
        report['wall_seconds']=time.monotonic()-start
        temp=a.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(a.out)
    save()
    for name,n,ar in [('sv60',60,math.pi),('la120',120,2*math.pi/3)]:
        op=CTOperator(a.size,n,angle_range=ar,device=a.device);y=op(x);lam=.3*op.ata_scale
        rhs=op.A_lin_T(y-op.offset)+lam*x0
        def matvec(v):return op.A_lin_T(op.A_lin(v))+lam*v
        adjoints=[op.A_lin_T(y-op.offset) for _ in range(3)]
        for steps in (5,12):
            base=[op.cg_data_consistency(x0,y,lam=lam,n_iter=steps,relative=False) for _ in range(3)]
            weighted=[pixel_weighted_dc(op,x0,y,torch.ones_like(y),lam,steps) for _ in range(3)]
            row=dict(setting=name,n_iter=steps,lambda_absolute=lam,
                adjoint_repeat=[difference(z,adjoints[0]) for z in adjoints[1:]],
                legacy_repeat=[difference(z,base[0]) for z in base[1:]],
                weighted_repeat=[difference(z,weighted[0]) for z in weighted[1:]],
                cross=[difference(weighted[i],base[i]) for i in range(3)],
                relative_equation_residual={method:[float((matvec(z)-rhs).double().norm()/rhs.double().norm().clamp_min(1e-12)) for z in values]
                    for method,values in [('legacy',base),('weighted',weighted)]})
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
    report['complete']=True;save()

if __name__=='__main__':
    import sys,traceback
    try:
        main()
    except Exception as exc:
        # Preserve prior diagnostic rows even when a later operation fails.
        if '--out' in sys.argv:
            target=Path(sys.argv[sys.argv.index('--out')+1])
            failure=target.with_name(target.name+'.failure.json')
            failure.parent.mkdir(parents=True,exist_ok=True)
            if not failure.exists():
                failure.write_text(json.dumps(dict(complete=False,error_type=type(exc).__name__,
                    error=str(exc),traceback=traceback.format_exc()),indent=2,allow_nan=False))
        raise
