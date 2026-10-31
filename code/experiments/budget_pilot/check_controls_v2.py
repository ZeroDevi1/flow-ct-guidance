"""Synthetic numerical controls only; does not read any patient images."""
import argparse, json, math, time
from pathlib import Path
import torch
from src.physics.radon import CTOperator, measured_mask, weighted_cg_dc
from src.data.dataset import make_phantom
from experiments.budget_pilot.methods import pixel_weighted_dc, measured_wls_weights, sample_dc


def run(device='cpu', size=32, dtype=torch.float32, cg_iters=12, records=None):
    torch.manual_seed(13)
    x = make_phantom(size, torch.Generator().manual_seed(13))[None,None].to(device=device,dtype=dtype)
    x0 = torch.randn_like(x)*0.1
    results = records if records is not None else []
    for name, views, angle_range in [('sv60',60,math.pi), ('la120',120,2*math.pi/3)]:
        full = CTOperator(size, 180, device=device,dtype=dtype)
        op = CTOperator(size, views, angle_range=angle_range, device=device,dtype=dtype)
        mask = measured_mask(180, views, angle_range).to(device)
        y = op(x)
        yf = torch.zeros_like(full(x)); yf[:,:,mask] = y
        row_weights = mask.float()[None]
        absolute = .3*op.ata_scale
        sparse = op.cg_data_consistency(x0,y,lam=absolute,n_iter=cg_iters,relative=False)
        raw_matched = weighted_cg_dc(full,x0,yf,row_weights,lam=absolute,n_iter=cg_iters,relative=False)
        raw_delta = float((sparse-raw_matched).abs().max())
        # Require EXACT measurement angles and affine offsets for the equivalence control.
        # Separate linspace constructions can differ by ulps and amplify in finite CG.
        # Report that raw discrepancy rather than silently treating it as identical physics.
        angle_delta = float((full.angles[mask]-op.angles).abs().max())
        full.angles[mask] = op.angles
        full.offset[:,:,mask] = op.offset
        matched = weighted_cg_dc(full,x0,yf,row_weights,lam=absolute,n_iter=cg_iters,relative=False)
        historical = weighted_cg_dc(full,x0,yf,row_weights,lam=.3,n_iter=cg_iters,relative=True)
        delta = float((sparse-matched).abs().max())
        scale = max(float(sparse.abs().max()), 1.)
        passed = delta < 3e-4*scale
        results.append(dict(setting=name, lambda_absolute=absolute,
            sparse_ata_scale=op.ata_scale,full_ata_scale=full.ata_scale,
            measured_forward_max_abs=float((full(x)[:,:,mask]-y).abs().max()),
            matched_max_abs=delta, raw_grid_matched_max_abs=raw_delta, raw_angle_max_abs=angle_delta, historical_same_relative_max_abs=float((historical-sparse).abs().max()),
            atol_scaled=3e-4*scale, passed=passed))
        assert passed, results[-1]
        weighted = pixel_weighted_dc(op,x0,y,torch.ones_like(y),absolute,cg_iters)
        results[-1]['unit_weight_max_abs']=float((sparse-weighted).abs().max())
        results[-1]['unit_weight_allclose']=bool(torch.allclose(sparse,weighted,atol=1e-5))
        assert results[-1]['unit_weight_allclose'], 'unit pixel-weight CG differs'
        # Reference property: weighted operator remains symmetric under exact adjoint.
        w = measured_wls_weights(y,1e5)
        a,b = torch.randn_like(x),torch.randn_like(x)
        lhs = (op.A_lin(a)*w*op.A_lin(b)).sum()
        rhs = (a*op.A_lin_T(w*op.A_lin(b))).sum()
        assert torch.allclose(lhs,rhs,atol=1e-4,rtol=1e-4)
        assert torch.isfinite(w).all() and (w>0).all()
    # Tests values derived from known positive counts, not a GT-weighted method.
    counts=torch.tensor([1.,10.,100.],device=device,dtype=dtype).view(1,1,1,3)
    y=-torch.log(counts/1e5)/.06
    got=measured_wls_weights(y,1e5)
    assert torch.allclose(got,counts/counts.mean(),atol=1e-5)
    return dict(device=device,size=size,dtype=str(dtype),cg_iters=cg_iters,controls=results,all_passed=True,
                scope='Synthetic numerical controls; not clinical or held-out evaluation.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cpu'); ap.add_argument('--size',type=int,default=32);ap.add_argument('--out',required=True);ap.add_argument('--dtype',choices=['float32','float64'],default='float32');ap.add_argument('--cg-iters',type=int,default=12)
    a=ap.parse_args(); t=time.monotonic();p=Path(a.out)
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    try:
        if a.cg_iters<=0:raise ValueError('Positive CG iterations required')
        out=run(a.device,a.size,getattr(torch,a.dtype),a.cg_iters,records)
    except Exception as exc:
        import traceback
        out=dict(all_passed=False,device=a.device,size=a.size,dtype=a.dtype,
            cg_iters=a.cg_iters,controls=records,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-t)
        p.write_text(json.dumps(out,indent=2,allow_nan=False));raise
    out['wall_seconds']=time.monotonic()-t
    p.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2),flush=True)
