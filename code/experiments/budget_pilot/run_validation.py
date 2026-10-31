"""Bounded validation-only pilot. No test image loading, training or uploads."""
import argparse, csv, hashlib, json, math, platform, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from src.physics.radon import CTOperator
from src.eval.metrics import psnr, ssim, data_consistency
from src.utils import load_ema, save_grid
from experiments.budget_pilot.methods import sample_dc, measured_wls_weights

BASE_COMMIT='2311351b40adafd47cc922200a2c249cc22650e3'
PRIOR_SHA256='0e04c3e70c3d904b6ef93025faec465bb3b6a896d1c594088c18e7cfac8f72bb'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def select_validation(root, patients, slices, selection_seed):
    with open(root/'index_val.csv') as f: rows=list(csv.DictReader(f))
    by_patient=defaultdict(list)
    for i,r in enumerate(rows):by_patient[r['patient_id']].append(i)
    for split in ('train','test'):
        with open(root/f'index_{split}.csv') as f:
            others={r['patient_id'] for r in csv.DictReader(f)}
        if others & set(by_patient):raise ValueError(f'Patient overlap val/{split}')
    pids=sorted(by_patient); random.Random(selection_seed).shuffle(pids)
    chosen=pids[:patients]; items=[]
    for pid in chosen:
        ids=by_patient[pid]
        for k in np.linspace(0,len(ids)-1,min(slices,len(ids))).round().astype(int):
            i=ids[k]; items.append(dict(array_index=i,patient_id=pid,z=rows[i]['z']))
    return items, len(rows)


def atomic_json(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)

def require_finite(out, context, **values):
    bad=[name for name,value in values.items() if not bool(torch.isfinite(torch.as_tensor(value)).all())]
    if bad:
        atomic_json(Path(out)/'status.json',dict(complete=False,reason='non-finite numerical failure',
            failed_fields=bad,context=context,scope='validation only'))
        raise FloatingPointError('Non-finite values: '+', '.join(bad))

def summarize_group(group, selected, **metadata):
    expected={(e['patient_id'],e['array_index']) for e in selected}
    actual={(e['patient_id'],e['array_index']) for e in group}
    complete=bool(expected) and actual==expected and len(group)==len(expected)
    summary=metadata|dict(n_images=len(group),n_patients=len({r['patient_id'] for r in group}),
        complete=complete,eligible_for_comparison=complete,
        cohort=[dict(patient_id=p,array_index=i) for p,i in sorted(actual)])
    if complete:
        patient_means=defaultdict(list)
        for r in group:patient_means[r['patient_id']].append(r['psnr'])
        summary.update(patient_macro_psnr=float(np.mean([np.mean(v) for v in patient_means.values()])),
            image_mean_psnr=float(np.mean([r['psnr'] for r in group])),
            image_mean_ssim=float(np.mean([r['ssim'] for r in group])))
    else:
        summary['warning']='Incomplete fixed cohort: raw rows retained, no quality aggregate or ranking.'
    return summary

def comparison_status(summaries, lambdas):
    result={}
    for label in ('sv60','la120','ld180_i1e5'):
        groups=[s for s in summaries if s['setting']==label]
        expected=(2 if label=='ld180_i1e5' else 1)*len(lambdas)
        eligible=len(groups)==expected and all(s['complete'] for s in groups)
        eligible=eligible and all(s['cohort']==groups[0]['cohort'] for s in groups)
        result[label]=dict(eligible_for_comparison=eligible,expected_groups=expected,observed_groups=len(groups),
            rule='Compare only complete groups with identical fixed patient/slice cohorts; incomplete groups are not ranked.')
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed',type=Path,required=True);ap.add_argument('--ckpt',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--patients',type=int,default=4)
    ap.add_argument('--slices-per-patient',type=int,default=2);ap.add_argument('--nfe',type=int,default=20)
    ap.add_argument('--cg-iters',type=int,default=5);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--max-seconds',type=int,default=1200);ap.add_argument('--lambdas',type=float,nargs='+',default=[.3,1.,3.])
    a=ap.parse_args()
    if min(a.patients,a.slices_per_patient,a.nfe,a.cg_iters,a.max_seconds)<=0 or not all(math.isfinite(v) and v>0 for v in a.lambdas):
        raise ValueError("Positive finite pilot settings required")
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for validation; run synthetic controls separately')
    if a.out.exists():raise FileExistsError('Use a fresh output directory; do not overwrite prior evidence')
    a.out.mkdir(parents=True); started=time.monotonic()
    if sha(a.ckpt)!=PRIOR_SHA256:raise ValueError('Prior hash differs from verified release; do not load pickle')
    selected,nrows=select_validation(a.processed,a.patients,a.slices_per_patient,20261031)
    if not selected:raise ValueError('Empty validation cohort')
    xarr=np.load(a.processed/'slices_val.npy',mmap_mode='r')
    if len(xarr)!=nrows or tuple(xarr.shape[1:])!=(256,256):raise ValueError('Validation array/index mismatch')
    manifest=dict(base_commit=BASE_COMMIT,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        purpose='Validation-only exploratory pilot, not final-test or clinical evidence',
        selected=selected, selection_seed=20261031,prior_sha256=PRIOR_SHA256,
        val_index_sha256=sha(a.processed/'index_val.csv'),val_array_sha256=sha(a.processed/'slices_val.npy'),
        settings=vars(a)|{'processed':str(a.processed),'ckpt':str(a.ckpt),'out':str(a.out)},
        versions={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
        metric_roi='PSNR and SSIM within CTOperator circle mask, range [-1,1]. No LPIPS in this pilot.',
        noise='Poisson i0=1e5, mu_scale=.06, floor counts=1, sigma=0; observed-count delta WLS, per-image mean normalized.',
        privacy='Selection and images stay private; do not publish without review.')
    atomic_json(a.out/'manifest.json',manifest)
    model,cfg=load_ema(a.ckpt,'cuda');records=[];summaries=[]
    cases=[('sv60',60,math.pi,None),('la120',120,2*math.pi/3,None),('ld180_i1e5',180,math.pi,1e5)]
    for label,views,arange,i0 in cases:
        if time.monotonic()-started>a.max_seconds:break
        op=CTOperator(256,views,angle_range=arange,device='cuda')
        methods=[('dc',lam) for lam in a.lambdas]
        if i0:methods += [('wls',lam) for lam in a.lambdas]
        for method,lam in methods:
            if time.monotonic()-started>a.max_seconds:break
            group=[];torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
            for si,entry in enumerate(selected):
                if time.monotonic()-started>a.max_seconds:break
                index=entry['array_index'];x=torch.from_numpy(np.array(xarr[index],dtype=np.float32))[None,None].cuda()
                g=torch.Generator(device='cuda').manual_seed(1000003*(a.seed+1)+index)
                y=op.add_noise(op(x),i0=i0,mu_scale=.06,generator=g)
                weights=measured_wls_weights(y,i0) if method=='wls' else None
                torch.cuda.synchronize();t_sample=time.monotonic()
                pred=sample_dc(model,op,y,x.shape,nfe=a.nfe,lam_relative=lam,n_iter=a.cg_iters,
                    seed=2000003*(a.seed+1)+index,weights=weights)
                torch.cuda.synchronize();seconds=time.monotonic()-t_sample
                require_finite(a.out,dict(setting=label,method=method,lambda_relative=lam,array_index=index),pred=pred,truth=x,measurement=y)
                fbp=op.fbp(y)
                require_finite(a.out,dict(setting=label,method=method,array_index=index),fbp=fbp)
                row=dict(setting=label,method=method,lambda_relative=lam,array_index=index,
                    patient_id=entry['patient_id'],psnr=float(psnr(pred,x,op.mask)[0]),
                    ssim=float(ssim(pred,x,op.mask)[0]),residual=float(data_consistency(op,pred,y)[0]),
                    fbp_psnr=float(psnr(fbp,x,op.mask)[0]),fbp_ssim=float(ssim(fbp,x,op.mask)[0]),seconds=seconds)
                if i0:
                    est_counts=i0*torch.exp(-.06*y)
                    row['count_floor_fraction']=float((est_counts<=1.0001).float().mean())
                    w=measured_wls_weights(y,i0)
                    row['weighted_residual_rms']=float(((op(pred)-y).square()*w).mean().sqrt())
                require_finite(a.out,dict(setting=label,method=method,array_index=index),
                    **{k:v for k,v in row.items() if isinstance(v,(float,int))})
                records.append(row);group.append(row)
                # Incremental evidence survives timeout. Outputs contain patient IDs, keep private.
                with open(a.out/'per_image.jsonl','a') as f:f.write(json.dumps(row)+'\n')
                if si==0:save_grid(torch.cat([x,fbp,pred]),a.out/f'{label}_{method}_lam{lam}.png',nrow=3)
                print('[METRIC]',json.dumps(row),flush=True)
            if group:
                summary=summarize_group(group,selected,setting=label,method=method,lambda_relative=lam,
                    wall_seconds=time.monotonic()-t0,peak_cuda_bytes=torch.cuda.max_memory_allocated())
                summaries.append(summary);atomic_json(a.out/'summary.json',summaries)
                atomic_json(a.out/'comparison_status.json',comparison_status(summaries,a.lambdas))
                print('[SUMMARY]',json.dumps(summary),flush=True)
    # Generalize completion for caller-specified lambda grids.
    done=len(summaries)==4*len(a.lambdas) and all(s['complete'] for s in summaries)
    atomic_json(a.out/'status.json',dict(complete=done,wall_seconds=time.monotonic()-started,
        reason='completed' if done else 'pilot time budget reached',scope='validation only'))
    print('[DONE]',done,flush=True)

if __name__=='__main__':main()
