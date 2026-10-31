"""Read-only matched-development-cohort FBPConvNet evaluation.
Private rows stay on the instance; only summary_anonymous.json is exportable.
No optimization, checkpoint selection, test image loading or public upload.
"""
import argparse,csv,json,math,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from src.models.unet import UNet
from src.physics.radon import CTOperator
from src.eval.metrics import psnr,ssim,data_consistency
from experiments.budget_pilot.train_fbp_safe import sha,atomic_json,finite
from experiments.budget_pilot.methods import measured_wls_weights


def read_json(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def cohort(entries):
    keys=[(r['patient_id'],int(r['array_index'])) for r in entries]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate cohort records')
    return set(keys)


def load_p1(batch):
    batch=Path(batch);cfg=read_json(batch/'batch_config.json');status=read_json(batch/'batch_status.json')
    if not status.get('complete'):raise ValueError('Incomplete P1 batch')
    if cfg['seeds']!=[0,1,2] or cfg['patients']!=20 or cfg['slices_per_patient']!=2 or cfg['cases']!=['ld180_i1e5']:raise ValueError('Unexpected P1 protocol')
    if len(cfg['dc_lambdas'])!=1 or len(cfg['wls_lambdas'])!=1:raise ValueError('Frozen one-lambda methods required')
    if {r['seed'] for r in status['seeds']}!=set(cfg['seeds']) or len(status['seeds'])!=3 or not all(r['complete'] and r['exit_code']==0 for r in status['seeds']):raise ValueError('P1 seed status incomplete')
    expected=None;data_hashes=None;samples={};manifests=[]
    for seed in cfg['seeds']:
        root=batch/f'seed_{seed}';m=read_json(root/'manifest.json');manifests.append(m)
        if not read_json(root/'status.json').get('complete'):raise ValueError('Incomplete P1 seed')
        current=cohort(m['selected']);counts=Counter(p for p,i in current)
        if len(counts)!=20 or len(current)!=40 or set(counts.values())!={2}:raise ValueError('P1 cohort not exactly 20 patients x 2 slices')
        s=m['settings']
        if s['seed']!=seed or s['patients']!=20 or s['slices_per_patient']!=2 or s['cases']!=cfg['cases'] or s['lambdas']!=cfg['dc_lambdas'] or s['wls_lambdas']!=cfg['wls_lambdas'] or s['nfe']!=cfg['nfe'] or s['cg_iters']!=cfg['cg_iters']:raise ValueError('P1 manifest settings mismatch')
        hashes={'slices_val.npy':m['val_array_sha256'],'index_val.csv':m['val_index_sha256']}
        if expected is None:expected=current;data_hashes=hashes
        if current!=expected or hashes!=data_hashes:raise ValueError('P1 cross-seed cohort/data mismatch')
        for row in rows(root/'per_image.jsonl'):
            method=row['method'];key=(seed,row['patient_id'],int(row['array_index']),method)
            if method not in ('dc','wls') or key in samples:raise ValueError('Unexpected/duplicate P1 method row')
            if row['setting']!='ld180_i1e5' or row['lambda_relative']!=cfg[f'{method}_lambdas'][0]:raise ValueError('Unexpected P1 setting/lambda')
            finite(**{k:row[k] for k in ('psnr','ssim','fbp_psnr','fbp_ssim')});samples[key]=row
        needed={(seed,p,i,meth) for p,i in expected for meth in ('dc','wls')}
        if {k for k in samples if k[0]==seed}!=needed:raise ValueError('Incomplete P1 paired rows')
    return cfg,expected,data_hashes,samples,manifests


def validate_checkpoint(ck,expected,hashes):
    if ck.get('format')!='flowct_fbp_safe_v1' or ck.get('step',0)<=0:raise ValueError('Not a trained safe-trainer checkpoint')
    cfg=ck['config'];p=cfg['physics']
    if p.get('n_angles')!=180 or p.get('i0')!=1e5 or p.get('sigma',0.)!=0. or p.get('mu_scale',.06)!=.06 or p.get('angle_range',math.pi)!=math.pi:raise ValueError('Checkpoint physics differs from low-dose P1')
    if cfg['validation']['patients']!=20 or cfg['validation']['slices_per_patient']!=2 or cfg['validation']['selection_seed']!=20261031:raise ValueError('Checkpoint development selection protocol differs')
    if cohort(ck['validation_selection'])!=expected:raise ValueError('Checkpoint cohort differs from completed P1')
    if any(ck['data_sha256'].get(k)!=v for k,v in hashes.items()):raise ValueError('Checkpoint/P1 validation hashes differ')
    lv=ck.get('last_val')
    if not lv or not lv.get('complete') or lv.get('step')!=ck['step']:raise ValueError('Best checkpoint lacks complete same-step validation')
    return cfg


@torch.no_grad()
def predict_residual(model,fbp):
    raw=fbp+model(fbp,torch.zeros(fbp.shape[0],device=fbp.device))
    finite(raw_prediction=raw)
    return raw.clamp(-1,1)


def anonymous_summary(evaluated,expected,seeds,p1):
    needed={(s,p,i) for s in seeds for p,i in expected}
    if set(evaluated)!=needed:raise ValueError('Incomplete FBP evaluation: no comparison allowed')
    patients=sorted({p for p,i in expected});by=defaultdict(lambda:defaultdict(list));seconds=[]
    for (seed,p,i),r in evaluated.items():
        for metric in ('psnr','ssim','residual','weighted_residual_rms'):
            finite(value=r[metric]);by[('fbpconvnet',metric)][p].append(r[metric])
        seconds.append(r['seconds'])
        for method in ('dc','wls'):
            old=p1[(seed,p,i,method)]
            # Confirm regenerated measurement before accepting paired metrics.
            if abs(old['fbp_psnr']-r['fbp_psnr'])>1e-5 or abs(old['fbp_ssim']-r['fbp_ssim'])>1e-5:raise ValueError('Regenerated FBP mismatch: not a valid paired comparison')
            for metric in ('psnr','ssim'):by[(method,metric)][p].append(old[metric])
        by[('fbp','psnr')][p].append(r['fbp_psnr']);by[('fbp','ssim')][p].append(r['fbp_ssim'])
    ag={}
    for method in ('fbpconvnet','dc','wls','fbp'):
        ag[method]={metric:float(np.mean([np.mean(v) for v in by[(method,metric)].values()])) for metric in ('psnr','ssim')}
    for metric in ('residual','weighted_residual_rms'):ag['fbpconvnet'][metric]=float(np.mean([np.mean(v) for v in by[('fbpconvnet',metric)].values()]))
    pairs={};rng=np.random.default_rng(20261031)
    bootstrap_indices=rng.integers(0,len(patients),size=(10000,len(patients)))
    for method in ('dc','wls','fbp'):
        delta=np.array([np.mean(by[('fbpconvnet','psnr')][p])-np.mean(by[(method,'psnr')][p]) for p in patients])
        pairs['fbpconvnet_minus_'+method]={'patient_mean_db':float(delta.mean()),'patient_median_db':float(np.median(delta)),
            'fraction_patients_positive':float((delta>0).mean()),'exploratory_patient_bootstrap95':[float(v) for v in np.quantile(delta[bootstrap_indices].mean(axis=1),[.025,.975])]}
    return {'complete':True,'eligible_for_comparison':True,'n_patients':len(patients),'n_slices_per_seed':len(expected),'n_seeds':len(seeds),
        'n_fbpconvnet_predictions':len(evaluated),'aggregates':ag,'paired_psnr':pairs,
        'prediction_seconds_mean':float(np.mean(seconds)),'prediction_seconds_p95':float(np.quantile(seconds,.95)),
        'limitations':['Same development cohort selected training checkpoint and prior-method parameters; not independent validation or clinical evidence.',
            'Bootstrap resamples patients after averaging slices/seeds; intervals are exploratory and omit training-seed/checkpoint-selection uncertainty.',
            'FBPConvNet is deterministic given each noisy input; P1 seeds additionally vary diffusion initialization.',
            'Prediction timing excludes FBP generation; no claim of matched total computational cost.']}


def evaluate(batch,checkpoint,expected_sha,processed,out,device='cuda',max_seconds=900):
    out=Path(out);checkpoint=Path(checkpoint);processed=Path(processed)
    if out.exists():raise FileExistsError(out)
    if checkpoint.name!='best.pt':raise ValueError('Freeze safe-trainer best.pt, not last or a post-hoc chosen artifact')
    if sha(checkpoint)!=expected_sha:raise ValueError('Checkpoint SHA256 mismatch; deserialization refused')
    if max_seconds<=0:raise ValueError('Positive time budget required')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    cfg,expected,hashes,p1,manifests=load_p1(Path(batch))
    if any(sha(processed/k)!=v for k,v in hashes.items()):raise ValueError('Actual validation data differs from P1')
    # SHA verified above; only trusted locally-produced training checkpoints accepted.
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);modelcfg=validate_checkpoint(ck,expected,hashes)
    with open(processed/'index_val.csv') as f:index=list(csv.DictReader(f))
    if any(i<0 or i>=len(index) or index[i]['patient_id']!=p for p,i in expected):raise ValueError('Cohort/index lookup mismatch')
    xarr=np.load(processed/'slices_val.npy',mmap_mode='r');n=modelcfg['model']['image_size']
    if len(xarr)!=len(index) or tuple(xarr.shape[1:])!=(n,n):raise ValueError('Validation array shape mismatch')
    out.mkdir(parents=True);started=time.monotonic();evaluated={}
    atomic_json(out/'manifest_private.json',{'checkpoint_sha256':expected_sha,'checkpoint_step':ck['step'],'p1_config':cfg,
        'data_sha256':hashes,'selected':[{'patient_id':p,'array_index':i} for p,i in sorted(expected)],'seeds':cfg['seeds'],
        'scope':'Read-only same-development-cohort comparison; private identity-bearing manifest'})
    try:
        model=UNet(**modelcfg['model']).to(device);model.load_state_dict(ck['model']);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        op=CTOperator(n,180,device=device)
        for seed in cfg['seeds']:
            for patient_id,idx in sorted(expected):
                if time.monotonic()-started>=max_seconds:raise TimeoutError('Evaluation budget reached; no partial comparison')
                x=torch.from_numpy(np.array(xarr[idx],dtype=np.float32))[None,None].to(device)
                gen=torch.Generator(device=device).manual_seed(1000003*(seed+1)+idx)
                y=op.add_noise(op(x),i0=1e5,sigma=0.,mu_scale=.06,generator=gen);fbp=op.fbp(y)
                finite(truth=x,measurement=y,fbp=fbp)
                if device=='cuda':torch.cuda.synchronize()
                t=time.monotonic();pred=predict_residual(model,fbp)
                if device=='cuda':torch.cuda.synchronize()
                sec=time.monotonic()-t;weights=measured_wls_weights(y,1e5)
                row={'seed':seed,'patient_id':patient_id,'array_index':idx,'method':'fbpconvnet','setting':'ld180_i1e5',
                    'psnr':float(psnr(pred,x,op.mask)[0]),'ssim':float(ssim(pred,x,op.mask)[0]),
                    'fbp_psnr':float(psnr(fbp,x,op.mask)[0]),'fbp_ssim':float(ssim(fbp,x,op.mask)[0]),
                    'residual':float(data_consistency(op,pred,y)[0]),'weighted_residual_rms':float(((op(pred)-y).square()*weights).mean().sqrt()),'seconds':sec}
                finite(**{k:v for k,v in row.items() if isinstance(v,(int,float))})
                evaluated[(seed,patient_id,idx)]=row
                with open(out/'per_image_private.jsonl','a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        summary=anonymous_summary(evaluated,expected,cfg['seeds'],p1);summary.update(checkpoint_step=ck['step'],wall_seconds=time.monotonic()-started)
        atomic_json(out/'summary_anonymous.json',summary);atomic_json(out/'status.json',{'complete':True,'n_records':len(evaluated)})
        return summary
    except Exception as exc:
        atomic_json(out/'status.json',{'complete':False,'eligible_for_comparison':False,'error_type':type(exc).__name__,'error':str(exc),'n_records':len(evaluated)})
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--checkpoint-sha256',required=True)
    ap.add_argument('--processed',required=True);ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda',choices=['cpu','cuda']);ap.add_argument('--max-seconds',type=int,default=900)
    a=ap.parse_args();print(json.dumps(evaluate(a.batch,a.checkpoint,a.checkpoint_sha256,a.processed,a.out,a.device,a.max_seconds),indent=2,allow_nan=False))
