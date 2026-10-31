"""Validation-only FBPConvNet trainer; never reads test arrays or evaluates test.

Fresh outputs only. Training samples are drawn uniformly with replacement using a
checkpointed generator (not historical epoch-based shuffling). bf16 needs no scaler.
External process timeout and verified platform budget deadline remain mandatory.
"""
import argparse,csv,hashlib,json,math,random,signal,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models.unet import UNet
from src.physics.radon import CTOperator
from src.eval.metrics import psnr,ssim
from src.train_flow import lr_at


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

def atomic_torch(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(value,tmp);tmp.replace(p)

def read_index(root,split):
    with open(root/f'index_{split}.csv') as f:return list(csv.DictReader(f))

def prepare_data(root,n,patients,slices,selection_seed):
    # Test IDs are checked for leakage, but no test image array is opened.
    rows={s:read_index(root,s) for s in ('train','val','test')}
    ids={s:{r['patient_id'] for r in rs} for s,rs in rows.items()}
    if any(ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Patient split overlap')
    arrays={s:np.load(root/f'slices_{s}.npy',mmap_mode='r') for s in ('train','val')}
    for s,x in arrays.items():
        if len(x)!=len(rows[s]) or tuple(x.shape[1:])!=(n,n) or not len(x):raise ValueError('Array/index mismatch')
    by=defaultdict(list)
    for i,r in enumerate(rows['val']):by[r['patient_id']].append(i)
    pids=sorted(by)
    if len(pids)<patients:raise ValueError(f'Requested {patients} validation patients but only {len(pids)} are available')
    random.Random(selection_seed).shuffle(pids);chosen=pids[:patients]
    selected=[]
    for pid in chosen:
        indices=by[pid]
        for k in np.linspace(0,len(indices)-1,min(slices,len(indices))).round().astype(int):selected.append({'patient_id':pid,'array_index':indices[k]})
    if not selected:raise ValueError('Empty validation cohort')
    hashes={f:sha(root/f) for f in ('slices_train.npy','slices_val.npy','index_train.csv','index_val.csv','index_test.csv')}
    return arrays,selected,hashes


def finite(**values):
    bad=[k for k,v in values.items() if not torch.isfinite(torch.as_tensor(v)).all()]
    if bad:raise FloatingPointError('Nonfinite '+','.join(bad))


def train(config,out,resume=None,stop_after=None,max_seconds=13800,device='cuda'):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    out.mkdir(parents=True);start=time.monotonic();stop={'requested':False}
    old_handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):
        old_handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.update(requested=True))
    cfg=json.loads(json.dumps(config));seed=int(cfg.get('seed',0));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    n=cfg['model']['image_size'];root=Path(cfg['data']['processed_dir']);tc=cfg['train'];vc=cfg['validation']
    if min(tc['batch_size'],tc['max_steps'],tc['eval_every'],tc['save_every'],max_seconds,vc['patients'],vc['slices_per_patient'])<=0:raise ValueError('Positive limits required')
    arrays,selected,hashes=prepare_data(root,n,vc['patients'],vc['slices_per_patient'],vc['selection_seed'])
    manifest={'config':cfg,'data_sha256':hashes,'validation_selection':selected,'created_epoch':time.time(),
        'train_sampling':'Uniform with replacement; checkpointed generator and flips',
        'scope':'Validation development only. Never reads test images. No convergence guarantee.',
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda},'resume_from':str(resume) if resume else None}
    atomic_json(out/'manifest.json',manifest)
    p=cfg['physics'];op=CTOperator(n,p['n_angles'],angle_range=p.get('angle_range',math.pi),device=device)
    model=UNet(**cfg['model']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc.get('weight_decay',.01))
    bg=torch.Generator(device='cpu').manual_seed(seed+331);ng=torch.Generator(device=device).manual_seed(seed+773)
    step=0;best=-1e100;bad_evals=0;last_val=None;train_seconds=0.;timed_steps=0;stop_reason='max_steps'
    if resume:
        # Only resume a checkpoint created by this trainer, with a recorded hash supplied by operator.
        ck=torch.load(resume,map_location=device,weights_only=False)
        if ck['format']!='flowct_fbp_safe_v1' or ck['config']!=cfg or ck['data_sha256']!=hashes or ck['validation_selection']!=selected:raise ValueError('Resume config/data/cohort mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];best=ck['best'];bad_evals=ck['bad_evals'];last_val=ck['last_val']
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in ck['rng']['cuda']])
        bg.set_state(ck['rng']['batch'].cpu());ng.set_state(ck['rng']['noise'].cpu())
    def checkpoint(name):
        value={'format':'flowct_fbp_safe_v1','model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':None,
            'step':step,'config':cfg,'data_sha256':hashes,'validation_selection':selected,'best':best,'bad_evals':bad_evals,'last_val':last_val,
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all() if device=='cuda' else [],'batch':bg.get_state(),'noise':ng.get_state()}}
        atomic_torch(out/name,value);atomic_json(out/(name+'.sha256.json'),{'sha256':sha(out/name),'step':step})
    @torch.no_grad()
    def evaluate():
        model.eval();by=defaultdict(list);ss=[]
        for entry in selected:
            idx=entry['array_index'];x=torch.from_numpy(np.array(arrays['val'][idx],dtype=np.float32))[None,None].to(device)
            g=torch.Generator(device=device).manual_seed(1000003*(vc['noise_seed']+1)+idx)
            y=op.add_noise(op(x),i0=p.get('i0'),sigma=p.get('sigma',0.),mu_scale=p.get('mu_scale',.06),generator=g);f=op.fbp(y)
            raw=f+model(f,torch.zeros(1,device=device));finite(raw=raw,truth=x);pred=raw.clamp(-1,1)
            value=float(psnr(pred,x,op.mask)[0]);sim=float(ssim(pred,x,op.mask)[0]);finite(psnr=value,ssim=sim)
            by[entry['patient_id']].append(value);ss.append(sim)
        model.train()
        return {'step':step,'patient_macro_psnr':float(np.mean([np.mean(v) for v in by.values()])),
            'image_mean_ssim':float(np.mean(ss)),'n_images':len(selected),'n_patients':len(by),'complete':True}
    try:
        while step<tc['max_steps']:
            if stop['requested'] or time.monotonic()-start>=max_seconds:
                stop_reason='signal' if stop['requested'] else 'time_budget';break
            if stop_after is not None and timed_steps>=stop_after:stop_reason='throughput_step_limit';break
            if device=='cuda':torch.cuda.synchronize()
            t=time.monotonic();indices=torch.randint(len(arrays['train']),(tc['batch_size'],),generator=bg)
            x=torch.from_numpy(np.array(arrays['train'][indices.numpy()],dtype=np.float32))[:,None]
            flips=torch.rand(tc['batch_size'],generator=bg)<.5;x[flips]=x[flips].flip(-1);x=x.to(device)
            with torch.no_grad():y=op.add_noise(op(x),i0=p.get('i0'),sigma=p.get('sigma',0.),mu_scale=p.get('mu_scale',.06),generator=ng);f=op.fbp(y)
            for group in opt.param_groups:group['lr']=lr_at(step,tc)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                pred=f+model(f,torch.zeros(x.shape[0],device=device)).float();loss=(pred-x).square().mean()
            finite(loss=loss,pred=pred);opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);finite(gradient_norm=norm);opt.step();step+=1
            if device=='cuda':torch.cuda.synchronize()
            duration=time.monotonic()-t;train_seconds+=duration;timed_steps+=1
            rec={'step':step,'loss':float(loss.detach()),'seconds':duration,'lr':opt.param_groups[0]['lr']}
            with open(out/'train.jsonl','a') as h:h.write(json.dumps(rec,allow_nan=False)+'\n')
            if step%tc['eval_every']==0:
                last_val=evaluate();score=last_val['patient_macro_psnr']
                if score>best+tc['min_delta_db']:best=score;bad_evals=0;checkpoint('best.pt')
                else:bad_evals+=1
                with open(out/'validation.jsonl','a') as h:h.write(json.dumps(last_val,allow_nan=False)+'\n')
                if step>=tc['min_steps'] and bad_evals>=tc['patience']:
                    stop_reason='validation_plateau';break
            if step%tc['save_every']==0:checkpoint('last.pt')
        if last_val is None or last_val['step']!=step:
            # Do not spend unbounded evaluation time after deadline or TERM.
            if not stop['requested'] and time.monotonic()-start<max_seconds-30:
                last_val=evaluate()
                with open(out/'validation.jsonl','a') as h:h.write(json.dumps(last_val,allow_nan=False)+'\n')
                if last_val['patient_macro_psnr']>best+tc['min_delta_db']:
                    best=last_val['patient_macro_psnr'];bad_evals=0;checkpoint('best.pt')
        checkpoint('last.pt')
        result={'step':step,'reason':stop_reason,'wall_seconds':time.monotonic()-start,'timed_train_steps':timed_steps,
            'mean_train_step_seconds':train_seconds/max(timed_steps,1),'last_validation':last_val,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'convergence_claim':False,'note':'Plateau is this fixed development cohort only, not external convergence.'}
        atomic_json(out/'status.json',result);return result
    except Exception as exc:
        atomic_json(out/'failure.json',{'step':step,'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()});raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--resume');ap.add_argument('--resume-sha256');ap.add_argument('--stop-after',type=int)
    ap.add_argument('--max-seconds',type=int,default=13800);ap.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=ap.parse_args()
    if a.resume and (not a.resume_sha256 or sha(a.resume)!=a.resume_sha256):raise ValueError('Resume checkpoint hash required and must match')
    try:
        print(json.dumps(train(yaml.safe_load(open(a.config)),a.out,a.resume,a.stop_after,a.max_seconds,a.device),indent=2))
    except Exception as exc:
        target=Path(a.out)
        # Never overwrite an earlier run when the failure is an existing output.
        if not isinstance(exc,FileExistsError) and target.exists() and not (target/'failure.json').exists():
            atomic_json(target/'failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise
