import csv,math
import numpy as np
import pytest
import torch
from src.physics.radon import CTOperator
from src.sample_posterior import posterior_sample
from experiments.budget_pilot.methods import sample_dc, measured_wls_weights
from experiments.budget_pilot.run_validation import select_validation

class Zero(torch.nn.Module):
    def forward(self,x,t):return x*.1

def test_sampler_matches_existing_dc():
    op=CTOperator(16,12);x=torch.zeros(1,1,16,16);y=op(x);model=Zero()
    a=sample_dc(model,op,y,x.shape,nfe=4,lam_relative=.3,n_iter=5,seed=12)
    b=posterior_sample(model,op,y,x.shape,'dc',4,dc_lam=.3,dc_iters=5,generator=torch.Generator().manual_seed(12))
    c=sample_dc(model,op,y,x.shape,nfe=4,lam_relative=.3,n_iter=5,seed=12,weights=torch.ones_like(y))
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.allclose(a,c,atol=1e-6)

def test_invalid_inputs():
    y=torch.zeros(1,1,3,4)
    for bad in [0,-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):measured_wls_weights(y,bad)

def test_selection_and_overlap(tmp_path):
    def write(split,rows):
        with open(tmp_path/f'index_{split}.csv','w') as f:
            w=csv.writer(f);w.writerow(['patient_id','z']);w.writerows(rows)
    write('val',[(p,z) for p in ['v1','v2','v3'] for z in range(4)])
    write('train',[('tr1',1)]);write('test',[('te1',1)])
    a,n=select_validation(tmp_path,2,2,17);b,_=select_validation(tmp_path,2,2,17)
    assert a==b and n==12 and len(a)==4 and len({r['patient_id'] for r in a})==2
    write('test',[('v1',1)])
    with pytest.raises(ValueError):select_validation(tmp_path,2,2,17)

def test_finite_guard_records_failure(tmp_path):
    import json
    from experiments.budget_pilot.run_validation import require_finite
    require_finite(tmp_path,{},pred=torch.ones(2))
    with pytest.raises(FloatingPointError):require_finite(tmp_path,{'method':'dc'},pred=torch.tensor([float('nan')]))
    status=json.loads((tmp_path/'status.json').read_text())
    assert not status['complete'] and status['failed_fields']==['pred']

def test_partial_groups_not_comparable():
    from experiments.budget_pilot.run_validation import summarize_group,comparison_status
    chosen=[dict(patient_id='a',array_index=0),dict(patient_id='b',array_index=1)]
    rows=[r|dict(psnr=20.+i,ssim=.7) for i,r in enumerate(chosen)]
    partial=summarize_group(rows[:1],chosen,setting='sv60')
    assert not partial['complete'] and 'patient_macro_psnr' not in partial
    assert not comparison_status([partial],[1.])['sv60']['eligible_for_comparison']
    full=summarize_group(rows,chosen,setting='sv60')
    assert full['complete'] and full['patient_macro_psnr']==20.5
    assert comparison_status([full],[1.])['sv60']['eligible_for_comparison']
    assert not comparison_status([full],[1.,3.])['sv60']['eligible_for_comparison']
