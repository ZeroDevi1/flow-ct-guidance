import pytest,torch
from experiments.budget_pilot.run_p1_validation import comparison_status,summarize_group
from experiments.budget_pilot.methods_checked import sample_dc
from experiments.budget_pilot.methods import sample_dc as original
from src.physics.radon import CTOperator

def test_independent_equal_budget_grid():
    cohort=[dict(patient_id='x',array_index=0)]
    r=cohort[0]|dict(psnr=20.,ssim=.8)
    groups=[summarize_group([r],cohort,setting='ld180_i1e5',method=m,lambda_relative=l) for m,ls in [('dc',[1,3,10]),('wls',[.03,.1,.3])] for l in ls]
    s=comparison_status(groups,[1,3,10],['ld180_i1e5'],[.03,.1,.3])
    assert s['ld180_i1e5']['eligible_for_comparison']
    assert not comparison_status(groups[:-1],[1,3,10],['ld180_i1e5'],[.03,.1,.3])['ld180_i1e5']['eligible_for_comparison']

def test_checked_sampler_same_values_rejects_nonfinite():
    op=CTOperator(16,12);x=torch.zeros(1,1,16,16);y=op(x)
    class Model(torch.nn.Module):
        def forward(self,x,t):return x*.1
    kwargs=dict(nfe=4,lam_relative=.3,n_iter=5,seed=12)
    assert torch.equal(sample_dc(Model(),op,y,x.shape,**kwargs),original(Model(),op,y,x.shape,**kwargs))
    class Bad(torch.nn.Module):
        def forward(self,x,t):return torch.full_like(x,float('inf'))
    with pytest.raises(FloatingPointError):sample_dc(Bad(),op,y,x.shape,**kwargs)
