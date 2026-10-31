import json,copy
import pytest
import torch
from experiments.budget_pilot.eval_fbp_safe import predict_residual,anonymous_summary,load_p1,validate_checkpoint,evaluate

class Constant(torch.nn.Module):
    def __init__(self,value):super().__init__();self.value=value
    def forward(self,x,t):return torch.full_like(x,self.value)

def test_residual_raw_finite_then_clip():
    x=torch.full((1,1,4,4),.9)
    assert torch.equal(predict_residual(Constant(.2),x),torch.ones_like(x))
    assert torch.allclose(predict_residual(Constant(-.2),x),torch.full_like(x,.7))
    with pytest.raises(FloatingPointError):predict_residual(Constant(float('inf')),x)

def fixtures():
    cohort={(f'PRIVATE_PATIENT_{p}',2*p+i) for p in range(20) for i in range(2)};ev={};p1={}
    for seed in [0,1,2]:
        for p,i in cohort:
            ev[(seed,p,i)]={'psnr':30.,'ssim':.9,'fbp_psnr':20.,'fbp_ssim':.7,'residual':.01,'weighted_residual_rms':.1,'seconds':.01}
            for m,ps in [('dc',29.),('wls',31.)]:p1[(seed,p,i,m)]={'psnr':ps,'ssim':.8,'fbp_psnr':20.,'fbp_ssim':.7}
    return cohort,ev,p1

def test_complete_patient_summary_no_ids_and_incomplete_rejected():
    cohort,ev,p1=fixtures();result=anonymous_summary(ev,cohort,[0,1,2],p1)
    assert result['n_patients']==20 and result['n_fbpconvnet_predictions']==120
    assert result['paired_psnr']['fbpconvnet_minus_dc']['patient_mean_db']==1.
    assert result['paired_psnr']['fbpconvnet_minus_wls']['exploratory_patient_bootstrap95']==[-1.,-1.]
    assert 'PRIVATE_PATIENT' not in json.dumps(result) and 'patient_id' not in json.dumps(result) and 'array_index' not in json.dumps(result)
    ev.pop(next(iter(ev)))
    with pytest.raises(ValueError,match='Incomplete'):anonymous_summary(ev,cohort,[0,1,2],p1)

def test_mismatched_regenerated_fbp_rejected():
    cohort,ev,p1=fixtures();next(iter(ev.values()))['fbp_psnr']=19.
    with pytest.raises(ValueError,match='FBP mismatch'):anonymous_summary(ev,cohort,[0,1,2],p1)

def test_checkpoint_protocol_and_cohort():
    cohort,_,_=fixtures();hashes={'slices_val.npy':'a','index_val.csv':'b'}
    ck={'format':'flowct_fbp_safe_v1','step':500,'config':{'physics':{'n_angles':180,'i0':1e5,'sigma':0.,'mu_scale':.06},
        'validation':{'patients':20,'slices_per_patient':2,'selection_seed':20261031}},'validation_selection':[{'patient_id':p,'array_index':i} for p,i in cohort],
        'data_sha256':hashes,'last_val':{'complete':True,'step':500}}
    validate_checkpoint(ck,cohort,hashes)
    bad=copy.deepcopy(ck);bad['config']['physics']['n_angles']=60
    with pytest.raises(ValueError,match='physics'):validate_checkpoint(bad,cohort,hashes)
    bad=copy.deepcopy(ck);bad['validation_selection'].pop()
    with pytest.raises(ValueError,match='cohort'):validate_checkpoint(bad,cohort,hashes)

def test_hash_checked_before_unpickle(tmp_path,monkeypatch):
    ck=tmp_path/'best.pt';ck.write_bytes(b'not a checkpoint')
    def forbidden(*a,**kw):raise AssertionError('must not deserialize mismatched hash')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='SHA256'):evaluate(tmp_path,ck,'0'*64,tmp_path,tmp_path/'out',device='cpu')

def test_p1_incomplete_rows_rejected_even_with_complete_flag(tmp_path):
    c,ev,p1=fixtures();config={'seeds':[0,1,2],'patients':20,'slices_per_patient':2,'cases':['ld180_i1e5'],
        'dc_lambdas':[3.],'wls_lambdas':[.3],'nfe':50,'cg_iters':30}
    (tmp_path/'batch_config.json').write_text(json.dumps(config))
    (tmp_path/'batch_status.json').write_text(json.dumps({'complete':True,'seeds':[{'seed':s,'complete':True,'exit_code':0} for s in [0,1,2]]}))
    for seed in [0,1,2]:
        root=tmp_path/f'seed_{seed}';root.mkdir();(root/'status.json').write_text('{"complete":true}')
        manifest={'selected':[{'patient_id':p,'array_index':i} for p,i in c],'val_array_sha256':'a','val_index_sha256':'b',
            'settings':{'seed':seed,'patients':20,'slices_per_patient':2,'cases':['ld180_i1e5'],'lambdas':[3.],'wls_lambdas':[.3],'nfe':50,'cg_iters':30}}
        (root/'manifest.json').write_text(json.dumps(manifest))
        rr=[]
        for (s,p,i,m),r in p1.items():
            if s==seed:rr.append(r|{'patient_id':p,'array_index':i,'method':m,'setting':'ld180_i1e5','lambda_relative':3. if m=='dc' else .3})
        if seed==2:rr.pop()
        (root/'per_image.jsonl').write_text('\n'.join(json.dumps(r) for r in rr))
    with pytest.raises(ValueError,match='Incomplete P1 paired'):load_p1(tmp_path)
