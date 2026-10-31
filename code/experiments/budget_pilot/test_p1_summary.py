import json,pytest
from experiments.budget_pilot.summarize_p1_private import summarize

def test_aggregator_uses_patients_not_replicates(tmp_path):
    def write(p,x):p.write_text(json.dumps(x))
    write(tmp_path/'batch_config.json',dict(seeds=[0,1],dc_lambdas=[3.],wls_lambdas=[.3],nfe=50,cg_iters=30,cases=['ld180_i1e5']))
    write(tmp_path/'batch_status.json',dict(complete=True,wall_seconds=1.))
    for seed in [0,1]:
        root=tmp_path/f'seed_{seed}';root.mkdir();write(root/'status.json',dict(complete=True))
        write(root/'manifest.json',dict(selected=[dict(patient_id=p,array_index=i) for i,p in enumerate(['a','b'])],val_array_sha256='synthetic'))
        rows=[]
        for i,p in enumerate(['a','b']):
            for method,lam in [('dc',3.),('wls',.3)]:
                rows.append(dict(patient_id=p,array_index=i,method=method,lambda_relative=lam,setting='ld180_i1e5',psnr=20+(i+1 if method=='wls' else 0),ssim=.8,fbp_psnr=18.,fbp_ssim=.7,seconds=.1,weighted_residual_rms=.2,residual=.1,count_floor_fraction=0.))
        (root/'per_image.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    result=summarize(tmp_path)
    assert result['paired_wls_minus_dc_psnr']['patient_mean_db']==1.5
    assert result['n_patients']==2 and result['n_method_image_evaluations']==8
    r=tmp_path/'seed_1'/'per_image.jsonl';r.write_text(r.read_text().split('\n')[0])
    with pytest.raises(ValueError):summarize(tmp_path)
