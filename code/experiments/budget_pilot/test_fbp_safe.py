import csv
import numpy as np
import pytest
import torch
from experiments.budget_pilot.train_fbp_safe import train,prepare_data

def fixture_data(tmp):
    root=tmp/'data';root.mkdir()
    for split,pid,count in [('train','tr',4),('val','va',2),('test','te',1)]:
        with open(root/f'index_{split}.csv','w') as f:
            w=csv.writer(f);w.writerow(['patient_id','z']);w.writerows([(pid,i) for i in range(count)])
        if split!='test':np.save(root/f'slices_{split}.npy',np.random.default_rng(3).uniform(-1,1,(count,16,16)).astype('float32'))
    return root

def config(root):
    return {'seed':4,'data':{'processed_dir':str(root)},'physics':{'n_angles':6,'i0':None},
        'model':{'in_ch':1,'out_ch':1,'base_ch':8,'ch_mult':[1,2],'num_res':1,'attn_res':[],'image_size':16},
        'validation':{'patients':1,'slices_per_patient':2,'selection_seed':1,'noise_seed':0},
        'train':{'batch_size':2,'lr':.0002,'lr_min':.00001,'warmup':1,'max_steps':4,'eval_every':2,'save_every':1,'min_steps':4,'patience':10,'min_delta_db':.01}}

def test_resume_matches_uninterrupted_and_no_test_array(tmp_path):
    root=fixture_data(tmp_path);cfg=config(root)
    train(cfg,tmp_path/'first',stop_after=2,max_seconds=120,device='cpu')
    a=train(cfg,tmp_path/'resumed',resume=tmp_path/'first/last.pt',max_seconds=120,device='cpu')
    b=train(cfg,tmp_path/'full',max_seconds=120,device='cpu')
    ca=torch.load(tmp_path/'resumed/last.pt',weights_only=False);cb=torch.load(tmp_path/'full/last.pt',weights_only=False)
    assert a['step']==b['step']==4
    for k in ca['model']:assert torch.equal(ca['model'][k],cb['model'][k]),k
    assert not (root/'slices_test.npy').exists()
    assert a['last_validation']['complete']
    with pytest.raises(FileExistsError):train(cfg,tmp_path/'full',max_seconds=120,device='cpu')

def test_overlap_rejected(tmp_path):
    root=fixture_data(tmp_path)
    (root/'index_test.csv').write_text('patient_id,z\nva,1\n')
    with pytest.raises(ValueError):prepare_data(root,16,1,2,1)


def test_requested_patient_count_not_silently_reduced(tmp_path):
    root=fixture_data(tmp_path)
    with pytest.raises(ValueError,match='Requested 2 validation patients but only 1'):
        prepare_data(root,16,2,2,1)
