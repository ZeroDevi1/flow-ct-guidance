"""Post-hoc read-only audit on AutoDL; export aggregates, no patient identifiers."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.budget_pilot.run_validation import atomic_json

def read_rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def audit(batch,diagnostic,processed):
    cfg=json.loads((batch/'batch_config.json').read_text());root=batch/'seed_0'
    if not json.loads((root/'status.json').read_text())['complete']:raise ValueError('Seed0 incomplete')
    manifest=json.loads((root/'manifest.json').read_text());dm=json.loads((diagnostic/'manifest_private.json').read_text())
    if manifest['val_array_sha256']!=dm['val_sha256'] or manifest['prior_sha256']!=dm['prior_sha256']:raise ValueError('Data/prior mismatch')
    selected={(r['patient_id'],r['array_index']) for r in dm['selected']}
    rows=read_rows(root/'per_image.jsonl');old=read_rows(diagnostic/'per_image_private.jsonl')
    out=dict(scope='Post-hoc same-image NFE and cohort audit; not additional tuning or confirmatory testing',comparisons={})
    for method,lam in [('dc',cfg['dc_lambdas'][0]),('wls',cfg['wls_lambdas'][0])]:
        now={(r['patient_id'],r['array_index']):r for r in rows if r['method']==method and r['lambda_relative']==lam}
        before={(r['patient_id'],r['array_index']):r for r in old if r['method']==method and r['lambda_relative']==lam and r['cg_iters']==30}
        if set(before)!=selected or not selected<=set(now):raise ValueError('Exact paired diagnostic cohort missing')
        deltas=[now[k]['psnr']-before[k]['psnr'] for k in sorted(selected)]
        prior_pids={p for p,i in selected};by=defaultdict(list)
        for (p,i),r in now.items():by[p].append(r['psnr'])
        other=[np.mean(v) for p,v in by.items() if p not in prior_pids]
        out['comparisons'][method]=dict(same4_first_slice_nfe20_psnr=float(np.mean([before[k]['psnr'] for k in selected])),
            same4_first_slice_nfe50_psnr=float(np.mean([now[k]['psnr'] for k in selected])),
            matched_nfe50_minus20_mean_db=float(np.mean(deltas)),matched_nfe_delta_range=[float(min(deltas)),float(max(deltas))],
            original4_both_slices_nfe50_patient_macro_psnr=float(np.mean([np.mean(by[p]) for p in prior_pids])),
            other16_both_slices_nfe50_patient_macro_psnr=float(np.mean(other)),
            other_patient_count=len(other),nfe50_all_patient_psnr_quantiles=[float(q) for q in np.quantile([np.mean(v) for v in by.values()],[0,.1,.25,.5,.75,.9,1])])
    arr=np.load(processed/'slices_val.npy',mmap_mode='r');indices=sorted({r['array_index'] for r in manifest['selected']})
    x=np.asarray(arr[indices]);finite=np.isfinite(x)
    out['selected_truth_range_audit']=dict(n_images=len(indices),n_nonfinite=int((~finite).sum()),minimum=float(x[finite].min()),maximum=float(x[finite].max()),
        fraction_outside_minus1_plus1=float(((x< -1)|(x>1)).mean()),fraction_at_minus1=float((x==-1).mean()),fraction_at_plus1=float((x==1).mean()),
        note='Checks stored normalized pixel values only; does not establish original HU calibration or diagnostic image quality.')
    out['limitations']=['Matched4 slice comparison changes NFE20 to50 with same seed/prior/data/CG/lambda; minor CUDA nondeterminism may remain.',
        'Other16 anatomy not labelled here; lower values alone cannot identify anatomy or pathology causes.',
        'Quantiles are development-patient summaries; no post-hoc parameter changes.']
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--processed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=audit(a.batch,a.diagnostic,a.processed);atomic_json(a.out,result);print(json.dumps(result,indent=2,allow_nan=False))
