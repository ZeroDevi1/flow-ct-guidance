"""Run on AutoDL only: emit identity-free aggregates, never export private rows."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.budget_pilot.run_validation import atomic_json

def summarize(batch):
    config=json.loads((batch/'batch_config.json').read_text());status=json.loads((batch/'batch_status.json').read_text())
    if not status['complete']:raise ValueError('Incomplete batch: do not compare')
    if len(config['dc_lambdas'])!=1 or len(config['wls_lambdas'])!=1:raise ValueError('Only frozen one-lambda-per-method analysis supported')
    samples={};cohort=None;valhash=None;timings=defaultdict(list);metrics=defaultdict(list)
    for seed in config['seeds']:
        root=batch/f'seed_{seed}';s=json.loads((root/'status.json').read_text());m=json.loads((root/'manifest.json').read_text())
        if not s['complete']:raise ValueError('Incomplete seed')
        selected={(q['patient_id'],q['array_index']) for q in m['selected']}
        if cohort is None:cohort=selected;valhash=m['val_array_sha256']
        if selected!=cohort or m['val_array_sha256']!=valhash:raise ValueError('Cohort/data mismatch')
        rows=[json.loads(line) for line in (root/'per_image.jsonl').read_text().splitlines()]
        for r in rows:
            method=r['method'];expected=config['dc_lambdas'][0] if method=='dc' else config['wls_lambdas'][0]
            key=(seed,r['patient_id'],r['array_index'],method)
            if method not in ('dc','wls') or key in samples or r['lambda_relative']!=expected or r['setting']!='ld180_i1e5':raise ValueError('Unexpected or duplicate group')
            vals={k:r[k] for k in ['psnr','ssim','fbp_psnr','fbp_ssim','seconds','weighted_residual_rms','residual','count_floor_fraction']}
            if not np.isfinite(list(vals.values())).all():raise ValueError('Nonfinite metrics')
            samples[key]=vals;timings[method].append(vals['seconds'])
        expectedkeys={(seed,p,i,method) for p,i in cohort for method in ('dc','wls')}
        if {k for k in samples if k[0]==seed}!=expectedkeys:raise ValueError('Incomplete or unmatched paired records')
    by=defaultdict(lambda:defaultdict(list));fbp=defaultdict(list)
    for seed in config['seeds']:
        for p,i in sorted(cohort):
            dc=samples[(seed,p,i,'dc')];wls=samples[(seed,p,i,'wls')]
            if abs(dc['fbp_psnr']-wls['fbp_psnr'])>1e-5:raise ValueError('Unpaired measurement or FBP')
            fbp[p].append(dc['fbp_psnr'])
            for method,r in [('dc',dc),('wls',wls)]:
                for metric in ('psnr','ssim','residual','weighted_residual_rms','count_floor_fraction'):by[(method,metric)][p].append(r[metric])
    patients=sorted({p for p,_ in cohort});aggregates={}
    for method in ('dc','wls'):
        aggregates[method]={metric:float(np.mean([np.mean(by[(method,metric)][p]) for p in patients])) for metric in ('psnr','ssim','residual','weighted_residual_rms','count_floor_fraction')}
        aggregates[method].update(seconds_mean=float(np.mean(timings[method])),seconds_p95=float(np.quantile(timings[method],.95)))
    delta=np.array([np.mean(by[('wls','psnr')][p])-np.mean(by[('dc','psnr')][p]) for p in patients])
    rng=np.random.default_rng(20261031);boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return dict(complete=True,n_patients=len(patients),n_slices_per_seed=len(cohort),n_seeds=len(config['seeds']),n_method_image_evaluations=len(samples),
        settings={k:config[k] for k in ['nfe','cg_iters','cases','dc_lambdas','wls_lambdas','seeds']},
        aggregates=aggregates,fbp_patient_macro_psnr=float(np.mean([np.mean(v) for v in fbp.values()])),
        paired_wls_minus_dc_psnr=dict(patient_mean_db=float(delta.mean()),patient_median_db=float(np.median(delta)),
            patient_min_db=float(delta.min()),patient_max_db=float(delta.max()),fraction_patients_positive=float((delta>0).mean()),
            exploratory_patient_bootstrap95=[float(v) for v in np.quantile(boot,[.025,.975])]),
        wall_seconds=status['wall_seconds'],limitations=['Development cohort includes four tuning patients; not an independent performance estimate.',
            'Bootstrap resamples patients after averaging slices and seeds; only20 patients, exploratory interval, not clinical inference.',
            'NFE50 CG30 frozen settings, no guarantee globally optimal lambdas or exact convergence.',
            'Seeds jointly vary measurement noise and initialization. No image or patient identifier exported.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=summarize(a.batch);atomic_json(a.out,result);print(json.dumps(result,indent=2,allow_nan=False))
