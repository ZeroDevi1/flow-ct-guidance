"""Budget-limited complete validation batch; never shuts the instance down."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from experiments.budget_pilot.run_validation import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());deadline=int(os.environ['PLATFORM_DEADLINE_EPOCH'])
    if os.environ.get('PLATFORM_DEADLINE_VERIFIED')!='YES':raise ValueError('Verified platform deadline required')
    if deadline-time.time()<c['max_seconds']+120:raise ValueError('Insufficient platform time for complete batch')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);atomic_json(a.out/'batch_config.json',c);start=time.monotonic();statuses=[]
    for seed in c['seeds']:
        remain=int(c['max_seconds']-(time.monotonic()-start))
        if remain<60:raise TimeoutError('Global batch budget exhausted')
        out=a.out/f'seed_{seed}'
        cmd=[sys.executable,'-m','experiments.budget_pilot.run_p1_validation','--processed',c['processed'],'--ckpt',c['ckpt'],
            '--out',str(out),'--patients',str(c['patients']),'--slices-per-patient',str(c['slices_per_patient']),
            '--seed',str(seed),'--nfe',str(c['nfe']),'--cg-iters',str(c['cg_iters']),
            '--max-seconds',str(remain-30),'--cases',*c['cases'],'--lambdas',*map(str,c['dc_lambdas']),
            '--wls-lambdas',*map(str,c['wls_lambdas'])]
        with open(a.out/f'seed_{seed}.log','w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=remain)
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else dict(complete=False)
        statuses.append(dict(seed=seed,exit_code=result.returncode,complete=status['complete'],wall_seconds=status.get('wall_seconds')))
        atomic_json(a.out/'batch_status.json',dict(complete=False,seeds=statuses,wall_seconds=time.monotonic()-start))
        if result.returncode or not status['complete']:raise RuntimeError('Seed failed or incomplete; no partial-cohort comparison')
    atomic_json(a.out/'batch_status.json',dict(complete=True,seeds=statuses,wall_seconds=time.monotonic()-start,
        scope='Development validation only; paired methods within seed, not independent holdout results'))

if __name__=='__main__':
    import traceback
    try:main()
    except Exception as exc:
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
            failure=out/'batch_failure.json'
            if not failure.exists():atomic_json(failure,dict(complete=False,error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc()))
        raise
