"""Launch a frozen evaluate_dit_vae_v2 plan through guarded Remote API sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vae_v2.config import atomic_json,digest,storage_path
from utils.dit_vae_v2 import identity
from scripts.train_seg_vae_v2_remote import api_client,API_ROOT,BOARD,NODES,PYTHON,dispatch,remote_worker_prefix
from scripts.dit_vae_v2_remote import environment


def run_node(args,plan):
    ngpu=plan['workers']//3
    cache=storage_path(Path(plan['vae_config']['storage_root'])/'runtime_cache'/args.output_dir.name/f'node{args.node_rank}',plan['vae_config'])
    env=dict(os.environ,**environment(cache))
    for key,value in environment(cache).items():
        if key.endswith('_DIR') or key in ('XDG_CACHE_HOME','HF_HOME','TORCH_HOME','CUDA_CACHE_PATH','TMPDIR'):
            Path(value).mkdir(parents=True,exist_ok=True)
    def one(local):
        rank=args.node_rank*ngpu+local
        with (args.output_dir/f'.worker{rank}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            command=[PYTHON,'-B','-u',str(ROOT/'scripts/evaluate_dit_vae_v2.py'),
                     '--output-dir',str(args.output_dir),'--worker',str(rank)]
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=str(local))
            with (args.output_dir/'logs'/f'worker{rank:02d}.log').open('a') as log:
                r=subprocess.run(command,cwd=ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(args.output_dir/'logs'/f'worker{rank:02d}_exit.json',{'worker':rank,'exit_code':r.returncode})
            print('Worker exit',rank,r.returncode,flush=True)
            return r.returncode
    with ThreadPoolExecutor(ngpu) as pool:codes=list(pool.map(one,range(ngpu)))
    if any(codes):raise RuntimeError(f'Evaluation worker failures: {codes}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',action='store_true');mode.add_argument('--node-rank',type=int)
    args=ap.parse_args();args.output_dir=args.output_dir.resolve()
    plan=json.loads((args.output_dir/'plan.json').read_text())
    if plan['workers']%3 or not 1<=plan['workers']//3<=8:raise ValueError('Expected 3 to24 workers in three equal shards')
    for k in ('checkpoint','vae_checkpoint'):
        if identity(plan[k]['path'])!=plan[k]:raise ValueError('Frozen checkpoint changed')
    if args.node_rank is not None:
        if not 0<=args.node_rank<3:ap.error('node-rank must be 0..2')
        return run_node(args,plan)
    target=args.output_dir/'remote_launch.json'
    if target.exists():raise ValueError('Launch already recorded; inspect before retry')
    client=api_client(API_ROOT,'master');board=next(b for b in client.list_boards() if b['id']==BOARD)
    nodes={n['ip']:n for n in board['machines']}
    if any(nodes[ip]['status']!='online' or nodes[ip]['gpu_count']!=8 for ip in NODES):raise ValueError('Expected three online8GPU nodes')
    if any(s['session_name'].startswith(('dit_eval_','ditcheck_','ditv2_')) for s in client.list_sessions(board_id=BOARD)):
        raise ValueError('Existing DiT session; inspect before overlapping work')
    launch={'created_at':datetime.now().astimezone().isoformat(),'plan_digest':digest(plan),'status':'planned','jobs':[]}
    tag=datetime.now().strftime('%Y%m%d_%H%M%S')
    for rank,ip in enumerate(NODES):
        log=args.output_dir/'logs'/f'node{rank}.log';ex=args.output_dir/'logs'/f'node{rank}.exit'
        cmd=shlex.join([*remote_worker_prefix(),str(Path(__file__).resolve()),'--output-dir',str(args.output_dir),'--node-rank',str(rank)])
        cmd+=' > '+shlex.quote(str(log))+' 2>&1; eval_rc=$?; printf "%s\\n" "$eval_rc" > '+shlex.quote(str(ex))
        launch['jobs'].append({'rank':rank,'ip':ip,'session_name':f'dit_eval_{tag}_n{rank}',
                              'command':cmd,'log':str(log),'exit_file':str(ex)})
    from common.security import check_agent_command
    for job in launch['jobs']:
        safe,reason=check_agent_command(job['command'],protected_dirs=board.get('agent_readonly_dirs',[]))
        if not safe:raise ValueError(reason)
    atomic_json(target,launch);dispatch(client,launch,target)
    print('LAUNCH_RECORD',target,flush=True)


if __name__=='__main__':main()
