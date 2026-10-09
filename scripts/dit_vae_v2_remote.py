"""Remote API cache preparation / DiT training. No TAIJI or SSH; persistent guarded sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.dit_vae_v2 import config
from vae_v2.config import atomic_json,storage_path
from scripts.train_seg_vae_v2_remote import api_client,API_ROOT,NODES,PYTHON,BOARD,dispatch,remote_worker_prefix


def environment(cache):
    env={'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
         'NCCL_SOCKET_IFNAME':'bond1','GLOO_SOCKET_IFNAME':'bond1','NCCL_NVLS_ENABLE':'0',
         'NCCL_DEBUG':'WARN','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1','FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE':'0'}
    for key,name in [('XDG_CACHE_HOME','xdg'),('HF_HOME','huggingface'),('TORCH_HOME','torch'),
                     ('TORCH_EXTENSIONS_DIR','torch_extensions'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),
                     ('CUDA_CACHE_PATH','cuda'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        env[key]=str(cache/name)
    env['FLEX_GEMM_AUTOTUNE_CACHE_PATH']=str(cache/'flex_gemm.json')
    return env


def node(args,cfg,vcfg):
    cache=storage_path(Path(vcfg['storage_root'])/'runtime_cache'/Path(cfg['output_dir']).name/f'node{args.node_rank}',vcfg)
    base_env=environment(cache)
    for k,v in base_env.items():
        if k.endswith('_DIR') or k in ('XDG_CACHE_HOME','HF_HOME','TORCH_HOME','CUDA_CACHE_PATH','TMPDIR'):Path(v).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,**base_env)
    logs=Path(args.log_dir);logs.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':
        def run(local):
            rank=args.node_rank*args.gpus+local
            command=[PYTHON,'-B','-u',str(ROOT/'scripts/prepare_dit_vae_v2.py'),'--config',str(args.config),
                     '--cache-dir',cfg['cache_dir'],'--worker',str(rank)]
            with (logs/f'worker{rank:02d}.log').open('a') as f:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            atomic_json(logs/f'worker{rank:02d}_exit.json',{'rank':rank,'exit_code':result.returncode})
            print('worker exit',rank,result.returncode,flush=True)
            return result.returncode
        with ThreadPoolExecutor(max_workers=args.gpus) as pool:codes=list(pool.map(run,range(args.gpus)))
        if any(codes):raise RuntimeError(f'Preparation failures: {codes}')
    else:
        command=[PYTHON,'-B','-u','-m','torch.distributed.run','--nnodes','3','--nproc-per-node',str(args.gpus),
                 '--node-rank',str(args.node_rank),'--master-addr',NODES[0],'--master-port',str(args.port),
                 '--max-restarts','0',str(ROOT/'scripts/train_dit_vae_v2.py'),'--config',str(args.config),
                 '--cache-dir',cfg['cache_dir'],'--output-dir',cfg['output_dir'],'--log-dir',str(logs)]
        if args.steps:command+=['--steps',str(args.steps)]
        if args.smoke:command+=['--smoke']
        if args.smoke_ids:command+=['--smoke-ids',*args.smoke_ids]
        if args.resume:command+=['--resume']
        result=subprocess.run(command,cwd=ROOT,env=env)
        if result.returncode:raise SystemExit(result.returncode)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/dit_vae_v2_parts64.json')
    ap.add_argument('--phase',choices=['prepare','train'],required=True)
    ap.add_argument('--cache-dir',type=Path);ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--log-dir',type=Path);ap.add_argument('--gpus',type=int,default=8)
    ap.add_argument('--launch',action='store_true');ap.add_argument('--status',action='store_true')
    ap.add_argument('--node-rank',type=int);ap.add_argument('--port',type=int,default=0)
    ap.add_argument('--steps',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke-ids',nargs='+')
    args=ap.parse_args();args.config=args.config.resolve();cfg,vcfg=config(args.config)
    if not 1<=args.gpus<=8:ap.error('gpus must be 1..8')
    if args.smoke_ids and not args.smoke:ap.error('--smoke-ids requires --smoke')
    if args.cache_dir:cfg['cache_dir']=str(storage_path(args.cache_dir,vcfg))
    if args.output_dir:cfg['output_dir']=str(storage_path(args.output_dir,vcfg))
    name=Path(cfg['cache_dir'] if args.phase=='prepare' else cfg['output_dir']).name
    logs=(args.log_dir or ROOT/'logs'/name/args.phase).resolve();args.log_dir=logs
    if args.node_rank is not None:
        if not 0<=args.node_rank<3:ap.error('Invalid node rank')
        return node(args,cfg,vcfg)
    if args.status:
        plans=sorted(logs.glob('remote_*.json'))
        if not plans:ap.error('No launch record')
        plan=json.loads(plans[-1].read_text());client=api_client(API_ROOT,'master');active={s['id'] for s in client.list_sessions()}
        print('PLAN',plans[-1],plan['status'])
        for j in plan['jobs']:
            f=Path(j['exit_file']);print(j['ip'],'active',j.get('session_id') in active,'exit',f.read_text().strip() if f.exists() else None)
        progress=[]
        for f in sorted(logs.glob('train_rank*.jsonl')):
            with f.open('rb') as handle:
                handle.seek(0,2);handle.seek(max(0,handle.tell()-65536));lines=handle.read().splitlines()
            for line in reversed(lines):
                try:
                    r=json.loads(line)
                    progress.append({'rank':r['rank'],'step':r['step'],'loss':r['batch'][-1]['loss'],
                                     'max_gpu_gb':r['max_gpu_gb']});break
                except (ValueError,KeyError):continue
        print(json.dumps({'ranks_with_steps':len(progress),'ranks':progress},indent=2))
        return
    if args.phase=='prepare':
        p=json.loads((Path(cfg['cache_dir'])/'plan.json').read_text())
        if p['workers']!=3*args.gpus:raise ValueError('Cache plan worker count differs')
    elif not args.smoke:
        from utils.dit_vae_v2 import load_cache
        load_cache(cfg)
    if args.phase=='train' and not args.resume and Path(cfg['output_dir']).exists() and any(Path(cfg['output_dir']).iterdir()):
        raise ValueError('New training needs empty output')
    tag=datetime.now().strftime('%Y%m%d_%H%M%S');port=args.port or 29515
    plan={'phase':args.phase,'status':'planned','config':cfg,'jobs':[]}
    client=None
    if args.launch:
        client=api_client(API_ROOT,'master');board=next(b for b in client.list_boards() if b['id']==BOARD)
        machines={m['ip']:m for m in board['machines']}
        if any(machines[ip]['status']!='online' or machines[ip]['gpu_count']!=8 for ip in NODES):raise ValueError('Nodes unavailable')
        if any(s['session_name'].startswith('ditv2_') for s in client.list_sessions()):raise ValueError('Existing DiT v2 session; inspect before duplicate launch')
        if args.phase=='train' and not args.port:
            r=client.exec_sync(NODES[0],"python3 -c 'import socket; s=socket.socket(); s.bind((\"\",0)); print(s.getsockname()[1])'",timeout=10)
            if r['exit_code']!=0:raise RuntimeError(r)
            port=int(r['output'].strip())
    for rank,ip in enumerate(NODES):
        command=[*remote_worker_prefix(),str(Path(__file__).resolve()),'--config',str(args.config),'--phase',args.phase,
                 '--cache-dir',cfg['cache_dir'],'--output-dir',cfg['output_dir'],'--log-dir',str(logs),
                 '--node-rank',str(rank),'--gpus',str(args.gpus),'--port',str(port)]
        if args.steps:command+=['--steps',str(args.steps)]
        if args.smoke:command+=['--smoke']
        if args.smoke_ids:command+=['--smoke-ids',*args.smoke_ids]
        if args.resume:command+=['--resume']
        log=logs/f'{tag}_node{rank}.log';ex=logs/f'{tag}_node{rank}.exit'
        cmd=shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1; dit_rc=$?; printf "%s\\n" "$dit_rc" > '+shlex.quote(str(ex))
        plan['jobs'].append({'rank':rank,'ip':ip,'session_name':f'ditv2_{args.phase}_{tag}_n{rank}',
                             'command':cmd,'log':str(log),'exit_file':str(ex)})
    if not args.launch:print(json.dumps(plan,indent=2));return
    from common.security import check_agent_command
    for j in plan['jobs']:
        safe,reason=check_agent_command(j['command'],protected_dirs=board.get('agent_readonly_dirs',[]))
        if not safe:raise ValueError(reason)
    logs.mkdir(parents=True,exist_ok=True);path=logs/f'remote_{tag}.json';atomic_json(path,plan)
    dispatch(client,plan,path);print('LAUNCH_RECORD',path,flush=True)


if __name__=='__main__':main()
