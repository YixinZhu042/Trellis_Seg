"""Launch VAE training in guarded Remote API sessions (no TAIJI/SSH/nohup).

Default is an offline dry run. --launch dispatches all nodes; --status reads the
most recent launch. API submission is never reported as completed training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vae_v2.config import atomic_json, default_run_dir, load_config, storage_path

API_ROOT = Path(os.environ.get('TRELLIS_REMOTE_API_ROOT', str(ROOT.parent / 'remote_api')))
PYTHON = os.environ.get('TRELLIS_PYTHON', sys.executable)
NODES = [n.strip() for n in os.environ.get('TRELLIS_REMOTE_NODES', 'node0,node1,node2').split(',')]
BOARD = os.environ.get('TRELLIS_REMOTE_BOARD', '')


def remote_worker_prefix():
    """Forward only non-secret deployment settings to fresh remote shells."""
    values={'TRELLIS_REMOTE_API_ROOT':str(API_ROOT),'TRELLIS_PYTHON':PYTHON,
            'TRELLIS_REMOTE_NODES':','.join(NODES),'TRELLIS_REMOTE_BOARD':BOARD}
    return ['env',*[f'{k}={v}' for k,v in values.items()],'python3','-B','-u']


def api_client(api_root, server):
    if not BOARD or len(NODES)!=3 or len(set(NODES))!=3 or any(not n or n in ('node0','node1','node2') for n in NODES):
        raise ValueError('Configure TRELLIS_REMOTE_BOARD and TRELLIS_REMOTE_NODES (three comma-separated hosts)')
    if not (Path(api_root)/'client/remote_api_client.py').is_file():
        raise ValueError('Configure TRELLIS_REMOTE_API_ROOT to your installed Remote API SDK')
    sys.path.insert(0, str(api_root))
    from client.remote_api_client import RemoteAPIClient
    return RemoteAPIClient(server_name=server, timeout=60)


def node_command(job, config, output, logs, cache, python, nodes, port, steps,
                 resume_from=None, resume=False, manifest_workers=1):
    """Quote every argument; runtime caches stay on the designated storage."""
    node_cache = cache / f'node{job["rank"]}'
    env = {
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS': '4',
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'NCCL_SOCKET_IFNAME': 'bond1', 'GLOO_SOCKET_IFNAME': 'bond1',
        'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'WARN',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
        'XDG_CACHE_HOME': str(node_cache / 'xdg'),
        'HF_HOME': str(node_cache / 'huggingface'),
        'TORCH_HOME': str(node_cache / 'torch'),
        'TORCH_EXTENSIONS_DIR': str(node_cache / 'torch_extensions'),
        'TORCHINDUCTOR_CACHE_DIR': str(node_cache / 'inductor'),
        'TRITON_CACHE_DIR': str(node_cache / 'triton'),
        'CUDA_CACHE_PATH': str(node_cache / 'cuda'),
        'NUMBA_CACHE_DIR': str(node_cache / 'numba'),
        'FLEX_GEMM_AUTOTUNE_CACHE_PATH': str(node_cache / 'flex_gemm.json'),
        # Explicit GEMM needs no shared-file autotuner writes from eight ranks.
        'FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE': '0',
        'TMPDIR': str(node_cache / 'tmp'),
    }
    args = [python, '-B', '-u', '-m', 'torch.distributed.run',
            '--nnodes', str(len(nodes)), '--nproc-per-node', '8',
            '--node-rank', str(job['rank']), '--master-addr', nodes[0],
            '--master-port', str(port), '--max-restarts', '0',
            str(ROOT / 'scripts/train_seg_vae_v2.py'), '--config', str(config),
            '--output_dir', str(output), '--log_dir', str(logs), '--steps', str(steps)]
    if resume_from:
        args += ['--resume_from', str(resume_from)]
    elif resume:
        args += ['--resume']
    if manifest_workers != 1:
        args += ['--manifest_workers', str(manifest_workers)]
    cache_dirs = [v for k, v in env.items() if k.endswith('_DIR') or k in
                  ('XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'TMPDIR')]
    train = shlex.join(['env', *[f'{k}={v}' for k, v in env.items()], *args])
    # Direct, inspectable commands: Remote API disallows shell -c wrappers.
    # This runs in the session's existing shell, in the foreground.
    return (f'cd {shlex.quote(str(ROOT))} && '
            f'{shlex.join(["mkdir", "-p", *cache_dirs])} && '
            f'{train} > {shlex.quote(job["log"])} 2>&1; '
            'vae_exit=$?; '
            f'printf "%s\\n" "$vae_exit" > {shlex.quote(job["exit_file"])}')


def probe_command(python, config, cfg, output, port, resume_from=None, resume=False):
    paths = [str(ROOT), str(config), cfg['prepared_dir'] + '/manifest.json',
             cfg['encoder_init'], cfg['decoder_init'], cfg['vae_config']]
    if resume_from:
        paths.append(str(resume_from))
    if resume:
        paths.append(str(output/'trainer_latest.pt'))
    code = f'''import json,os,socket,sys
from pathlib import Path
import torch
sys.path.insert(0,{str(ROOT.parent / 'TRELLIS.2')!r})
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder
import flex_gemm
paths={paths!r}
assert all(Path(p).exists() for p in paths), 'Missing required path'
assert torch.cuda.is_available() and torch.cuda.device_count()==8, 'Expected eight CUDA GPUs'
assert not (os.statvfs({cfg['storage_root']!r}).f_flag & os.ST_RDONLY), 'Storage is read-only'
out=Path({str(output)!r})
assert {resume!r} or not out.exists() or not any(out.iterdir()), 'Output is not empty; refuse to overwrite'
duplicates=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  if any(a.endswith(b'/train_seg_vae_v2.py') or a==b'train_seg_vae_v2.py' for a in args): duplicates.append(p.name)
 except OSError: pass
assert not duplicates, 'Existing VAE trainer PIDs: '+str(duplicates)
sock=socket.socket(); sock.bind(('',{port})); free_port=sock.getsockname()[1]; sock.close()
print('VAE_PREFLIGHT='+json.dumps({{'torch':torch.__version__,'gpus':torch.cuda.device_count(),'free_port':free_port,'paths_ok':True}}))
'''
    return shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1',
                       'FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0', python, '-B', '-c', code])


def read_probe(result):
    if result.get('exit_code') != 0:
        raise RuntimeError(f'Remote preflight failed: {result}')
    rows = [s for s in result['output'].splitlines() if s.startswith('VAE_PREFLIGHT=')]
    if len(rows) != 1:
        raise RuntimeError(f'No unique preflight result: {result}')
    return json.loads(rows[0].split('=', 1)[1])


def dispatch(client, plan, path):
    """Persist IDs before dispatch and execute immediately after each create."""
    try:
        for job in plan['jobs']:
            session = client.create_session(job['ip'], job['session_name'])
            job['session_id'] = session['id']
            job['session_confirmed'] = session.get('confirmed')
            atomic_json(path, plan)
            if not session.get('write_guard', True):
                raise RuntimeError('Refusing unguarded session')
            result = client.exec_command(session['id'], job['command'], timeout=10)
            job['submission'] = result
            atomic_json(path, plan)
            if result.get('exit_code') not in (None, 0):
                raise RuntimeError(f'Command submission failed: {result}')
            print(f'Submitted node{job["rank"]}: {session["id"]}', flush=True)
        plan['status'] = 'submitted_not_yet_verified'
        atomic_json(path, plan)
    except BaseException as exc:
        plan['status'] = 'submission_failed'
        plan['error'] = str(exc)
        # Cleanup only sessions whose IDs this invocation just created.
        for job in plan['jobs']:
            if 'session_id' in job:
                try:
                    job['cleanup'] = client.delete_session(job['session_id'])
                except Exception as cleanup_exc:
                    job['cleanup_error'] = str(cleanup_exc)
        atomic_json(path, plan)
        raise


def report_status(client, path):
    plan = json.loads(path.read_text())
    active = {s['id'] for s in client.list_sessions(board_id=plan['board'])}
    print(f'Launch plan: {path}')
    print(f'Recorded launch state: {plan["status"]} (current sessions/steps below)')
    for job in plan['jobs']:
        exit_path = Path(job['exit_file'])
        print(json.dumps({'node': job['rank'], 'ip': job['ip'],
                          'session_active': job.get('session_id') in active,
                          'exit_code': exit_path.read_text().strip() if exit_path.exists() else None,
                          'log': job['log']}))
        if Path(job['log']).exists():
            print('\n'.join(Path(job['log']).read_text(errors='replace').splitlines()[-4:]))
    ranks = []
    for path in sorted(Path(plan['log_dir']).glob('train_rank*.jsonl')):
        # Read only the tail; a concurrent final partial JSON line is ignored.
        with path.open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            lines = handle.read().splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
                ranks.append({'rank': path.stem, 'step': row['step'],
                              'loss': row['loss']['loss'],
                              'finite': all(math.isfinite(v) for v in row['loss'].values())
                                        and math.isfinite(row['grad_norm'])})
                break
            except (ValueError, KeyError):
                continue
    print(json.dumps({'ranks_with_steps': len(ranks), 'ranks': ranks}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--status', action='store_true')
    ap.add_argument('--config', type=Path, default=ROOT/'configs/seg_vae_v2_parts64.json')
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--log-dir', type=Path)
    ap.add_argument('--plan', type=Path, help='Specific launch plan for --status')
    ap.add_argument('--python', default=PYTHON)
    ap.add_argument('--api-root', type=Path, default=API_ROOT)
    ap.add_argument('--server', default='master')
    resume_mode = ap.add_mutually_exclusive_group()
    resume_mode.add_argument('--resume-from', type=Path, help='Fork full state and enable label permutation in a fresh output directory')
    resume_mode.add_argument('--resume', action='store_true', help='Resume this experiment without changing its settings')
    ap.add_argument('--manifest-workers', type=int, default=1)
    ap.add_argument('--master-port', type=int, default=0, help='0 selects a free port during preflight')
    args = ap.parse_args()
    if not 0 <= args.master_port < 65536:
        ap.error('master-port must be 0..65535')
    if args.manifest_workers < 1:
        ap.error('manifest-workers must be positive')
    cfg = load_config(args.config)
    output = storage_path(args.output_dir, cfg) if args.output_dir else default_run_dir(args.config, cfg)
    if args.resume_from:
        args.resume_from = storage_path(args.resume_from, cfg)
        if output == args.resume_from.parent:
            ap.error('resume-from requires a new output directory')
    logs = (args.log_dir or ROOT/'logs'/output.name).resolve()
    cache = storage_path(Path(cfg['storage_root'])/'runtime_cache'/output.name, cfg)
    if args.status:
        plans = sorted(logs.glob('remote_launch_*.json'))
        if not args.plan and not plans:
            ap.error('No Remote API launch plan found')
        report_status(api_client(args.api_root, args.server), args.plan or plans[-1])
        return
    token = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    plan = {'created_at': datetime.now().astimezone().isoformat(), 'board': BOARD,
            'server': args.server, 'config': str(args.config.resolve()),
            'output_dir': str(output), 'log_dir': str(logs), 'cache_dir': str(cache),
            'world_size': 24, 'steps': cfg['train']['steps'], 'status': 'planned', 'jobs': []}
    plan.update(resume_from=str(args.resume_from) if args.resume_from else None,
                resume=args.resume, manifest_workers=args.manifest_workers,
                label_permutation=cfg['train'].get('label_permutation', 'none'))
    for rank, ip in enumerate(NODES):
        plan['jobs'].append({'rank': rank, 'ip': ip,
                             'session_name': f'vae64_{token}_n{rank}',
                             'log': str(logs/f'launcher_{token}_node{rank}.log'),
                             'exit_file': str(logs/f'exit_{token}_node{rank}.txt')})
    def commands(port):
        plan['master_port'] = port
        for job in plan['jobs']:
            job['command'] = node_command(job, args.config.resolve(), output, logs, cache,
                                          args.python, NODES, port, cfg['train']['steps'],
                                          args.resume_from, args.resume, args.manifest_workers)
    if not args.launch:
        commands(args.master_port or 29500)
        print(json.dumps(plan, indent=2))
        print('Dry run only. Use --launch to check the cluster and submit.')
        return
    logs.mkdir(parents=True, exist_ok=True)
    # Serialize launch requests; additionally check output, sessions and remote PIDs.
    with (logs/'.remote_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = api_client(args.api_root, args.server)
        board = next((b for b in client.list_boards() if b['id'] == BOARD), None)
        if not board:
            raise RuntimeError('Authorized board not found; will not change board configuration')
        machines = {m['ip']: m for m in board['machines']}
        if any(ip not in machines or machines[ip]['status'] != 'online'
               or machines[ip]['gpu_count'] != 8 for ip in NODES):
            raise RuntimeError('All three configured nodes must be online with eight GPUs')
        if any(s.get('session_name', '').startswith('vae64_') for s in client.list_sessions(board_id=BOARD)):
            raise RuntimeError('An existing VAE64 session was found; use --status before launching again')
        def probe(item):
            rank, ip = item
            command = probe_command(args.python, args.config.resolve(), cfg, output,
                                    args.master_port if rank == 0 else 0,args.resume_from,args.resume)
            return read_probe(api_client(args.api_root, args.server).exec_sync(ip, command, timeout=50))
        with ThreadPoolExecutor(max_workers=3) as pool:
            checks = list(pool.map(probe, enumerate(NODES)))
        plan['preflight'] = dict(zip(NODES, checks))
        commands(checks[0]['free_port'])
        # Use the installed API's own validator before creating any session.
        from common.security import check_agent_command
        for job in plan['jobs']:
            safe, reason = check_agent_command(job['command'], protected_dirs=board.get('agent_readonly_dirs', []))
            if not safe:
                raise RuntimeError(f'Remote API command validation failed: {reason}')
        path = logs/f'remote_launch_{token}.json'
        atomic_json(path, plan)
        print(f'Preflight passed. Launch plan: {path}', flush=True)
        dispatch(client, plan, path)
        print('All commands submitted. Check --status for actual training progress.', flush=True)


if __name__ == '__main__':
    main()
