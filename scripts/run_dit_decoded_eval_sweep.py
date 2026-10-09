"""Evaluate the completed paired DiT experiment without any parameter updates.

Validation phase: original model plus steps 100/200/300 in each arm, seed 42.
Select each arm independently by validation projection mIoU (surface mIoU then
earlier step break ties). Test phase: original and the two selected checkpoints,
using three fixed seeds; test results never affect checkpoint selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['validation', 'test'], required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--validation_dir', type=Path)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or not gpus:
        ap.error('Use distinct GPUs')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Use a new output directory')
    jobs = [{'name': 'pretrained', 'checkpoint': None}]
    selection = None
    if args.phase == 'validation':
        for arm in ['flow', 'decoded']:
            for step in [100, 200, 300]:
                checkpoint = ROOT / f'experiments/dit_{arm}_ddp_v1/dit_step{step:06d}.pt'
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                jobs.append({'name': f'{arm}_{step}', 'checkpoint': str(checkpoint), 'step': step, 'arm': arm})
        seeds = [42]
    else:
        if args.validation_dir is None:
            ap.error('Test phase needs --validation_dir')
        validation = json.loads((args.validation_dir / 'sweep.json').read_text())
        if validation['failures'] or validation['phase'] != 'validation':
            raise ValueError('Validation sweep is incomplete')
        selection = {}
        for arm in ['flow', 'decoded']:
            candidates = [x for x in validation['results'] if x.get('arm') == arm]
            if len(candidates) != 3:
                raise ValueError('Expected all 3 validation checkpoints per arm')
            chosen = max(candidates, key=lambda x: (x['summary']['projection']['miou'],
                                                    x['summary']['surface']['miou'], -x['step']))
            selection[arm] = {'checkpoint': chosen['checkpoint'], 'step': chosen['step'],
                              'validation_summary': chosen['summary']}
            jobs.append({'name': arm, 'checkpoint': chosen['checkpoint'], 'step': chosen['step'], 'arm': arm})
        seeds = [42, 123, 2026]
    args.output_dir.mkdir(parents=True)
    protocol = {'phase': args.phase, 'seeds': seeds, 'jobs': jobs, 'selection': selection,
                'selection_rule': 'projection mIoU, then surface mIoU, then earlier step',
                'postprocessing': False, 'parameter_updates': False, 'results': [], 'failures': []}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    def worker(gpu, assigned):
        results = []
        for job in assigned:
            output = args.output_dir / job['name']
            cmd = [sys.executable, str(ROOT / 'scripts/evaluate_dit_decoded.py'),
                   '--config', str(ROOT / 'configs/dit_decoded_aux_v1.json'),
                   '--output_dir', str(output), '--split', args.phase,
                   '--device', f'cuda:{gpu}', '--seeds', *map(str, seeds)]
            if job['checkpoint']:
                cmd += ['--checkpoint', job['checkpoint']]
            start = time.monotonic()
            with (args.output_dir / f'{job["name"]}.log').open('w') as log:
                status = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result = {**job, 'gpu': gpu, 'command': cmd, 'returncode': status.returncode,
                      'seconds': time.monotonic() - start}
            if status.returncode == 0:
                metrics = json.loads((output / 'metrics.json').read_text())
                expected = 32 * len(seeds)
                if len(metrics['samples']) != expected:
                    raise ValueError(f'{job["name"]}: expected {expected} observations')
                result.update(summary=metrics['summary'], oracle_summary=metrics['oracle_summary'])
                print(f"{job['name']}: projection={result['summary']['projection']['miou']:.5f}, "
                      f"surface={result['summary']['surface']['miou']:.5f}, "
                      f"boundary={result['summary']['boundary_f1']}", flush=True)
            else:
                print(f"FAIL {job['name']}: see its log", flush=True)
            results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        for future in as_completed(futures):
            for result in future.result():
                protocol['results' if result['returncode'] == 0 else 'failures'].append(result)
            (args.output_dir / 'sweep.json').write_text(json.dumps(protocol, indent=2) + '\n')
    protocol['results'].sort(key=lambda x: x['name'])
    (args.output_dir / 'sweep.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    if protocol['failures']:
        raise RuntimeError('Evaluation failures; see sweep.json and logs')


if __name__ == '__main__':
    main()
