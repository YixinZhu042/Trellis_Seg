"""Run fresh neural inference for the exact frozen evaluation cohort on local GPUs.

Uses cached input masks and the original meshes; GT is not read. Subprocesses
load the configured checkpoints and save complete outputs to a new directory.
No existing process is killed and no network/API calls are issued.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, default=ROOT / 'results_2d_consistent/report.json')
    ap.add_argument('--output_dir', required=True, type=Path)
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--decoder')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus):
        ap.error('Duplicate GPUs')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Use a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.cohort.read_text())['samples']
    if args.limit:
        rows = rows[:args.limit]
    report = {'cohort': str(args.cohort.resolve()), 'gt_used_for_predictions': False,
              'gpus': gpus, 'samples': [], 'failures': []}
    def worker(gpu, samples):
        done = []
        for sample in samples:
            uid = sample['instance']
            output = args.output_dir / f'{uid}.glb'
            cmd = [sys.executable, str(ROOT / 'scripts/inference_wild.py'),
                   '--mesh', sample['mesh'], '--seg_maps', sample['seg_maps'],
                   '--output', str(output), '--device', f'cuda:{gpu}', '--seed', '42', '--steps', '50']
            if args.decoder:
                cmd += ['--seg_decoder', args.decoder]
            start = time.monotonic()
            with (args.output_dir / f'{uid}.log').open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry = {'instance': uid, 'gpu': gpu, 'returncode': result.returncode,
                     'seconds': time.monotonic() - start, 'command': cmd,
                     'mesh': sample['mesh'], 'prediction': sample['prediction'], 'seg_maps': sample['seg_maps']}
            if result.returncode == 0:
                score = json.loads(output.with_name(uid + '_consistency.json').read_text())
                entry.update(score)
                entry['glb_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
                print(f"{uid}: GPU {gpu}, acc={score['exported']['accuracy']:.5f}, "
                      f"mIoU={score['exported']['miou']:.5f}, {entry['seconds']:.1f}s", flush=True)
            else:
                print(f'FAIL {uid}: return code {result.returncode}; see log', flush=True)
            done.append(entry)
        return done
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, rows[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        for future in as_completed(futures):
            for entry in future.result():
                report['samples' if entry['returncode'] == 0 else 'failures'].append(entry)
            (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['samples'].sort(key=lambda x: x['instance'])
    report['summary'] = {'num_samples': len(report['samples']), 'num_failures': len(report['failures'])}
    if report['samples']:
        for kind in ['baseline', 'exported']:
            report['summary'][kind] = {
                key: sum(x[kind][key] for x in report['samples']) / len(report['samples'])
                for key in ['accuracy', 'miou']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))
    if report['failures']:
        raise RuntimeError('Fresh pipeline validation had failures')


if __name__ == '__main__':
    main()
