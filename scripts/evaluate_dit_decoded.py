"""Raw DiT evaluation on a predeclared validation/test split, without cleanup."""
import argparse
from datetime import timedelta
import faulthandler
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))
from scripts.train_dit_decoded import (load_models, records, get_sample, spatial_template,
                                     normalize, make_conditions, decode_clean, resolve_config)
from trainers.decoded_seg_losses import face_logits_from_voxels
from trellis2.pipelines.samplers import FlowEulerSampler
from utils.decoded_dit_runtime import atomic_json, distributed_topology, master_action


def classification_metrics(target, prediction, weights):
    valid = (target > 0) & (weights > 0)
    t, p, w = target[valid], prediction[valid], weights[valid]
    if not len(t):
        return {'accuracy':None,'miou':None,'small_part_recall':None}
    parts = np.unique(t)
    ious, recalls, fractions = [], [], []
    for part in parts:
        gt, pr = t==part, p==part
        intersection = w[gt & pr].sum()
        ious.append(float(intersection/w[gt | pr].sum()))
        recalls.append(float(intersection/w[gt].sum()))
        fractions.append(w[gt].sum()/w.sum())
    small = np.array(fractions) < .01
    return {'accuracy':float(w[t==p].sum()/w.sum()),'miou':float(np.mean(ious)),
            'small_part_recall':float(np.mean(np.array(recalls)[small])) if small.any() else None,
            'part_iou':{int(p):v for p,v in zip(parts,ious)}}


def boundary_f1(target, pred, adjacency):
    if not len(adjacency):return None
    left,right=adjacency.T
    valid=(target[left]>0)&(target[right]>0)
    gt=(target[left]!=target[right])&valid
    guess=(pred[left]!=pred[right])&valid
    if not gt.any():return None
    return float(2*np.count_nonzero(gt&guess)/max(int(gt.sum()+guess.sum()),1))


def fmt(value):
    return 'n/a' if value is None else f'{value:.4f}'


def summarize_metrics(rows):
    result={}
    for group in ['voxel','surface','projection']:
        result[group]={}
        for key in ['accuracy','miou','small_part_recall']:
            values=[r[group][key] for r in rows if r[group][key] is not None]
            result[group][key]=float(np.mean(values)) if values else None
    boundaries=[r['boundary_f1'] for r in rows if r['boundary_f1'] is not None]
    result['boundary_f1']=float(np.mean(boundaries)) if boundaries else None
    return result


def merge_results(args, rank, world, full_cohort, report=None):
    def merge():
        base = report or {'checkpoint': str(args.checkpoint) if args.checkpoint else None,
                          'split': args.split, 'seeds': args.seeds,
                          'postprocessing': False,
                          'label_matching': 'fixed input IDs, no permutation',
                          'save_predictions': not args.no_save_predictions}
        merged = dict(base, rank=None, world_size=world, samples=[], oracle=[])
        for shard in range(world):
            path = args.output_dir / f'metrics_rank{shard:04d}.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text())
            merged['samples'].extend(data['samples'])
            merged['oracle'].extend(data['oracle'])
        actual = {(row['instance'], row['seed']) for row in merged['samples']}
        expected = {(row['instance'], seed) for row in full_cohort for seed in args.seeds}
        oracle_ids = [row['instance'] for row in merged['oracle']]
        if actual != expected or len(actual) != len(merged['samples']):
            raise ValueError('Evaluation shards are incomplete or duplicated')
        if set(oracle_ids) != {row['instance'] for row in full_cohort} or len(oracle_ids) != len(set(oracle_ids)):
            raise ValueError('Oracle shards are incomplete or duplicated')
        merged['samples'].sort(key=lambda row: (row['instance'], row['seed']))
        merged['oracle'].sort(key=lambda row: row['instance'])
        merged['summary'] = summarize_metrics(merged['samples'])
        merged['oracle_summary'] = summarize_metrics(merged['oracle'])
        atomic_json(args.output_dir / 'metrics.json', merged)
        print(json.dumps(merged['summary'], indent=2), flush=True)
    master_action(merge, rank, world > 1 and not args.independent_worker)


def evaluate(args, rank, world):
    device = args.device
    def setup():
        if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError('Use a new output directory')
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.independent_worker:
        setup()
    else:
        master_action(setup, rank, world > 1)
    config = resolve_config(args.config)
    if args.checkpoint:
        config['dit_ckpt'] = str(args.checkpoint.resolve())
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    full_cohort = sorted([r for r in records(config['data_dir']) if r['split'] == args.split], key=lambda r: r['instance'])
    if args.limit:
        full_cohort = full_cohort[:args.limit]
    cohort = full_cohort[rank::world]
    shard_path = args.output_dir / f'metrics_rank{rank:04d}.json'
    if args.resume and shard_path.is_file():
        report = json.loads(shard_path.read_text())
        expected_header = (config['dit_ckpt'], args.split, args.seeds, rank, world)
        actual_header = (report['checkpoint'], report['split'], report['seeds'],
                         report['rank'], report['world_size'])
        if actual_header != expected_header:
            raise ValueError(f'Resume shard header mismatch: {actual_header} != {expected_header}')
    else:
        report = {'checkpoint': config['dit_ckpt'], 'split': args.split, 'seeds': args.seeds,
                  'postprocessing': False, 'label_matching': 'fixed input IDs, no permutation',
                  'save_predictions': not args.no_save_predictions,
                  'conv_algorithm': args.conv_algorithm,
                  'rank': rank, 'world_size': world, 'samples': [], 'oracle': [],
                  'split_note': 'Excluded from this continuation, may have been seen during original pretraining.'}
    sampler = FlowEulerSampler(sigma_min=config['sigma_min'])
    done = {(row['instance'], row['seed']) for row in report['samples']}
    oracle_done = {row['instance'] for row in report['oracle']}
    progress_path = args.output_dir / f'progress_rank{rank:04d}.json'
    pending = [row for row in cohort if row['instance'] not in oracle_done or
               any((row['instance'], seed) not in done for seed in args.seeds)]
    if not pending:
        print(f'[rank {rank}] shard already complete; skipping model load', flush=True)
        if args.no_merge:
            return
        if world > 1:
            dist.barrier()
        merge_results(args, rank, world, full_cohort, report)
        return
    model, encoder, decoder, base = load_models(
        config, device, False, conv_algorithm=args.conv_algorithm)
    model.eval()
    for row in cohort:
        if row['instance'] in oracle_done and all((row['instance'], seed) in done for seed in args.seeds):
            continue
        atomic_json(progress_path, {'rank': rank, 'instance': row['instance'],
                                    'phase': 'load_sample', 'updated': time.time()})
        if args.watchdog_seconds:
            faulthandler.dump_traceback_later(args.watchdog_seconds, repeat=False)
        sample = get_sample(row, device)
        template = spatial_template(sample, encoder)
        normalized = normalize(sample, base)
        target = sample['voxel_labels'].cpu().numpy()
        face_target = sample['face_labels'].cpu().numpy()
        area = sample['face_area'].cpu().numpy()
        pixels = sample['face_pixels'].cpu().numpy()
        adjacency = sample['adjacency'].cpu().numpy()
        if row['instance'] not in oracle_done:
            atomic_json(progress_path, {'rank': rank, 'instance': row['instance'],
                                        'phase': 'oracle_decode', 'updated': time.time()})
            oracle_logits = decode_clean(decoder, template, normalized['latent_feats'], sample, base)
            oracle_faces = face_logits_from_voxels(
                oracle_logits, sample['face_knn'], sample['face_knn_weights']).argmax(1).cpu().numpy()
            report['oracle'].append({'instance': row['instance'],
                'voxel': classification_metrics(target, oracle_logits.argmax(1).cpu().numpy(), np.ones(len(target))),
                'surface': classification_metrics(face_target, oracle_faces, area),
                'projection': classification_metrics(face_target, oracle_faces, pixels),
                'boundary_f1': boundary_f1(face_target, oracle_faces, adjacency)})
            oracle_done.add(row['instance'])
            atomic_json(shard_path, report)
        for seed in args.seeds:
            if (row['instance'], seed) in done:
                continue
            atomic_json(progress_path, {'rank': rank, 'instance': row['instance'],
                                        'seed': seed, 'phase': 'sampling', 'updated': time.time()})
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(normalized['latent_feats'].shape, device=device, generator=generator)
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = sampler.sample(model, noise=template.replace(noise),
                    **make_conditions(template, sample, normalized), steps=50, verbose=False)
            logits = decode_clean(decoder, template, result.samples.feats, sample, base)
            faces = face_logits_from_voxels(logits, sample['face_knn'], sample['face_knn_weights']).argmax(1).cpu().numpy()
            metric = {'instance': row['instance'], 'seed': seed, 'seconds': time.monotonic() - start,
                'voxel': classification_metrics(target, logits.argmax(1).cpu().numpy(), np.ones(len(target))),
                'surface': classification_metrics(face_target, faces, area),
                'projection': classification_metrics(face_target, faces, pixels),
                'boundary_f1': boundary_f1(face_target, faces, adjacency)}
            report['samples'].append(metric)
            done.add((row['instance'], seed))
            atomic_json(shard_path, report)
            if not args.no_save_predictions:
                np.savez_compressed(args.output_dir / f'{row["instance"]}_seed{seed}.npz', face_labels=faces,
                                     voxel_labels=logits.argmax(1).cpu().numpy())
            print(f"[rank {rank}] {row['instance']} seed={seed}: proj mIoU={fmt(metric['projection']['miou'])}, "
                  f"surface mIoU={fmt(metric['surface']['miou'])}, boundary={metric['boundary_f1']}", flush=True)
            atomic_json(shard_path, report)
        atomic_json(shard_path, report)
        faulthandler.cancel_dump_traceback_later()
        atomic_json(progress_path, {'rank': rank, 'instance': row['instance'],
                                    'phase': 'complete', 'updated': time.time()})
    atomic_json(shard_path, report)
    faulthandler.cancel_dump_traceback_later()
    if args.no_merge:
        return
    if world > 1:
        dist.barrier()
    merge_results(args, rank, world, full_cohort, report)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output_dir',type=Path,required=True)
    ap.add_argument('--split',choices=['validation','test'],default='validation')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42])
    ap.add_argument('--limit',type=int)
    ap.add_argument('--no_save_predictions', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no_merge', action='store_true')
    ap.add_argument('--merge_only', action='store_true')
    ap.add_argument('--independent_worker', action='store_true')
    ap.add_argument('--worker_rank', type=int)
    ap.add_argument('--worker_world_size', type=int)
    ap.add_argument('--watchdog_seconds', type=int, default=600)
    ap.add_argument('--conv_algorithm', choices=['explicit_gemm', 'masked_implicit_gemm_splitk'],
                    default='explicit_gemm')
    ap.add_argument('--local-rank', '--local_rank', type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--distributed_timeout', type=int, default=1800)
    args=ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or (args.limit is not None and args.limit < 1):
        ap.error('Seeds must be unique and limit must be positive')
    if (args.worker_rank is None) != (args.worker_world_size is None):
        ap.error('--worker_rank and --worker_world_size must be used together')
    if args.worker_rank is not None:
        rank, world = args.worker_rank, args.worker_world_size
        if not 0 <= rank < world:
            ap.error('Invalid independent worker rank')
        local = 0
        args.independent_worker = True
    else:
        rank, local, world = distributed_topology()
    if args.merge_only:
        config = resolve_config(args.config)
        args.checkpoint = Path(config['dit_ckpt']) if args.checkpoint is None else args.checkpoint.resolve()
        full_cohort = sorted([row for row in records(config['data_dir']) if row['split'] == args.split],
                             key=lambda row: row['instance'])
        if args.limit:
            full_cohort = full_cohort[:args.limit]
        merge_results(args, 0, world, full_cohort)
        return
    if world > 1 and not args.independent_worker:
        args.device = f'cuda:{local}'
    torch.cuda.set_device(args.device)
    if world > 1 and not args.independent_worker:
        dist.init_process_group('nccl', timeout=timedelta(seconds=args.distributed_timeout))
    try:
        evaluate(args, rank, world)
    finally:
        if world > 1 and not args.independent_worker and dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':main()
