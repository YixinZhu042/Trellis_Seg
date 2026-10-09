"""Paired multi-node DiT continuation with a frozen VAE (torchrun/DDP).

VAE encoder and decoder weights are frozen. Decoder input gradients remain
enabled. Baseline and treatment share initialization, data order, noise, times,
dropout, optimizer and evaluation seeds when launched with the same topology.
No segmentation postprocessing is used. --check_only validates without a GPU.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))
sys.path.insert(0, str(ROOT))
from utils.decoded_dit_runtime import (
    FeatureOutputModel, atomic_json, atomic_torch_save, canonical_digest,
    distributed_topology, master_action, training_sample_index,
)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_models(config, device, training, dit_state=None, conv_algorithm='explicit_gemm'):
    from models.seg_flow_model import SegSLatFlowModel
    import trellis2.models as vae_models
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = conv_algorithm
    try:
        from flex_gemm.ops import spconv
        spconv.set_algorithm(conv_algorithm)
    except (ImportError, AttributeError):
        pass
    base = json.loads(Path(config['base_config']).read_text())
    vae = json.loads(Path(config['vae_config']).read_text())['models']
    encoder = vae_models.SparseUnetVaeEncoder(**vae['encoder']['args']).eval()
    decoder = vae_models.SparseUnetVaeDecoder(**dict(vae['decoder']['args'], use_fp16=False)).eval()
    encoder.load_state_dict(torch.load(config['encoder'], map_location='cpu', weights_only=True))
    decoder.load_state_dict(torch.load(config['decoder'], map_location='cpu', weights_only=True))
    encoder = encoder.to(device).requires_grad_(False)
    decoder = decoder.to(device).requires_grad_(False)
    # Same architecture/state keys as ElasticSegSLatFlowModel; the plain class
    # removes only the memory-controller wrapper; DDP manages synchronization.
    model = SegSLatFlowModel(**dict(base['models']['denoiser']['args'], use_checkpoint=training))
    model.load_state_dict(dit_state if dit_state is not None else
                         torch.load(config['dit_ckpt'], map_location='cpu', weights_only=True))
    model = model.to(device)
    # Keep fp32 master weights for AdamW; autocast computes attention in BF16.
    model.dtype = torch.bfloat16
    return model, encoder, decoder, base


def load_teacher(config, device):
    """Load only the original frozen DiT, without duplicating either VAE."""
    from models.seg_flow_model import SegSLatFlowModel
    base = json.loads(Path(config['base_config']).read_text())
    teacher = SegSLatFlowModel(**dict(base['models']['denoiser']['args'], use_checkpoint=False))
    teacher.load_state_dict(torch.load(config['dit_ckpt'], map_location='cpu', weights_only=True))
    teacher = teacher.to(device).eval().requires_grad_(False)
    teacher.dtype = torch.bfloat16
    return teacher


def records(data_dir, require_audit=False, check_files=False):
    data_dir = Path(data_dir)
    prepared_path = data_dir / 'prepared_manifest.json'
    audit_path = data_dir / 'prepared_audit.json'
    audit = None
    if require_audit or check_files:
        if not prepared_path.is_file() or not audit_path.is_file():
            raise FileNotFoundError('Prepared manifest and deep audit are required before training')
        prepared = json.loads(prepared_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (prepared['prepared'] != prepared['expected'] or prepared['failures'] or prepared['missing'] or
                prepared['extra'] or prepared['wrong_split'] or prepared.get('orphan_files') or
                prepared.get('missing_ranks') or prepared.get('extra_ranks') or prepared.get('duplicate_ranks')):
            raise ValueError('Prepared dataset manifest is incomplete')
        if audit['errors_count'] or audit['objects'] != prepared['expected']:
            raise ValueError('Prepared dataset deep audit did not pass')
        if audit['prepared_manifest_sha256'] != hashlib.sha256(prepared_path.read_bytes()).hexdigest():
            raise ValueError('Deep audit does not match prepared_manifest.json')
    rows = []
    failures = []
    for path in sorted(data_dir.glob('prep_rank*.json')):
        data = json.loads(path.read_text()); rows.extend(data['samples']); failures.extend(data['failures'])
    if failures:
        raise ValueError(f'Unresolved preparation failures: {failures}')
    selection_path = data_dir / 'selection.json'
    if audit is not None and audit['selection_sha256'] != hashlib.sha256(selection_path.read_bytes()).hexdigest():
        raise ValueError('Deep audit does not match selection.json')
    expected = json.loads(selection_path.read_text())['samples']
    if len(rows) != len(expected) or len({r['instance'] for r in rows}) != len(rows):
        raise ValueError('Prepared cohort is incomplete or duplicated')
    expected_splits = {r['model_id']: r['split'] for r in expected}
    if {r['instance']: r['split'] for r in rows} != expected_splits:
        raise ValueError('Prepared samples differ from the declared train/validation/test split')
    if check_files:
        file_state=[]
        for row in sorted(rows,key=lambda x:x['instance']):
            path=Path(row['path'])
            if not path.is_file():raise FileNotFoundError(path)
            stat=path.stat();file_state.append((row['instance'],stat.st_size,stat.st_mtime_ns))
        digest=hashlib.sha256(json.dumps(file_state,separators=(',',':')).encode()).hexdigest()
        if audit.get('file_state_sha256')!=digest:
            raise ValueError('Prepared sample files changed after the deep audit')
    return rows


def get_sample(row, device):
    keys = ['coords', 'voxel_labels', 'latent_coords', 'latent_feats', 'shape_feats', 'shape_valid', 'mv_coords', 'mv_feats',
            'face_labels', 'face_area', 'face_knn', 'face_knn_weights', 'adjacency', 'face_pixels']
    with np.load(row['path']) as d:
        data = {key: torch.as_tensor(d[key], device=device) for key in keys}
    for key in ['voxel_labels', 'face_labels', 'face_knn', 'adjacency']:
        data[key] = data[key].long()
    data['shape_valid'] = data['shape_valid'].bool()
    for key in ['latent_feats', 'shape_feats', 'mv_feats', 'face_area', 'face_knn_weights', 'face_pixels']:
        data[key] = data[key].float()
    return data


def spatial_template(sample, encoder):
    from trellis2.modules.sparse.basic import SparseTensor
    coords = sample['coords'].int()
    coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    dummy = torch.zeros(len(coords), 20, device=coords.device)
    dummy[:, 0] = 1
    with torch.no_grad():
        template = encoder(SparseTensor(feats=dummy, coords=coords), sample_posterior=False)
    if not torch.equal(template.coords[:, 1:], sample['latent_coords'].int()):
        raise ValueError('Cached target latent and decoder template coordinates differ')
    # Frozen no-grad encoder caches cannot be reused for decoder backward.
    # Retain only spatial mappings, never features or convolution neighbor maps.
    template._spatial_cache = {
        scale: {key: val for key, val in cache.items() if not key.startswith('SubMConv3d_neighbor_cache_')}
        for scale, cache in template._spatial_cache.items()}
    return template


def normalize_aligned_features(features, mean, std, valid=None):
    normalized = (features - mean) / std.clamp_min(1e-6)
    if valid is not None:
        normalized = normalized.masked_fill(~valid[:, None], 0)
    return normalized


def normalize(sample, base):
    args = base['dataset']['args']
    output = {}
    for key, field in [('latent_feats', 'normalization'), ('shape_feats', 'shape_cond_normalization'), ('mv_feats', 'mv_cond_normalization')]:
        mean = torch.tensor(args[field]['mean'], device=sample[key].device)
        std = torch.tensor(args[field]['std'], device=sample[key].device).clamp_min(1e-6)
        valid = sample['shape_valid'] if key == 'shape_feats' else None
        output[key] = normalize_aligned_features(sample[key], mean, std, valid)
    return output


def unnormalize(latent, base):
    args = base['dataset']['args']['normalization']
    mean = torch.tensor(args['mean'], device=latent.device)
    std = torch.tensor(args['std'], device=latent.device).clamp_min(1e-6)
    return latent * std + mean


def make_conditions(template, sample, normalized, drop=False):
    shape = template.replace(normalized['shape_feats'])
    cond = torch.zeros(1, 1029, 1024, device=template.device)
    return dict(cond=cond, concat_cond=shape,
                mv_cond=[None if drop else normalized['mv_feats']],
                mv_coords=[None if drop else sample['mv_coords'].float()])


def decode_clean(decoder, template, predicted_clean, sample, base):
    # No no_grad here: frozen weights, differentiable input.
    latent = unnormalize(predicted_clean.float(), base)
    output = decoder(template.replace(latent))
    if not torch.equal(output.coords[:, 1:], sample['coords'].int()):
        raise ValueError('Decoded voxel coordinates do not match supervision')
    return output.feats.float()


def resolve_config(path, accumulate=None):
    config = json.loads(Path(path).read_text())
    for key in ('base_config', 'vae_config', 'data_dir', 'encoder', 'decoder', 'dit_ckpt'):
        value = Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else ROOT / value).resolve())
        if not Path(config[key]).exists():
            raise FileNotFoundError(f'{key}: {config[key]}')
    if accumulate is not None:
        config['accumulate'] = accumulate
    for key in ('accumulate', 'warmup_steps', 'aux_warmup_steps', 'save_every'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('condition_dropout', 'aux_t_max', 'sigma_min'):
        if not 0 <= config[key] < 1:
            raise ValueError(f'{key} must be in [0, 1)')
    if not np.isfinite(config['lr']) or config['lr'] <= 0:
        raise ValueError('lr must be finite and positive')
    if set(config['loss_weights']) != {'segmentation', 'boundary', 'reprojection'}:
        raise ValueError('Expected all three decoded task weights')
    if any(not np.isfinite(v) or v <= 0 for v in config['loss_weights'].values()):
        raise ValueError('Task weights must be finite and positive')
    ema = config.setdefault('ema', {'enabled': False, 'rate': .9999})
    if not isinstance(ema.get('enabled'), bool) or not 0 < ema.get('rate', 0) < 1:
        raise ValueError('ema requires boolean enabled and rate in (0,1)')
    replay = config.setdefault('teacher_replay', {'probability': 0., 'weight': 0.})
    if not 0 <= replay.get('probability', -1) <= 1 or replay.get('weight', -1) < 0:
        raise ValueError('teacher_replay probability/weight are invalid')
    if replay['weight'] > 0 and replay['probability'] == 0:
        raise ValueError('teacher replay weight needs positive probability')
    return config


def rng_state(generator, times, device, gradient_checked):
    return {'noise': generator.get_state().cpu(), 'times': times.bit_generator.state,
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(device).cpu(),
            'gradient_checked': gradient_checked}


def compose_training_loss(flow, *, teacher_loss=None, teacher_weight=0.,
                          auxiliary_loss=None, auxiliary_gate=0.):
    """Combine every enabled objective without allowing one branch to replace another."""
    loss = flow
    if teacher_loss is not None:
        loss = loss + teacher_weight * teacher_loss
    if auxiliary_loss is not None:
        loss = loss + auxiliary_gate * auxiliary_loss
    return loss


def save_training_checkpoint(output, step, model, optimizer, metadata, local_rng, rank, world,
                             vae_digests=None, ema_state=None):
    all_rng = [None] * world
    if world > 1:
        dist.all_gather_object(all_rng, local_rng)
    else:
        all_rng[0] = local_rng
    def save():
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        atomic_torch_save(output / f'dit_step{step:06d}.pt', weights)
        ema_weights = ({k: v.detach().cpu() for k, v in ema_state.items()}
                       if ema_state is not None else None)
        if ema_state is not None:
            atomic_torch_save(output / f'dit_ema_step{step:06d}.pt', ema_weights)
        # Only the latest full optimizer/RNG state is retained. Portable weights
        # are kept at each interval for evaluation with evaluate_dit_decoded.py.
        atomic_torch_save(output / 'trainer_latest.pt', {
            'step': step, 'model': weights, 'optimizer': optimizer.state_dict(),
            'metadata': metadata, 'rank_rng': all_rng, 'vae_digests': vae_digests,
            'ema': ema_weights,
        })
        atomic_json(output / 'checkpoint_latest.json', {'step': step, 'metadata': metadata})
    master_action(save, rank, world > 1)


def train(args, config):
    rank, local_rank, world = distributed_topology()
    distributed = world > 1
    device = torch.device(f'cuda:{local_rank}' if distributed else args.device)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group('nccl', timeout=timedelta(seconds=args.distributed_timeout))
    try:
        _train(args, config, device, rank, world)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, config, device, rank, world):
    from trainers.decoded_seg_losses import decoded_task_losses, recover_clean_latent
    output = args.output_dir.resolve()
    distributed = world > 1
    resume_path = output / 'trainer_latest.pt' if args.resume == 'latest' else (Path(args.resume).resolve() if args.resume else None)
    rows = records(config['data_dir'], require_audit=True)
    train_rows = sorted([r for r in rows if r['split'] == 'train'], key=lambda r: r['instance'])
    random.Random(config['seed']).shuffle(train_rows)
    if not train_rows:
        raise ValueError('Training split is empty')
    metadata = {'mode': args.mode, 'world_size': world, 'config_digest': canonical_digest(config),
                'base_config_digest': canonical_digest(json.loads(Path(config['base_config']).read_text())),
                'vae_config_digest': canonical_digest(json.loads(Path(config['vae_config']).read_text())),
                'data_digest': canonical_digest(sorted((r['instance'], r['split'], r['path']) for r in rows))}
    if distributed:
        all_metadata = [None] * world
        dist.all_gather_object(all_metadata, metadata)
        if any(m != metadata for m in all_metadata):
            raise ValueError('Ranks disagree on config, data, mode or topology')
    def setup_output():
        if resume_path:
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
        elif output.exists() and any(output.iterdir()):
            raise ValueError('Use a new output directory or --resume latest')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'ranks').mkdir(exist_ok=True)
    master_action(setup_output, rank, distributed)
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path else None
    if checkpoint and checkpoint['metadata'] != metadata:
        raise ValueError('Resume requires identical mode, config, data and world size')
    start_step = checkpoint['step'] + 1 if checkpoint else 1
    if start_step > args.steps:
        raise ValueError('Requested total --steps must exceed the saved checkpoint step')
    torch.manual_seed(config['seed'])
    model, encoder, decoder, base = load_models(config, device, True, checkpoint['model'] if checkpoint else None)
    model.train()
    teacher = None
    if config['teacher_replay']['weight'] > 0:
        teacher = load_teacher(config, device)
    encoder_digest, decoder_digest = state_digest(encoder), state_digest(decoder)
    vae_digests = [None] * world
    current_digest = {'encoder': encoder_digest, 'decoder': decoder_digest}
    if checkpoint and checkpoint.get('vae_digests') != current_digest:
        raise ValueError('Frozen VAE weights differ from the resume checkpoint')
    if distributed:
        dist.all_gather_object(vae_digests, current_digest)
    else:
        vae_digests[0] = current_digest
    if any(d != current_digest for d in vae_digests):
        raise ValueError('Ranks loaded different frozen VAE weights')
    manifest = {'config': config, **metadata, 'global_batch_size': world * config['accumulate'],
                'microbatch_per_gpu': 1, 'accumulate_per_gpu': config['accumulate'],
                'train_ids': [r['instance'] for r in train_rows],
                'validation_ids': [r['instance'] for r in rows if r['split'] == 'validation'],
                'test_ids': [r['instance'] for r in rows if r['split'] == 'test'],
                'encoder_digest_before': encoder_digest, 'decoder_digest_before': decoder_digest,
                'requested_steps': args.steps, 'start_step': start_step,
                'resume_from': str(resume_path) if resume_path else None,
                'note': 'Original pretraining may include this data; continuation splits are disjoint.'}
    master_action(lambda: atomic_json(output / 'manifest.json', manifest), rank, distributed)
    forward_model = FeatureOutputModel(model)
    if distributed:
        forward_model = DistributedDataParallel(
            forward_model, device_ids=[device.index], output_device=device.index,
            broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9, .95), weight_decay=.01, foreach=False)
    ema_state = None
    if config['ema']['enabled'] and rank == 0:
        ema_state = ({k: v.detach().to(device).clone() for k,v in checkpoint['ema'].items()}
                     if checkpoint and checkpoint.get('ema') is not None else
                     {k: v.detach().clone() for k,v in model.state_dict().items()})
    generator = torch.Generator(device=device).manual_seed(config['seed'] + 1 + 100003 * rank)
    times = np.random.default_rng(config['seed'] + 2 + 100003 * rank)
    gradient_checked = False
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng = checkpoint['rank_rng'][rank]
        generator.set_state(rng['noise'])
        times.bit_generator.state = rng['times']
        torch.set_rng_state(rng['torch_cpu'])
        torch.cuda.set_rng_state(rng['torch_cuda'], device)
        gradient_checked = rng['gradient_checked']
        del checkpoint
    start = time.monotonic()
    scalar_keys = ('flow', 'loss', 'segmentation', 'boundary', 'reprojection', 'aux_gate', 'teacher')
    epoch_size = len(train_rows)
    if rank == 0:
        print(f'{args.mode}: ranks={world}, accumulate/GPU={config["accumulate"]}, '
              f'global batch={manifest["global_batch_size"]}, start={start_step}, end={args.steps}', flush=True)
    with (output / 'ranks' / f'rank{rank:04d}.jsonl').open('a' if resume_path else 'w', buffering=1) as rank_log:
        global_log = (output / 'train.jsonl').open('a' if resume_path else 'w', buffering=1) if rank == 0 else None
        try:
            for step in range(start_step, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                metrics = []
                for group in optimizer.param_groups:
                    group['lr'] = config['lr'] * min(1., step / config['warmup_steps'])
                for micro in range(config['accumulate']):
                    index = training_sample_index(step, micro, rank, world, config['accumulate'],
                                                  len(train_rows), seed=config['seed'], epoch_size=epoch_size)
                    row = train_rows[index]
                    sample = get_sample(row, device)
                    template = spatial_template(sample, encoder)
                    normalized = normalize(sample, base)
                    clean = normalized['latent_feats']
                    noise = torch.randn(clean.shape, device=device, generator=generator)
                    t = float(times.uniform())
                    drop = bool(times.uniform() < config['condition_dropout'])
                    use_teacher = bool(teacher is not None and times.uniform() < config['teacher_replay']['probability'])
                    sigma = config['sigma_min']
                    noisy = (1-t)*clean + (sigma+(1-sigma)*t)*noise
                    target_v = (1-sigma)*noise-clean
                    sync_context = forward_model.no_sync() if distributed and micro < config['accumulate'] - 1 else nullcontext()
                    with sync_context:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            predicted = forward_model(template.replace(noisy), torch.tensor([t*1000], device=device),
                                                      **make_conditions(template, sample, normalized, drop))
                        flow = F.mse_loss(predicted.float(), target_v)
                        teacher_loss = None
                        aux = None
                        gate = 0.
                        row_metrics = {'instance': row['instance'], 't': t, 'drop': drop,
                                       'teacher_replay': use_teacher, 'flow': float(flow.detach())}
                        if use_teacher:
                            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                                teacher_pred = teacher(template.replace(noisy), torch.tensor([t*1000], device=device),
                                                       **make_conditions(template, sample, normalized, drop)).feats.float()
                            teacher_loss = F.mse_loss(predicted.float(), teacher_pred)
                            row_metrics['teacher'] = float(teacher_loss.detach())
                        if args.mode == 'decoded' and t <= config['aux_t_max'] and not drop:
                            clean_pred = recover_clean_latent(noisy, predicted.float(), t, sigma)
                            logits = decode_clean(decoder, template, clean_pred, sample, base)
                            terms, face_logits = decoded_task_losses(logits, sample)
                            aux = sum(config['loss_weights'][k] * val for k, val in terms.items())
                            gate = (1-t)**2 * min(1., step / config['aux_warmup_steps'])
                            row_metrics.update({k: float(val.detach()) for k, val in terms.items()})
                            row_metrics['aux_gate'] = gate
                            if not gradient_checked:
                                grads = {name: float(torch.autograd.grad(term, predicted, retain_graph=True)[0].float().norm())
                                         for name, term in terms.items()}
                                if not all(np.isfinite(v) for v in grads.values()):
                                    raise ValueError(f'Nonfinite auxiliary gradients: {grads}')
                                # A sample with no annotated boundary can legitimately
                                # yield zero boundary gradient. Try subsequent batches.
                                if all(v > 0 for v in grads.values()):
                                    atomic_json(output / 'ranks' / f'gradient_check_rank{rank:04d}.json', grads)
                                    gradient_checked = True
                                    print(f'[rank {rank}] Auxiliary gradients reach DiT: {grads}', flush=True)
                        loss = compose_training_loss(
                            flow, teacher_loss=teacher_loss,
                            teacher_weight=config['teacher_replay']['weight'],
                            auxiliary_loss=aux, auxiliary_gate=gate)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Nonfinite loss at step {step}, rank {rank}')
                        row_metrics['loss'] = float(loss.detach())
                        (loss / config['accumulate']).backward()
                    if any(p.grad is not None for p in encoder.parameters()) or any(p.grad is not None for p in decoder.parameters()):
                        raise RuntimeError('Frozen VAE accumulated gradients')
                    metrics.append(row_metrics)
                    # Release each sparse decoder graph before constructing the next.
                    if 'terms' in locals():
                        del terms, logits, aux, clean_pred, face_logits
                    del loss, flow, predicted, template, sample, normalized
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if ema_state is not None:
                    rate = config['ema']['rate']
                    for name, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema_state[name].mul_(rate).add_(value.detach(), alpha=1-rate)
                        else:
                            ema_state[name].copy_(value.detach())
                elapsed = time.monotonic() - start
                rank_log.write(json.dumps({'step': step, 'rank': rank, 'seconds': elapsed,
                                          'grad_norm': float(norm), 'batch': metrics}) + '\n')
                stats = torch.tensor([[sum(m.get(k, 0.) for m in metrics), sum(k in m for m in metrics)]
                                      for k in scalar_keys], dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(stats)
                if rank == 0:
                    values = stats.cpu().tolist()
                    summary = {key: total / count for key, (total, count) in zip(scalar_keys, values) if count}
                    global_log.write(json.dumps({'step': step, 'seconds': elapsed, 'mean': summary,
                                                 'lr': optimizer.param_groups[0]['lr'], 'grad_norm': float(norm),
                                                 'global_samples_seen': step * manifest['global_batch_size']}) + '\n')
                    if step == start_step or step % 10 == 0:
                        print(f"{args.mode} step={step} flow={summary['flow']:.5f} loss={summary['loss']:.5f} time={elapsed:.1f}s", flush=True)
                if step % config['save_every'] == 0 or step == args.steps:
                    save_training_checkpoint(output, step, model, optimizer, metadata,
                                             rng_state(generator, times, device, gradient_checked), rank, world,
                                             vae_digests=current_digest, ema_state=ema_state)
        finally:
            if global_log:
                global_log.close()
    after = {'encoder': state_digest(encoder), 'decoder': state_digest(decoder), 'gradient_checked': gradient_checked}
    all_after = [None] * world
    if distributed:
        dist.all_gather_object(all_after, after)
    else:
        all_after[0] = after
    frozen = all(d['encoder'] == encoder_digest and d['decoder'] == decoder_digest for d in all_after)
    if not frozen:
        raise RuntimeError('VAE weights changed during DiT training')
    manifest.update(encoder_digest_after=after['encoder'], decoder_digest_after=after['decoder'],
                    vae_frozen_verified=True, completed_steps=args.steps, seconds=time.monotonic()-start,
                    ranks_with_verified_aux_gradients=sum(d['gradient_checked'] for d in all_after))
    master_action(lambda: atomic_json(output / 'manifest.json', manifest), rank, distributed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--mode', choices=['flow','decoded'], required=True)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--device', default='cuda:0', help='Single-process fallback; torchrun uses LOCAL_RANK')
    ap.add_argument('--accumulate', type=int, help='Override microbatches per GPU and optimizer step')
    ap.add_argument('--resume', nargs='?', const='latest', help='Restore full optimizer/RNG state; default trainer_latest.pt')
    ap.add_argument('--check_only', action='store_true', help='Read config/data and print settings without loading models')
    ap.add_argument('--distributed_timeout', type=int, default=1800)
    ap.add_argument('--local-rank', '--local_rank', type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.steps < 1 or args.distributed_timeout < 1:
        ap.error('steps and distributed_timeout must be positive')
    config = resolve_config(args.config, args.accumulate)
    if args.check_only:
        rows = records(config['data_dir'], require_audit=True, check_files=True)
        _, _, world = distributed_topology()
        print(json.dumps({'mode': args.mode, 'steps': args.steps, 'world_size': world,
                          'global_batch_size': world * config['accumulate'],
                          'accumulate_per_gpu': config['accumulate'],
                          'split_counts': {s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
                          'resolved_config': config, 'output_dir': str(args.output_dir.resolve())}, indent=2))
        return
    train(args, config)


if __name__ == '__main__':
    main()
