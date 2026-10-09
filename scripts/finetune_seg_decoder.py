"""Frozen-encoder Seg VAE decoder refinement with part-balanced supervision.

The 64 deterministic VAE diagnostic objects are excluded from updates. This is
a held-out split for this refinement, not evidence that the original VAE never
saw those objects. Existing encoders and DiT latents remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import faulthandler

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))
sys.path.insert(0, str(ROOT))
import trellis2.models as models
from trellis2.modules.sparse.basic import SparseTensor
from scripts.inference_wild import SEG_VAE_ENC_CKPT, SEG_VAE_DEC_CKPT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--accumulate', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--device', default='cuda:3')
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--conv_algorithm', choices=['explicit_gemm', 'masked_implicit_gemm_splitk'],
                    default='explicit_gemm')
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Use an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    from flex_gemm.ops import spconv
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = args.conv_algorithm
    spconv.set_algorithm(args.conv_algorithm)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = json.loads((ROOT / 'configs/seg_vae_train.json').read_text())['models']
    enc = models.SparseUnetVaeEncoder(**config['encoder']['args'])
    dec_args = dict(config['decoder']['args'], use_fp16=False)
    dec = models.SparseUnetVaeDecoder(**dec_args)
    enc.load_state_dict(torch.load(SEG_VAE_ENC_CKPT, map_location='cpu', weights_only=True))
    dec.load_state_dict(torch.load(SEG_VAE_DEC_CKPT, map_location='cpu', weights_only=True))
    enc = enc.eval().to(args.device).requires_grad_(False)
    dec = dec.train().to(args.device)
    optimizer = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0)
    files = sorted((args.data_dir / 'seg_voxels').glob('*.npz'),
                   key=lambda p: hashlib.sha256(('vae-validation-v1:' + p.stem).encode()).hexdigest())
    validation, training = files[:64], files[64:]
    random.shuffle(training)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(encoder=SEG_VAE_ENC_CKPT, source_decoder=SEG_VAE_DEC_CKPT,
                    encoder_frozen=True, validation_ids=[p.stem for p in validation],
                    training_ids=[p.stem for p in training],
                    loss='sqrt inverse-frequency CE (max weight 5) + 0.25 macro soft Dice',
                    note='Validation split excludes updates in this refinement only')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    log = (args.output_dir / 'train.jsonl').open('w', buffering=1)
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        metrics = []
        for micro in range(args.accumulate):
            path = training[((step - 1) * args.accumulate + micro) % len(training)]
            with np.load(path) as d:
                coords, labels = d['coords'].astype(np.int32), d['labels'].astype(np.int64)
            if len(labels) > 30000:
                idx = rng.choice(len(labels), 30000, replace=False)
                coords, labels = coords[idx], labels[idx]
            if not len(labels):
                raise ValueError(f'Empty training sample: {path}')
            c = torch.tensor(np.column_stack([np.zeros(len(coords), np.int32), coords]), device=args.device)
            target = torch.tensor(labels, device=args.device)
            x = SparseTensor(feats=F.one_hot(target, 20).float(), coords=c)
            with torch.no_grad():
                if step == 1:
                    print(f'First batch {path.stem}: encoder forward', flush=True)
                z = enc(x, sample_posterior=False)
            # The frozen encoder's flex_gemm neighbor cache omits backward
            # signals. Retain the multiscale upsample maps but rebuild conv
            # neighbor entries under the decoder's gradient-enabled forward.
            z._spatial_cache = {
                scale: {k: value for k, value in cache.items()
                        if not k.startswith('SubMConv3d_neighbor_cache_')}
                for scale, cache in z._spatial_cache.items()
            }
            if step == 1:
                print('Decoder forward', flush=True)
            output = dec(z.replace(z.feats.float()))
            if not torch.equal(output.coords, c):
                raise ValueError('Decoder coordinate order differs from targets')
            logits = output.feats.float()
            counts = torch.bincount(target, minlength=20).float()
            present = counts > 0
            weights = torch.zeros_like(counts)
            weights[present] = torch.sqrt(counts[present].mean() / counts[present]).clamp(.25, 5.)
            ce = F.cross_entropy(logits, target, weight=weights)
            prob = logits.softmax(1)
            truth = F.one_hot(target, 20).float()
            dice = 1 - ((2 * (prob * truth).sum(0) + 1) / (prob.sum(0) + counts + 1))[present].mean()
            loss = ce + .25 * dice
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            if step == 1:
                print('Decoder backward', flush=True)
            (loss / args.accumulate).backward()
            metrics.append({'instance': path.stem, 'ce': float(ce.detach()), 'dice': float(dice.detach()),
                            'acc': float((logits.argmax(1) == target).float().mean())})
            del x, z, output, logits, prob, truth, loss
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(60, repeat=True)
        row = {'step': step, 'elapsed': time.monotonic() - start, 'grad_norm': float(norm), 'batch': metrics}
        log.write(json.dumps(row) + '\n')
        if step % 10 == 0 or step == 1:
            print(f"step={step} ce={np.mean([m['ce'] for m in metrics]):.5f} "
                  f"acc={np.mean([m['acc'] for m in metrics]):.5f} "
                  f"seconds={row['elapsed']:.1f}", flush=True)
        if step % 100 == 0 or step == args.steps:
            path = args.output_dir / f'decoder_step{step:06d}.pt'
            torch.save({k: v.detach().cpu() for k, v in dec.state_dict().items()}, path)
            print(f'Saved {path}', flush=True)
    log.close()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
