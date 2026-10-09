"""Evaluate Seg VAE reconstruction separately from DiT and mesh label baking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))
sys.path.insert(0, str(ROOT))
from trellis2.modules.sparse.basic import SparseTensor
import trellis2.models as models
from scripts.inference_wild import SEG_VAE_ENC_CKPT, SEG_VAE_DEC_CKPT


def scores(target, prediction):
    parts = np.unique(target)
    ious, recalls = [], []
    for part in parts:
        t, p = target == part, prediction == part
        ious.append(float((t & p).sum() / (t | p).sum()))
        recalls.append(float((t & p).sum() / t.sum()))
    counts = np.array([(target == p).sum() for p in parts])
    small = counts < len(target) * .01
    return {'accuracy': float(np.mean(target == prediction)), 'miou': float(np.mean(ious)),
            'small_part_recall': float(np.mean(np.array(recalls)[small])) if small.any() else None,
            'parts': [{'id': int(p), 'count': int(n), 'iou': i, 'recall': r}
                      for p, n, i, r in zip(parts, counts, ious, recalls)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--decoder', default=SEG_VAE_DEC_CKPT)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--device', default='cuda:2')
    ap.add_argument('--conv_algorithm', choices=['explicit_gemm', 'masked_implicit_gemm_splitk'],
                    default='explicit_gemm')
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = args.conv_algorithm
    config = json.loads((ROOT / 'configs/seg_vae_train.json').read_text())['models']
    enc = models.SparseUnetVaeEncoder(**config['encoder']['args'])
    dec = models.SparseUnetVaeDecoder(**config['decoder']['args'])
    enc.load_state_dict(torch.load(SEG_VAE_ENC_CKPT, map_location='cpu', weights_only=True))
    dec.load_state_dict(torch.load(args.decoder, map_location='cpu', weights_only=True))
    enc, dec = enc.eval().to(args.device), dec.eval().to(args.device)
    files = sorted((args.data_dir / 'seg_voxels').glob('*.npz'),
                   key=lambda p: hashlib.sha256(('vae-validation-v1:' + p.stem).encode()).hexdigest())[:args.samples]
    report = {'encoder': SEG_VAE_ENC_CKPT, 'decoder': args.decoder,
              'conv_algorithm': args.conv_algorithm, 'samples': []}
    for path in files:
        d = np.load(path)
        coords, labels = d['coords'].astype(np.int32), d['labels'].astype(np.int64)
        if len(labels) == 0:
            raise ValueError(f'Empty voxel sample {path}')
        if len(labels) > 30000:
            idx = np.random.default_rng(42).choice(len(labels), 30000, replace=False)
            coords, labels = coords[idx], labels[idx]
        c = torch.as_tensor(np.column_stack([np.zeros(len(coords), np.int32), coords]), device=args.device)
        target = torch.as_tensor(labels, device=args.device)
        x = SparseTensor(feats=F.one_hot(target, 20).float(), coords=c)
        z = enc(x, sample_posterior=False)
        output = dec(z)
        if not torch.equal(output.coords, c):
            raise ValueError('Decoder coordinates are reordered')
        row = {'instance': path.stem, 'voxels': len(labels), 'tokens': len(z.feats),
               'reconstruction': scores(labels, output.feats.argmax(1).cpu().numpy())}
        latent_path = args.data_dir / 'seg_latent' / path.name
        if latent_path.exists():
            cached = np.load(latent_path)
            same_coords = np.array_equal(z.coords[:, 1:].cpu().numpy(), cached['coords'])
            row['cached_coords_equal'] = same_coords
            if same_coords:
                feats = torch.tensor(cached['feats'], device=args.device, dtype=z.feats.dtype)
                row['cached_feature_rmse'] = float((feats - z.feats).float().square().mean().sqrt())
                decoded = dec(z.replace(feats))
                row['cached_reconstruction'] = scores(labels, decoded.feats.argmax(1).cpu().numpy())
        report['samples'].append(row)
        print(path.stem, row['reconstruction']['accuracy'], row['reconstruction']['miou'],
              row.get('cached_feature_rmse'), flush=True)
        del x, z, output
    report['summary'] = {}
    for kind in ['reconstruction', 'cached_reconstruction']:
        rows = [r[kind] for r in report['samples'] if kind in r]
        report['summary'][kind] = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
                                   for k in ['accuracy', 'miou', 'small_part_recall']} if rows else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
