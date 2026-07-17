"""
Debug: verify Seg VAE encode→decode round-trip.

Test 1: Load seg_voxels (real coords + labels), run encoder → decoder, check accuracy.
Test 2: Load precomputed seg_latent, try to decode it (check if cache issue).
"""
import os
import sys
import json
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
TRELLIS2_ROOT = os.path.join(PROJECT_DIR, '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor
import trellis2.models as trellis2_models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)

DATA_DIR = '/path/to/data'
SEG_VAE_ENC_CKPT = './checkpoints/seg_vae/encoder_ema0.9999_step0001000.pt'
SEG_VAE_DEC_CKPT = './checkpoints/seg_vae/decoder_ema0.9999_step0001000.pt'
SEG_VAE_CONFIG = os.path.join(PROJECT_DIR, 'configs/seg_vae_train.json')
NUM_CLASSES = 20


def main():
    device = 'cuda'

    # Load Seg VAE
    seg_vae_cfg = json.load(open(SEG_VAE_CONFIG))
    enc_args = seg_vae_cfg['models']['encoder']['args']
    dec_args = seg_vae_cfg['models']['decoder']['args']

    print('Loading Seg VAE encoder...')
    encoder = trellis2_models.SparseUnetVaeEncoder(**enc_args).eval().to(device)
    enc_state = torch.load(SEG_VAE_ENC_CKPT, map_location=device, weights_only=True)
    encoder.load_state_dict(enc_state)
    del enc_state

    print('Loading Seg VAE decoder...')
    decoder = trellis2_models.SparseUnetVaeDecoder(**dec_args).eval().to(device)
    dec_state = torch.load(SEG_VAE_DEC_CKPT, map_location=device, weights_only=True)
    decoder.load_state_dict(dec_state)
    del dec_state

    # Pick a sample
    seg_voxels_dir = os.path.join(DATA_DIR, 'seg_voxels')
    seg_latent_dir = os.path.join(DATA_DIR, 'seg_latent')
    files = sorted(os.listdir(seg_voxels_dir))[:5]

    for fname in files:
        instance = fname[:-4]
        print(f'\n{"="*60}')
        print(f'Instance: {instance}')

        # ============================================================
        # TEST 1: Full round-trip (encode with real labels → decode)
        # ============================================================
        vox_data = np.load(os.path.join(seg_voxels_dir, fname))
        coords = torch.tensor(vox_data['coords'], dtype=torch.int32, device=device)
        labels = torch.tensor(vox_data['labels'], dtype=torch.long, device=device)

        # One-hot
        feats = torch.zeros(coords.shape[0], NUM_CLASSES, dtype=torch.float32, device=device)
        feats.scatter_(1, labels.unsqueeze(1).clamp(0, NUM_CLASSES - 1), 1.0)

        batch_coords = torch.cat([
            torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=device),
            coords
        ], dim=-1)
        x = sp.SparseTensor(feats, batch_coords)

        print(f'  Input: {coords.shape[0]} voxels, {labels.unique().numel()} unique labels')

        # Encode
        z = encoder(x, sample_posterior=False)
        print(f'  Latent z: {z.feats.shape[0]} tokens × {z.feats.shape[1]}ch')
        print(f'  Latent coords range: [{z.coords[:, 1:].min().item()}, {z.coords[:, 1:].max().item()}]')
        print(f'  Spatial cache keys: {list(z._spatial_cache.keys()) if z._spatial_cache else "NONE"}')
        print(f'  z.feats stats: mean={z.feats.mean().item():.4f}, std={z.feats.std().item():.4f}')

        # Decode
        dec_dtype = next(decoder.parameters()).dtype
        z_cast = z.replace(z.feats.to(dec_dtype))
        logits = decoder(z_cast)
        pred_labels = logits.feats.float().argmax(dim=-1)
        pred_coords = logits.coords[:, 1:]

        print(f'  Decoded: {pred_coords.shape[0]} voxels × {logits.feats.shape[1]}ch')
        print(f'  Decoded coords range: [{pred_coords.min().item()}, {pred_coords.max().item()}]')
        print(f'  Unique predicted labels: {pred_labels.unique().tolist()}')

        # Check accuracy vs input (match by coords)
        # The decoded coords should match input coords (round-trip)
        in_lin = (coords[:, 0].long() * 512 * 512 + coords[:, 1].long() * 512 + coords[:, 2].long())
        out_lin = (pred_coords[:, 0].long() * 512 * 512 + pred_coords[:, 1].long() * 512 + pred_coords[:, 2].long())
        
        # Find matching coords
        in_set = set(in_lin.cpu().numpy().tolist())
        out_set = set(out_lin.cpu().numpy().tolist())
        overlap = in_set & out_set
        print(f'  Coord overlap: {len(overlap)}/{len(in_set)} input, {len(overlap)}/{len(out_set)} output')

        # ============================================================
        # TEST 2: Load precomputed seg_latent and try decode
        # ============================================================
        latent_path = os.path.join(seg_latent_dir, f'{instance}.npz')
        if os.path.exists(latent_path):
            lat_data = np.load(latent_path)
            lat_coords = torch.tensor(lat_data['coords'], dtype=torch.int32, device=device)
            lat_feats = torch.tensor(lat_data['feats'], dtype=torch.float32, device=device)
            print(f'\n  Precomputed seg_latent: {lat_coords.shape[0]} tokens')
            print(f'  Precomputed coords range: [{lat_coords.min().item()}, {lat_coords.max().item()}]')
            print(f'  Precomputed feats stats: mean={lat_feats.mean().item():.4f}, std={lat_feats.std().item():.4f}')

            # Compare with encoder output z
            z_coords = z.coords[:, 1:].int()
            print(f'  Encoder z coords range: [{z_coords.min().item()}, {z_coords.max().item()}]')
            print(f'  Encoder z tokens: {z_coords.shape[0]}, precomputed tokens: {lat_coords.shape[0]}')

            # Check coord overlap between encoder z and precomputed
            z_lin = (z_coords[:, 0].long() * 32 * 32 + z_coords[:, 1].long() * 32 + z_coords[:, 2].long())
            lat_lin = (lat_coords[:, 0].long() * 32 * 32 + lat_coords[:, 1].long() * 32 + lat_coords[:, 2].long())
            z_set = set(z_lin.cpu().numpy().tolist())
            lat_set = set(lat_lin.cpu().numpy().tolist())
            overlap_lat = z_set & lat_set
            print(f'  Latent coord overlap: {len(overlap_lat)}/{len(z_set)} encoder, {len(overlap_lat)}/{len(lat_set)} precomputed')

            # Check feat similarity
            if len(overlap_lat) > 0:
                # cos similarity of feats at matching coords
                from scipy.spatial import cKDTree
                z_np = z_coords.cpu().numpy().astype(np.float32)
                lat_np = lat_coords.cpu().numpy().astype(np.float32)
                tree = cKDTree(lat_np)
                dists, nn_idx = tree.query(z_np)
                matched = dists < 0.5  # exact match
                if matched.sum() > 0:
                    z_feats_matched = z.feats[matched].float()
                    lat_feats_matched = lat_feats[nn_idx[matched]]
                    cos = torch.nn.functional.cosine_similarity(
                        z_feats_matched, lat_feats_matched, dim=-1
                    ).mean().item()
                    print(f'  Feat cosine similarity (matched coords): {cos:.4f} ({matched.sum()} matches)')

            # Try to decode precomputed latent using encoder's cache
            from scripts.batch_inference import lookup_feats_at_coords
            lat_at_z = lookup_feats_at_coords(z_coords, lat_coords, lat_feats)
            nonzero_ratio = (lat_at_z.abs().sum(dim=-1) > 0).float().mean().item()
            print(f'  Lookup hit ratio (precomputed→encoder coords): {nonzero_ratio:.4f}')

            z_with_precomputed = z.replace(lat_at_z.to(dec_dtype))
            z_with_precomputed._shape = torch.Size([1, 32])
            logits2 = decoder(z_with_precomputed)
            pred2 = logits2.feats.float().argmax(dim=-1)
            print(f'  Decode precomputed via cache: {pred2.unique().tolist()} ({logits2.feats.shape[0]} voxels)')


if __name__ == '__main__':
    main()
