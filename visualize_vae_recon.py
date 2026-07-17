"""
Visualize texture VAE reconstruction: save GT and reconstructed point clouds
as JSON for 3D visualization in HTML (using Three.js).

Outputs per sample:
    - GT point cloud with class colors
    - Reconstructed point cloud with class colors
    - Accuracy metrics
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)


def generate_class_colors(num_classes):
    """Generate visually distinct colors for classes."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        h = hue * 6.0
        if h < 1:
            rgb = (1.0, h, 0)
        elif h < 2:
            rgb = (2.0 - h, 1.0, 0)
        elif h < 3:
            rgb = (0, 1.0, h - 2.0)
        elif h < 4:
            rgb = (0, 4.0 - h, 1.0)
        elif h < 5:
            rgb = (h - 4.0, 0, 1.0)
        else:
            rgb = (1.0, 0, 6.0 - h)
        colors.append([int(c * 255) for c in rgb])
    return colors


def labels_to_feats_color(labels, num_classes):
    """Encode labels as 6ch features for texture VAE."""
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for i in range(num_classes):
        hue = i / num_classes
        h = hue * 6.0
        if h < 1:
            rgb = (1.0, h, 0)
        elif h < 2:
            rgb = (2.0 - h, 1.0, 0)
        elif h < 3:
            rgb = (0, 1.0, h - 2.0)
        elif h < 4:
            rgb = (0, 4.0 - h, 1.0)
        elif h < 5:
            rgb = (h - 4.0, 0, 1.0)
        else:
            rgb = (1.0, 0, 6.0 - h)
        palette[i] = rgb

    palette_t = torch.from_numpy(palette).to(labels.device)
    rgb = palette_t[labels]
    label_norm = labels.float() / max(num_classes - 1, 1)
    
    feats = torch.zeros(labels.shape[0], 6, device=labels.device)
    feats[:, 0:3] = rgb
    feats[:, 3] = label_norm
    feats[:, 4] = label_norm
    feats[:, 5] = 1.0
    feats = feats * 2 - 1
    return feats


def feats_to_labels(feats, num_classes):
    """Decode 6ch features back to labels via nearest RGB."""
    feats = (feats + 1) / 2
    rgb_recon = feats[:, 0:3]
    
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for i in range(num_classes):
        hue = i / num_classes
        h = hue * 6.0
        if h < 1:
            rgb = (1.0, h, 0)
        elif h < 2:
            rgb = (2.0 - h, 1.0, 0)
        elif h < 3:
            rgb = (0, 1.0, h - 2.0)
        elif h < 4:
            rgb = (0, 4.0 - h, 1.0)
        elif h < 5:
            rgb = (h - 4.0, 0, 1.0)
        else:
            rgb = (1.0, 0, 6.0 - h)
        palette[i] = rgb
    
    palette_t = torch.from_numpy(palette).to(feats.device)
    dists = torch.cdist(rgb_recon.unsqueeze(0), palette_t.unsqueeze(0)).squeeze(0)
    return dists.argmin(dim=-1)


@torch.no_grad()
def process_sample(coords, labels, encoder, decoder, num_classes, device='cuda', max_vis_points=30000):
    """Run VAE encode/decode and prepare visualization data."""
    coords_t = torch.tensor(coords, dtype=torch.int32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    
    feats = labels_to_feats_color(labels_t, num_classes)
    
    batch_coords = torch.cat([
        torch.zeros(coords_t.shape[0], 1, dtype=torch.int32, device=device),
        coords_t
    ], dim=-1)
    
    x = sp.SparseTensor(feats.float(), batch_coords)
    z = encoder(x)
    recon = decoder(z)
    
    recon_feats = recon.feats
    recon_coords = recon.coords[:, 1:]
    
    # Decode reconstructed features to labels
    pred_labels = feats_to_labels(recon_feats, num_classes)
    
    # Match coordinates
    RES = 2048
    coord_key = coords_t[:, 0].long() * RES * RES + coords_t[:, 1].long() * RES + coords_t[:, 2].long()
    recon_coord_key = recon_coords[:, 0].long() * RES * RES + recon_coords[:, 1].long() * RES + recon_coords[:, 2].long()
    
    # Build GT lookup
    coord_to_label = {}
    for i in range(coords_t.shape[0]):
        coord_to_label[coord_key[i].item()] = labels_t[i].item()
    
    # Compute accuracy on matched voxels
    matched = 0
    correct = 0
    for i in range(recon_coords.shape[0]):
        key = recon_coord_key[i].item()
        if key in coord_to_label:
            matched += 1
            if pred_labels[i].item() == coord_to_label[key]:
                correct += 1
    
    accuracy = correct / matched if matched > 0 else 0.0
    
    # Subsample for visualization
    n_gt = coords_t.shape[0]
    n_recon = recon_coords.shape[0]
    
    if n_gt > max_vis_points:
        idx = np.random.choice(n_gt, max_vis_points, replace=False)
    else:
        idx = np.arange(n_gt)
    
    gt_coords_vis = coords_t[idx].cpu().numpy()
    gt_labels_vis = labels_t[idx].cpu().numpy()
    
    if n_recon > max_vis_points:
        idx_r = np.random.choice(n_recon, max_vis_points, replace=False)
    else:
        idx_r = np.arange(n_recon)
    
    recon_coords_vis = recon_coords[idx_r].cpu().numpy()
    recon_labels_vis = pred_labels[idx_r].cpu().numpy()
    
    # Normalize coords to [-1, 1] for visualization
    all_coords = np.concatenate([gt_coords_vis, recon_coords_vis], axis=0)
    center = all_coords.mean(axis=0)
    scale = max(all_coords.max() - all_coords.min(), 1)
    
    gt_coords_norm = ((gt_coords_vis - center) / scale * 2).tolist()
    recon_coords_norm = ((recon_coords_vis - center) / scale * 2).tolist()
    
    return {
        'gt_coords': gt_coords_norm,
        'gt_labels': gt_labels_vis.tolist(),
        'recon_coords': recon_coords_norm,
        'recon_labels': recon_labels_vis.tolist(),
        'accuracy': accuracy,
        'num_voxels': int(n_gt),
        'num_parts': int(len(np.unique(labels))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_dir', type=str, default='./data/seg_labels_test')
    parser.add_argument('--output', type=str, default='./output/vis_data.json')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--max_vis_points', type=int, default=20000)
    parser.add_argument('--encoder_path', type=str,
                        default='/path/to/trellis2_ckpts/tex_enc')
    parser.add_argument('--decoder_path', type=str,
                        default='/path/to/trellis2_ckpts/tex_dec')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("Loading models...")
    encoder = models.from_pretrained(args.encoder_path).eval().to(args.device)
    decoder = models.from_pretrained(args.decoder_path).eval().to(args.device)
    
    # Load metadata
    files = sorted([f for f in os.listdir(args.seg_dir) if f.endswith('.npz') and f != 'metadata.csv'])
    
    results = []
    for fname in tqdm(files, desc="Processing"):
        data = np.load(os.path.join(args.seg_dir, fname), allow_pickle=True)
        coords = data['coords']
        labels = data['labels']
        num_parts = int(data.get('num_parts', len(np.unique(labels))))
        model_id = fname.replace('.npz', '')
        type_id = str(data.get('type_id', ''))
        
        try:
            vis_data = process_sample(
                coords, labels, encoder, decoder,
                num_classes=args.num_classes, device=args.device,
                max_vis_points=args.max_vis_points
            )
            vis_data['model_id'] = model_id
            vis_data['type_id'] = type_id
            vis_data['filename'] = fname
            results.append(vis_data)
            print(f"  {fname}: acc={vis_data['accuracy']:.4f}, parts={vis_data['num_parts']}, voxels={vis_data['num_voxels']}")
        except Exception as e:
            print(f"  Error with {fname}: {e}")
            torch.cuda.empty_cache()
    
    # Save
    palette = generate_class_colors(args.num_classes)
    output = {
        'samples': results,
        'palette': palette,
        'num_classes': args.num_classes,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f)
    
    print(f"\nSaved visualization data to {args.output}")
    print(f"Samples: {len(results)}, Avg accuracy: {np.mean([r['accuracy'] for r in results]):.4f}")


if __name__ == '__main__':
    main()
