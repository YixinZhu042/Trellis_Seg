"""
Render 6 orthographic views of a mesh for segmentation input.

Output: 6 PNG images (512x512) with white background.
View order matches the training pipeline (step1/step2):
  front(+Z), right(+X), back(-Z), left(-X), top(+Y), bottom(-Y)

Usage:
    python scripts/render_views.py \
        --mesh /path/to/input.glb \
        --output_dir ./results/my_model_renders \
        --resolution 512
"""
import os
import sys
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

VIEW_CONFIGS = [
    {'name': 'front',  'axis': 2, 'sign': +1, 'up': 1, 'right': 0, 'flip_right': False},
    {'name': 'right',  'axis': 0, 'sign': +1, 'up': 1, 'right': 2, 'flip_right': True},
    {'name': 'back',   'axis': 2, 'sign': -1, 'up': 1, 'right': 0, 'flip_right': True},
    {'name': 'left',   'axis': 0, 'sign': -1, 'up': 1, 'right': 2, 'flip_right': False},
    {'name': 'top',    'axis': 1, 'sign': +1, 'up': 2, 'right': 0, 'flip_right': False},
    {'name': 'bottom', 'axis': 1, 'sign': -1, 'up': 2, 'right': 0, 'flip_right': False},
]


def render_views(mesh, resolution=512, device='cuda'):
    """Render 6 orthographic RGB views using nvdiffrast.

    Args:
        mesh: trimesh.Trimesh (will be normalized to [-1, 1])
        resolution: output image size
        device: torch device

    Returns:
        list of 6 numpy arrays (H, W, 3) uint8
    """
    import nvdiffrast.torch as dr

    verts = mesh.vertices.astype(np.float32)
    bbox_min = verts.min(0)
    bbox_max = verts.max(0)
    center = (bbox_min + bbox_max) / 2
    max_dim = (bbox_max - bbox_min).max()
    if max_dim > 0:
        verts = (verts - center) / max_dim * 2

    # Get vertex colors from texture or default
    try:
        vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32) / 255.0
    except Exception:
        vc = np.ones((len(mesh.vertices), 3), dtype=np.float32) * 0.7

    V = torch.from_numpy(verts).to(device)
    tri = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
    colors = torch.from_numpy(vc).to(device)

    glctx = dr.RasterizeCudaContext()

    clips = []
    for cfg in VIEW_CONFIGS:
        px = V[:, cfg['right']].clone()
        py = V[:, cfg['up']]
        pd = V[:, cfg['axis']] * cfg['sign']
        if cfg['flip_right']:
            px = -px
        w = torch.ones_like(px)
        clips.append(torch.stack([px, py, -pd, w], dim=-1))

    pos = torch.stack(clips, dim=0).contiguous()
    rast, _ = dr.rasterize(glctx, pos, tri, resolution=[resolution, resolution])

    colors_attr = colors[None].expand(6, -1, -1).contiguous()
    rendered, _ = dr.interpolate(colors_attr, rast, tri)

    mask = (rast[..., 3:4] > 0).float()
    rendered = rendered * mask + (1.0 - mask)  # white background
    rendered = rendered.flip(1)  # flip Y to match image convention

    # Occupancy mask: (6, H, W) bool — True where mesh is visible
    occupancy = rast[..., 3].flip(1) > 0  # same flip as rendered

    images = []
    for i in range(6):
        img = (rendered[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        images.append(img)

    return images, occupancy.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Render 6 orthographic views of a mesh')
    parser.add_argument('--mesh', type=str, required=True, help='Input mesh path (.glb/.obj)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for rendered PNGs')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading mesh: {args.mesh}')
    scene = trimesh.load(args.mesh)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    else:
        mesh = scene
    print(f'  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces')

    print(f'Rendering 6 views at {args.resolution}x{args.resolution}...')
    images, occupancy = render_views(mesh, args.resolution, args.device)

    for i, cfg in enumerate(VIEW_CONFIGS):
        path = os.path.join(args.output_dir, f'{cfg["name"]}.png')
        Image.fromarray(images[i]).save(path)
        print(f'  Saved: {path}')

    # Save occupancy mask (used to mask seg_maps background)
    occ_path = os.path.join(args.output_dir, 'occupancy.npy')
    np.save(occ_path, occupancy)
    print(f'  Saved: {occ_path} (6, {args.resolution}, {args.resolution}) bool')

    print(f'\nDone! All views saved to {args.output_dir}')
    print(f'Next: run 2D segmentation, then mask with occupancy:')
    print(f'  seg_maps[~occupancy] = 0')


if __name__ == '__main__':
    main()
