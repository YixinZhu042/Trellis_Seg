"""
Render 6 orthographic views of a mesh for segmentation input.

Output: 6 RGBA PNG images (512x512) with transparent background.
Supports HDR environment map lighting (EXR) for realistic shading.
View order matches the training pipeline (step1/step2):
  front(+Z), right(+X), back(-Z), left(-X), top(+Y), bottom(-Y)

Usage:
    python scripts/render_views.py \
        --mesh /path/to/input.glb \
        --output_dir ./results/my_model_renders \
        --resolution 512 \
        --env_map /path/to/studio.exr
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

DEFAULT_ENV_MAP = '/path/to/Trellis_Seg/studio_small_08_4k.exr'


def load_env_map(path, device='cuda'):
    """Load HDR environment map as (H, W, 3) float tensor."""
    import OpenEXR, Imath

    f = OpenEXR.InputFile(path)
    header = f.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = {}
    for ch in ['R', 'G', 'B']:
        raw = f.channel(ch, float_type)
        channels[ch] = np.frombuffer(raw, dtype=np.float32).reshape(h, w)

    env = np.stack([channels['R'], channels['G'], channels['B']], axis=-1)
    return torch.from_numpy(env).to(device)


def sample_env_map(env_map, normals):
    """Sample environment map using normals (equirectangular projection)."""
    H, W = env_map.shape[:2]
    theta = torch.acos(normals[:, 1].clamp(-1, 1))
    phi = torch.atan2(normals[:, 0], normals[:, 2])

    u = (phi / (2 * np.pi) + 0.5) % 1.0
    v = theta / np.pi

    px = (u * (W - 1)).long().clamp(0, W - 1)
    py = (v * (H - 1)).long().clamp(0, H - 1)
    return env_map[py, px]


def render_views(mesh, resolution=512, device='cuda', env_map_path=None):
    """Render 6 orthographic views with IBL or simple lighting."""
    import nvdiffrast.torch as dr

    verts = mesh.vertices.astype(np.float32)
    bbox_min = verts.min(0)
    bbox_max = verts.max(0)
    center = (bbox_min + bbox_max) / 2
    max_dim = (bbox_max - bbox_min).max()
    if max_dim > 0:
        verts = (verts - center) / max_dim * 2

    try:
        vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32) / 255.0
    except Exception:
        vc = np.ones((len(mesh.vertices), 3), dtype=np.float32) * 0.7

    V = torch.from_numpy(verts).to(device)
    tri = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
    colors = torch.from_numpy(vc).to(device)

    # Compute vertex normals
    v0 = V[tri[:, 0]]; v1 = V[tri[:, 1]]; v2 = V[tri[:, 2]]
    face_normals = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    vert_normals = torch.zeros_like(V)
    vert_normals.index_add_(0, tri[:, 0], face_normals)
    vert_normals.index_add_(0, tri[:, 1], face_normals)
    vert_normals.index_add_(0, tri[:, 2], face_normals)
    vert_normals = torch.nn.functional.normalize(vert_normals, dim=-1)

    # Load environment map; fall back to directional lighting if OpenEXR is unavailable.
    env_map = None
    if env_map_path and os.path.exists(env_map_path):
        try:
            env_map = load_env_map(env_map_path, device)
            env_map = torch.nn.functional.interpolate(
                env_map.permute(2, 0, 1).unsqueeze(0), size=(256, 512), mode='bilinear'
            ).squeeze(0).permute(1, 2, 0)
        except (ImportError, OSError, ValueError) as exc:
            print(f'Warning: cannot load environment map ({exc}); using directional lighting instead.')

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

    normals_attr = vert_normals[None].expand(6, -1, -1).contiguous()
    normals_interp, _ = dr.interpolate(normals_attr, rast, tri)

    mask = (rast[..., 3:4] > 0).float()

    # Apply lighting
    for i, cfg in enumerate(VIEW_CONFIGS):
        n = torch.nn.functional.normalize(normals_interp[i].reshape(-1, 3), dim=-1)
        if env_map is not None:
            irradiance = sample_env_map(env_map, n)
            irradiance = irradiance.reshape(resolution, resolution, 3)
            irradiance = irradiance / (irradiance + 1.0)  # Reinhard tone mapping
            # Stronger lighting: ambient floor + boosted env lighting
            shading = 0.5 + 1.2 * irradiance
            rendered[i] = rendered[i] * shading
        else:
            light_dir = torch.zeros(3, device=device)
            light_dir[cfg['axis']] = -cfg['sign']
            ndotl = (n * light_dir).sum(dim=-1, keepdim=True).clamp(0, 1).reshape(resolution, resolution, 1)
            shading = 0.5 + 0.8 * ndotl
            rendered[i] = rendered[i] * shading

    rendered = rendered * mask + (1.0 - mask)  # white bg
    rendered = rendered.flip(1)

    occupancy = rast[..., 3].flip(1) > 0

    images = []
    for i in range(6):
        img = (rendered[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        images.append(img)

    return images, occupancy.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Render 6 orthographic views of a mesh')
    parser.add_argument('--mesh', type=str, required=True, help='Input mesh path (.glb/.obj)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution')
    parser.add_argument('--env_map', type=str, default=DEFAULT_ENV_MAP, help='HDR environment map (.exr)')
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

    env_path = args.env_map if os.path.exists(args.env_map) else None
    if env_path:
        print(f'Using environment map: {env_path}')
    else:
        print('No env map found, using simple directional lighting')

    print(f'Rendering 6 views at {args.resolution}x{args.resolution}...')
    images, occupancy = render_views(mesh, args.resolution, args.device, env_path)

    for i, cfg in enumerate(VIEW_CONFIGS):
        alpha = (occupancy[i] * 255).astype(np.uint8)
        rgba = np.concatenate([images[i], alpha[:, :, None]], axis=-1)
        path = os.path.join(args.output_dir, f'{cfg["name"]}.png')
        Image.fromarray(rgba, 'RGBA').save(path)
        print(f'  Saved: {path}')

    occ_path = os.path.join(args.output_dir, 'occupancy.npy')
    np.save(occ_path, occupancy)

    print(f'\nDone! All views saved to {args.output_dir}')


if __name__ == '__main__':
    main()
