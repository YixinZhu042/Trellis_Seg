"""
Step 5: Render ONE condition image per instance for DINO cross-attention condition.

Renders a single high-quality RGBA image from a random viewpoint.
DINO feature extraction happens ONLINE during training (trainer auto-encodes).
Optionally can pre-extract DINO features offline for faster training.

Input:
    GLB meshes from PartNeXt

Output:
    {data_dir}/render_cond/{instance}.png  (RGBA image, 512x512)
    
    Optional (--extract_dino):
    {data_dir}/dino_features/{instance}.npy  (N, D) float16 patch tokens

Usage:
    # Render only (DINO extracted online during training)
    python preprocess/step5_render_cond_images.py \
        --data_dir ./data \
        --resolution 512 \
        --num_samples 5 \
        --renderer blender

    # Render + offline DINO extraction
    python preprocess/step5_render_cond_images.py \
        --data_dir ./data \
        --extract_dino \
        --dino_model dinov2_vitl14_reg
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
import numpy as np
import trimesh
import pyarrow.ipc as ipc
from glob import glob
from tqdm import tqdm
from PIL import Image

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)


# ============================================================
# Blender renderer
# ============================================================

BLENDER_SCRIPT = '''
import bpy
import sys
import math
import mathutils
import numpy as np
import os
import json

# Batch manifest: one JSON per line, each {mesh_path, output_path, resolution, elevation, azimuth, seed}
argv = sys.argv[sys.argv.index("--") + 1:]
manifest_path = argv[0]
with open(manifest_path) as f:
    jobs = [json.loads(line) for line in f if line.strip()]

# ---- One-time GPU device setup (OptiX; persists across all renders in this process) ----
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for device in prefs.devices:
    device.use = (device.type in ('OPTIX', 'CUDA'))


def render_one(mesh_path, output_path, resolution, elevation, azimuth, seed):
    np.random.seed(seed)

    # Clear
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import
    if mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        bpy.ops.import_mesh.stl(filepath=mesh_path)

    # Normalize to [-0.5, 0.5] (same as TRELLIS.2)
    objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    all_verts = []
    for obj in objects:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)

    if all_verts:
        coords = [list(v) for v in all_verts]
        min_c = [min(c[i] for c in coords) for i in range(3)]
        max_c = [max(c[i] for c in coords) for i in range(3)]
        center = mathutils.Vector([(min_c[i] + max_c[i]) / 2 for i in range(3)])
        size = max(max_c[i] - min_c[i] for i in range(3))
        scale_f = 1.0 / size if size > 0 else 1.0

        # Clear all parents (keep transform) then apply
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Bake normalization into mesh vertices
        for obj in objects:
            me = obj.data
            for v in me.vertices:
                v.co = (v.co - center) * scale_f

    # ---- Render settings (matching TRELLIS.2) ----
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3

    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # ---- Random lighting (matching TRELLIS.2) ----
    cam_elev = math.radians(elevation)
    cam_azim = math.radians(azimuth)
    cam_dir = np.array([
        math.cos(cam_elev) * math.sin(cam_azim),
        -math.cos(cam_elev) * math.cos(cam_azim),
        math.sin(cam_elev)
    ])
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    total_energy = 1.5
    num_lights = np.random.randint(1, 4)  # 1~3 lights

    for i in range(num_lights):
        light_dir = np.random.randn(3)
        light_dir[2] += 0.6  # bias upward
        light_dir = light_dir / np.linalg.norm(light_dir)

        distance = 1.0 / np.random.uniform(1.0/100, 1.0/10)

        alignment = max(np.dot(light_dir, cam_dir), 0.1)
        light_energy = (total_energy / num_lights) * alignment
        actual_energy = light_energy * distance * distance * 31.4
        total_energy -= light_energy

        loc = light_dir * distance
        bpy.ops.object.light_add(type='POINT', location=tuple(loc))
        light = bpy.context.active_object
        light.data.energy = actual_energy
        light.data.color = (1.0, 1.0, 1.0)
        light.data.shadow_soft_size = np.random.uniform(0.1, 0.1 * distance)

    # Remaining energy as white environment light
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs['Strength'].default_value = max(total_energy, 0.1)
    output_node = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # ---- Camera (matching TRELLIS.2 style) ----
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    camera.data.sensor_width = 32
    camera.data.sensor_height = 32

    fov_min_deg, fov_max_deg = 10, 70
    radius_min = math.sqrt(3)/2 / math.sin(math.radians(fov_max_deg) / 2)
    radius_max = math.sqrt(3)/2 / math.sin(math.radians(fov_min_deg) / 2)
    k = np.random.uniform(1.0/radius_max**2, 1.0/radius_min**2)
    dist = 1.0 / math.sqrt(k)
    fov = 2 * math.asin(math.sqrt(3)/2 / dist)
    camera.data.lens = 16.0 / math.tan(fov / 2)

    elev = math.radians(elevation)
    azim = math.radians(azimuth)
    x = dist * math.cos(elev) * math.sin(azim)
    y = -dist * math.cos(elev) * math.cos(azim)
    z = dist * math.sin(elev)
    camera.location = (x, y, z)

    track = camera.constraints.new(type='TRACK_TO')
    track.target = bpy.data.objects.new('CamTarget', None)
    bpy.context.scene.collection.objects.link(track.target)
    track.target.location = (0, 0, 0)
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


# ---- Batch loop: one process renders all jobs (kernel compiled once, no per-mesh startup) ----
for job in jobs:
    try:
        render_one(job['mesh_path'], job['output_path'], int(job['resolution']),
                   float(job['elevation']), float(job['azimuth']), int(job['seed']))
        print("RENDER_DONE " + job['output_path'], flush=True)
    except Exception as e:
        print("RENDER_FAIL " + job['output_path'] + " " + repr(e), flush=True)
'''


def render_blender_batch(jobs, blender_path='blender'):
    """Render a batch of jobs in a single Blender process (OptiX).

    jobs: list of dicts {mesh_path, output_path, resolution, elevation, azimuth, seed}.
    Returns the set of output_paths that rendered successfully.
    Running one process for many meshes avoids per-mesh startup + kernel reload.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for job in jobs:
            f.write(json.dumps(job) + '\n')
        manifest_path = f.name

    done = set()
    try:
        proc = subprocess.Popen(
            [blender_path, '--background', '--python', _write_batch_script(), '--', manifest_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            if line.startswith('RENDER_DONE '):
                out = line[len('RENDER_DONE '):].strip()
                done.add(out)
                _BATCH_PROGRESS['bar'].update(1)
            elif line.startswith('RENDER_FAIL '):
                _BATCH_PROGRESS['bar'].update(1)
        proc.wait()
    finally:
        os.unlink(manifest_path)
    return done


_BATCH_SCRIPT_PATH = None
_BATCH_PROGRESS = {'bar': None}


def _write_batch_script():
    global _BATCH_SCRIPT_PATH
    if _BATCH_SCRIPT_PATH is None or not os.path.exists(_BATCH_SCRIPT_PATH):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        f.write(BLENDER_SCRIPT)
        f.close()
        _BATCH_SCRIPT_PATH = f.name
    return _BATCH_SCRIPT_PATH


def render_blender(mesh_path, output_path, resolution, elevation, azimuth, blender_path='blender', seed=42):
    """Render a single view (kept for one-off use / fallback)."""
    done = render_blender_batch([{
        'mesh_path': mesh_path, 'output_path': output_path, 'resolution': resolution,
        'elevation': elevation, 'azimuth': azimuth, 'seed': seed,
    }], blender_path=blender_path)
    return output_path in done


# ============================================================
# Pyrender fallback
# ============================================================

def render_pyrender(mesh_path, output_path, resolution, elevation, azimuth):
    """Render single view with pyrender."""
    try:
        import pyrender
    except ImportError:
        return render_trimesh_fallback(mesh_path, output_path, resolution)
    
    mesh = trimesh.load(mesh_path, force='mesh')
    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    scale = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    mesh.vertices = (mesh.vertices - center) / scale * 2
    
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    dist = 3.5
    el, az = np.radians(elevation), np.radians(azimuth)
    cam_pos = np.array([dist * np.cos(el) * np.sin(az), dist * np.sin(el), dist * np.cos(el) * np.cos(az)])
    
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 1, 0.])
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0, 0, 1.])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    
    cam_matrix = np.eye(4)
    cam_matrix[:3, 0] = right
    cam_matrix[:3, 1] = up
    cam_matrix[:3, 2] = -forward
    cam_matrix[:3, 3] = cam_pos
    scene.add(camera, pose=cam_matrix)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_matrix)
    
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    alpha = (depth > 0).astype(np.uint8) * 255
    rgba = np.concatenate([color, alpha[:, :, None]], axis=2)
    Image.fromarray(rgba).save(output_path)
    return True


def render_nvdiffrast(mesh_path, output_path, resolution, elevation, azimuth):
    """High-quality GPU rendering using nvdiffrast."""
    import torch
    import nvdiffrast.torch as dr
    
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Normalize to [-1, 1]
    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    scale = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    verts = (mesh.vertices - center) / scale * 1.8
    
    # Camera
    el, az = np.radians(elevation), np.radians(azimuth)
    dist = 3.0
    cam_pos = np.array([
        dist * np.cos(el) * np.sin(az),
        dist * np.sin(el),
        dist * np.cos(el) * np.cos(az),
    ])
    
    # Look-at matrix
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 1, 0.])
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0, 0, 1.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    
    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -forward
    view[:3, 3] = cam_pos
    view = np.linalg.inv(view)
    
    # Perspective projection
    fov = 45.0
    aspect = 1.0
    near, far = 0.1, 10.0
    f = 1.0 / np.tan(np.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1
    
    mvp = proj @ view
    
    # To torch
    verts_t = torch.tensor(verts, dtype=torch.float32, device='cuda')
    faces_t = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')
    mvp_t = torch.tensor(mvp, dtype=torch.float32, device='cuda').unsqueeze(0)
    
    # Transform vertices
    verts_homo = torch.cat([verts_t, torch.ones(verts_t.shape[0], 1, device='cuda')], dim=-1)
    verts_clip = (mvp_t @ verts_homo.unsqueeze(-1)).squeeze(-1)  # (1, V, 4)
    verts_clip = verts_clip.unsqueeze(0) if verts_clip.dim() == 2 else verts_clip
    
    # Rasterize
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, verts_clip, faces_t, resolution=[resolution, resolution])
    
    # rast shape: (1, H, W, 4) where channel 3 is triangle_id (1-indexed, 0=background)
    face_idx = rast[0, :, :, 3].long() - 1  # (H, W), -1 means background
    mask = face_idx >= 0  # (H, W) bool
    
    # Compute face normals for shading
    face_normals = torch.tensor(mesh.face_normals, dtype=torch.float32, device='cuda')
    
    # Shade: simple diffuse lighting
    light_dir = torch.tensor([0.5, 0.7, 0.3], device='cuda')
    light_dir = light_dir / light_dir.norm()
    
    image = torch.zeros(resolution, resolution, 4, device='cuda')
    if mask.any():
        valid_face_idx = face_idx[mask]  # (N_pixels,)
        normals = face_normals[valid_face_idx]  # (N_pixels, 3)
        shade = (normals * light_dir).sum(-1).clamp(0.2, 1.0)  # (N_pixels,)
        
        # Base color (light gray with shading)
        color = torch.ones(valid_face_idx.shape[0], 3, device='cuda') * 0.75
        color = color * shade.unsqueeze(-1)
        
        # Write to image using flat indexing
        flat_mask = mask.reshape(-1)  # (H*W,)
        image_flat = image.reshape(-1, 4)  # (H*W, 4)
        image_flat[flat_mask, :3] = color
        image_flat[flat_mask, 3] = 1.0
        image = image_flat.reshape(resolution, resolution, 4)
    
    # To uint8
    image_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_np, 'RGBA').save(output_path)
    return True


def render_trimesh_fallback(mesh_path, output_path, resolution):
    """Minimal fallback."""
    Image.new('RGBA', (resolution, resolution), (200, 200, 200, 255)).save(output_path)
    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/path/to/data')
    parser.add_argument('--mesh_dir', type=str,
                        default='/path/to/PartNeXt_mesh/glbs')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0, help='Worker rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of workers')
    parser.add_argument('--renderer', type=str, default='nvdiffrast',
                        choices=['nvdiffrast', 'blender', 'pyrender', 'trimesh'])
    parser.add_argument('--blender_path', type=str, default='blender')
    parser.add_argument('--extract_dino', action='store_true',
                        help='Also extract DINO features offline (optional, normally done online)')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitl14_reg',
                        help='DINOv2 model name for offline extraction')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.data_dir, 'render_cond')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get instances
    seg_views_dir = os.path.join(args.data_dir, 'seg_views')
    if not os.path.isdir(seg_views_dir):
        print(f"Error: {seg_views_dir} not found. Run step1 first.")
        return
    target_instances = sorted([f[:-4] for f in os.listdir(seg_views_dir) if f.endswith('.npz')])
    if args.num_samples:
        target_instances = target_instances[:args.num_samples]
    # Shard by rank (num_samples applied first, so a fixed cap splits evenly across workers)
    target_instances = target_instances[args.rank::args.world_size]
    
    # Build id→type mapping
    arrow_dir = '/path/to/PartNeXt'
    id_to_type = {}
    for af in sorted(glob(os.path.join(arrow_dir, 'data-*.arrow'))):
        with open(af, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                for i in range(batch.num_rows):
                    id_to_type[batch.column('model_id')[i].as_py()] = batch.column('type_id')[i].as_py()
        if len(id_to_type) > 50000:
            break
    
    # Check renderer availability
    if args.renderer == 'nvdiffrast':
        try:
            import nvdiffrast.torch as dr
            dr.RasterizeCudaContext()
            print("Using nvdiffrast (GPU rasterizer)")
        except:
            print("nvdiffrast not available, falling back to blender")
            args.renderer = 'blender'
    
    if args.renderer == 'blender':
        try:
            subprocess.run([args.blender_path, '--version'], capture_output=True, timeout=10)
            print(f"Using Blender: {args.blender_path}")
        except:
            print("Blender not found, falling back to pyrender")
            args.renderer = 'pyrender'
    
    np.random.seed(42)

    print(f"Rendering {len(target_instances)} instances × 1 view ({args.renderer})...")
    success, fail = 0, 0

    if args.renderer == 'blender':
        # Build the full job list first (deterministic viewpoints), then render the
        # whole batch in ONE Blender process: OptiX kernel compiles once, no per-mesh
        # process startup. This is ~15-30x faster than one subprocess per mesh.
        jobs = []
        for instance in target_instances:
            out_path = os.path.join(output_dir, f'{instance}.png')
            if os.path.exists(out_path):
                success += 1
                continue
            type_id = id_to_type.get(instance)
            if not type_id:
                fail += 1
                continue
            mesh_path = os.path.join(args.mesh_dir, type_id, f'{instance}.glb')
            if not os.path.exists(mesh_path):
                fail += 1
                continue
            # Draw viewpoints in the same order as before (seed 42) for reproducibility
            elevation = float(np.random.uniform(-10, 40))
            azimuth = float(np.random.uniform(0, 360))
            render_seed = int(np.random.randint(0, 100000))
            jobs.append({
                'mesh_path': mesh_path, 'output_path': out_path,
                'resolution': args.resolution, 'elevation': elevation,
                'azimuth': azimuth, 'seed': render_seed,
            })

        if jobs:
            _BATCH_PROGRESS['bar'] = tqdm(total=len(jobs), desc="Blender render")
            done = render_blender_batch(jobs, blender_path=args.blender_path)
            _BATCH_PROGRESS['bar'].close()
            for job in jobs:
                if job['output_path'] in done and os.path.exists(job['output_path']):
                    success += 1
                else:
                    fail += 1
    else:
        for instance in tqdm(target_instances):
            out_path = os.path.join(output_dir, f'{instance}.png')
            if os.path.exists(out_path):
                success += 1
                continue

            type_id = id_to_type.get(instance)
            if not type_id:
                fail += 1
                continue

            mesh_path = os.path.join(args.mesh_dir, type_id, f'{instance}.glb')
            if not os.path.exists(mesh_path):
                fail += 1
                continue

            # Random viewpoint (TRELLIS.2 style: elevation biased upward)
            elevation = np.random.uniform(-10, 40)
            azimuth = np.random.uniform(0, 360)
            render_seed = np.random.randint(0, 100000)

            try:
                if args.renderer == 'nvdiffrast':
                    ok = render_nvdiffrast(mesh_path, out_path, args.resolution, elevation, azimuth)
                elif args.renderer == 'pyrender':
                    ok = render_pyrender(mesh_path, out_path, args.resolution, elevation, azimuth)
                else:
                    ok = render_trimesh_fallback(mesh_path, out_path, args.resolution)

                if ok:
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  Error {instance[:8]}: {e}")

    print(f"\nRendering done! Success: {success}, Failed: {fail}")
    print(f"Output: {output_dir}/*.png")
    
    # Optional: offline DINO feature extraction
    if args.extract_dino:
        print(f"\nExtracting DINO features ({args.dino_model})...")
        import torch
        from trellis2.modules.image_feature_extractor import DinoV2FeatureExtractor
        
        dino_dir = os.path.join(args.data_dir, 'dino_features')
        os.makedirs(dino_dir, exist_ok=True)
        
        extractor = DinoV2FeatureExtractor(args.dino_model)
        extractor.cuda()
        
        rendered_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        
        for fname in tqdm(rendered_files, desc="DINO"):
            instance = fname[:-4]
            feat_path = os.path.join(dino_dir, f'{instance}.npy')
            if os.path.exists(feat_path):
                continue
            
            img = Image.open(os.path.join(output_dir, fname)).convert('RGBA')
            # White background composite
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            rgb = bg.convert('RGB')
            
            with torch.no_grad():
                features = extractor([rgb])  # (1, N, D)
            
            np.save(feat_path, features[0].cpu().numpy().astype(np.float16))
        
        print(f"DINO features saved to {dino_dir}/")
    else:
        print("\nNote: DINO features will be extracted ONLINE during training.")
        print("  To pre-extract offline, add --extract_dino flag.")


if __name__ == '__main__':
    main()
