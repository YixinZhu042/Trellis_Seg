"""
Preprocess: Save voxelized seg labels for Seg VAE training.

Reuses step4's voxelization logic but saves the intermediate format
(coords + integer labels) instead of encoding through the Texture VAE.

Input:
    PartNeXt annotations + meshes

Output:
    {data_dir}/seg_voxels/{instance}.npz
        coords: (N, 3) int32 — voxel coordinates at resolution 512
        labels: (N,) int32 — per-voxel class labels [0, num_classes-1]

Usage:
    python preprocess/step4b_save_seg_voxels.py \
        --data_dir ./data \
        --num_samples 5
"""
import os
import sys
import json
import argparse
import numpy as np
import trimesh
import pyarrow.ipc as ipc
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from preprocess.step4_encode_seg_latent import parse_face_labels, voxelize_with_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/path/to/data')
    parser.add_argument('--arrow_dir', type=str, default='/path/to/PartNeXt')
    parser.add_argument('--mesh_dir', type=str, default='/path/to/PartNeXt_mesh/glbs')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    output_dir = os.path.join(args.data_dir, 'seg_voxels')
    os.makedirs(output_dir, exist_ok=True)

    # Get target instances (from seg_views)
    seg_views_dir = os.path.join(args.data_dir, 'seg_views')
    if not os.path.isdir(seg_views_dir):
        print(f"Error: {seg_views_dir} not found.")
        return
    target_instances = set(f[:-4] for f in os.listdir(seg_views_dir) if f.endswith('.npz'))

    # Load annotations
    arrow_files = sorted(glob(os.path.join(args.arrow_dir, 'data-*.arrow')))
    records = {}
    for af in arrow_files:
        with open(af, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                for i in range(batch.num_rows):
                    mid = batch.column('model_id')[i].as_py()
                    if mid in target_instances:
                        records[mid] = {
                            'type_id': batch.column('type_id')[i].as_py(),
                            'masks': batch.column('masks')[i].as_py(),
                            'hierarchy': batch.column('hierarchyList')[i].as_py(),
                        }
        if len(records) >= len(target_instances):
            break

    instances = sorted(records.keys())
    if args.num_samples:
        instances = instances[:args.num_samples]
    instances = instances[args.rank::args.world_size]

    print(f"[Rank {args.rank}] Processing {len(instances)} instances...")
    success, fail = 0, 0

    for instance in tqdm(instances, desc="Save seg_voxels"):
        out_path = os.path.join(output_dir, f'{instance}.npz')
        if os.path.exists(out_path):
            success += 1
            continue

        rec = records[instance]
        mesh_path = os.path.join(args.mesh_dir, rec['type_id'], f'{instance}.glb')
        if not os.path.exists(mesh_path):
            fail += 1
            continue

        try:
            scene = trimesh.load(mesh_path, force='scene')
            mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
            if not isinstance(mesh, trimesh.Trimesh):
                fail += 1
                continue

            face_labels, num_parts = parse_face_labels(
                rec['masks'], len(mesh.faces), hierarchy_json=rec.get('hierarchy'),
                max_parts=args.num_classes - 1,
            )
            if num_parts > args.num_classes - 1:
                raise ValueError(
                    f'too many parts after hierarchy merge: {num_parts} > {args.num_classes - 1}')
            coords, labels = voxelize_with_labels(mesh, face_labels, args.resolution)
            if coords is None:
                fail += 1
                continue

            # Silent clipping changes the part meaning and can disagree with
            # seg_views / seg_latent. Invalid labels must fail visibly.
            if labels.min(initial=0) < 0 or labels.max(initial=0) >= args.num_classes:
                raise ValueError(
                    f'label range [{labels.min()}, {labels.max()}] outside [0, {args.num_classes - 1}]')

            np.savez_compressed(out_path, coords=coords.astype(np.int32), labels=labels.astype(np.int32))
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  Error {instance[:8]}: {e}")

    print(f"\nDone! Success: {success}, Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
