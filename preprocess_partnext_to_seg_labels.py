"""
Preprocess PartNeXt dataset to voxelized seg_labels for VAE reconstruction testing.

Pipeline:
    1. Read PartNeXt arrow dataset (model_id, type_id, masks, hierarchyList)
    2. Load corresponding GLB mesh from PartNeXt_mesh/glbs/{type_id}/{model_id}.glb
    3. Assign per-face part labels from masks annotation
    4. Voxelize the mesh: sample points on surface → quantize to voxel grid
    5. Each voxel gets the part label of the face it was sampled from
    6. Save as seg_labels/{model_id}.npz with coords (N,3) and labels (N,)

Usage:
    python preprocess_partnext_to_seg_labels.py \
        --arrow_dir /path/to/PartNeXt \
        --mesh_dir /path/to/PartNeXt_mesh/glbs \
        --output_dir ./data/seg_labels \
        --resolution 256 \
        --num_samples 100 \
        --points_per_face 10
"""
import os
import sys
import json
import argparse
import numpy as np
import trimesh
import pyarrow.ipc as ipc
from tqdm import tqdm
from glob import glob


def load_partnext_annotations(arrow_dir, max_records=None):
    """
    Load all PartNeXt annotations from arrow files.
    
    Returns list of dicts with keys: model_id, type_id, masks, hierarchyList
    """
    arrow_files = sorted(glob(os.path.join(arrow_dir, 'data-*.arrow')))
    records = []
    
    for arrow_file in arrow_files:
        with open(arrow_file, 'rb') as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                
                for i in range(batch.num_rows):
                    rec = {
                        'model_id': batch.column('model_id')[i].as_py(),
                        'type_id': batch.column('type_id')[i].as_py(),
                        'masks': batch.column('masks')[i].as_py(),
                        'hierarchyList': batch.column('hierarchyList')[i].as_py(),
                    }
                    records.append(rec)
                    
                    if max_records and len(records) >= max_records:
                        return records
    
    return records


def parse_face_labels(masks_json, num_faces):
    """
    Parse PartNeXt masks annotation to per-face labels.
    
    masks format: {"part_id": {"submesh_id": [face_indices...]}, ...}
    Each top-level key is a part, value contains face indices for that part.
    
    Returns:
        face_labels: (num_faces,) int array, -1 for unlabeled faces
        num_parts: number of unique parts
    """
    masks = json.loads(masks_json) if isinstance(masks_json, str) else masks_json
    
    face_labels = np.full(num_faces, -1, dtype=np.int32)
    
    part_id = 0
    for part_key in sorted(masks.keys(), key=int):
        submeshes = masks[part_key]
        for submesh_key, face_indices in submeshes.items():
            for fi in face_indices:
                if fi < num_faces:
                    face_labels[fi] = part_id
        part_id += 1
    
    return face_labels, part_id


def voxelize_mesh_with_labels(mesh, face_labels, resolution=256, points_per_face=10):
    """
    Voxelize a mesh with per-face labels.
    
    Strategy:
        1. Sample points uniformly on each face
        2. Each sampled point inherits the face's label
        3. Quantize points to voxel grid
        4. For duplicate voxels, use majority vote for label
    
    Args:
        mesh: trimesh.Trimesh
        face_labels: (F,) int array
        resolution: voxel grid resolution
        points_per_face: points to sample per face
    
    Returns:
        coords: (N, 3) int array - unique voxel coordinates
        labels: (N,) int array - per-voxel labels
    """
    # Sample points on mesh surface with face index tracking
    num_faces = len(mesh.faces)
    
    # Compute area-weighted sampling count per face
    face_areas = mesh.area_faces
    total_points = min(num_faces * points_per_face, 500000)  # Cap total points
    
    # Sample proportionally to area, but ensure at least 1 point per labeled face
    labeled_mask = face_labels >= 0
    if labeled_mask.sum() == 0:
        return None, None
    
    # Use trimesh's built-in sampling with face tracking
    points, face_indices = mesh.sample(total_points, return_index=True)
    
    # Get labels for sampled points
    point_labels = face_labels[face_indices]
    
    # Filter out points from unlabeled faces
    valid = point_labels >= 0
    points = points[valid]
    point_labels = point_labels[valid]
    
    if len(points) == 0:
        return None, None
    
    # Normalize points to [0, resolution-1] range
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Avoid division by zero
    bbox_size = np.maximum(bbox_size, 1e-8)
    
    # Scale to fit in [1, resolution-2] (leave 1 voxel border)
    scale = (resolution - 3) / bbox_size.max()
    points_normalized = (points - bbox_min) * scale + 1.0
    
    # Quantize to integer voxel coordinates
    voxel_coords = points_normalized.astype(np.int32)
    voxel_coords = np.clip(voxel_coords, 0, resolution - 1)
    
    # Deduplicate voxels with majority vote for labels
    # Hash coordinates for grouping
    coord_hash = (voxel_coords[:, 0].astype(np.int64) * resolution * resolution + 
                  voxel_coords[:, 1].astype(np.int64) * resolution + 
                  voxel_coords[:, 2].astype(np.int64))
    
    # Find unique voxels
    unique_hashes, inverse_indices = np.unique(coord_hash, return_inverse=True)
    num_unique = len(unique_hashes)
    
    # Majority vote for each unique voxel
    unique_coords = np.zeros((num_unique, 3), dtype=np.int32)
    unique_labels = np.zeros(num_unique, dtype=np.int32)
    
    for i in range(num_unique):
        mask = inverse_indices == i
        unique_coords[i] = voxel_coords[mask][0]
        # Majority vote
        lbls = point_labels[mask]
        counts = np.bincount(lbls)
        unique_labels[i] = counts.argmax()
    
    return unique_coords, unique_labels


def process_one_sample(record, mesh_dir, resolution, points_per_face):
    """
    Process a single PartNeXt record: load mesh, assign labels, voxelize.
    
    Returns:
        coords: (N, 3) or None
        labels: (N,) or None  
        num_parts: int
        error: str or None
    """
    model_id = record['model_id']
    type_id = record['type_id']
    
    # Load mesh
    mesh_path = os.path.join(mesh_dir, type_id, f'{model_id}.glb')
    if not os.path.exists(mesh_path):
        return None, None, 0, f"Mesh not found: {mesh_path}"
    
    try:
        scene = trimesh.load(mesh_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            mesh = scene.dump(concatenate=True)
        else:
            mesh = scene
        
        if not isinstance(mesh, trimesh.Trimesh):
            return None, None, 0, "Failed to get Trimesh"
            
    except Exception as e:
        return None, None, 0, f"Failed to load mesh: {e}"
    
    num_faces = len(mesh.faces)
    
    # Parse face labels
    try:
        face_labels, num_parts = parse_face_labels(record['masks'], num_faces)
    except Exception as e:
        return None, None, 0, f"Failed to parse masks: {e}"
    
    # Check labeling coverage
    labeled_ratio = (face_labels >= 0).sum() / num_faces
    if labeled_ratio < 0.5:
        return None, None, num_parts, f"Low label coverage: {labeled_ratio:.2%}"
    
    # Voxelize
    try:
        coords, labels = voxelize_mesh_with_labels(
            mesh, face_labels, resolution, points_per_face
        )
    except Exception as e:
        return None, None, num_parts, f"Voxelization failed: {e}"
    
    if coords is None:
        return None, None, num_parts, "No valid voxels"
    
    return coords, labels, num_parts, None


def main():
    parser = argparse.ArgumentParser(description='Preprocess PartNeXt to voxelized seg_labels')
    parser.add_argument('--arrow_dir', type=str, 
                        default='/path/to/PartNeXt',
                        help='Directory containing PartNeXt arrow files')
    parser.add_argument('--mesh_dir', type=str,
                        default='/path/to/PartNeXt_mesh/glbs',
                        help='Directory containing GLB meshes')
    parser.add_argument('--output_dir', type=str, default='./data/seg_labels',
                        help='Output directory for seg_labels npz files')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Voxel grid resolution')
    parser.add_argument('--points_per_face', type=int, default=10,
                        help='Number of sample points per face for voxelization')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Max number of samples to process (None=all)')
    parser.add_argument('--min_voxels', type=int, default=5000,
                        help='Minimum number of voxels to keep a sample')
    parser.add_argument('--max_voxels', type=int, default=200000,
                        help='Maximum number of voxels (subsample if exceeded)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {args.arrow_dir}...")
    records = load_partnext_annotations(args.arrow_dir, max_records=args.num_samples)
    print(f"Loaded {len(records)} annotations")
    
    # Process each sample
    stats = {
        'total': len(records),
        'success': 0,
        'skipped': 0,
        'errors': [],
        'num_parts_distribution': {},
        'voxel_counts': [],
    }
    
    metadata = []
    
    for record in tqdm(records, desc="Processing"):
        model_id = record['model_id']
        output_path = os.path.join(args.output_dir, f'{model_id}.npz')
        
        # Skip if already processed
        if os.path.exists(output_path):
            stats['success'] += 1
            continue
        
        coords, labels, num_parts, error = process_one_sample(
            record, args.mesh_dir, args.resolution, args.points_per_face
        )
        
        if error:
            stats['skipped'] += 1
            stats['errors'].append(f"{model_id}: {error}")
            continue
        
        # Filter by voxel count
        if coords.shape[0] < args.min_voxels:
            stats['skipped'] += 1
            stats['errors'].append(f"{model_id}: Too few voxels ({coords.shape[0]})")
            continue
        
        # Subsample if too many voxels
        if coords.shape[0] > args.max_voxels:
            indices = np.random.choice(coords.shape[0], args.max_voxels, replace=False)
            coords = coords[indices]
            labels = labels[indices]
        
        # Remap labels to consecutive integers [0, num_actual_parts)
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int32)
        num_actual_parts = len(unique_labels)
        
        # Save
        np.savez_compressed(
            output_path,
            coords=coords,
            labels=labels,
            num_parts=num_actual_parts,
            model_id=model_id,
            type_id=record['type_id'],
        )
        
        stats['success'] += 1
        stats['voxel_counts'].append(coords.shape[0])
        stats['num_parts_distribution'][num_actual_parts] = \
            stats['num_parts_distribution'].get(num_actual_parts, 0) + 1
        
        metadata.append({
            'model_id': model_id,
            'type_id': record['type_id'],
            'num_voxels': int(coords.shape[0]),
            'num_parts': int(num_actual_parts),
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total records:  {stats['total']}")
    print(f"Success:        {stats['success']}")
    print(f"Skipped:        {stats['skipped']}")
    
    if stats['voxel_counts']:
        vc = np.array(stats['voxel_counts'])
        print(f"\nVoxel count statistics:")
        print(f"  Min:    {vc.min()}")
        print(f"  Max:    {vc.max()}")
        print(f"  Mean:   {vc.mean():.0f}")
        print(f"  Median: {np.median(vc):.0f}")
    
    if stats['num_parts_distribution']:
        print(f"\nParts distribution:")
        for k in sorted(stats['num_parts_distribution'].keys()):
            print(f"  {k} parts: {stats['num_parts_distribution'][k]} samples")
    
    if stats['errors'][:10]:
        print(f"\nFirst 10 errors:")
        for e in stats['errors'][:10]:
            print(f"  {e}")
    
    # Save metadata
    if metadata:
        import csv
        meta_path = os.path.join(args.output_dir, 'metadata.csv')
        with open(meta_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model_id', 'type_id', 'num_voxels', 'num_parts'])
            writer.writeheader()
            writer.writerows(metadata)
        print(f"\nMetadata saved to {meta_path}")
    
    print(f"\nOutput saved to {args.output_dir}")


if __name__ == '__main__':
    main()
