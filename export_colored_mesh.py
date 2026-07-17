"""
Export PartNeXt meshes with face colors = part class color.
Saves as GLB files for visualization in HTML (Three.js).
"""
import os
import sys
import json
import numpy as np
import trimesh
import pyarrow.ipc as ipc
from glob import glob


def generate_class_colors(num_classes):
    """Generate distinct colors matching the palette used in VAE test."""
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


def parse_face_labels(masks_json, num_faces):
    """Parse masks to per-face labels."""
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


def export_colored_mesh(mesh, face_labels, num_parts, output_path):
    """Export mesh with per-face class colors as GLB.
    
    To avoid color bleeding at shared vertices, we split vertices
    so each face has its own 3 vertices with exact per-face color.
    """
    palette = generate_class_colors(max(num_parts, 2))
    
    # Create face colors array (RGBA)
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for i, lbl in enumerate(face_labels):
        if lbl >= 0 and lbl < len(palette):
            face_colors[i] = [palette[lbl][0], palette[lbl][1], palette[lbl][2], 255]
        else:
            face_colors[i] = [128, 128, 128, 255]  # gray for unlabeled
    
    # Normalize mesh to [-1, 1] centered at origin (no rotation)
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    max_dim = (bbox_max - bbox_min).max()
    if max_dim > 0:
        vertices_norm = (mesh.vertices - center) / max_dim * 2  # [-1, 1]
    else:
        vertices_norm = mesh.vertices - center
    
    # Split vertices: each face gets its own 3 vertices (no sharing)
    # This prevents color interpolation at shared vertices
    new_vertices = vertices_norm[mesh.faces].reshape(-1, 3)  # (F*3, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)  # (F, 3)
    
    # Vertex colors: repeat each face color 3 times (one per vertex in that face)
    vertex_colors = np.repeat(face_colors, 3, axis=0)  # (F*3, 4)
    
    # Build new mesh with split vertices and vertex colors
    colored_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )
    colored_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=colored_mesh,
        vertex_colors=vertex_colors
    )
    
    # Export as GLB
    colored_mesh.export(output_path, file_type='glb')
    return True


def main():
    arrow_dir = '/path/to/PartNeXt'
    mesh_dir = '/path/to/PartNeXt_mesh/glbs'
    output_dir = './output/colored_meshes'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load first 5 samples (matching our test data)
    arrow_files = sorted(glob(os.path.join(arrow_dir, 'data-*.arrow')))
    
    records = []
    with open(arrow_files[0], 'rb') as f:
        reader = ipc.open_stream(f)
        batch = reader.read_next_batch()
        for i in range(min(5, batch.num_rows)):
            records.append({
                'model_id': batch.column('model_id')[i].as_py(),
                'type_id': batch.column('type_id')[i].as_py(),
                'masks': batch.column('masks')[i].as_py(),
                'hierarchy': batch.column('hierarchyList')[i].as_py(),
            })
    
    results = []
    for rec in records:
        model_id = rec['model_id']
        type_id = rec['type_id']
        mesh_path = os.path.join(mesh_dir, type_id, f'{model_id}.glb')
        
        if not os.path.exists(mesh_path):
            print(f"  Skip {model_id}: mesh not found")
            continue
        
        # Load mesh
        scene = trimesh.load(mesh_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene
        
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"  Skip {model_id}: not a Trimesh")
            continue
        
        # Parse labels
        face_labels, num_parts = parse_face_labels(rec['masks'], len(mesh.faces))
        
        # Parse hierarchy for part names
        hierarchy = json.loads(rec['hierarchy']) if isinstance(rec['hierarchy'], str) else rec['hierarchy']
        
        # Export colored mesh
        out_path = os.path.join(output_dir, f'{model_id}.glb')
        export_colored_mesh(mesh, face_labels, num_parts, out_path)
        
        # Collect part names from hierarchy
        part_names = []
        def extract_names(node, depth=0):
            part_names.append(node.get('name', f'Part_{len(part_names)}'))
            for child in node.get('children', []):
                extract_names(child, depth+1)
        if isinstance(hierarchy, list):
            for h in hierarchy:
                extract_names(h)
        
        results.append({
            'model_id': model_id,
            'type_id': type_id,
            'num_parts': num_parts,
            'num_faces': len(mesh.faces),
            'num_vertices': len(mesh.vertices),
            'part_names': part_names[:num_parts],
            'glb_file': f'{model_id}.glb',
        })
        
        labeled_pct = (face_labels >= 0).sum() / len(face_labels) * 100
        print(f"  {model_id[:8]}... | parts={num_parts} | faces={len(mesh.faces)} | labeled={labeled_pct:.1f}% | -> {out_path}")
    
    # Save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nExported {len(results)} colored meshes to {output_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == '__main__':
    main()
