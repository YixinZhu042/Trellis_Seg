"""Unique palettes and exact face-color exports for configurable class counts."""
import colorsys
from pathlib import Path
import numpy as np
import trimesh


def class_colors(num_classes):
    if not 2 <= num_classes <= 256:
        raise ValueError('Expected 2..256 channels')
    # Neutral gray is explicitly unknown, not a colored semantic part.
    colors = [(128, 128, 128)]
    for i in range(1, num_classes):
        rgb = colorsys.hsv_to_rgb(((i-1)*.61803398875 + .07) % 1, [.65,.85,.5][(i-1)%3], [.92,.72][(i-1)%2])
        colors.append(tuple(round(v*255) for v in rgb))
    palette = np.asarray(colors, np.uint8)
    if len(np.unique(palette, axis=0)) != num_classes:
        raise ValueError('Palette colors must be unique')
    return palette


def render_palette(num_classes):
    return np.vstack([np.full((1,3),255,np.uint8), class_colors(num_classes)])


def export_labels(mesh, labels, path, num_classes):
    labels = np.asarray(labels)
    if labels.shape != (len(mesh.faces),) or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError('Invalid face labels for GLB export')
    palette = class_colors(num_classes)
    color = np.column_stack([palette[labels],np.full(len(labels),255,np.uint8)])
    vertices = mesh.triangles.reshape(-1,3)
    result = trimesh.Trimesh(vertices, np.arange(len(vertices)).reshape(-1,3),
                             vertex_colors=np.repeat(color,3,axis=0),process=False)
    path = Path(path)
    result.export(str(path))
    loaded = trimesh.load(str(path),force='scene',process=False).to_geometry()
    if len(loaded.faces) != len(labels) or not np.allclose(loaded.triangles,mesh.triangles,rtol=1e-6,atol=max(float(mesh.extents.max()),1e-12)*1e-6):
        raise ValueError('Export changed face geometry/order')
    triangle_colors = loaded.visual.vertex_colors[loaded.faces,:3]
    if not np.array_equal(triangle_colors,np.repeat(palette[labels,None],3,axis=1)):
        raise ValueError('Export changed exact face colors')
    return labels
