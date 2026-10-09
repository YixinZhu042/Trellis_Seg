"""
Call GPT-Image API to segment 6 rendered views into part masks — in ONE call.

Strategy: stitch all 6 views into a single 3x2 grid image, send once to the API,
then split the result back into 6 individual seg maps. This ensures globally
consistent part colors across all views.

Input:  directory with 6 rendered views (front.png, right.png, back.png, left.png, top.png, bottom.png)
Output: {output_dir}/seg_maps.npz with seg_maps (6, H, W) int32, pixel value = part_id (0=bg, 1,2,3...=parts)
        {output_dir}/seg_grid.png uses the same ID palette as training (0=black bg, 1→training class 0 color)

Usage:
    python scripts/segment_views_via_api.py \
        --input_dir ./batch_results/9f980a50_renders \
        --output_dir ./batch_results/9f980a50_seg
"""
import os
import sys
import io
import json
import base64
import argparse
import numpy as np
import requests
from PIL import Image, ImageFilter

# Credentials must stay outside source control.
API_URL = os.environ.get("SEG_API_URL", "https://api.openai.com/v1/images/edits")
API_KEY = os.environ.get("SEG_API_KEY", "")

VIEW_NAMES = ['front', 'right', 'back', 'left', 'top', 'bottom']

SEGMENT_PROMPT = (
    "This image shows 6 views of the same 3D object arranged in a 3x2 grid "
    "(top row: front, right, back; bottom row: left, top, bottom). "
    "Segment this object into 3 to 10 large, coarse semantic parts. "
    "Think in terms of major structural components — for example a chair has: seat, backrest, legs, armrests; "
    "a car has: body, wheels, windows, bumper; a human has: head, torso, arms, legs. "
    "Do NOT over-segment: merge small details (screws, edges, trim) into their parent part. "
    "Each part should cover a substantial contiguous region. "
    "Output a segmentation mask in the SAME 3x2 grid layout, "
    "where each part is filled with a unique solid flat color, consistent across all 6 views "
    "(the same part in different views must use the same color). "
    "Use at most 15 distinct colors for parts. Background (empty space) should be black. "
    "Preserve the input object's silhouette and geometry exactly; do not invent, repaint, or alter any object details. "
    "Every part must be one completely uniform RGB color with hard, clean boundaries. "
    "Do not produce speckles, scattered patches, color leaks, internal holes, shadows, highlights, anti-aliasing, or semi-transparent pixels. "
    "No gradients, no textures, no outlines, no labels, no text. "
)

MAX_PARTS = 15  # part IDs will be 1..MAX_PARTS, background=0

# Same palette as training: raw seg label 1 uses class-0 color, label 2 uses class-1 color, ...
TRAINING_SEG_COLORMAP = np.array([
    [230, 25, 75],    # red
    [60, 180, 75],    # green
    [67, 99, 216],    # blue
    [244, 212, 30],   # yellow
    [133, 20, 177],   # purple
    [244, 133, 29],   # orange
    [70, 240, 240],   # cyan
    [240, 50, 230],   # magenta
    [210, 245, 60],   # lime
    [250, 190, 212],  # pink
    [0, 128, 128],    # teal
    [230, 190, 255],  # lavender
    [170, 110, 40],   # brown
    [255, 250, 200],  # beige
    [128, 0, 0],      # maroon
    [170, 255, 195],  # mint
    [128, 128, 0],    # olive
    [255, 215, 180],  # apricot
    [0, 0, 128],      # navy
    [128, 128, 128],  # grey
], dtype=np.uint8)


def colorize_label_map(seg_map):
    """Colorize raw training-format labels: 0=background, 1+=parts."""
    colored = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    fg = seg_map > 0
    palette_ids = np.clip(seg_map[fg] - 1, 0, len(TRAINING_SEG_COLORMAP) - 1)
    colored[fg] = TRAINING_SEG_COLORMAP[palette_ids]
    return colored


def stitch_label_maps(seg_maps):
    """Stitch 6 label maps into a 3x2 RGB grid using the training palette."""
    _, h, w = seg_maps.shape
    grid = Image.new('RGB', (w * 3, h * 2), (0, 0, 0))
    for i, seg_map in enumerate(seg_maps):
        col = i % 3
        row = i // 3
        grid.paste(Image.fromarray(colorize_label_map(seg_map)), (col * w, row * h))
    return grid


def stitch_views(input_dir, target_size=512):
    """Stitch 6 views into a 3x2 grid. Returns PIL Image."""
    images = []
    for name in VIEW_NAMES:
        path = os.path.join(input_dir, f'{name}.png')
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            img = img.resize((target_size, target_size), Image.LANCZOS)
        else:
            img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        images.append(img)

    # 3 columns x 2 rows
    grid = Image.new('RGB', (target_size * 3, target_size * 2), (0, 0, 0))
    for i, img in enumerate(images):
        col = i % 3
        row = i // 3
        grid.paste(img, (col * target_size, row * target_size))
    return grid, target_size


def split_grid(grid_img, cell_size):
    """Split a 3x2 grid image back into 6 individual images."""
    views = []
    for i in range(6):
        col = i % 3
        row = i // 3
        x0, y0 = col * cell_size, row * cell_size
        view = grid_img.crop((x0, y0, x0 + cell_size, y0 + cell_size))
        views.append(view)
    return views


def build_foreground_grid_mask(input_dir, cell_size):
    """Build a 3x2 foreground mask from render alpha/occupancy to avoid API background noise."""
    masks = []
    has_mask = False
    for name in VIEW_NAMES:
        render_path = os.path.join(input_dir, f'{name}.png')
        mask = None
        if os.path.exists(render_path):
            render_img = Image.open(render_path)
            if render_img.mode == 'RGBA':
                alpha = render_img.getchannel(3).resize((cell_size, cell_size), Image.NEAREST)
                mask = np.array(alpha) >= 128
                has_mask = True
        if mask is None:
            mask = np.ones((cell_size, cell_size), dtype=bool)
        masks.append(mask)

    occ_path = os.path.join(input_dir, 'occupancy.npy')
    if os.path.exists(occ_path):
        occupancy = np.load(occ_path).astype(bool)
        if occupancy.shape[0] == 6:
            occ_masks = []
            for i in range(6):
                occ = occupancy[i]
                if occ.shape != (cell_size, cell_size):
                    occ = np.array(Image.fromarray(occ.astype(np.uint8) * 255).resize(
                        (cell_size, cell_size), Image.NEAREST)) > 128
                occ_masks.append(occ)
            masks = occ_masks
            has_mask = True

    if not has_mask:
        return None

    grid_mask = np.zeros((cell_size * 2, cell_size * 3), dtype=bool)
    for i, mask in enumerate(masks):
        col = i % 3
        row = i // 3
        grid_mask[row * cell_size:(row + 1) * cell_size,
                  col * cell_size:(col + 1) * cell_size] = mask
    return grid_mask


def pick_dominant_color_seeds(rgb, fg_mask, max_parts=MAX_PARTS, bin_size=8,
                              min_color_dist=65.0, min_pixels=32, min_brightness=80):
    """Pick dominant, well-separated RGB seeds from a shaded API segmentation image."""
    pixels = rgb[fg_mask]
    if pixels.size == 0:
        return np.array([[255, 255, 255]], dtype=np.float32)

    q = ((pixels.astype(np.int32) // bin_size) * bin_size + bin_size // 2).clip(0, 255).astype(np.uint8)
    colors, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]

    seeds = []
    min_dist2 = min_color_dist ** 2
    for idx in order:
        if counts[idx] < min_pixels:
            break
        color = colors[idx].astype(np.float32)
        if color.sum() < min_brightness:
            continue
        if not seeds:
            seeds.append(color)
        else:
            seed_arr = np.stack(seeds, axis=0)
            dist2 = ((seed_arr - color) ** 2).sum(axis=1)
            seed_norm = seed_arr / np.maximum(np.linalg.norm(seed_arr, axis=1, keepdims=True), 1e-6)
            color_norm = color / max(np.linalg.norm(color), 1e-6)
            same_hue = (seed_norm @ color_norm).max() > 0.985
            if dist2.min() >= min_dist2 and not same_hue:
                seeds.append(color)
        if len(seeds) >= max_parts:
            break

    if not seeds:
        seeds.append(colors[order[0]].astype(np.float32))
    return np.stack(seeds, axis=0).astype(np.float32)


def assign_pixels_to_seeds(rgb, fg_mask, seeds, chunk_size=262144):
    """Nearest-seed RGB assignment; background stays 0, parts are 1..K."""
    labels = np.zeros(rgb.shape[:2], dtype=np.int32)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float32)
    flat_fg = fg_mask.reshape(-1)
    fg_indices = np.flatnonzero(flat_fg)
    flat_labels = labels.reshape(-1)
    seeds = seeds.astype(np.float32)

    for start in range(0, len(fg_indices), chunk_size):
        idx = fg_indices[start:start + chunk_size]
        pix = flat_rgb[idx]
        d2 = ((pix[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
        flat_labels[idx] = d2.argmin(axis=1).astype(np.int32) + 1
    return labels


def mode_filter_labels(labels, fg_mask, size=3):
    """Remove isolated label speckles while preserving the render silhouette."""
    if size <= 1:
        labels[~fg_mask] = 0
        return labels
    filtered = np.array(Image.fromarray(labels.astype(np.uint8)).filter(ImageFilter.ModeFilter(size)))
    cleaned = labels.copy()
    valid = fg_mask & (filtered > 0)
    cleaned[valid] = filtered[valid].astype(np.int32)
    cleaned[~fg_mask] = 0
    return cleaned


def api_grid_to_label_maps(seg_grid, input_dir, cell_size, max_parts=MAX_PARTS):
    """Convert shaded/anti-aliased API RGB output to discrete training-format labels."""
    rgb = np.array(seg_grid.convert('RGB'))
    fg_grid_mask = build_foreground_grid_mask(input_dir, cell_size)
    if fg_grid_mask is None:
        fg_grid_mask = rgb.sum(axis=2) > 30

    min_pixels = max(32, int(fg_grid_mask.sum() * 0.00005))
    seeds = pick_dominant_color_seeds(rgb, fg_grid_mask, max_parts=max_parts, min_pixels=min_pixels)
    print(f'  Selected {len(seeds)} dominant API colors: {np.round(seeds).astype(int).tolist()}')

    label_grid = assign_pixels_to_seeds(rgb, fg_grid_mask, seeds)
    seg_maps = []
    for i, name in enumerate(VIEW_NAMES):
        col = i % 3
        row = i // 3
        y0, y1 = row * cell_size, (row + 1) * cell_size
        x0, x1 = col * cell_size, (col + 1) * cell_size
        labels = label_grid[y0:y1, x0:x1].copy()
        fg = fg_grid_mask[y0:y1, x0:x1]
        labels = mode_filter_labels(labels, fg, size=3)
        seg_maps.append(labels)
        print(f'  {name}: {len(np.unique(labels[labels > 0]))} foreground parts')
    return np.stack(seg_maps, axis=0)


def call_api(grid_img):
    if not API_KEY:
        raise ValueError('Set SEG_API_KEY in the environment before calling the segmentation API')
    """Send stitched grid to API, get segmented grid back."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-image-2",
        "prompt": SEGMENT_PROMPT,
        "size": "1536x1024",  # 3:2 aspect ratio matches our 3x2 grid
        "background": "opaque",
        "output_format": "png",
        "quality": "high",
        "n": 1,
    }

    buf = io.BytesIO()
    grid_img.save(buf, format='PNG')
    buf.seek(0)

    files = [("image", ("grid.png", buf, "image/png"))]
    print('  Calling API (single request for all 6 views)...')
    response = requests.post(API_URL, headers=headers, data=payload, files=files, timeout=600)

    if response.status_code != 200:
        print(f"  API error {response.status_code}: {response.text[:300]}")
        return None

    data = response.json()
    if "data" not in data or not data["data"]:
        print(f"  Unexpected response: {json.dumps(data, ensure_ascii=False)[:300]}")
        return None

    item = data["data"][0]
    if item.get("b64_json"):
        img_bytes = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        img_bytes = requests.get(item["url"], timeout=120).content
    else:
        print("  No image data in response")
        return None

    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_npz', type=str, default=None)
    parser.add_argument('--api_seg_grid', type=str, default=None,
                        help='Reuse an existing API output PNG and only rerun RGB→label conversion')
    parser.add_argument('--cell_size', type=int, default=512)
    parser.add_argument('--save_color', action='store_true',
                        help='Save per-view segmentation images colorized with the training palette')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir + '_seg'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Input: {args.input_dir}')
    print(f'Output: {args.output_dir}')

    # Step 1: Stitch into grid
    print('  Stitching 6 views into 3x2 grid...')
    grid, cell_size = stitch_views(args.input_dir, args.cell_size)
    grid.save(os.path.join(args.output_dir, 'input_grid.png'))

    # Step 2: Call API once, or reuse a saved API output
    if args.api_seg_grid:
        print(f'  Reusing API segmentation grid: {args.api_seg_grid}')
        seg_grid = Image.open(args.api_seg_grid).convert('RGB')
    else:
        seg_grid = call_api(grid)
        if seg_grid is None:
            print('ERROR: API call failed')
            sys.exit(1)
        seg_grid.save(os.path.join(args.output_dir, 'api_seg_grid.png'))
    print(f'  Got segmentation grid: {seg_grid.size}')

    # Step 3: Resize to match expected grid if needed
    expected_w, expected_h = cell_size * 3, cell_size * 2
    if seg_grid.size != (expected_w, expected_h):
        print(f'  Resizing API output {seg_grid.size} -> ({expected_w}, {expected_h})')
        seg_grid = seg_grid.resize((expected_w, expected_h), Image.BILINEAR)

    # Step 4: Robust shaded-RGB → discrete label maps conversion
    seg_maps = api_grid_to_label_maps(seg_grid, args.input_dir, cell_size, MAX_PARTS)

    bg_pct = (seg_maps == 0).sum() / seg_maps.size * 100
    print(f'  After masking: {bg_pct:.1f}% background')

    # Verify range [0, MAX_PARTS]
    assert seg_maps.max() <= MAX_PARTS, f"max label {seg_maps.max()} > {MAX_PARTS}"
    assert seg_maps.min() >= 0

    # Save training-palette visualization: 0=black background, 1→palette[0], 2→palette[1], ...
    stitch_label_maps(seg_maps).save(os.path.join(args.output_dir, 'seg_grid.png'))
    if args.save_color:
        for i, name in enumerate(VIEW_NAMES):
            Image.fromarray(colorize_label_map(seg_maps[i])).save(
                os.path.join(args.output_dir, f'{name}_seg.png')
            )

    # Save
    npz_path = args.output_npz or os.path.join(args.output_dir, 'seg_maps.npz')
    np.savez_compressed(npz_path, seg_maps=seg_maps, num_parts=int(seg_maps.max()))
    print(f'\nSaved: {npz_path}')
    print(f'  Shape: {seg_maps.shape}, labels in [0, {seg_maps.max()}], {len(np.unique(seg_maps))} unique values')


if __name__ == '__main__':
    main()
