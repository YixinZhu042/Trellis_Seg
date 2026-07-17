"""
Call GPT-Image API to segment 6 rendered views into part masks — in ONE call.

Strategy: stitch all 6 views into a single 3x2 grid image, send once to the API,
then split the result back into 6 individual seg maps. This ensures globally
consistent part colors across all views.

Input:  directory with 6 rendered views (front.png, right.png, back.png, left.png, top.png, bottom.png)
Output: {output_dir}/seg_maps.npz with seg_maps (6, H, W) int32, pixel value = part_id (0=bg, 1,2,3...=parts)

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
from PIL import Image

# API config — set your own API key here or via environment variable
API_URL = os.environ.get("SEG_API_URL", "https://api.openai.com/v1/images/edits")
API_KEY = os.environ.get("SEG_API_KEY", "your-api-key-here")

VIEW_NAMES = ['front', 'right', 'back', 'left', 'top', 'bottom']

SEGMENT_PROMPT = (
    "This image shows 6 views of the same 3D object arranged in a 3x2 grid "
    "(top row: front, right, back; bottom row: left, top, bottom). "
    "Segment this object into at most 19 semantic parts (e.g. legs, seat, back, arms, etc). "
    "Output a segmentation mask in the SAME 3x2 grid layout, "
    "where each part is filled with a unique solid flat color, consistent across all 6 views "
    "(the same part in different views must use the same color). "
    "Use at most 19 distinct colors for parts. "
    "No gradients, no textures, no outlines, no labels. "
)

MAX_PARTS = 19  # part IDs will be 1..19, background=0, total 20 classes


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


def call_api(grid_img):
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
    parser.add_argument('--cell_size', type=int, default=512)
    parser.add_argument('--save_color', action='store_true',
                        help='Save the raw color segmentation images')
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

    # Step 2: Call API once
    seg_grid = call_api(grid)
    if seg_grid is None:
        print('ERROR: API call failed')
        sys.exit(1)

    seg_grid.save(os.path.join(args.output_dir, 'seg_grid.png'))
    print(f'  Got segmentation grid: {seg_grid.size}')

    # Step 3: Resize to match expected grid if needed
    expected_w, expected_h = cell_size * 3, cell_size * 2
    if seg_grid.size != (expected_w, expected_h):
        print(f'  Resizing API output {seg_grid.size} -> ({expected_w}, {expected_h})')
        seg_grid = seg_grid.resize((expected_w, expected_h), Image.NEAREST)

    seg_views = split_grid(seg_grid, cell_size)

    # Step 4: Build global color→label map from entire grid
    seg_grid_np = np.array(seg_grid)
    # Quantize colors to reduce noise from compression
    quantized_grid = ((seg_grid_np.astype(np.int32) // 8) * 8).astype(np.uint8)
    all_pixels = quantized_grid.reshape(-1, 3)
    unique_colors = np.unique(all_pixels, axis=0)

    bg_threshold = 30
    color_to_label = {}
    next_id = 1

    # Sort by frequency (most common colors first) to assign stable IDs
    all_pixels_flat = quantized_grid.reshape(-1, 3)
    # Count occurrences of each unique color
    _, inverse_all, counts_all = np.unique(
        all_pixels_flat, axis=0, return_inverse=True, return_counts=True
    )
    # Sort unique_colors by count descending (skip bg)
    color_counts = []
    for i, color in enumerate(unique_colors):
        s = int(color.sum())
        if s > bg_threshold:
            color_counts.append((counts_all[i] if i < len(counts_all) else 0, tuple(color.tolist())))
        else:
            color_to_label[tuple(color.tolist())] = 0
    color_counts.sort(key=lambda x: -x[0])

    # Assign IDs 1..MAX_PARTS, merge remaining into nearest
    for _, key in color_counts:
        if next_id <= MAX_PARTS:
            color_to_label[key] = next_id
            next_id += 1
        else:
            # Over 19 parts: merge into nearest existing color by RGB distance
            best_label = 1
            best_dist = float('inf')
            for assigned_key, assigned_label in color_to_label.items():
                if assigned_label == 0:
                    continue
                dist = sum((a - b) ** 2 for a, b in zip(key, assigned_key))
                if dist < best_dist:
                    best_dist = dist
                    best_label = assigned_label
            color_to_label[key] = best_label

    actual_parts = min(next_id - 1, MAX_PARTS)
    print(f'  Total unique parts: {actual_parts} (capped at {MAX_PARTS})')

    # Step 5: Convert each view to label map
    seg_maps = []
    for i, (name, view_img) in enumerate(zip(VIEW_NAMES, seg_views)):
        view_np = np.array(view_img)
        quantized = ((view_np.astype(np.int32) // 8) * 8).astype(np.uint8)
        H, W = quantized.shape[:2]

        # Vectorized: hash each pixel color to lookup
        flat = quantized.reshape(-1, 3)
        # Use a hash for fast lookup
        color_hash = flat[:, 0].astype(np.int64) * 256 * 256 + flat[:, 1].astype(np.int64) * 256 + flat[:, 2].astype(np.int64)

        # Build hash→label lookup
        hash_to_label = {}
        for color, label in color_to_label.items():
            h = color[0] * 256 * 256 + color[1] * 256 + color[2]
            hash_to_label[h] = label

        labels = np.array([hash_to_label.get(h, 0) for h in color_hash], dtype=np.int32)
        labels = labels.reshape(H, W)
        seg_maps.append(labels)

        if args.save_color:
            view_img.save(os.path.join(args.output_dir, f'{name}_seg.png'))
        print(f'  {name}: {len(np.unique(labels))} parts')

    seg_maps = np.stack(seg_maps, axis=0)  # (6, H, W)

    # Verify range [0, 19]
    assert seg_maps.max() <= MAX_PARTS, f"max label {seg_maps.max()} > {MAX_PARTS}"
    assert seg_maps.min() >= 0

    # Save
    npz_path = args.output_npz or os.path.join(args.output_dir, 'seg_maps.npz')
    np.savez_compressed(npz_path, seg_maps=seg_maps, num_parts=int(seg_maps.max()))
    print(f'\nSaved: {npz_path}')
    print(f'  Shape: {seg_maps.shape}, labels in [0, {seg_maps.max()}], {len(np.unique(seg_maps))} unique values')


if __name__ == '__main__':
    main()
