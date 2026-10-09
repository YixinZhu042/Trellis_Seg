"""Audit a UID-to-GLB index and visualize cached color/shape/part views on CPU."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import colorsys
import html
import io
import json
import math
from pathlib import Path
import random
import re
import struct
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_INDEX = "/path/to/part_data/matedate/part_biaozhu_complex_outputtexturemesh_zw.json"
ROOT = Path(__file__).resolve().parents[1]
FILES = ("full_object_texture.glb", "full_object.glb", "parts.glb", "vis.glb",
         "watertight.ply", "watertight_wt_label.txt", "vc_render.npz", "surface_color.npz")
BG = (245, 247, 251)
INK = (30, 42, 63)


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def describe_glb(path):
    with path.open("rb") as f:
        magic, version, total = struct.unpack("<4sII", f.read(12))
        size, kind = struct.unpack("<II", f.read(8))
        if magic != b"glTF" or version != 2 or kind != 0x4E4F534A:
            raise ValueError(f"Invalid GLB: {path}")
        g = json.loads(f.read(size))
    meshes = []
    for mesh in g.get("meshes", []):
        primitives = mesh.get("primitives", [])
        if any(p.get("mode", 4) != 4 for p in primitives):
            raise ValueError(f"Non-triangle primitive in {path}")
        meshes.append({
            "name": mesh.get("name"),
            "primitives": len(primitives),
            "vertices": sum(g["accessors"][p["attributes"]["POSITION"]]["count"] for p in primitives),
            "faces": sum(g["accessors"][p["indices"]]["count"] // 3 for p in primitives),
            "attributes": sorted(set(k for p in primitives for k in p.get("attributes", {}))),
        })
    return {
        "bytes": path.stat().st_size, "declared_bytes": total,
        "generator": g.get("asset", {}).get("generator"),
        "mesh_count": len(meshes), "node_count": len(g.get("nodes", [])),
        "mesh_node_count": sum("mesh" in n for n in g.get("nodes", [])),
        "faces": sum(m["faces"] for m in meshes),
        "vertices_in_primitives": sum(m["vertices"] for m in meshes),
        "material_count": len(g.get("materials", [])), "image_count": len(g.get("images", [])),
        "meshes": meshes,
    }


def ply_header(path):
    lines = []
    with path.open("rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        else:
            raise ValueError("Missing PLY end_header")
    return {m.group(1): int(m.group(2)) for line in lines
            if (m := re.fullmatch(r"element (vertex|face) (\d+)", line))}


def npz_headers(path):
    result = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f"Unsupported NPY header version {version}")
            result[name] = {"shape": list(shape), "dtype": str(dtype), "object": dtype.hasobject}
    return result


def label_color(label):
    if label == -1:
        return BG
    if label < -1:
        return (65, 65, 65)
    rgb = colorsys.hsv_to_rgb((label * 0.61803398875 + 0.07) % 1, 0.65, 0.91)
    return tuple(round(c * 255) for c in rgb)


def colorize(labels):
    result = np.empty((*labels.shape, 3), dtype=np.uint8)
    for value in np.unique(labels):
        result[labels == value] = label_color(int(value))
    return Image.fromarray(result)


def decode_image(value):
    with Image.open(io.BytesIO(value.tobytes())) as image:
        encoding = image.format
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A")) > 0
        bg = Image.new("RGBA", rgba.size, (*BG, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB"), alpha, encoding


def crop_view(image, box, size, nearest=False):
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.crop(box).resize((size, size), method)


def get_crop(mask, alphas):
    foreground = mask != -1
    for alpha in alphas:
        if alpha.any() and not alpha.all():
            foreground |= alpha
    ys, xs = np.where(foreground)
    if not len(xs):
        return (0, 0, mask.shape[1], mask.shape[0])
    side = min(max(mask.shape), math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 1.14))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x = max(0, min(mask.shape[1] - side, round(cx - side / 2)))
    y = max(0, min(mask.shape[0] - side, round(cy - side / 2)))
    return (x, y, x + side, y + side)


def inspect_sample(uid, source, groups, out, n_views, tile):
    folder = Path(source).parent
    destination = out / uid
    destination.mkdir(parents=True, exist_ok=True)
    result = {"id": uid, "path": source, "quality_groups": groups,
              "files": {name: (folder / name).stat().st_size if (folder / name).is_file() else None for name in FILES}}
    result["glb"] = {name: describe_glb(folder / name) for name in FILES[:4]}
    result["ply"] = ply_header(folder / "watertight.ply")
    raw = np.fromfile(folder / "watertight_wt_label.txt", dtype="<i4")
    if len(raw) < 2 or raw[0] != len(raw) - 1 or raw[0] != result["ply"]["face"]:
        raise ValueError(f"Binary label header/PLY face count mismatch: {uid}")
    ids, counts = np.unique(raw[1:], return_counts=True)
    result["labels"] = {
        "encoding": "little-endian int32: face_count followed by face_count labels",
        "face_count": int(raw[0]), "counts": {str(k): int(v) for k, v in zip(ids, counts)},
        "nonnegative_part_count": int((ids >= 0).sum()),
        "special_negative_face_fraction": float((raw[1:] < 0).mean()),
        "special_negative_semantics": "Not established; do not silently remap or drop.",
    }
    positive = counts[ids >= 0]
    result["labels"]["largest_part_fraction_of_all_faces"] = float(positive.max() / raw[0]) if len(positive) else None
    result["surface_color_headers"] = npz_headers(folder / "surface_color.npz")
    result["render_headers"] = npz_headers(folder / "vc_render.npz")
    rendered = []
    with np.load(folder / "vc_render.npz", allow_pickle=False) as z:
        views = sorted(k.removesuffix("_part.png") for k in z.files if k.endswith("_part.png"))
        selected = [views[i] for i in np.linspace(0, len(views) - 1, min(n_views, len(views)), dtype=int)]
        result["view_count"] = len(views)
        result["selected_views"] = selected
        result["views"] = []
        for view in selected:
            labels = z[f"{view}_part.png"]
            color, alpha, encoding = decode_image(z[f"{view}_color.png"])
            shape, shape_alpha, _ = decode_image(z[f"{view}_baimo.png"])
            if color.size != labels.shape[::-1] or shape.size != color.size:
                raise ValueError(f"View shape mismatch: {uid}/{view}")
            crop = get_crop(labels, (alpha, shape_alpha))
            part_image = colorize(labels)
            images = [crop_view(im, crop, tile, nearest=(i == 2)) for i, im in enumerate((color, shape, part_image))]
            rendered.append(images)
            result["views"].append({
                "view": view, "shape": list(labels.shape), "mask_dtype": str(labels.dtype),
                "label_ids": [int(i) for i in np.unique(labels)], "background_fraction": float((labels == -1).mean()),
                "foreground_fraction": float((labels != -1).mean()), "crop_xyxy": list(crop), "color_encoding": encoding,
            })
    result["visible_nonnegative_parts_in_selected_views"] = sorted({label for v in result["views"] for label in v["label_ids"] if label >= 0})
    gap, left, top = 12, 108, 112
    width = left + len(rendered) * (tile + gap) + 12
    height = top + 3 * (tile + 32) + 96
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 16), uid, fill=INK, font=font(21))
    mesh = result["glb"]["full_object_texture.glb"]
    detail = f'{mesh["mesh_count"]} GLB meshes | {mesh["faces"]:,} textured faces | {result["labels"]["nonnegative_part_count"]} WT part IDs | {result["view_count"]} cached views'
    draw.text((20, 48), detail, fill=INK, font=font(17))
    draw.text((20, 75), "Cached surface-color / white model / part IDs (not a new GLB render)", fill=(83, 98, 122), font=font(16))
    for col, (view, images) in enumerate(zip(result["selected_views"], rendered)):
        x = left + col * (tile + gap)
        draw.text((x, top - 22), f"View {view}", fill=INK, font=font(16))
        for row, image in enumerate(images):
            y = top + row * (tile + 32)
            sheet.paste(image, (x, y))
    for row, label in enumerate(("COLOR", "SHAPE", "PART ID")):
        draw.text((12, top + row * (tile + 32) + tile // 2), label, fill=INK, font=font(17))
    y = top + 3 * (tile + 32) + 5
    draw.text((20, y), "Same part ID = same color within this sample; IDs are not semantic classes across samples.", fill=INK, font=font(16))
    draw.text((20, y + 25), "2D -1: background. WT negative labels: special IDs, meaning unverified. No object-array unpickling.", fill=INK, font=font(16))
    sheet.save(destination / "views.png")
    preview = Image.new("RGB", (tile * 2, tile + 80), "white")
    preview.paste(rendered[0][0], (0, 72))
    preview.paste(rendered[0][2], (tile, 72))
    pd = ImageDraw.Draw(preview)
    pd.text((10, 8), uid[:24] + ("..." if len(uid) > 24 else ""), fill=INK, font=font(17))
    pd.text((10, 32), f'{mesh["mesh_count"]} meshes | {mesh["faces"]:,} faces | View {selected[0]}', fill=INK, font=font(15))
    pd.text((10, 53), "CACHED COLOR", fill=INK, font=font(13))
    pd.text((tile + 10, 53), "PART IDS", fill=INK, font=font(13))
    preview.save(destination / "preview.png")
    result["images"] = {"views": f"{uid}/views.png", "preview": f"{uid}/preview.png"}
    dump(destination / "metadata.json", result)
    return result


def check_path(item):
    uid, path = item
    try:
        p = Path(path)
        size = p.stat().st_size
        return uid, size, None if p.is_file() else "Not a file"
    except OSError as exc:
        return uid, None, str(exc)


def write_gallery(out, summary, samples, failures):
    cards = []
    for s in samples:
        uid = html.escape(s["id"])
        groups = " / ".join(s["quality_groups"]) or "无配套分类"
        info = s["glb"]["full_object_texture.glb"]
        cards.append(f'<article><h2>{uid}</h2><p>{html.escape(groups)} · {info["mesh_count"]} 个 mesh · {info["faces"]:,} 个纹理三角面 · {s["labels"]["nonnegative_part_count"]} 个 WT 非负标签</p><a href="{s["images"]["views"]}" target="_blank"><img loading="lazy" src="{s["images"]["views"]}" alt="六视角颜色、白模、部件标签对照"></a><details><summary>原始路径与统计</summary><pre>{html.escape(s["path"])}</pre><a href="{s["id"]}/metadata.json">完整 metadata.json</a><p>WT 特殊负标签面占比：{s["labels"]["special_negative_face_fraction"]:.2%}；纹理、parts、WT 并非同一拓扑，标签不能直接按面序复制。</p></details></article>')
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Complex Part Dataset · 样本检查</title><style>body{margin:0;background:#f2f5fa;color:#1e2a3f;font:16px/1.65 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:38px 24px}h1{font-size:34px;margin:0}small{color:#59708d}.metrics{display:flex;gap:18px;flex-wrap:wrap;margin:25px 0}.metric,article,.note{background:white;border:1px solid #dde5ef;border-radius:14px;padding:22px}.metric strong{display:block;font-size:28px}.note{border-left:5px solid #557bee}article{margin:25px 0}h2{font-size:19px;overflow-wrap:anywhere}img{width:100%;height:auto}a{color:#3559c7}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}p{margin:10px 0}</style><main><small>DATASET INSPECTION / CPU ONLY / LOCAL ARTIFACTS</small><h1>复杂部件标注 · 数据检查与可视化</h1>'''
    doc += f'<div class="metrics"><div class="metric"><strong>{summary["records"]:,}</strong>索引样本</div><div class="metric"><strong>{summary["accessible"]:,}</strong>路径可访问</div><div class="metric"><strong>{len(samples)}</strong>可视化样本 / seed {summary["seed"]}</div><div class="metric"><strong>{summary["duplicate_keys"]}</strong>重复 UID</div></div>'
    doc += '<div class="note"><b>阅读说明</b><p>每个样本展示预渲染中的六个等间隔视角：第一行为缓存表面颜色，第二行为白模，第三行为部件 ID。列内使用相同裁剪框。颜色图不是本次重新渲染的纹理 GLB；视角编号不代表已知正面/侧面。背景 -1 单独显示，部件 0 为有效标签。</p><p>不同样本同一颜色不代表相同语义。配套质量分类可能重叠；“非目标数据剔除”等样本仍可能在此索引中。原始数据未修改，含 Python 对象的 NPZ 数组未反序列化。</p><a href="report.md">研究报告</a> · <a href="summary.json">全局统计</a> · <a href="overview.png">12 样本总览</a></div>'
    doc += "".join(cards)
    if failures:
        doc += f'<article><h2>失败样本</h2><pre>{html.escape(json.dumps(failures,ensure_ascii=False,indent=2))}</pre></article>'
    (out / "index.html").write_text(doc + "</main></html>", encoding="utf-8")


def write_report(out, summary, samples, failures):
    lines = ["# 复杂部件数据检查", "", f'源索引：`{summary["index"]}`', "",
             "## 全量索引", "", f'- 记录数：{summary["records"]:,}；重复 UID：{summary["duplicate_keys"]}；重复路径：{summary["duplicate_paths"]}。',
             f'- 文件可访问：{summary["accessible"]:,}；失败：{len(summary["inaccessible"])}（仅检查路径/文件大小，非全量 GLB 解码）。',
             f'- UID 与父目录不一致：{summary["id_parent_mismatches"]}。',
             f'- GLB 文件总大小：{summary["total_glb_bytes"] / 1024**3:.2f} GiB。',
             '- JSON 仅保存 UID → GLB 路径，不含类别名、语义部件名称、模型分数或评估指标。', "",
             "## 同目录质量分类交叉检查", "", "| 分类 | 配套文件条数 | 与本索引交集 |", "|---|---:|---:|"]
    for name, values in summary["quality_groups"].items():
        lines.append(f'| {name} | {values["source_count"]} | {values["intersection_count"]} |')
    lines += ["", f'- 在多个分类中出现的索引 UID：{summary["quality_overlap_ids"]}；不在任何分类的 UID：{summary["quality_unclassified_ids"]}。',
              '- 分类可能交叠，不能直接加总，也不能把文件名中的“标注结果”理解为所有样本合格。', "",
              "## 抽样方法", "", f'- 对原始 JSON 顺序执行 `random.Random({summary["seed"]}).sample`，请求 {summary["requested_samples"]} 个；成功 {len(samples)} 个，失败 {len(failures)} 个。',
              '- 不按好坏筛选、不替换失败样本；仅抽样检查，不代表全量网格质量。',
              '- 从每个 `vc_render.npz` 等间隔选取六个视角；颜色、白模和 ID 图采用同一裁剪框。无需 GPU。',
              '- 展示的是已有缓存的表面颜色，不是本次重新渲染 `full_object_texture.glb`。', "",
              "## 配套内容及格式", "",
              "| 文件 | 本次确认的内容 |", "|---|---|",
              "| `full_object_texture.glb` | 纹理 GLB，多个 Part mesh，可有多个材质 primitive；mesh 数与 primitive 数不是同一概念。 |",
              "| `full_object.glb` | 无嵌入纹理图片的整体 GLB；具体拓扑统计见各样本 metadata。 |",
              "| `parts.glb` | 分部件、顶点颜色网格；不假定与纹理 GLB 面序相同。 |",
              "| `vis.glb` | 包含 `part_*` 命名的可视化网格，可能带特殊负标签。 |",
              "| `watertight.ply` | 二进制 PLY；顶点/三角面数量写于 header。 |",
              "| `watertight_wt_label.txt` | 虽为 TXT 后缀，实为 little-endian int32：首个整数为面数，随后每个整数为逐面 ID；抽样逐一与 PLY 面数核对。 |",
              "| `vc_render.npz` | 部件 ID 图为 int16 数组，颜色和白模为压缩图像字节；`.png` 键不保证 PNG 编码，抽样发现 WEBP。`frames` 为 object 数组，本次不反序列化。 |",
              "| `surface_color.npz` | `full_mesh`、`full_mesh_sdf`、`full_mesh_deform`、`full_mesh_part_super` 为 object 数组。本次只检查 header，未加载对象内容。 |", "",
              "## 抽样统计", "", "| UID | 质量分类 | 纹理 mesh 数 | 纹理面数 | WT 面数 | WT 非负 ID 数 | WT 负标签面占比 |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in samples:
        g = s["glb"]["full_object_texture.glb"]
        lines.append(f'| `{s["id"]}` | {" / ".join(s["quality_groups"])} | {g["mesh_count"]} | {g["faces"]:,} | {s["ply"]["face"]:,} | {s["labels"]["nonnegative_part_count"]} | {s["labels"]["special_negative_face_fraction"]:.2%} |')
    lines += ["", "## 使用注意", "",
              "- 2D ID 图中的 -1 为背景，0 是有效 part ID。WT 的 -1、-2 等特殊负值语义尚未得到生成端确认，不应默默丢弃或当普通部件。",
              "- 部件 ID 是样本内部的编号，不是跨样本一致的语义类别。",
              "- 纹理 GLB、parts GLB、watertight PLY 的面数可不同，不能直接把 WT 标签按 face index 贴到原始纹理模型。",
              "- 是否 watertight、语义切分是否正确、是否缺件均未做全量验证；这里不产生模型精度或 IoU 结论。",
              "- 当前查看六个相机视角，遮挡或微小部件可能看不到；完整 metadata 保留原始标签分布和缓存视角数量。", "",
              "## 输出", "", "- `index.html`：本地画廊。", "- `overview.png`：样本总览。",
              "- `<UID>/views.png`：六视角、三行对照。", "- `<UID>/metadata.json`：模型、标签、NPZ 头和抽样视图统计。",
              "- `summary.json`：全量索引、路径和质量分类统计。", "- `samples.json`：抽样结果。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "complex_dataset_inspection")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.num_samples, args.num_views, args.tile_size, args.workers) < 1:
        parser.error("Counts and sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(args.index.read_text(encoding="utf-8"), object_pairs_hook=list)
    if not isinstance(pairs, list) or not all(isinstance(k, str) and isinstance(v, str) for k, v in pairs):
        raise ValueError("Expected an object mapping string IDs to string mesh paths")
    index = dict(pairs)
    quality_path = args.index.parent / "part_biaozhu_complex_labelcount_zw.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    memberships = {uid: [] for uid in index}
    for group, values in quality.items():
        for uid in set(values) & index.keys():
            memberships[uid].append(group)
    summary = {"index": str(args.index), "records": len(pairs), "unique_ids": len(index),
               "duplicate_keys": len(pairs) - len(index), "duplicate_paths": len(index) - len(set(index.values())),
               "id_parent_mismatches": sum(uid != Path(path).parent.name for uid, path in index.items()),
               "filenames": dict(Counter(Path(p).name for p in index.values())),
               "roots": dict(Counter(str(Path(p).parent.parent) for p in index.values())),
               "quality_source": str(quality_path),
               "quality_groups": {k: {"source_count": len(v), "intersection_count": len(set(v) & index.keys())} for k, v in quality.items()},
               "quality_overlap_ids": sum(len(v) > 1 for v in memberships.values()),
               "quality_unclassified_ids": sum(not v for v in memberships.values()),
               "seed": args.seed, "requested_samples": args.num_samples}
    print(f"Checking {len(index):,} mesh paths...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stats = list(pool.map(check_path, index.items()))
    sizes = [size for _, size, error in stats if error is None]
    summary.update(accessible=len(sizes), total_glb_bytes=sum(sizes),
                   inaccessible=[{"id": uid, "error": error} for uid, _, error in stats if error],
                   glb_bytes_quantiles={str(q): float(np.quantile(sizes, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 1)} if sizes else {})
    dump(args.output / "summary.json", summary)
    selected = random.Random(args.seed).sample(list(index.items()), min(args.num_samples, len(index)))
    dump(args.output / "selected.json", [{"id": uid, "path": path} for uid, path in selected])
    samples, failures = [], []
    for number, (uid, path) in enumerate(selected, 1):
        print(f"[{number}/{len(selected)}] {uid}", flush=True)
        try:
            sample = inspect_sample(uid, path, memberships[uid], args.output, args.num_views, args.tile_size)
            samples.append(sample)
            print(f'  {sample["glb"]["full_object_texture.glb"]["mesh_count"]} meshes; {sample["view_count"]} views; OK', flush=True)
        except Exception as exc:
            failures.append({"id": uid, "path": path, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {exc}", flush=True)
        dump(args.output / "samples.json", {"samples": samples, "failures": failures})
    if samples:
        columns = min(3, len(samples))
        cw, ch = args.tile_size * 2, args.tile_size + 80
        overview = Image.new("RGB", (columns * (cw + 16) + 16, math.ceil(len(samples) / columns) * (ch + 16) + 80), BG)
        draw = ImageDraw.Draw(overview)
        draw.text((18, 14), f"COMPLEX PART DATASET | {len(samples)} samples | seed {args.seed}", fill=INK, font=font(26))
        draw.text((18, 47), "Cached surface color vs part IDs | per-sample IDs, not semantic classes", fill=INK, font=font(18))
        for i, sample in enumerate(samples):
            with Image.open(args.output / sample["images"]["preview"]) as image:
                overview.paste(image, (16 + (i % columns) * (cw + 16), 80 + (i // columns) * (ch + 16)))
        overview.save(args.output / "overview.png")
    write_gallery(args.output, summary, samples, failures)
    write_report(args.output, summary, samples, failures)
    print(json.dumps({"output": str(args.output), "accessible": summary["accessible"], "samples": len(samples), "failures": failures}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
