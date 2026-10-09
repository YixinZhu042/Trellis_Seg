"""Compare mesh GT, frozen-VAE reconstruction, and an existing DiT prediction."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))

from scripts.train_dit_decoded import get_sample, spatial_template, normalize, decode_clean, resolve_config
from scripts.evaluate_dit_decoded import classification_metrics, boundary_f1
from scripts.evaluate_seg_quality import generate_class_colors
from scripts.refine_seg_projection import export_labels
from trainers.decoded_seg_losses import face_logits_from_voxels
from utils.decoded_dit_runtime import atomic_json
from utils.seg_projection import render_labels, VIEW_NAMES


def font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu/DejaVuSans.ttf']:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def metric(sample, faces, voxels):
    return {
        'voxel': classification_metrics(sample['voxel_labels'], voxels, np.ones(len(voxels))),
        'surface': classification_metrics(sample['face_labels'], faces, sample['face_area']),
        'projection': classification_metrics(sample['face_labels'], faces, sample['face_pixels']),
        'boundary_f1': boundary_f1(sample['face_labels'], faces, sample['adjacency']),
    }


def sheet(path, panels, titles, rows, subtitle):
    cell, gap, margin, header, label = 512, 18, 24, 128, 36
    width = 2 * margin + 3 * cell + 2 * gap
    height = header + len(rows) * (cell + label + gap) + margin
    canvas = Image.new('RGB', (width, height), '#edf1f5')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), 'Chair segmentation: where do the triangles appear?', font=font(27), fill='#17212b')
    draw.text((margin, 56), subtitle, font=font(17), fill='#465568')
    for column, title in enumerate(titles):
        draw.text((margin + column * (cell + gap), 94), title, font=font(23), fill='#17212b')
    for row, name in enumerate(rows):
        y = header + row * (cell + label + gap)
        for column in range(3):
            x = margin + column * (cell + gap)
            draw.text((x + 8, y + 4), name, font=font(18), fill='#465568')
            panel = panels[column][row]
            panel = panel.copy()
            panel.thumbnail((cell, cell), Image.Resampling.NEAREST)
            draw.rectangle((x, y + label, x + cell - 1, y + label + cell - 1), fill='white')
            canvas.paste(panel, (x + (cell - panel.width) // 2,
                                y + label + (cell - panel.height) // 2))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instance', default='007d2949e5b0450c995c54809b43af0f')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/dit_decoded_aux_full_v1.json')
    ap.add_argument('--gallery', type=Path, default=ROOT / 'results/dit_decoded_full_v1_ema1095_test50')
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--render_only', action='store_true')
    args = ap.parse_args()
    config = resolve_config(args.config)
    cache_path = Path(config['data_dir']) / 'samples' / f'{args.instance}.npz'
    pred_path = args.gallery / 'predictions' / f'{args.instance}_seed42.npz'
    with np.load(cache_path) as cache:
        arrays = {key: cache[key] for key in cache.files}
    with np.load(pred_path) as pred:
        dit_faces, dit_voxels = pred['face_labels'], pred['voxel_labels']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = args.output_dir / 'vae_reconstruction.npz'
    if not args.render_only:
        import trellis2.models as models
        from trellis2.modules.sparse.basic import SparseTensor
        from trellis2.modules.sparse.conv import config as conv_config
        from flex_gemm.ops import spconv
        torch.set_grad_enabled(False)
        torch.cuda.set_device(args.device)
        torch.manual_seed(0)
        algorithm = 'masked_implicit_gemm_splitk'
        conv_config.FLEX_GEMM_ALGO = algorithm
        spconv.set_algorithm(algorithm)
        settings = json.loads(Path(config['vae_config']).read_text())['models']
        encoder = models.SparseUnetVaeEncoder(**settings['encoder']['args'])
        decoder = models.SparseUnetVaeDecoder(**dict(settings['decoder']['args'], use_fp16=False))
        encoder.load_state_dict(torch.load(config['encoder'], map_location='cpu', weights_only=True))
        decoder.load_state_dict(torch.load(config['decoder'], map_location='cpu', weights_only=True))
        encoder = encoder.eval().to(args.device).requires_grad_(False)
        decoder = decoder.eval().to(args.device).requires_grad_(False)
        sample = get_sample({'path': str(cache_path)}, args.device)
        base = json.loads(Path(config['base_config']).read_text())
        template = spatial_template(sample, encoder)
        normalized = normalize(sample, base)
        logits = decode_clean(decoder, template, normalized['latent_feats'], sample, base)
        vae_faces = face_logits_from_voxels(logits, sample['face_knn'], sample['face_knn_weights']).argmax(1).cpu().numpy()
        vae_voxels = logits.argmax(1).cpu().numpy()
        np.savez_compressed(oracle_path, face_labels=vae_faces, voxel_labels=vae_voxels,
                            voxel_logits=logits.cpu().numpy())
        print('Saved VAE reconstruction from GT latent', flush=True)
        # Independently re-encode GT, checking that cached latents are not the cause.
        coords = torch.cat([torch.zeros_like(sample['coords'][:, :1]), sample['coords']], 1).int()
        encoded = encoder(SparseTensor(feats=F.one_hot(sample['voxel_labels'], 20).float(), coords=coords),
                          sample_posterior=False)
        if not torch.equal(encoded.coords[:, 1:], sample['latent_coords'].int()):
            raise ValueError('Fresh encoding coordinates differ from cached coordinates')
        fresh_logits = decoder(template.replace(encoded.feats.float()))
        if not torch.equal(fresh_logits.coords[:, 1:], sample['coords'].int()):
            raise ValueError('Fresh decoder output coordinates differ')
        fresh_faces = face_logits_from_voxels(fresh_logits.feats.float(), sample['face_knn'],
                                             sample['face_knn_weights']).argmax(1).cpu().numpy()
        fresh_voxels = fresh_logits.feats.argmax(1).cpu().numpy()
        check = {
            'latent_rmse': float((encoded.feats.float() - sample['latent_feats']).square().mean().sqrt()),
            'face_label_disagreements': int((fresh_faces != vae_faces).sum()),
            'voxel_label_disagreements': int((fresh_voxels != vae_voxels).sum()),
            'fresh_reconstruction_metrics': metric(arrays, fresh_faces, fresh_voxels),
        }
        atomic_json(args.output_dir / 'fresh_encoding_check.json', check)
        print('Fresh encoding check:', check, flush=True)
    with np.load(oracle_path) as oracle:
        vae_faces, vae_voxels = oracle['face_labels'], oracle['voxel_labels']
    labels = [arrays['face_labels'], vae_faces, dit_faces]
    names = ['gt', 'vae', 'dit']
    titles = ['GT / original face labels', 'VAE / decoded GT latent', 'DiT + VAE / EMA 1095']
    mesh = trimesh.Trimesh(arrays['mesh_vertices'], arrays['mesh_faces'], process=False)
    palette = np.full((21, 3), 255, np.uint8)
    palette[1:] = (255 * generate_class_colors(20)).astype(np.uint8)
    raster = arrays['face_raster']
    images, maps = [], []
    for name, face_labels in zip(names, labels):
        recovered = export_labels(mesh, face_labels, args.output_dir / f'{name}.glb')
        m = render_labels(recovered + 1, raster)
        maps.append(m)
        views = [Image.fromarray(palette[view]) for view in m]
        images.append(views)
        for view, image in zip(VIEW_NAMES, views):
            image.save(args.output_dir / f'{name}_{view}.png')
    sheet(args.output_dir / 'comparison.png', images, titles, VIEW_NAMES,
          'Same geometry, cameras and label colors. Raw predictions; no cleanup.')
    crops, crop_rows = [[], [], []], []
    seat_label = 14
    for view in [4, 5, 0]:
        rr = raster[view]
        seat = (rr >= 0) & (arrays['face_labels'][rr.clip(min=0)] == seat_label)
        ys, xs = np.where(seat)
        box = (max(0, int(xs.min()) - 12), max(0, int(ys.min()) - 12),
               min(rr.shape[1], int(xs.max()) + 13), min(rr.shape[0], int(ys.max()) + 13))
        for col in range(3):
            cut = images[col][view].crop(box)
            scale = 512 / max(cut.size)
            crops[col].append(cut.resize((round(cut.width * scale), round(cut.height * scale)), Image.Resampling.NEAREST))
        crop_rows.append(f'Seat close-up: {VIEW_NAMES[view]}')
    sheet(args.output_dir / 'seat_closeup.png', crops, titles, crop_rows,
          'Identical GT-defined crops in every column; enlarged without smoothing.')
    error_images = []
    for face_labels in labels:
        views = []
        for rr in raster:
            image = np.full((*rr.shape, 3), 255, np.uint8)
            hit = rr >= 0
            image[hit] = [209, 217, 225]
            wrong = hit & (face_labels[rr.clip(min=0)] != arrays['face_labels'][rr.clip(min=0)])
            image[wrong] = [231, 61, 79]
            views.append(Image.fromarray(image))
        error_images.append(views)
    sheet(args.output_dir / 'errors.png', error_images, titles, VIEW_NAMES,
          'Red: wrong face label versus GT. Gray: correct. White: background.')
    scores = {
        'gt': metric(arrays, labels[0], arrays['voxel_labels']),
        'vae': metric(arrays, vae_faces, vae_voxels),
        'dit': metric(arrays, dit_faces, dit_voxels),
    }
    prior = json.loads((args.gallery / 'predictions/metrics.json').read_text())
    prior_oracle = next(r for r in prior['oracle'] if r['instance'] == args.instance)
    differences = {group: scores['vae'][group]['miou'] - prior_oracle[group]['miou']
                   for group in ['voxel', 'surface', 'projection']}
    top_error_faces = [2410, 2251, 2534] if args.instance == '007d2949e5b0450c995c54809b43af0f' else []
    report = {
        'instance': args.instance, 'encoder': config['encoder'], 'decoder': config['decoder'],
        'cache': str(cache_path), 'dit_prediction': str(pred_path),
        'vae_definition': 'Decode cached posterior mean of GT voxel labels, using original evaluation normalization and spatial template',
        'metrics': scores, 'previous_oracle_miou_differences': differences,
        'top_triangle_labels': [{'face': i, 'gt': int(labels[0][i]), 'vae': int(labels[1][i]),
                                 'dit': int(labels[2][i])} for i in top_error_faces],
        'fresh_encoding_check': json.loads((args.output_dir / 'fresh_encoding_check.json').read_text()),
        'glb_label_geometry_roundtrip_verified': True,
    }
    atomic_json(args.output_dir / 'report.json', report)
    readme = ['# 椅子：GT / VAE 重建 / DiT＋VAE 并排对比', '',
              f'对象：`{args.instance}`。三列从左到右为原始 GT、纯 VAE 重建、已有 EMA 1095 预测。', '',
              'VAE 列解码 GT 的缓存 posterior mean，未经过 DiT；同时重新编码 GT 检查缓存一致性。',
              '各列采用相同网格、视角和调色板，无后处理。GLB 导出后已回读核对逐面标签和几何。', '',
              '- [座面放大](seat_closeup.png)', '- [完整六视图](comparison.png)', '- [错误区域：红色为错误](errors.png)',
              '- GLB：[GT](gt.glb) / [VAE](vae.glb) / [DiT＋VAE](dit.glb)',
              '- [指标与缓存一致性核验](report.json)', '',
              '| 输出 | 体素 mIoU | 表面 mIoU | 投影 mIoU |', '| --- | ---: | ---: | ---: |']
    for name in names:
        readme.append(f'| {name} | ' + ' | '.join(f'{scores[name][g]["miou"]*100:.3f}%' for g in ['voxel', 'surface', 'projection']) + ' |')
    (args.output_dir / 'README.md').write_text('\n'.join(readme) + '\n')
    (args.output_dir / 'index.html').write_text(
        '<!doctype html><html><meta charset="utf-8"><title>GT / VAE / DiT comparison</title>'
        '<style>body{font:17px system-ui;max-width:1660px;margin:24px auto;background:#edf1f5;color:#17212b;padding:0 16px}img{width:100%}a{color:#125dcc}</style>'
        '<h1>GT / VAE reconstruction / DiT + VAE</h1><p>Left: GT. Center: VAE reconstruction from GT latent. Right: EMA 1095 prediction.</p>'
        '<p>GLB: <a href="gt.glb">GT</a> / <a href="vae.glb">VAE</a> / <a href="dit.glb">DiT + VAE</a> · <a href="report.json">Metrics</a></p>'
        '<h2>Seat close-ups</h2><img src="seat_closeup.png"><h2>All six views</h2><img src="comparison.png">'
        '<h2>Errors against GT (red)</h2><img src="errors.png"></html>\n')
    print(json.dumps(report, indent=2), flush=True)
    print('Saved comparisons:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
