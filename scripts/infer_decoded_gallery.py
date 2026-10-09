"""Run the validation-selected DiT on a fixed test subset and export a gallery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    selection_path = ROOT / 'eval/dit_decoded_full_v1/selection.json'
    selection = json.loads(selection_path.read_text())
    selected = selection['best_overall_on_validation']
    config_path = ROOT / 'configs/dit_decoded_aux_full_v1.json'
    config = json.loads(config_path.read_text())
    data_dir = ROOT / config['data_dir']
    cohort = json.loads((data_dir / 'selection.json').read_text())['samples']
    cohort = sorted((r for r in cohort if r['split'] == 'test'), key=lambda r: r['model_id'])[:args.limit]
    if len(cohort) != args.limit:
        raise ValueError('Not enough test objects')
    checkpoint = Path(selected['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    samples = []
    for row in cohort:
        cache = data_dir / 'samples' / (row['model_id'] + '.npz')
        if not cache.is_file():
            raise FileNotFoundError(cache)
        samples.append(dict(row, instance=row['model_id'], cache=str(cache)))
    manifest = {
        'checkpoint_name': selected['name'], 'checkpoint': str(checkpoint),
        'selection_source': str(selection_path), 'selection_rule': selection['selection_rule'],
        'checkpoint_bytes': checkpoint.stat().st_size,
        'config': str(config_path), 'split': 'test', 'seed': args.seed,
        'sampling_steps': 50, 'limit': args.limit,
        'sample_selection': 'First N test objects sorted by instance ID; no quality filtering',
        'condition_source': 'Prepared ground-truth six-view segmentation and shape latent',
        'postprocessing': False, 'label_matching': False,
        'conv_algorithm': 'masked_implicit_gemm_splitk',
        'samples': samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing output manifest differs; choose a new output directory')
    elif any(args.output_dir.iterdir()):
        raise ValueError('Output directory is not empty and has no manifest')
    else:
        write_json(path, manifest)
    return manifest


def infer(args, manifest):
    output = args.output_dir / 'predictions'
    logs = args.output_dir / 'logs'
    logs.mkdir(exist_ok=True)
    common = [sys.executable, str(ROOT / 'scripts/evaluate_dit_decoded.py'),
              '--config', manifest['config'], '--checkpoint', manifest['checkpoint'],
              '--output_dir', str(output), '--split', 'test', '--limit', str(args.limit),
              '--seeds', str(args.seed), '--conv_algorithm', manifest['conv_algorithm']]
    commands = []
    for rank, gpu in enumerate(args.gpus):
        command = common + ['--resume', '--no_merge', '--independent_worker', '--device', 'cuda:0',
                            '--worker_rank', str(rank), '--worker_world_size', str(len(args.gpus))]
        commands.append({'gpu': gpu, 'command': command})
    launch_path = args.output_dir / 'launch.json'
    launch = {'workers': commands, 'python': sys.executable}
    if launch_path.exists() and json.loads(launch_path.read_text()) != launch:
        raise ValueError('Resume requires the same worker topology and Python environment')
    write_json(launch_path, launch)
    def worker(rank):
        spec = commands[rank]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu']),
                   OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        with (logs / f'inference_rank{rank:04d}.log').open('a') as handle:
            handle.write(f'\nLaunch {datetime.now(timezone.utc).isoformat()}\n')
            handle.flush()
            result = subprocess.run(spec['command'], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        print(f'Inference worker {rank}: exit={result.returncode}', flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        statuses = list(pool.map(worker, range(len(commands))))
    if any(statuses):
        raise RuntimeError(f'Inference failed: {statuses}; inspect {logs}')
    subprocess.run(common + ['--merge_only', '--worker_rank', '0',
                             '--worker_world_size', str(len(args.gpus))], cwd=ROOT, check=True)


def export(args, manifest):
    import numpy as np
    from PIL import Image, ImageDraw
    import trimesh
    from scripts.evaluate_seg_quality import generate_class_colors
    from scripts.refine_seg_projection import export_labels
    from utils.seg_projection import render_labels, VIEW_NAMES

    prediction_dir = args.output_dir / 'predictions'
    metrics = json.loads((prediction_dir / 'metrics.json').read_text())
    by_id = {r['instance']: r for r in metrics['samples']}
    expected = {r['instance'] for r in manifest['samples']}
    if set(by_id) != expected or len(metrics['samples']) != args.limit or metrics['checkpoint'] != manifest['checkpoint']:
        raise ValueError('Prediction metrics do not match the fixed manifest')
    if metrics['seeds'] != [args.seed] or not metrics['save_predictions']:
        raise ValueError('Unexpected prediction protocol')
    palette = np.full((21, 3), 255, dtype=np.uint8)
    palette[1:] = (generate_class_colors(20) * 255).astype(np.uint8)
    glb_dir = args.output_dir / 'glb'
    render_dir = args.output_dir / 'renders'
    glb_dir.mkdir(exist_ok=True)
    render_dir.mkdir(exist_ok=True)
    artifacts = []
    overview = Image.new('RGB', (1600, ((args.limit + 4) // 5) * 250), 'white')
    overview_draw = ImageDraw.Draw(overview)
    for index, row in enumerate(manifest['samples']):
        uid = row['instance']
        with np.load(row['cache']) as data:
            mesh = trimesh.Trimesh(vertices=data['mesh_vertices'], faces=data['mesh_faces'], process=False)
            raster, inputs = data['face_raster'], data['seg_maps']
        pred_path = prediction_dir / f'{uid}_seed{args.seed}.npz'
        with np.load(pred_path) as prediction:
            labels = prediction['face_labels']
        if labels.shape != (len(mesh.faces),) or labels.min() < 0 or labels.max() >= 20:
            raise ValueError(f'Invalid face labels: {uid}')
        glb_path = glb_dir / f'{uid}.glb'
        # Shared exporter reloads the GLB and checks exact labels and face geometry.
        delivered = export_labels(mesh, labels, glb_path)
        # Geometry and face order are verified above, so cached visibility remains valid.
        maps = render_labels(delivered + 1, raster)
        images = [Image.fromarray(palette[view]) for view in maps]
        individual = render_dir / uid
        individual.mkdir(exist_ok=True)
        size = images[0].width
        grid = Image.new('RGB', (size * 3, (size + 24) * 2), 'white')
        draw = ImageDraw.Draw(grid)
        comparison = Image.new('RGB', (256 * 6, 280 * 2), 'white')
        comp_draw = ImageDraw.Draw(comparison)
        for view, name in enumerate(VIEW_NAMES):
            images[view].save(individual / f'{name}.png')
            x, y = (view % 3) * size, (view // 3) * (size + 24)
            draw.text((x + 8, y + 5), name, fill='black')
            grid.paste(images[view], (x, y + 24))
            for line, title, panel in [(0, 'Input 2D', Image.fromarray(palette[inputs[view]])),
                                       (1, 'EMA prediction', images[view])]:
                comp_draw.text((view * 256 + 5, line * 280 + 5), f'{title}: {name}', fill='black')
                comparison.paste(panel.resize((256, 256), Image.Resampling.NEAREST), (view * 256, line * 280 + 24))
        grid_path = render_dir / f'{uid}_views.png'
        comparison_path = render_dir / f'{uid}_comparison.png'
        grid.save(grid_path)
        comparison.save(comparison_path)
        x, y = (index % 5) * 320, (index // 5) * 250
        overview.paste(grid.resize((320, 223), Image.Resampling.LANCZOS), (x, y + 25))
        overview_draw.text((x + 5, y + 5), f'{index+1:02d}  {uid[:12]}', fill='black')
        metric = by_id[uid]
        artifact = {'instance': uid, 'seed': args.seed, 'faces': len(mesh.faces),
                    'predicted_labels': np.unique(labels).tolist(),
                    'glb': str(glb_path.relative_to(args.output_dir)),
                    'views': str(grid_path.relative_to(args.output_dir)),
                    'comparison': str(comparison_path.relative_to(args.output_dir)),
                    'individual_views': [str((individual / f'{v}.png').relative_to(args.output_dir)) for v in VIEW_NAMES],
                    'prediction': str(pred_path.relative_to(args.output_dir)),
                    'glb_roundtrip_verified': True, 'metrics': metric}
        artifacts.append(artifact)
        write_json(args.output_dir / 'export_progress.json', {'completed': len(artifacts), 'expected': args.limit, 'last_instance': uid})
        print(f'Export {index+1}/{args.limit}: {uid}', flush=True)
    overview.save(args.output_dir / 'overview.png')
    report = {'checkpoint': manifest['checkpoint'], 'objects': len(artifacts),
              'glb_count': len(artifacts), 'individual_render_count': 6 * len(artifacts),
              'summary': metrics['summary'], 'artifacts': artifacts,
              'rendering': 'Six orthographic 512px flat part-color views; cached face visibility; white background',
              'completed_at': datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / 'report.json', report)
    cards = []
    for index, item in enumerate(artifacts):
        score = item['metrics']['projection']['miou']
        score_text = 'n/a' if score is None else f'{score*100:.3f}%'
        cards.append(f'<article><h2>{index+1:02d} · {item["instance"]}</h2>'
                     f'<p>Projection mIoU: {score_text} · '
                     f'<a href="{item["glb"]}">GLB</a> · <a href="{item["comparison"]}">Input comparison</a></p>'
                     f'<a href="{item["views"]}"><img loading="lazy" src="{item["views"]}" alt="Six views"></a></article>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>DiT EMA · 50 test objects</title><style>body{font:16px system-ui;background:#f3f5f7;color:#17212b;max-width:1400px;margin:32px auto;padding:0 20px}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}article{background:white;padding:18px;border-radius:12px}'
            'h2{font-size:15px;overflow-wrap:anywhere}img{width:100%}a{color:#125dcc}header{margin-bottom:28px}</style>'
            f'<header><h1>{html.escape(manifest["checkpoint_name"])} · {args.limit} test objects</h1>'
            f'<p>Seed {args.seed} · 50 sampling steps · raw prediction · six orthographic views</p>'
            '<p><a href="overview.png">Overview</a> · <a href="report.json">Metrics and artifact manifest</a></p></header><main>'
            + ''.join(cards) + '</main></html>')
    (args.output_dir / 'index.html').write_text(page + '\n')
    print(f'Gallery ready: {args.output_dir / "index.html"}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--prepare_only', action='store_true')
    ap.add_argument('--export_only', action='store_true')
    args = ap.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.limit < 1 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        ap.error('Require a positive limit and unique nonnegative GPU indices')
    manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps({'checkpoint': manifest['checkpoint'], 'objects': len(manifest['samples']),
                          'output': str(args.output_dir)}, indent=2))
        return
    if not args.export_only:
        infer(args, manifest)
    export(args, manifest)


if __name__ == '__main__':
    main()
