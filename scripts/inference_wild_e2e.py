"""
End-to-end in-the-wild inference: mesh -> rendered views -> 2D seg maps -> colored mesh.

Usage:
    python scripts/inference_wild_e2e.py \
        --mesh /path/to/input.glb \
        --output ./results/input_colored.glb \
        --steps 50

Intermediate files are kept under ``--work_dir`` so failed API/model steps can be resumed.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
VIEW_NAMES = ['front', 'right', 'back', 'left', 'top', 'bottom']
DEFAULT_ENV_MAP = PROJECT_DIR / 'studio_small_08_4k.exr'


def abspath(path):
    return str(Path(path).expanduser().absolute())


def quote_cmd(cmd):
    return ' '.join(shlex.quote(str(x)) for x in cmd)


def run_step(title, cmd, dry_run=False):
    print(f'\n=== {title} ===')
    print(quote_cmd(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)


def render_outputs_exist(render_dir):
    render_dir = Path(render_dir)
    return all((render_dir / f'{name}.png').exists() for name in VIEW_NAMES) and (render_dir / 'occupancy.npy').exists()


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end in-the-wild inference: input mesh, output colored mesh.'
    )
    parser.add_argument('--mesh', type=str, required=True, help='Input mesh path (.glb/.obj)')
    parser.add_argument('--output', type=str, required=True, help='Output colored mesh .glb path')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Directory for intermediate renders and segmentation files')
    parser.add_argument('--render_dir', type=str, default=None,
                        help='Override render intermediate directory')
    parser.add_argument('--seg_dir', type=str, default=None,
                        help='Override 2D segmentation intermediate directory')
    parser.add_argument('--seg_maps', type=str, default=None,
                        help='Use existing seg_maps.npz and skip rendering/API segmentation')
    parser.add_argument('--api_seg_grid', type=str, default=None,
                        help='Reuse an existing API output grid PNG; still converts it to seg_maps.npz')
    parser.add_argument('--resolution', type=int, default=512, help='Render and segmentation map resolution')
    parser.add_argument('--env_map', type=str, default=str(DEFAULT_ENV_MAP), help='HDR environment map for rendering')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seg_prior_mode', type=str, default='consistency',
                        choices=['off', 'components', 'interior', 'override', 'raw_ids', 'consistency'])
    parser.add_argument('--prior_component_agreement', type=float, default=0.5)
    parser.add_argument('--prior_erode_rings', type=int, default=1)
    parser.add_argument('--projection_subdivide', type=int, default=None,
                        help='Boundary subdivision rounds (default: 5 in consistency mode; 0 preserves face count)')
    parser.add_argument('--projection_cleanup', choices=['graph', 'ties'], default='graph')
    parser.add_argument('--projection_accuracy_budget', type=float, default=.003)
    parser.add_argument('--projection_miou_budget', type=float, default=.01)
    parser.add_argument('--part_iou_refine', choices=['on', 'off'], default='on')
    parser.add_argument('--hidden_prior_confidence', type=float, default=.8)
    parser.add_argument('--hidden_smooth', type=float, default=.25)
    parser.add_argument('--bake_knn', type=int, default=8)
    parser.add_argument('--smooth_iterations', type=int, default=2)
    parser.add_argument('--min_component_faces', type=int, default=0)
    parser.add_argument('--postprocess_after_prior', action='store_true',
                        help='Apply component cleanup and smoothing after 2D prior fusion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None,
                        help='DIT config path passed to scripts/inference_wild.py')
    parser.add_argument('--dit_ckpt', type=str, default=None,
                        help='DIT checkpoint path passed to scripts/inference_wild.py')
    parser.add_argument('--seg_encoder', default=None)
    parser.add_argument('--seg_decoder', default=None)
    parser.add_argument('--save_color', action='store_true',
                        help='Save per-view colorized 2D segmentation images')
    parser.add_argument('--force', action='store_true',
                        help='Re-run render/API segmentation even if intermediate files already exist')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without executing them')
    args = parser.parse_args()

    mesh_path = Path(abspath(args.mesh))
    output_path = Path(abspath(args.output))
    if not args.dry_run and not mesh_path.exists():
        raise FileNotFoundError(f'Input mesh not found: {mesh_path}')

    if args.work_dir:
        work_dir = Path(abspath(args.work_dir))
    else:
        work_dir = output_path.parent / f'{mesh_path.stem}_wild_e2e'
    render_dir = Path(abspath(args.render_dir)) if args.render_dir else work_dir / 'renders'
    seg_dir = Path(abspath(args.seg_dir)) if args.seg_dir else work_dir / 'seg'
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

    print('End-to-end in-the-wild inference')
    print(f'  mesh     : {mesh_path}')
    print(f'  output   : {output_path}')
    print(f'  work_dir : {work_dir}')

    if args.seg_maps:
        seg_maps_path = Path(abspath(args.seg_maps))
        if not args.dry_run and not seg_maps_path.exists():
            raise FileNotFoundError(f'seg_maps not found: {seg_maps_path}')
        print(f'  seg_maps : {seg_maps_path} (provided; skip render/API segmentation)')
    else:
        render_cmd = [
            sys.executable, str(PROJECT_DIR / 'scripts' / 'render_views.py'),
            '--mesh', str(mesh_path),
            '--output_dir', str(render_dir),
            '--resolution', str(args.resolution),
            '--env_map', abspath(args.env_map),
            '--device', args.device,
        ]
        if args.force or not render_outputs_exist(render_dir):
            run_step('Step 1/3: render 6 orthographic views', render_cmd, args.dry_run)
        else:
            print(f'\n=== Step 1/3: render 6 orthographic views ===')
            print(f'Skip: existing renders found in {render_dir}')

        seg_maps_path = seg_dir / 'seg_maps.npz'
        seg_cmd = [
            sys.executable, str(PROJECT_DIR / 'scripts' / 'segment_views_via_api.py'),
            '--input_dir', str(render_dir),
            '--output_dir', str(seg_dir),
            '--output_npz', str(seg_maps_path),
            '--cell_size', str(args.resolution),
        ]
        if args.api_seg_grid:
            seg_cmd += ['--api_seg_grid', abspath(args.api_seg_grid)]
        if args.save_color:
            seg_cmd.append('--save_color')

        if args.force or not seg_maps_path.exists():
            run_step('Step 2/3: segment rendered views via API', seg_cmd, args.dry_run)
        else:
            print(f'\n=== Step 2/3: segment rendered views via API ===')
            print(f'Skip: existing seg_maps found at {seg_maps_path}')

    infer_cmd = [
        sys.executable, str(PROJECT_DIR / 'scripts' / 'inference_wild.py'),
        '--mesh', str(mesh_path),
        '--seg_maps', str(seg_maps_path),
        '--output', str(output_path),
        '--steps', str(args.steps),
        '--seg_prior_mode', args.seg_prior_mode,
        '--prior_component_agreement', str(args.prior_component_agreement),
        '--prior_erode_rings', str(args.prior_erode_rings),
        '--projection_cleanup', args.projection_cleanup,
        '--projection_accuracy_budget', str(args.projection_accuracy_budget),
        '--projection_miou_budget', str(args.projection_miou_budget),
        '--part_iou_refine', args.part_iou_refine,
        '--hidden_prior_confidence', str(args.hidden_prior_confidence),
        '--hidden_smooth', str(args.hidden_smooth),
        '--bake_knn', str(args.bake_knn),
        '--smooth_iterations', str(args.smooth_iterations),
        '--min_component_faces', str(args.min_component_faces),
        '--seed', str(args.seed),
        '--device', args.device,
    ]
    if args.postprocess_after_prior:
        infer_cmd.append('--postprocess_after_prior')
    if args.projection_subdivide is not None:
        infer_cmd += ['--projection_subdivide', str(args.projection_subdivide)]
    if args.config:
        infer_cmd += ['--config', abspath(args.config)]
    if args.dit_ckpt:
        infer_cmd += ['--dit_ckpt', abspath(args.dit_ckpt)]
    if args.seg_encoder:
        infer_cmd += ['--seg_encoder', abspath(args.seg_encoder)]
    if args.seg_decoder:
        infer_cmd += ['--seg_decoder', abspath(args.seg_decoder)]

    run_step('Step 3/3: run 3D segmentation and export colored mesh', infer_cmd, args.dry_run)
    print(f'\nDone. Colored mesh: {output_path}')
    print(f'Intermediates: {work_dir}')


if __name__ == '__main__':
    main()
