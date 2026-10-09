"""Configuration and provenance, usable without TRELLIS or CUDA."""
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path(os.environ.get('TRELLISSEG_STORAGE_ROOT', '/path/to/trellisseg_run'))


def storage_path(path, cfg):
    """All newly generated data/checkpoint caches live on the designated disk."""
    root = Path(cfg['storage_root']).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Data/checkpoint output must be a subdirectory of {root}: {path}')
    return path


def default_run_dir(config_path, cfg):
    return storage_path(Path(cfg['storage_root']) / 'checkpoints' / Path(config_path).stem, cfg)


def resume_metadata_compatible(old, new):
    """Allow relocating identical audited data, not changing data or training settings."""
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    for metadata in (old, new):
        for key in ('prepared_dir', 'storage_root'):
            metadata['config'].pop(key, None)
        metadata['config']['train'].pop('steps', None)
    return old == new


def permutation_resume_compatible(old, new):
    """Explicit fork may ONLY enable label renaming (and extend total steps).

    Never silently relax normal --resume or permit changed model/loss/data/DDP.
    """
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    if (old['config'].get('num_classes') != 64 or new['config'].get('num_classes') != 64
            or old['config']['train'].get('label_permutation', 'none') != 'none'
            or new['config']['train'].get('label_permutation') != 'all_nonzero_v1'):
        return False
    old['config']['train']['label_permutation'] = 'all_nonzero_v1'
    return resume_metadata_compatible(old, new)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def read_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError('Circular config inheritance')
    seen.add(path)
    config = json.loads(path.read_text())
    parent = config.pop('_base', None)
    if parent:
        config = merge(read_config(path.parent / parent, seen), config)
    return config


def load_config(path):
    cfg = read_config(path)
    cfg.setdefault('storage_root', str(STORAGE_ROOT))
    cfg['storage_root'] = str(Path(cfg['storage_root']).resolve())
    classes = cfg.get('num_classes', 20)
    if not isinstance(classes, int) or isinstance(classes, bool) or not 2 <= classes <= 256:
        raise ValueError('num_classes must be an integer in 2..256, including unknown channel 0')
    if cfg.get('annotation_policy'):
        from .annotations import ANNOTATION_POLICY
        if cfg['annotation_policy'] != ANNOTATION_POLICY:
            raise ValueError('Unsupported raw annotation policy')
        for key in ('arrow_dir','mesh_dir','annotation_dir'):
            cfg[key] = str((ROOT / cfg[key]).resolve())
        storage_path(cfg['annotation_dir'],cfg)
    elif classes != 20:
        raise ValueError('Expanded classes require corrected raw annotations, not legacy cached labels')
    if cfg.get('dataset_split'):
        from .splits import validate_spec
        validate_spec(cfg['dataset_split'])
        if not cfg.get('annotation_policy'):
            raise ValueError('Repartitioning requires the corrected raw annotation pipeline')
    for key in ('source_dir', 'prepared_dir', 'vae_config', 'encoder_init', 'decoder_init'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    storage_path(cfg['prepared_dir'], cfg)
    if cfg['model']['stages'] not in (4, 5) or cfg['model']['latent_channels'] not in (32, 64):
        raise ValueError('Supported capacity experiments: stages=4/5, latent_channels=32/64')
    if cfg['model']['mode'] not in ('surface', 'voxel'):
        raise ValueError('model.mode must be surface or voxel')
    if cfg['model']['stages'] != 5 and cfg['model']['initialize_backbone']:
        raise ValueError('Changed stage layout requires initialize_backbone=false')
    for key in ('max_voxels', 'area_samples', 'candidates', 'neighbors'):
        if cfg['geometry'][key] < 1:
            raise ValueError(f'geometry.{key} must be positive')
    if cfg['geometry']['candidates'] < cfg['geometry']['neighbors']:
        raise ValueError('candidates must cover neighbors')
    if not 0 <= cfg['geometry']['normal_min_dot'] <= 1:
        raise ValueError('normal_min_dot must be in [0,1]')
    if cfg['geometry']['near_pair_radius_voxels'] <= 0:
        raise ValueError('near_pair_radius_voxels must be positive')
    for key in ('steps', 'accumulate', 'query_faces', 'pair_count', 'save_every', 'validate_every', 'validation_limit'):
        if not isinstance(cfg['train'][key], int) or cfg['train'][key] < 1:
            raise ValueError(f'train.{key} must be positive')
    for key in ('warmup_steps', 'deterministic_steps', 'kl_warmup_steps'):
        if cfg['train'][key] < 0:
            raise ValueError(f'train.{key} must be nonnegative')
    if not 0 <= cfg['train']['balanced_fraction'] <= 1:
        raise ValueError('balanced_fraction must be in [0,1]')
    if not 0 < cfg['train']['ema_rate'] < 1:
        raise ValueError('ema_rate must be in (0,1)')
    if cfg['train']['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Use fp32 or bf16 (no unscaled FP16 training)')
    if cfg['train'].get('label_permutation', 'none') not in ('none', 'all_nonzero_v1'):
        raise ValueError('Unsupported label permutation policy')
    if cfg['train']['lr'] <= 0 or cfg['train']['surface_lr'] <= 0 or cfg['train']['kl_weight'] < 0:
        raise ValueError('Invalid optimizer or KL settings')
    expected = {'voxel_ce', 'voxel_dice', 'point_ce', 'surface_ce', 'surface_dice',
                'reprojection', 'consistency', 'boundary'}
    if set(cfg['loss']) != expected or any(v < 0 for v in cfg['loss'].values()):
        raise ValueError('Loss weights must specify all terms with nonnegative weights')
    if not any(cfg['loss'].values()):
        raise ValueError('At least one reconstruction loss must be active')
    import math
    if any(not math.isfinite(v) for v in cfg['loss'].values()) or any(not math.isfinite(cfg['train'][key]) for key in
          ('lr','surface_lr','kl_weight','grad_clip','weight_decay','ema_rate')):
        raise ValueError('Nonfinite training settings')
    if cfg['train']['grad_clip'] <= 0 or cfg['train']['weight_decay'] < 0:
        raise ValueError('Invalid gradient clipping or weight decay')
    if cfg['model']['conv_algorithm'] != 'explicit_gemm':
        raise ValueError('Training currently supports the verified explicit_gemm backend')
    return cfg


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
