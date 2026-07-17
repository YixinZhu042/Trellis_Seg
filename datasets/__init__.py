"""
Dataset registry.
"""
import sys
import os

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from .seg_dataset import SegSLatDataset, MultiViewSegCondSLatDataset
from .seg_vae_dataset import SegVaeDataset


def get_dataset(name, data_dir, **kwargs):
    """Get dataset by name."""
    datasets = {
        'MultiViewSegCondSLatDataset': MultiViewSegCondSLatDataset,
        'SegSLatDataset': SegSLatDataset,
        'SegVaeDataset': SegVaeDataset,
    }
    if name in datasets:
        return datasets[name](data_dir, **kwargs)
    raise ValueError(f'Unknown dataset: {name}')
