"""
Segmentation VAE Dataset.

Loads meshes + face annotations, voxelizes, and produces one-hot label voxels
for training the Seg VAE.

Each sample: SparseTensor with (N, num_classes) one-hot features on 512³ coords.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor
from trellis2.utils.data_utils import load_balanced_group_indices


class _DegenerateSample(Exception):
    """Raised when a sample has too few voxels to be safely fed to the model."""
    pass


class SegVaeDataset(Dataset):
    """
    Dataset for Seg VAE training.

    Loads pre-voxelized seg labels (from step4's intermediate output) or
    voxelizes on-the-fly from mesh + annotations.

    For efficiency, uses pre-saved voxelized data:
        {root}/seg_voxels/{instance}.npz
            coords: (N, 3) int — voxel coordinates at resolution 512
            labels: (N,) int — per-voxel class labels [0, num_classes-1]

    Args:
        roots: data directory
        resolution: voxel grid resolution
        num_classes: number of seg classes
        max_tokens: maximum voxels per sample
        min_tokens: minimum voxels per sample; samples with fewer voxels are
            treated as degenerate and skipped at load time. Empty / near-empty
            sparse tensors make the sparse-conv down/upsample kernels hang or
            fault in the backward pass (this was the cause of a rank-desync
            NCCL deadlock), so we never feed them to the model.
    """

    def __init__(
        self,
        roots: str,
        *,
        resolution: int = 512,
        num_classes: int = 20,
        max_tokens: int = 30000,
        min_tokens: int = 1,
    ):
        super().__init__()
        self.roots = roots.split(',') if ',' in roots else [roots]
        self.resolution = resolution
        self.num_classes = num_classes
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        # per-sample verbose logging for hang debugging: SEG_VAE_DEBUG_SAMPLES=1
        self.debug_samples = os.environ.get('SEG_VAE_DEBUG_SAMPLES', '0') == '1'

        self.instances = []
        self.loads = []
        for root in self.roots:
            voxel_dir = os.path.join(root, 'seg_voxels')
            if os.path.isdir(voxel_dir):
                # Try metadata cache for fast load
                cache_path = os.path.join(voxel_dir, '_token_counts.npy')
                fnames = sorted([f for f in os.listdir(voxel_dir) if f.endswith('.npz')])
                if os.path.exists(cache_path):
                    counts = np.load(cache_path)
                    if len(counts) == len(fnames):
                        for i, fname in enumerate(fnames):
                            self.instances.append((root, fname[:-4]))
                            self.loads.append(min(int(counts[i]), self.max_tokens))
                        continue
                # Build cache
                counts = []
                for fname in fnames:
                    data = np.load(os.path.join(voxel_dir, fname), mmap_mode='r')
                    n = data['coords'].shape[0]
                    counts.append(n)
                    self.instances.append((root, fname[:-4]))
                    self.loads.append(min(n, self.max_tokens))
                np.save(cache_path, np.array(counts))

        print(f'[SegVaeDataset] Loaded {len(self.instances)} instances, '
              f'voxel range: [{min(self.loads)}, {max(self.loads)}]')
        # NOTE: self.loads is clamped to max_tokens; raw counts <min_tokens are
        # what matter for degeneracy. Flag them up-front so bad data is visible.
        n_degenerate = sum(1 for n in self.loads if n < self.min_tokens)
        if n_degenerate > 0:
            print(f'[SegVaeDataset] WARNING: {n_degenerate} instance(s) have '
                  f'<{self.min_tokens} voxels and will be skipped at load time.')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        # Bounded resample loop (not recursion — a run of bad samples must not
        # blow the stack). Skips load errors and degenerate (too-few-voxel)
        # samples, logging the offending instance so bad data can be pinpointed.
        max_retries = 20
        for attempt in range(max_retries):
            root, instance = self.instances[index]
            try:
                return self._load(root, instance)
            except _DegenerateSample as e:
                print(f'[SegVaeDataset] skip degenerate sample {instance}: {e}', flush=True)
            except Exception as e:
                print(f'[SegVaeDataset] error loading {instance}: {e}', flush=True)
            index = np.random.randint(0, len(self))
        raise RuntimeError(
            f'[SegVaeDataset] failed to get a valid sample after {max_retries} retries '
            f'(last index {index}); check data / min_tokens={self.min_tokens}.'
        )

    def _load(self, root, instance):
        data = np.load(os.path.join(root, 'seg_voxels', f'{instance}.npz'))
        coords = torch.tensor(data['coords'], dtype=torch.int32)  # (N, 3)
        labels = torch.tensor(data['labels'], dtype=torch.long)    # (N,)

        # Guard degenerate samples: empty / near-empty sparse tensors make the
        # sparse-conv kernels hang or fault in backward (rank-desync deadlock).
        if coords.shape[0] < self.min_tokens:
            raise _DegenerateSample(f'{coords.shape[0]} voxels (< min_tokens={self.min_tokens})')

        if self.debug_samples:
            print(f'[SegVaeDataset] load {instance}: N={coords.shape[0]}', flush=True)

        # Subsample if too many
        if coords.shape[0] > self.max_tokens:
            idx = torch.randperm(coords.shape[0])[:self.max_tokens]
            coords = coords[idx]
            labels = labels[idx]

        # One-hot encode
        feats = torch.zeros(coords.shape[0], self.num_classes, dtype=torch.float32)
        feats.scatter_(1, labels.unsqueeze(1).clamp(0, self.num_classes - 1), 1.0)

        return {'coords': coords, 'feats': feats}

    @staticmethod
    def collate_fn(batch, split_size=None):
        """Collate into batched SparseTensor."""
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size
            )

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list = []
            feats_list = []
            for i, b in enumerate(sub_batch):
                coords_list.append(torch.cat([
                    torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32),
                    b['coords']
                ], dim=-1))
                feats_list.append(b['feats'])

            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            pack = {}
            pack['x_0'] = SparseTensor(coords=coords, feats=feats)
            pack['x_0']._shape = torch.Size([len(group), sub_batch[0]['feats'].shape[-1]])
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
