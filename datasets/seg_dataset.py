"""
Segmentation-conditioned Structured Latent Dataset.

Data format:
- Pre-encoded seg latent: {root}/seg_latent/{instance}.npz
  → coords (N, 3), feats (N, 32)
- 6-view segmentation maps: {root}/seg_views/{instance}.npz
  → (6, H, W) integer labels

Conditioning pipeline (reference from project/):
1. 6-view seg maps → one-hot → Seg VAE encode → sparse latent tokens
2. Each pixel gets 3D coordinates via box_surface_image_ids mapping
3. All 6 views' latent tokens are merged into a single SparseTensor (concat_cond)
4. Cross-attention condition is set to zeros (DINOv3 removed)
"""
import os
import json
from typing import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from einops import rearrange

import sys
TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules.sparse.basic import SparseTensor
from trellis2.utils.data_utils import load_balanced_group_indices


# Segmentation colormap (20 classes) — maximally distinguishable
SEG_COLORMAP = np.array([
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
], dtype=np.float32) / 255.0


def box_surface_image_ids(h, w, rotation_result=0):
    """Generate 3D position IDs for image pixels on a box surface.
    
    Maps 2D pixel coordinates to 3D coordinates on the 6 faces of a unit cube.
    
    Args:
        h, w: image height and width
        rotation_result: view index (0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom)
    
    Returns:
        Tensor of shape (h*w, 3) with 3D coordinates in [0, 1]
    """
    img_ids = torch.zeros(h, w, 3)

    if rotation_result == 0:  # front view (z=1)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = 1
    elif rotation_result == 1:  # right view (x=1)
        img_ids[..., 0] = 1
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = (w - torch.arange(1, w + 1)[None, :]) / (w - 1)
    elif rotation_result == 2:  # back view (z=0)
        img_ids[..., 0] = (w - torch.arange(1, w + 1)[None, :]) / (w - 1)
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = 0
    elif rotation_result == 3:  # left view (x=0)
        img_ids[..., 0] = 0
        img_ids[..., 1] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)
        img_ids[..., 2] = torch.arange(w)[None, :] / (w - 1)
    elif rotation_result == 4:  # top view (y=1)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = 1
        img_ids[..., 2] = torch.arange(h)[:, None] / (h - 1)
    elif rotation_result == 5:  # bottom view (y=0)
        img_ids[..., 0] = torch.arange(w)[None, :] / (w - 1)
        img_ids[..., 1] = 0
        img_ids[..., 2] = (h - torch.arange(1, h + 1)[:, None]) / (h - 1)

    img_ids = rearrange(img_ids, "h w c -> (h w) c")
    return img_ids


def colorize_seg_map(seg_map, num_classes=20):
    """Convert integer segmentation map to RGB using colormap.
    
    Args:
        seg_map: (H, W) integer labels
    Returns:
        (3, H, W) float tensor in [0, 1]
    """
    h, w = seg_map.shape
    color_map = torch.tensor(SEG_COLORMAP[:num_classes])  # (C, 3)
    seg_map_clamped = seg_map.clamp(0, num_classes - 1).long()
    colored = color_map[seg_map_clamped.reshape(-1)]  # (H*W, 3)
    colored = colored.reshape(h, w, 3).permute(2, 0, 1)  # (3, H, W)
    return colored


class SegSLatDataset(Dataset):
    """
    Base segmentation structured latent dataset.
    Loads pre-encoded seg latent for training the DIT.
    
    Args:
        roots: data directory path
        resolution: spatial resolution of the latent grid
        max_tokens: maximum number of sparse tokens per sample
        num_classes: number of segmentation classes
        normalization: dict with 'mean' and 'std' for latent normalization
    """
    def __init__(
        self,
        roots: str,
        *,
        resolution: int = 512,
        max_tokens: int = 8192,
        num_classes: int = 20,
        normalization: Optional[dict] = None,
        image_size: int = 512,
        mv_image_size: int = 512,
    ):
        super().__init__()
        self.roots = roots.split(',') if ',' in roots else [roots]
        self.resolution = resolution
        self.max_tokens = max_tokens
        self.num_classes = num_classes
        self.image_size = image_size
        self.mv_image_size = mv_image_size
        self.normalization = normalization
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1).clamp_min(1e-6)
        
        # Load instances
        self.instances = []
        self.loads = []
        for root in self.roots:
            metadata_path = os.path.join(root, 'metadata.csv')
            if os.path.exists(metadata_path):
                import pandas as pd
                metadata = pd.read_csv(metadata_path)
                for _, row in metadata.iterrows():
                    sha256 = row['sha256']
                    tokens = row.get('seg_latent_tokens', self.max_tokens)
                    if tokens <= self.max_tokens:
                        self.instances.append((root, sha256))
                        self.loads.append(tokens)
            else:
                # Fallback: scan directory
                latent_dir = os.path.join(root, 'seg_latent')
                if os.path.isdir(latent_dir):
                    for fname in os.listdir(latent_dir):
                        if fname.endswith('.npz'):
                            instance = fname[:-4]
                            self.instances.append((root, instance))
                            self.loads.append(self.max_tokens)
        
        print(f'[SegSLatDataset] Loaded {len(self.instances)} instances from {len(self.roots)} roots')
    
    def __len__(self):
        return len(self.instances)
    
    def get_instance(self, root, instance):
        """Load a single instance."""
        # Load seg latent
        latent_path = os.path.join(root, 'seg_latent', f'{instance}.npz')
        data = np.load(latent_path)
        coords = torch.tensor(data['coords']).int()  # (N, 3)
        feats = torch.tensor(data['feats']).float()   # (N, 32)
        
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        
        return {
            'coords': coords,
            'feats': feats,
        }
    
    def __getitem__(self, index):
        # Bounded retry: skip corrupt/missing samples without unbounded recursion.
        for attempt in range(20):
            idx = index if attempt == 0 else np.random.randint(0, len(self))
            root, instance = self.instances[idx]
            try:
                return self.get_instance(root, instance)
            except Exception as e:
                print(f'Error loading {instance}: {e} (attempt {attempt + 1}/20)')
        raise RuntimeError(
            f'Failed to load a valid sample after 20 attempts starting at index {index}; '
            f'dataset likely has systematically broken/missing files.'
        )
    
    @staticmethod
    def collate_fn(batch, split_size=None):
        """Collate sparse tensors into batched SparseTensor."""
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size
            )
        
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([
                    torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32),
                    b['coords']
                ], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(coords=coords, feats=feats)
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # Collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in keys:
                if k in ('mv_cond_feats', 'mv_cond_coords'):
                    # Variable-length per-sample mv tokens → keep as a list (no stacking).
                    pack[k] = [b[k] for b in sub_batch]
                elif isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    # Merge SparseTensors for concat_cond
                    all_coords = []
                    all_feats = []
                    cond_layout = []
                    cond_start = 0
                    for j, b in enumerate(sub_batch):
                        st = b[k]
                        # Remap batch dim
                        new_coords = st.coords.clone()
                        new_coords[:, 0] = j
                        all_coords.append(new_coords)
                        all_feats.append(st.feats)
                        cond_layout.append(slice(cond_start, cond_start + st.coords.shape[0]))
                        cond_start += st.coords.shape[0]
                    merged_coords = torch.cat(all_coords)
                    merged_feats = torch.cat(all_feats)
                    merged_st = SparseTensor(coords=merged_coords, feats=merged_feats)
                    merged_st._shape = torch.Size([len(group), *sub_batch[0][k].feats.shape[1:]])
                    merged_st.register_spatial_cache('layout', cond_layout)
                    pack[k] = merged_st
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            packs.append(pack)
        
        if split_size is None:
            return packs[0]
        return packs


class MultiViewSegCondSLatDataset(SegSLatDataset):
    """
    Multi-view segmentation conditioned structured latent dataset.

    Extends SegSLatDataset with:
    - Shape latent (geometry) → from Shape VAE encoding the mesh
    - 6-view segmentation maps → encoded as sparse latent
    - Both shape_latent and mv_seg_latent are merged as concat_cond
    - cond slot returns zeros (DINOv3 cross-attention removed)

    Data structure expected:
        {root}/seg_latent/{instance}.npz    - pre-encoded 3D seg latent (target)
        {root}/shape_latent/{instance}.npz  - pre-encoded shape latent (geometry cond)
        {root}/mv_seg_latent/{instance}.npz - pre-encoded multi-view seg cond latent
        {root}/seg_views/{instance}.npz     - 6 view seg maps (6, H, W) int

    Conditioning strategy (analogous to TRELLIS.2 texture generation):
        - shape_latent provides 3D geometry (spatial structure defines where to predict)
        - mv_seg_latent provides multi-view 2D segmentation hints
        - concat_cond = cat([shape_latent, mv_seg_latent], dim=-1) → 64ch
        - Together with seg_latent noise (32ch), DIT sees 96ch input
    """
    
    def __init__(
        self,
        roots: str,
        *,
        resolution: int = 512,
        max_tokens: int = 8192,
        num_classes: int = 20,
        normalization: Optional[dict] = None,
        mv_cond_normalization: Optional[dict] = None,
        shape_cond_normalization: Optional[dict] = None,
        image_size: int = 512,
        mv_image_size: int = 512,
        use_preencoded_mv_cond: bool = True,
        use_shape_cond: bool = True,
        mv_as_cross_attn: bool = False,
        mv_enc_resolution: int = 64,
    ):
        super().__init__(
            roots,
            resolution=resolution,
            max_tokens=max_tokens,
            num_classes=num_classes,
            normalization=normalization,
            image_size=image_size,
            mv_image_size=mv_image_size,
        )
        self.use_preencoded_mv_cond = use_preencoded_mv_cond
        self.use_shape_cond = use_shape_cond
        self.mv_as_cross_attn = mv_as_cross_attn
        self.mv_enc_resolution = mv_enc_resolution
        self.mv_cond_normalization = mv_cond_normalization
        self.shape_cond_normalization = shape_cond_normalization
        
        if self.mv_cond_normalization is not None:
            self.mv_cond_mean = torch.tensor(self.mv_cond_normalization['mean']).reshape(1, -1)
            self.mv_cond_std = torch.tensor(self.mv_cond_normalization['std']).reshape(1, -1).clamp_min(1e-6)

        if self.shape_cond_normalization is not None:
            self.shape_cond_mean = torch.tensor(self.shape_cond_normalization['mean']).reshape(1, -1)
            self.shape_cond_std = torch.tensor(self.shape_cond_normalization['std']).reshape(1, -1).clamp_min(1e-6)
    
    def get_instance(self, root, instance):
        """Load instance with geometry + multi-view conditioning."""
        pack = super().get_instance(root, instance)

        # seg_latent coords are the canonical spatial structure for concat_cond,
        # since sparse_cat requires matching token counts between x and concat_cond.
        seg_coords = pack['coords']  # (N, 3)

        # Load shape latent (geometry condition)
        if self.use_shape_cond:
            shape_path = os.path.join(root, 'shape_latent', f'{instance}.npz')
            shape_data = np.load(shape_path)
            shape_coords = torch.tensor(shape_data['coords']).int()   # (M1, 3)
            shape_feats = torch.tensor(shape_data['feats']).float()    # (M1, 32)

            if self.shape_cond_normalization is not None:
                shape_feats = (shape_feats - self.shape_cond_mean) / self.shape_cond_std

        # Load multi-view seg condition (VAE-encoded 32ch latent tokens)
        if self.use_preencoded_mv_cond:
            mv_cond_path = os.path.join(root, 'mv_seg_latent', f'{instance}.npz')
            mv_data = np.load(mv_cond_path)
            mv_coords = torch.tensor(mv_data['coords']).int()    # (M, 3) in [0, 31]
            mv_feats = torch.tensor(mv_data['feats']).float()    # (M, 32)

            if self.mv_cond_normalization is not None:
                mv_feats = (mv_feats - self.mv_cond_mean) / self.mv_cond_std

        # Build concat_cond aligned to seg_latent coords (same tokens as x_0).
        # Look up shape/mv features at seg_latent positions via coordinate matching.
        if self.mv_as_cross_attn:
            # NEW design (mirrors native_tex_dit): mv_seg latent is NOT concatenated on
            # aligned voxels (its box-surface coords don't align with the seg/shape grid).
            # concat_cond carries ONLY shape (32ch); mv_seg becomes separate cross-attn tokens.
            batch_coords = torch.cat([
                torch.zeros(seg_coords.shape[0], 1, dtype=torch.int32),
                seg_coords
            ], dim=-1)
            if self.use_shape_cond:
                shape_at_seg = self._lookup_feats_at_coords(seg_coords, shape_coords, shape_feats)
            else:
                shape_at_seg = torch.zeros(seg_coords.shape[0], 32)
            pack['concat_cond'] = SparseTensor(coords=batch_coords, feats=shape_at_seg)

            # mv tokens: VAE latent 32ch + coords on latent grid [0, 31].
            # Coords are already in the same latent space as seg_latent/shape_latent,
            # so pass directly to AbsolutePositionEmbedder (no rescaling needed).
            if self.use_preencoded_mv_cond:
                pack['mv_cond_feats'] = mv_feats                 # (M, 32)
                pack['mv_cond_coords'] = mv_coords.float()       # (M, 3)
            else:
                pack['mv_cond_feats'] = torch.zeros(0, 32)
                pack['mv_cond_coords'] = torch.zeros(0, 3)
        elif self.use_shape_cond and self.use_preencoded_mv_cond:
            shape_at_seg = self._lookup_feats_at_coords(seg_coords, shape_coords, shape_feats)
            mv_at_seg = self._lookup_feats_at_coords(seg_coords, mv_coords, mv_feats)
            merged_feats = torch.cat([shape_at_seg, mv_at_seg], dim=-1)  # (N, 64)
            batch_coords = torch.cat([
                torch.zeros(seg_coords.shape[0], 1, dtype=torch.int32),
                seg_coords
            ], dim=-1)
            pack['concat_cond'] = SparseTensor(coords=batch_coords, feats=merged_feats)
        elif self.use_shape_cond:
            shape_at_seg = self._lookup_feats_at_coords(seg_coords, shape_coords, shape_feats)
            padded_feats = torch.cat([shape_at_seg, torch.zeros_like(shape_at_seg)], dim=-1)
            batch_coords = torch.cat([
                torch.zeros(seg_coords.shape[0], 1, dtype=torch.int32),
                seg_coords
            ], dim=-1)
            pack['concat_cond'] = SparseTensor(coords=batch_coords, feats=padded_feats)
        elif self.use_preencoded_mv_cond:
            mv_at_seg = self._lookup_feats_at_coords(seg_coords, mv_coords, mv_feats)
            padded_feats = torch.cat([torch.zeros_like(mv_at_seg), mv_at_seg], dim=-1)
            batch_coords = torch.cat([
                torch.zeros(seg_coords.shape[0], 1, dtype=torch.int32),
                seg_coords
            ], dim=-1)
            pack['concat_cond'] = SparseTensor(coords=batch_coords, feats=padded_feats)
        else:
            pack['seg_views'] = self._load_seg_views(root, instance)
        
        # Condition image slot (zeros — DINOv3 removed, kept for interface compat)
        cond_image = self._load_cond_image(root, instance)
        pack['cond'] = cond_image
        
        return pack
    
    def _lookup_feats_at_coords(self, target_coords, source_coords, source_feats):
        """
        Look up source features at target coordinates via coordinate matching.

        For target voxels without a matching source voxel, returns zeros.
        For multiple source voxels mapping to the same target position, averages them.

        Args:
            target_coords: (N, 3) int - coordinates to look up at
            source_coords: (M, 3) int - coordinates with known features
            source_feats: (M, C) float - features at source coordinates

        Returns:
            matched_feats: (N, C) float - features at target coordinates
        """
        N = target_coords.shape[0]
        C = source_feats.shape[-1]
        matched_feats = torch.zeros(N, C, dtype=source_feats.dtype, device=source_feats.device)
        if source_coords.shape[0] == 0 or N == 0:
            return matched_feats

        res = max(target_coords.max().item(), source_coords.max().item()) + 1

        source_linear = (source_coords[:, 0].long() * res * res +
                         source_coords[:, 1].long() * res +
                         source_coords[:, 2].long())
        target_linear = (target_coords[:, 0].long() * res * res +
                         target_coords[:, 1].long() * res +
                         target_coords[:, 2].long())

        # Average source features sharing a linear index (vectorized, no Python loop).
        uniq_lin, inverse = torch.unique(source_linear, return_inverse=True)
        sums = torch.zeros(uniq_lin.shape[0], C, dtype=source_feats.dtype, device=source_feats.device)
        sums.index_add_(0, inverse, source_feats)
        counts = torch.zeros(uniq_lin.shape[0], dtype=source_feats.dtype, device=source_feats.device)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=source_feats.dtype))
        uniq_feats = sums / counts.unsqueeze(-1)

        # Match target linear indices against the sorted unique source indices.
        pos = torch.searchsorted(uniq_lin, target_linear).clamp(max=uniq_lin.shape[0] - 1)
        hit = uniq_lin[pos] == target_linear
        matched_feats[hit] = uniq_feats[pos[hit]]

        return matched_feats
    
    def _load_seg_views(self, root, instance):
        """Load 6-view segmentation maps (restore original label space)."""
        seg_views_path = os.path.join(root, 'seg_views', f'{instance}.npz')
        data = np.load(seg_views_path)
        seg_views = torch.tensor(data['seg_maps']).long()  # (6, H, W) — 0=bg, 1+=parts
        seg_views = (seg_views - 1).clamp(min=0)  # restore: bg→0 stays 0, parts→original labels
        return seg_views
    
    def _load_cond_image(self, root, instance):
        """Return zeros for the DINOv3 condition slot.

        DINOv3 image conditioning has been removed (the mv_seg_latent already
        provides complete spatial segmentation information). The dataset still
        returns a tensor here so the collate/trainer interface is unchanged —
        the trainer will ignore it and pass zeros to the DIT's cross-attention.
        """
        return torch.zeros(3, self.image_size, self.image_size)
