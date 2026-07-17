"""
Segmentation Flow Matching Trainer.

Extends TRELLIS.2's SparseFlowMatchingCFGTrainer with:
- Geometry condition (shape_latent from Shape VAE) as part of concat_cond
- Multi-view seg condition (mv_seg_latent from Seg VAE) as part of concat_cond
- Fine-tuning from existing shape DIT checkpoint

DINOv3 image conditioning has been removed — the mv_seg_latent already
provides complete spatial segmentation information, making the image cross-
attention redundant.  The DIT still receives a zero-valued `cond` tensor so
its forward signature remains unchanged.

Data flow:
  concat_cond = [shape_latent(32ch), mv_seg_latent(32ch)] = 64ch
  DIT input = [seg_noise(32ch) | concat_cond(64ch)] = 96ch → input_layer → model_channels
  DIT output = velocity prediction (32ch)
"""
import sys
import os
import functools
from typing import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.modules import sparse as sp
from trellis2.utils.general_utils import dict_reduce
from trellis2.utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from trellis2.trainers.flow_matching.flow_matching import FlowMatchingTrainer
from trellis2.trainers.flow_matching.mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin


class SegFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for segmentation generation with sparse flow matching.

    Handles:
    - Sparse latent x_0 (seg latent, 32ch)
    - concat_cond: multi-view seg condition (SparseTensor, 32ch)
    - cond: always zeros (DINOv3 removed; model learns to ignore this input)
    """

    # DINOv3 conditioning dimension: 1029 tokens x 1024 channels.
    # We pass zeros so the model learns that cross-attention cond carries no info.
    COND_SEQ_LEN = 1029
    COND_CHANNELS = 1024

    def finetune_from(self, finetune_ckpt):
        """Override to support safetensors format."""
        from safetensors.torch import load_file

        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                path = finetune_ckpt[name]
                if path.endswith('.safetensors'):
                    model_ckpt = load_file(path, device=str(self.device))
                else:
                    model_ckpt = torch.load(path, map_location=self.device, weights_only=True)
                # Start from the model's own (initialized) params so that:
                #  - params present only in the model (e.g. new mv_proj / mv_pos_embedder)
                #    keep their random init instead of being dropped (avoids Missing key),
                #  - params present only in the ckpt are ignored,
                #  - shape-mismatched params (e.g. input_layer 32/96 -> 64) keep model init.
                merged = dict(model_state_dict)
                loaded, skipped, mismatched, added = 0, 0, 0, 0
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        skipped += 1
                    elif v.shape != model_state_dict[k].shape:
                        mismatched += 1
                        if self.is_master:
                            print(f'  Shape mismatch: {k} {v.shape} vs {model_state_dict[k].shape}')
                    else:
                        merged[k] = v
                        loaded += 1
                added = sum(1 for k in model_state_dict if k not in model_ckpt)
                if self.is_master:
                    print(f'  Loaded {loaded} params, skipped {skipped} (ckpt-only), '
                          f'shape-mismatch {mismatched}, kept-init {added} (model-only)')
                model_ckpt = merged
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
            else:
                if self.is_master:
                    print(f'  Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier()
    
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """Skip dataset snapshot - base class slicing is incompatible with SparseTensor."""
        pass

    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """Skip snapshot entirely - base class slicing is incompatible with SparseTensor."""
        pass

    def prepare_dataloader(self, **kwargs):
        """Prepare dataloader with balanced sampling."""
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=min(8, int(np.ceil(os.cpu_count() / torch.cuda.device_count()))),
            pin_memory=False,  # SparseTensor not compatible with pin_memory
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        concat_cond: Optional[sp.SparseTensor] = None,
        mv_cond_feats=None,
        mv_cond_coords=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            x_0: Target seg latent SparseTensor (N, 32)
            cond: Ignored (kept for interface compat). Zeros are used instead.
            concat_cond: Geometry (shape) condition SparseTensor (N, 32)
            mv_cond_feats: list (len B) of (Li, 32) normalized mv_seg latent feats
            mv_cond_coords: list (len B) of (Li, 3) mv voxel coords (rescaled to grid)
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        # Zero-valued cross-attention condition (DINOv3 removed).
        batch_size = x_0.shape[0]
        cond = torch.zeros(
            batch_size, self.COND_SEQ_LEN, self.COND_CHANNELS,
            device=x_0.device, dtype=x_0.feats.dtype,
        )

        # Randomly drop mv_cond for CFG training (10% drop rate)
        if mv_cond_feats is not None and self.training:
            drop_mask = torch.rand(batch_size) < 0.1
            if drop_mask.any():
                mv_cond_feats = [
                    None if drop_mask[i] else mv_cond_feats[i]
                    for i in range(batch_size)
                ]
                mv_cond_coords = [
                    None if drop_mask[i] else mv_cond_coords[i]
                    for i in range(batch_size)
                ]

        # Forward through denoiser with concat_cond (shape) and mv cross-attn tokens.
        pred = self.training_models['denoiser'](
            x_t, t * 1000, cond,
            concat_cond=concat_cond,
            mv_cond=mv_cond_feats,
            mv_coords=mv_cond_coords,
            **kwargs
        )
        
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # Log loss per time bin
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """Run inference snapshot for visualization."""
        import copy
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))

        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, 'cuda')
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            cond_vis.append(self.vis_cond(**batch_data))
            
            # Prepare inference args
            x_0 = batch_data.pop('x_0')
            args = self.get_inference_cond(**batch_data)
            
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=12, guidance_strength=0.0, verbose=verbose,
            )
            sample.append(res.samples)
        
        sample = sp.sparse_cat(sample)
        sample_gt = {k: v for k, v in data.items()}
        sample = {k: v if k != 'x_0' else sample for k, v in data.items()}
        
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict


class SegFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SegFlowMatchingTrainer):
    """Seg trainer with classifier-free guidance."""
    pass


class ImageConditionedSegFlowMatchingCFGTrainer(SegFlowMatchingCFGTrainer):
    """
    Main trainer: Segmentation flow matching with CFG.

    DINOv3 image conditioning has been removed. The class name is kept for
    backward-compatibility with existing configs that reference it. Any
    ``image_cond_model`` key in the config is silently ignored.

    Combines:
    - SegFlowMatchingTrainer: handles concat_cond and sparse flow matching
    - ClassifierFreeGuidanceMixin: p_uncond dropout
    """

    def __init__(self, *args, image_cond_model: dict = None, **kwargs):
        # Accept and discard legacy image_cond_model config for backward compat.
        super().__init__(*args, **kwargs)
