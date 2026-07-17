"""
Segmentation VAE Trainer.

Fine-tunes the Texture VAE (SparseUnetVaeEncoder + SparseUnetVaeDecoder) for
segmentation label reconstruction.

Input: one-hot 20ch labels on sparse voxels
Output: 20ch logits → cross-entropy loss per voxel
Latent: 32ch (same as original Texture VAE)

Fine-tuning from pre-trained Texture VAE:
- Encoder input_layer: Linear(6, 64) → expanded to Linear(20, 64), first 6 cols kept
- Decoder output_layer: Linear(64, 6) → expanded to Linear(64, 20), first 6 rows kept
- All intermediate layers: loaded as-is
"""
import os
import sys
import copy
import functools
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.trainers.basic import BasicTrainer
from trellis2.utils.data_utils import cycle, BalancedResumableSampler


class SegVaeTrainer(BasicTrainer):
    """
    Trainer for Segmentation VAE fine-tuning.

    Reconstruction: cross-entropy on 20-class voxel labels.
    KL: standard VAE KL divergence regularization.
    """

    def __init__(
        self,
        *args,
        lambda_kl: float = 1e-6,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_kl = lambda_kl

    def finetune_from(self, finetune_ckpt):
        """Load pretrained Texture VAE, expanding input/output channels (6→20)."""
        from safetensors.torch import load_file

        if self.is_master:
            print('\nFinetuning Seg VAE from Texture VAE:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                path = finetune_ckpt[name]
                if path.endswith('.safetensors'):
                    ckpt = load_file(path, device=str(self.device))
                else:
                    ckpt = torch.load(path, map_location=self.device, weights_only=True)

                loaded, skipped, expanded = 0, 0, 0
                for k in list(ckpt.keys()):
                    if k not in model_state_dict:
                        skipped += 1
                        del ckpt[k]
                    elif ckpt[k].shape != model_state_dict[k].shape:
                        old_shape = ckpt[k].shape
                        new_shape = model_state_dict[k].shape
                        # Expand: copy old weights into new (larger) tensor
                        new_param = model_state_dict[k].clone()
                        # Copy overlapping region
                        slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                        new_param[slices] = ckpt[k][slices]
                        ckpt[k] = new_param
                        expanded += 1
                        if self.is_master:
                            print(f'    Expanded: {k} {list(old_shape)} → {list(new_shape)}')
                    else:
                        loaded += 1

                model_ckpts[name] = ckpt
                model.load_state_dict(ckpt)
                if self.is_master:
                    print(f'  [{name}] loaded={loaded}, expanded={expanded}, skipped={skipped}')
            else:
                if self.is_master:
                    print(f'  Warning: {name} not in finetune_ckpt')
                model_ckpts[name] = model_state_dict

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier()

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=min(8, int(np.ceil(os.cpu_count() / torch.cuda.device_count()))),
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0,  # SparseTensor: (N, 20) one-hot labels
        **kwargs
    ) -> Tuple[Dict, Dict]:
        # Per-step batch profile for hang debugging (SEG_VAE_DEBUG_SAMPLES=1).
        # Printed from the main process right before the forward/backward, so the
        # LAST line before a hang is exactly the batch that hung (no dataloader
        # prefetch skew). Helps pin the sparse-conv backward-GEMM stall to a batch.
        if os.environ.get('SEG_VAE_DEBUG_SAMPLES', '0') == '1':
            try:
                counts = torch.bincount(x_0.coords[:, 0].long()).tolist()
            except Exception:
                counts = [int(x_0.feats.shape[0])]
            print(f'[SegVaeTrainer] step~{getattr(self, "step", "?")} '
                  f'batch total_N={int(x_0.feats.shape[0])} per_sample_N={counts}', flush=True)

        # Encode
        z, mean, logvar = self.training_models['encoder'](
            x_0, sample_posterior=True, return_raw=True
        )

        # Decode
        logits = self.training_models['decoder'](z)

        terms = edict(loss=0.0)

        # Cross-entropy: logits (N, 20) vs target labels (N,)
        target = x_0.feats.argmax(dim=-1)  # one-hot → class index
        terms["ce"] = F.cross_entropy(logits.feats, target)
        terms["loss"] = terms["ce"]

        # Accuracy
        with torch.no_grad():
            pred = logits.feats.argmax(dim=-1)
            terms["acc"] = (pred == target).float().mean()

        # KL divergence
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        return terms, {}

    def snapshot_dataset(self, num_samples=100, batch_size=4):
        pass

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=16, batch_size=4, verbose=False):
        if self.step == 0:
            return
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        total_correct = 0
        total_voxels = 0

        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            x_0 = data['x_0'].cuda()

            z = self.models['encoder'](x_0, sample_posterior=False)
            logits = self.models['decoder'](z)

            target = x_0.feats.argmax(dim=-1)
            pred = logits.feats.argmax(dim=-1)
            total_correct += (pred == target).sum().item()
            total_voxels += target.shape[0]

        acc = total_correct / max(total_voxels, 1)
        if self.is_master:
            print(f'  [Snapshot] Accuracy: {acc:.4f} ({total_correct}/{total_voxels})')

        return {}
