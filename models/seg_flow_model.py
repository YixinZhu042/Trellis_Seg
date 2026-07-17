"""
Segmentation-conditioned Structured Latent Flow Model.

Extends TRELLIS.2's SLatFlowModel to support:
- Geometry conditioning via shape_latent (from Shape VAE), feature-concatenated
- Multi-view 2D segmentation conditioning via mv_seg_latent (from Seg/Texture VAE),
  injected as EXTRA cross-attention tokens (NOT feature-concat).

Design (mirrors native_tex_dit flux_pe_texture):
- The mv_seg latent lives on box-surface voxel coordinates that do NOT align with the
  shape/seg voxel grid. So instead of concatenating it on aligned voxels (which fails —
  ~0% coordinate overlap), we project the mv latent to the cross-attention dimension,
  add an absolute positional embedding from its (rescaled) voxel coords, and append the
  resulting tokens to the DINOv3 image tokens as a single variable-length context.
  Attention then mixes them into every voxel — no coordinate alignment required.

Channels:
- in_channels  = 64 (32 noised seg_latent + 32 shape_latent concat)
- out_channels = 32 (velocity for seg_latent)
- mv_seg latent (32ch) → mv_proj → cond_channels (1024) cross-attn tokens

Fine-tuning from TRELLIS.2 shape DIT:
- Original shape DIT has in_channels=32; input_layer (32/96→64) shape mismatch is skipped
  by finetune_from(); mv_proj / mv_pos_embedder are new params (random init).
"""
import sys
import os
from typing import *

import torch
import torch.nn as nn

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.models.structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
from trellis2.models.sparse_elastic_mixin import SparseTransformerElasticMixin
from trellis2.modules import sparse as sp
from trellis2.modules.transformer import AbsolutePositionEmbedder
from trellis2.modules.utils import manual_cast


class _MvCrossAttnMixin:
    """
    Adds multi-view seg latent as extra cross-attention tokens.

    Expects the base class to be an SLatFlowModel (defines self.blocks, self.cond_channels,
    self.input_layer, etc.). Overrides forward() to merge mv tokens into the context.

    mv conditioning is passed to forward() as:
        mv_cond:   list (len B) of (Li, 32) float tensors  — normalized mv_seg latent feats
        mv_coords: list (len B) of (Li, 3)  float/int tensors — mv voxel coords already
                   rescaled to the [0, resolution-1] positional space of the main grid.
    Samples with no mv condition may pass None / empty tensors.
    """

    def _init_mv_modules(self):
        self.mv_proj = nn.Linear(self.out_channels, self.cond_channels)
        self.mv_pos_embedder = AbsolutePositionEmbedder(self.cond_channels)
        nn.init.xavier_uniform_(self.mv_proj.weight)
        nn.init.zeros_(self.mv_proj.bias)

    def _build_context(self, cond, mv_cond, mv_coords, device, dtype):
        """
        Merge DINOv3 image tokens (cond) with mv tokens into a per-sample context list,
        returned as a VarLenTensor. If no mv is provided, returns cond unchanged.
        """
        if mv_cond is None:
            return cond

        # Normalize cond into a per-sample list of (Li, C) tensors.
        if isinstance(cond, torch.Tensor):
            assert cond.ndim == 3, f"expected (B, L, C) image cond, got {cond.shape}"
            cond_list = [cond[i] for i in range(cond.shape[0])]
        elif isinstance(cond, list):
            cond_list = cond
        else:  # VarLenTensor
            cond_list = cond.to_tensor_list()

        B = len(cond_list)
        assert len(mv_cond) == B, f"mv_cond len {len(mv_cond)} != batch {B}"

        merged = []
        for i in range(B):
            ctx = cond_list[i]
            feats = mv_cond[i]
            if feats is not None and feats.shape[0] > 0:
                coords = mv_coords[i].to(device=device, dtype=torch.float32)
                mv_tok = self.mv_proj(feats.to(device=device, dtype=self.mv_proj.weight.dtype))
                pe = self.mv_pos_embedder(coords).to(mv_tok.dtype)
                mv_tok = mv_tok + pe
                ctx = torch.cat([ctx.to(mv_tok.dtype), mv_tok], dim=0)
            merged.append(manual_cast(ctx, dtype))
        return sp.VarLenTensor.from_tensor_list(merged)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        mv_cond: Optional[List[torch.Tensor]] = None,
        mv_coords: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)

        # Build the (image + mv) cross-attention context.
        context = self._build_context(cond, mv_cond, mv_coords, x.device, self.dtype)
        if isinstance(context, list):
            context = sp.VarLenTensor.from_tensor_list(context)

        import torch.nn.functional as F
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        context = manual_cast(context, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, context)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


class SegSLatFlowModel(_MvCrossAttnMixin, SLatFlowModel):
    """
    SLatFlowModel for segmentation generation.

    - in_channels=64 (32 seg_latent noise + 32 shape_latent concat)
    - out_channels=32 (seg latent velocity)
    - mv_seg latent injected via cross-attention (see _MvCrossAttnMixin)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mv_modules()


class ElasticSegSLatFlowModel(SparseTransformerElasticMixin, _MvCrossAttnMixin, SLatFlowModel):
    """
    Elastic (memory-managed) version of SegSLatFlowModel.

    MRO ordering is deliberate: SparseTransformerElasticMixin (→ ElasticModuleMixin.forward,
    which records memory usage for the LinearMemoryController and then calls super().forward)
    must sit BEFORE _MvCrossAttnMixin, so that ElasticModuleMixin.forward wraps our mv-aware
    forward. If _MvCrossAttnMixin came first, its forward would shadow ElasticModuleMixin's and
    update_run_states() would never fire → the controller's mem-ratio buffer stays empty →
    ZeroDivisionError in MemoryController.record().

    Equivalent to ElasticSLatFlowModel (SparseTransformerElasticMixin + SLatFlowModel) with the
    mv cross-attn forward spliced in between the elastic mixin and the base model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mv_modules()
