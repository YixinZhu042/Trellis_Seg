"""Distributed bookkeeping shared by the decoded-supervision experiment.

No TRELLIS/CUDA imports are needed to check launch settings or batch indexing.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def distributed_topology(environ=None):
    env = os.environ if environ is None else environ
    world = int(env.get('WORLD_SIZE', '1'))
    rank = int(env.get('RANK', '0'))
    local = int(env.get('LOCAL_RANK', '0'))
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError(f'Invalid distributed topology: rank={rank}, local_rank={local}, world_size={world}')
    return rank, local, world


def training_sample_index(step, micro, rank, world_size, accumulate, num_samples,
                          seed=0, epoch_size=None):
    """Interleave ranks within each global batch; wrap at dataset boundaries."""
    if step < 1 or accumulate < 1 or num_samples < 1:
        raise ValueError('step, accumulate and num_samples must be positive')
    if not 0 <= micro < accumulate or not 0 <= rank < world_size:
        raise ValueError('Invalid microbatch or rank')
    epoch_size = num_samples if epoch_size is None else epoch_size
    if epoch_size < 1:
        raise ValueError('epoch_size must be positive')
    global_index = ((step - 1) * accumulate + micro) * world_size + rank
    epoch, offset = divmod(global_index, epoch_size)
    # Affine permutation modulo N. Pick a stride coprime with N, making the
    # mapping bijective without storing/sharing a large permutation array.
    stride = (seed + 2 * epoch + 1) % num_samples
    if stride == 0:
        stride = 1
    import math
    while math.gcd(stride, num_samples) != 1:
        stride += 1
        if stride == num_samples:
            stride = 1
    shift = (seed * 1103515245 + epoch * 12345) % num_samples
    return (offset * stride + shift) % num_samples


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def master_action(action, rank, distributed):
    """Propagate rank-zero filesystem errors instead of leaving peers waiting."""
    status = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            status[0] = f'{type(exc).__name__}: {exc}'
    if distributed:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


class FeatureOutputModel(torch.nn.Module):
    """Expose tensor outputs to DDP's unused-parameter traversal.

    TRELLIS returns SparseTensor objects, which DDP does not inspect for tensor
    leaves. Returning .feats permits find_unused_parameters=True when randomly
    dropped multi-view conditions leave mv_proj parameters unused.
    """
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, *args, **kwargs):
        return self.denoiser(*args, **kwargs).feats
