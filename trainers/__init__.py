"""
Trainer registry. Extends TRELLIS.2 trainers with seg-specific trainer.
"""
import sys
import os

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from .seg_trainer import SegFlowMatchingTrainer, ImageConditionedSegFlowMatchingCFGTrainer
from .seg_vae_trainer import SegVaeTrainer


def get_trainer(name, model_dict, dataset, **kwargs):
    """Get trainer by name."""
    trainers = {
        'SegFlowMatchingTrainer': SegFlowMatchingTrainer,
        'ImageConditionedSegFlowMatchingCFGTrainer': ImageConditionedSegFlowMatchingCFGTrainer,
        'SegVaeTrainer': SegVaeTrainer,
    }
    if name in trainers:
        return trainers[name](model_dict, dataset, **kwargs)
    
    # Fall back to TRELLIS.2 trainers
    from trellis2 import trainers as trellis2_trainers
    return getattr(trellis2_trainers, name)(model_dict, dataset, **kwargs)
