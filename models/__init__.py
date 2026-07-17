"""
Model registry. Reuses TRELLIS.2 models and adds seg-specific models.
"""
import sys
import os

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'TRELLIS.2')
sys.path.insert(0, TRELLIS2_ROOT)

from trellis2.models import *
from trellis2.models import from_pretrained
from .seg_flow_model import SegSLatFlowModel, ElasticSegSLatFlowModel


def get_model(name, **kwargs):
    """Get model by name. Supports both TRELLIS.2 models and seg-specific models."""
    import trellis2.models as trellis2_models
    
    # Check local models first
    local_models = {
        'SegSLatFlowModel': SegSLatFlowModel,
        'ElasticSegSLatFlowModel': ElasticSegSLatFlowModel,
    }
    
    if name in local_models:
        return local_models[name](**kwargs)
    
    # Fall back to TRELLIS.2 models
    return getattr(trellis2_models, name)(**kwargs)
