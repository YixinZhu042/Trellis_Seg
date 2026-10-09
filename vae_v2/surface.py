"""Learned geometric attention over sparse decoder features, with residual labels."""
import math
import torch
from torch import nn


class SurfaceHead(nn.Module):
    def __init__(self, feature_channels=64, hidden_channels=64, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.attention = nn.Sequential(nn.Linear(6, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, 1))
        self.refine = nn.Sequential(nn.Linear(feature_channels + 7, hidden_channels), nn.SiLU(),
                                    nn.Linear(hidden_channels, num_classes))
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, features, logits, query):
        index = query['neighbor_index']
        scores = self.attention(query['neighbor_geometry'].float()).squeeze(-1).float() + query['neighbor_prior']
        scores = scores.masked_fill(~query['neighbor_mask'], -1e4)
        attention = scores.softmax(-1)
        fused = (features[index].float() * attention[..., None]).sum(1)
        base = (logits[index].float() * attention[..., None]).sum(1)
        geometry = torch.cat([query['query_normal'], query['query_position'], query['query_fallback'][:, None]], -1)
        prediction = base + self.refine(torch.cat([fused, geometry], -1)).float()
        points = prediction.reshape(-1, 4, self.num_classes)
        # Average probabilities from four surface points; no single centroid decides a face.
        face = torch.logsumexp(points.log_softmax(-1), dim=1) - math.log(4)
        return points, face
