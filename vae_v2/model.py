"""VAE backbone plus surface head. GT-free decode is a first-class interface."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torch import nn
from .surface import SurfaceHead

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TRELLIS.2'))


class SegVaeV2(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
        from trellis2.modules.sparse.conv import config as conv_config
        from flex_gemm.ops import spconv
        conv_config.FLEX_GEMM_ALGO = cfg['model']['conv_algorithm']
        spconv.set_algorithm(cfg['model']['conv_algorithm'])
        settings = json.loads(Path(cfg['vae_config']).read_text())['models']
        enc, dec = dict(settings['encoder']['args']), dict(settings['decoder']['args'])
        self.num_classes = cfg.get('num_classes', 20)
        enc['in_channels'] = self.num_classes
        dec['out_channels'] = self.num_classes
        for args in (enc, dec):
            args['use_fp16'] = False
            args['latent_channels'] = cfg['model']['latent_channels']
            args['block_args'] = [{'use_checkpoint': cfg['model']['gradient_checkpointing']} for _ in args['num_blocks']]
        if cfg['model']['stages'] == 4:
            enc['model_channels'] = enc['model_channels'][:-1]
            enc['num_blocks'] = enc['num_blocks'][:-1]
            dec['model_channels'] = dec['model_channels'][1:]
            dec['num_blocks'] = dec['num_blocks'][1:]
            for args, resample in ((enc, 'down_block_type'), (dec, 'up_block_type')):
                args['block_type'] = args['block_type'][:4]
                args['block_args'] = args['block_args'][:4]
                args[resample] = args[resample][:3]
        self.encoder = SparseUnetVaeEncoder(**enc)
        self.decoder = SparseUnetVaeDecoder(**dec)
        self.surface = SurfaceHead(dec['model_channels'][-1], cfg['model']['surface_hidden'], self.num_classes) if cfg['model']['mode'] == 'surface' else None
        self.cfg = cfg
        self.initialization = {}
        if initialize and cfg['model']['initialize_backbone']:
            for name in ('encoder', 'decoder'):
                module = getattr(self, name)
                state = torch.load(cfg[name+'_init'], map_location='cpu', weights_only=True)
                from .initialization import load_backbone
                reset = load_backbone(module,state,name,cfg['model']['latent_channels'],self.num_classes)
                self.initialization[name] = {'source': cfg[name+'_init'], 'new_parameters': reset}

    def encode(self, sample, sample_posterior):
        from trellis2.modules.sparse.basic import SparseTensor
        coords = sample['coords'].int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1)
        if sample['voxel_target'].shape != (len(coords),self.num_classes):
            raise ValueError('Cache class count does not match VAE')
        x = SparseTensor(feats=sample['voxel_target'].float(), coords=coords)
        # flex_gemm explicit backward requires matching feature/gradient dtypes.
        with torch.autocast(device_type=x.feats.device.type, enabled=False):
            latent, mean, logvar = self.encoder(x, sample_posterior=sample_posterior, return_raw=True)
        return latent, mean, logvar

    def decode_voxels(self, latent):
        """Accepts a latent plus geometry-only spatial cache; never consumes labels."""
        with torch.autocast(device_type=latent.feats.device.type, enabled=False):
            return self._decode_voxels(latent.replace(latent.feats.float()))

    def _decode_voxels(self, latent):
        h = self.decoder.from_latent(latent)
        h = h.type(self.decoder.dtype)
        for stage in self.decoder.blocks:
            for block in stage:
                h = block(h)
        h = h.type(latent.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        features = h.feats
        output = self.decoder.output_layer(h)
        return features, output.feats.float(), output.coords

    @torch.no_grad()
    def geometry_template(self, coords):
        """Build decoder spatial mappings at inference without any GT labels."""
        from trellis2.modules.sparse.basic import SparseTensor
        c = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).int()
        features = torch.zeros(len(c), self.num_classes, device=c.device)
        features[:, 0] = 1
        with torch.autocast(device_type=c.device.type, enabled=False):
            template = self.encoder(SparseTensor(feats=features, coords=c), sample_posterior=False)
        template._spatial_cache = {
            scale: {k:v for k,v in cache.items() if not k.startswith('SubMConv3d_neighbor_cache_')}
            for scale,cache in template._spatial_cache.items()}
        return template

    def decode_surface(self, features, logits, query):
        if self.surface is not None:
            return self.surface(features, logits, query)
        # Backbone-only control: fixed geometric interpolation, no learned surface head.
        weights = query['neighbor_prior'].masked_fill(~query['neighbor_mask'], -1e4).softmax(-1)
        point = (logits[query['neighbor_index']] * weights[..., None]).sum(1).reshape(-1, 4, self.num_classes)
        import math
        return point, torch.logsumexp(point.log_softmax(-1), dim=1)-math.log(4)

    def forward(self, sample, query=None, sample_posterior=True):
        latent, mean, logvar = self.encode(sample, sample_posterior)
        features, logits, coords = self.decode_voxels(latent)
        if not torch.equal(coords[:, 1:].int(), sample['coords'].int()):
            raise ValueError('Decoder voxel coordinate order differs from target')
        result = {'voxel_logits': logits, 'features': features, 'mean': mean, 'logvar': logvar}
        if query is not None:
            result['point_logits'], result['face_logprob'] = self.decode_surface(features, logits, query)
        return result
