"""Explicitly reset semantic interface layers; reuse only compatible backbone weights."""
def load_backbone(module, state, name, latent_channels, num_classes):
    current = module.state_dict()
    interfaces = {'input_layer.weight','input_layer.bias'} if name=='encoder' else {'output_layer.weight','output_layer.bias'}
    latent = {'to_latent.weight','to_latent.bias'} if name=='encoder' else {'from_latent.weight'}
    reset = set(interfaces) if num_classes != 20 else set()
    if latent_channels != 32:
        reset |= latent
    mismatched = {k for k in state if k in current and state[k].shape != current[k].shape}
    if mismatched - reset:
        raise ValueError(f'Unexpected {name} initialization shapes: {sorted(mismatched-reset)}')
    copied = {k:v for k,v in state.items() if k not in reset}
    result = module.load_state_dict(copied,strict=False)
    expected_missing = reset & current.keys()
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(f'Incomplete {name} initialization: {result}')
    return sorted(expected_missing)
