"""Training-only label renaming; geometry, source annotations and cache stay intact."""
import numpy as np

POLICY = 'all_nonzero_v1'


def label_permutation(num_classes, seed, sample_index):
    """old -> new IDs, deterministic per global training sample, including resume.

    Draw from every nonzero channel, not just the labels present in the object.
    A separate stateless RNG leaves query sampling / posterior RNG unchanged.
    """
    if num_classes < 2 or seed < 0 or sample_index < 0:
        raise ValueError('Invalid permutation size or seed/index')
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index, 0x56414532]))
    return np.r_[0, rng.permutation(np.arange(1, num_classes))]


def permute_sample(sample, old_to_new):
    """Return an in-memory view with consistently renamed inputs and supervision.

    Probability columns use the inverse permutation; integer labels use forward.
    Hard metric targets retain their original tie choice under renaming. Original
    annotation_audit is provenance only (never validated/exported as augmented GT).
    """
    mapping = np.asarray(old_to_new)
    classes = sample['voxel_target'].shape[-1]
    if (mapping.shape != (classes,) or not np.issubdtype(mapping.dtype, np.integer)
            or mapping[0] != 0 or not np.array_equal(np.sort(mapping), np.arange(classes))):
        raise ValueError('Permutation must fix zero and bijectively map all channels')
    result = dict(sample)
    result['voxel_target'] = sample['voxel_target'][:, np.argsort(mapping)].copy()
    for key in ('face_labels', 'voxel_labels'):
        labels = sample[key]
        if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= classes:
            raise ValueError(f'Invalid {key}')
        result[key] = mapping[labels].astype(labels.dtype, copy=False)
    return result
