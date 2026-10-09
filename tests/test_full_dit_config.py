import json
from pathlib import Path
import unittest

from scripts.train_dit_decoded import resolve_config
from utils.decoded_dit_runtime import training_sample_index


ROOT = Path(__file__).resolve().parents[1]


class FullDitConfigTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'data/dit_decoded_full_v1/selection.json').is_file(),
                         'requires external audited full-dataset manifest; not shipped in Git')
    def test_full_split_and_training_config(self):
        selection = json.loads((ROOT / 'data/dit_decoded_full_v1/selection.json').read_text())
        self.assertEqual(selection['format_version'], 2)
        self.assertEqual(selection['split_counts'], {'train': 21885, 'validation': 512, 'test': 512})
        self.assertEqual(len(selection['excluded_over_max_tokens']), 0)
        self.assertEqual(len(selection['adaptive_resample_candidates']), 29)
        self.assertEqual(len({x['model_id'] for x in selection['samples']}), 22909)
        config = resolve_config(ROOT / 'configs/dit_decoded_aux_full_v1.json')
        self.assertEqual(config['lr'], 2e-6)
        self.assertEqual(config['loss_weights'], {'segmentation': .1, 'boundary': .02, 'reprojection': .1})
        self.assertEqual(config['ema'], {'enabled': True, 'rate': .9999})
        self.assertEqual(config['teacher_replay'], {'probability': .1, 'weight': .1})

    def test_each_full_epoch_is_bijective(self):
        n, world = 21885, 16
        for epoch in [0, 1]:
            values = []
            for offset in range(n):
                global_index = epoch * n + offset
                values.append(training_sample_index(
                    global_index // world + 1, 0, global_index % world,
                    world, 1, n, seed=20260922, epoch_size=n))
            self.assertEqual(len(set(values)), n)


if __name__ == '__main__':
    unittest.main()
