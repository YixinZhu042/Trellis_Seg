import unittest
import numpy as np

from datasets.seg_dataset import external_seg_ids_to_model_ids
from preprocess.step2_encode_mv_seg_latent import voxelize_foreground
from preprocess.step4_encode_seg_latent import labels_to_onehot


class SegIdConventionTests(unittest.TestCase):
    def test_external_ids_are_not_shifted(self):
        values = np.array([[0, 1, 2, 19]], np.int32)
        np.testing.assert_array_equal(external_seg_ids_to_model_ids(values), values)

    def test_training_render_offset_is_removed_once(self):
        # A rendered training map stores face/model IDs [0,1,19] as pixels
        # [1,2,20]. Background pixels remain 0 and are excluded.
        rendered = np.array([[0, 1, 2, 20], [0, 1, 2, 20]], np.int32)
        _, features = voxelize_foreground(rendered, 0, 20)
        observed = set(features.argmax(1).tolist())
        self.assertEqual(observed, {0, 1, 19})

    def test_invalid_external_ids_fail(self):
        with self.assertRaises(ValueError):
            external_seg_ids_to_model_ids(np.array([-1]))
        with self.assertRaises(ValueError):
            external_seg_ids_to_model_ids(np.array([20]))

    def test_training_encoders_reject_out_of_range_ids(self):
        with self.assertRaises(ValueError):
            voxelize_foreground(np.array([[21]], np.int32), 0, 20)
        with self.assertRaises(ValueError):
            labels_to_onehot(np.array([20], np.int32), 20)


if __name__ == '__main__':
    unittest.main()
