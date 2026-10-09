"""CPU contract tests; no checkpoint loading, CUDA initialization or network."""
import copy
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import demo_paired_inference as paired
from vae_v2.config import digest
from vae_v2.geometry import build_geometry


def stats(mean=0., std=1.):
    return {key: {"mean": [mean] * 32, "std": [std] * 32}
            for key in ("latent_feats", "shape_feats", "mv_feats")}


class Template:
    def __init__(self, coords, feats=None):
        self.coords = torch.as_tensor(coords, dtype=torch.int32)
        self.feats = torch.full((len(coords), 32), float("nan")) if feats is None else feats

    def replace(self, feats):
        return Template(self.coords, feats)


class ProjectionTests(unittest.TestCase):
    def test_offset_unknown_background_and_exact_ids(self):
        maps = np.array([[0, 1, 2, 3, 3, 20]], np.int32)
        raster = np.array([[-1, 0, 1, 2, 1, 3]], np.int32)
        result = paired.projection_agreement(np.array([19, 1, 2, 19]), raster, maps)
        self.assertEqual(result["pixels"], 4)
        self.assertEqual(result["pixel_accuracy"], .75)
        self.assertEqual(result["condition_ids"], [1, 2, 19])
        self.assertAlmostEqual(result["miou"], 2 / 3)
        swapped = paired.projection_agreement(np.array([0, 2, 1, 1]), raster, maps)
        self.assertEqual(swapped["pixel_accuracy"], .25)

    def test_unknown_prediction_is_an_error_and_missing_faces_are_counted(self):
        result = paired.projection_agreement(np.array([0, 63]), np.array([[0, 1, -1]]),
                                              np.array([[2, 3, 2]]))
        self.assertEqual(result["pixels"], 2)
        self.assertEqual(result["pixel_accuracy"], 0)
        self.assertEqual(result["miou"], 0)
        self.assertEqual(result["condition_pixels_without_face"], 1)

    def test_empty_and_invalid_projection(self):
        result = paired.projection_agreement(np.array([1]), np.array([[-1, 0]]), np.array([[0, 1]]))
        self.assertIsNone(result["miou"])
        self.assertIsNone(result["pixel_accuracy"])
        for raster, maps in ((np.array([[1]]), np.array([[2]])),
                             (np.array([[0.]]), np.array([[2]])),
                             (np.array([[0]]), np.array([[2, 2]]))):
            with self.assertRaises(ValueError):
                paired.projection_agreement(np.array([1]), raster, maps)


class SamplingTests(unittest.TestCase):
    def test_native_exact_alignment_is_zero_after_each_normalization(self):
        target = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.int32)
        source = target[[2, 0]]
        values = np.stack([np.full(32, 9), np.full(32, 5)]).astype(np.float32)
        aligned, valid = paired.align_shape(target, source, values)
        np.testing.assert_array_equal(valid, [True, False, True])
        for normalization in (stats(3, 2), stats(1, 4)):
            normalized = paired.normalize_conditions(aligned, valid, values, normalization, "cpu")
            torch.testing.assert_close(normalized["shape_feats"][1], torch.zeros(32))
            self.assertEqual(normalized["shape_feats"][0, 0], (5-normalization["shape_feats"]["mean"][0]) /
                             normalization["shape_feats"]["std"][0])

    def test_noise_bytes_and_uniform_euler_ignore_template_values(self):
        old, new, hashes = paired.paired_noise(3, 42, "cpu")
        self.assertEqual(hashes["old"], paired.array_hash(old.numpy()))
        self.assertEqual(hashes["old"], hashes["new"])
        expected = torch.randn(3, 32, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(old, expected)
        template = Template([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]])
        calls = []

        class Model:
            cond_channels = 1024

            def __call__(self, x, t, **conditions):
                calls.append(float(t.item()))
                self_outer.assertTrue(torch.isfinite(x.feats).all())
                self_outer.assertEqual(conditions["concat_cond"].feats[1].count_nonzero(), 0)
                return x.replace(torch.ones_like(x.feats))

        self_outer = self
        mv = {"mv_feats": np.zeros((2, 32), np.float32), "mv_coords": np.zeros((2, 3), np.int32)}
        raw = paired.generate_latent(Model(), template, old, np.zeros((3, 32), np.float32),
                                      np.array([True, False, True]), mv, stats(2, 3), 20, "cpu")
        self.assertEqual(calls, list(range(1000, 0, -50)))
        torch.testing.assert_close(raw, (expected-1)*3+2)
        torch.testing.assert_close(old, new)
        torch.testing.assert_close(old, expected)

    def test_template_order_budget_and_downsampling_checked(self):
        coords = np.array([[0, 0, 0], [16, 16, 16]], np.int32)
        old = Template([[0, 0, 0, 0], [0, 1, 1, 1]])
        paired.check_templates(old, old, coords, 2)
        with self.assertRaisesRegex(ValueError, "order"):
            paired.check_templates(old, Template(old.coords.flip(0)), coords, 2)
        with self.assertRaisesRegex(ValueError, "budget"):
            paired.check_templates(old, old, coords, 1)
        with self.assertRaisesRegex(ValueError, "downsampling"):
            paired.check_templates(old, old, coords + 16, 2)

    def test_both_encoders_receive_unmodified_views_without_deduplication(self):
        maps = np.zeros((6, 2, 2), np.int32)
        maps[:, 0, 0] = 1
        maps[:, 1, 1] = 20
        original = maps.copy()
        calls = []
        stub = ModuleType("preprocess.step2_encode_mv_seg_latent")

        def encode(view, index, encoder, num_classes, device):
            calls.append((index, num_classes, view.copy()))
            return torch.zeros((1, 3), dtype=torch.int32), torch.full((1, 32), float(index))

        stub.encode_one_view = encode
        with patch.dict("sys.modules", {stub.__name__: stub}):
            old = paired.encode_views(maps, object(), 20, "cpu")
            new = paired.encode_views(maps, object(), 64, "cpu")
        np.testing.assert_array_equal(maps, original)
        np.testing.assert_array_equal(old["mv_coords"], new["mv_coords"])
        np.testing.assert_array_equal(old["view_offsets"], np.arange(7))
        self.assertEqual(len(old["mv_coords"]), 6)
        for i in range(6):
            np.testing.assert_array_equal(calls[i][2], calls[i+6][2])
            self.assertEqual((calls[i][1], calls[i+6][1]), (20, 64))

    def test_legacy_uses_its_own_three_statistics(self):
        config = {"dataset": {"args": {"normalization": stats(1)["latent_feats"],
                                      "shape_cond_normalization": stats(2)["shape_feats"],
                                      "mv_cond_normalization": stats(3)["mv_feats"]}}}
        result = paired.legacy_normalization(config)
        self.assertEqual([result[k]["mean"][0] for k in ("latent_feats", "shape_feats", "mv_feats")], [1, 2, 3])
        with self.assertRaises(ValueError):
            paired._stats(stats(0, 0))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vertices = np.array([[3, 2, 1], [5, 2, 1], [3, 4, 1], [5, 4, 1]], np.float32)
        faces = np.array([[1, 3, 2], [0, 1, 2]], np.int32)
        self.geometry = build_geometry(vertices, faces, {"area_samples": 8, "max_voxels": 100}, 42)
        self.geometry["face_raster"] = np.full((6, 512, 512), -1, np.int32)
        self.geometry["face_raster"][:, 0, :2] = [0, 1]
        self.maps = np.zeros((6, 512, 512), np.int32)
        self.maps[:, 0, :2] = [2, 3]
        self.row = {"id": "example", "title": "Example", "kind": "character", "source_name": "example.glb",
                    **{key: str(self.root / filename) for key, filename in (
                        ("geometry_path", "geometry.npz"), ("condition_path", "condition.npz"),
                        ("shape_path", "shape.npz"))}}
        np.savez(self.row["geometry_path"], **self.geometry)
        np.savez(self.row["condition_path"], render_maps=self.maps)
        self.latent_coords = np.unique(self.geometry["coords"] // 16, axis=0)
        np.savez(self.row["shape_path"], coords=self.latent_coords,
                 feats=np.ones((len(self.latent_coords), 32), np.float32))

    def test_input_cache_preserves_geometry_and_rejects_gt_or_new_ids(self):
        geometry, maps, _, _, sources = paired.load_inputs(self.row)
        np.testing.assert_array_equal(geometry["mesh_faces"], self.geometry["mesh_faces"])
        self.assertFalse(maps.flags.writeable)
        self.assertEqual(sources["geometry_path"]["sha256"], paired.file_sha256(self.row["geometry_path"]))
        np.savez(self.row["geometry_path"], **self.geometry, face_labels=np.array([1, 2]))
        with self.assertRaisesRegex(ValueError, "supervision"):
            paired.load_inputs(self.row)
        np.savez(self.row["geometry_path"], **self.geometry)
        self.maps[0, 0, 0] = 21
        np.savez(self.row["condition_path"], render_maps=self.maps)
        with self.assertRaisesRegex(ValueError, "render_maps"):
            paired.load_inputs(self.row)

    def test_legacy_bake_interpolates_logits_in_source_normalization(self):
        # Exact nearest voxel votes for 1; the seven other logits dominate class 2.
        center = self.geometry["mesh_vertices"][self.geometry["mesh_faces"][:1]].mean(1)[0]
        point = (center-self.geometry["center"]) / float(self.geometry["extent"])*509+256
        coords = np.array([point + [distance, 0, 0] for distance in range(1, 10)])
        logits = np.zeros((9, 20), np.float32)
        logits[0, 1] = 1
        logits[1:8, 2] = 10
        logits[8, 19] = 1e8  # ninth neighbor must be excluded
        geometry = dict(self.geometry, mesh_faces=self.geometry["mesh_faces"][:1])
        labels = paired.bake_legacy_faces(geometry, coords, logits, chunk_faces=1)
        np.testing.assert_array_equal(labels, [2])

    def test_surface_chunks_keep_all_500000_faces_without_targets(self):
        total = 500000
        geometry = {"coords": np.array([[0, 0, 0]], np.int32),
                    "mesh_faces": np.zeros((total, 3), np.int32)}
        calls = []

        class Queries:
            def __init__(self, sample, settings):
                pass

            def build(self, ids):
                calls.append(len(ids))
                return {"face_ids": ids, "query_face_ids": np.repeat(ids, 4),
                        "query_fallback": np.zeros(4*len(ids))}

        class Vae:
            cfg = {"geometry": {}}

            def decode_voxels(self, latent):
                return torch.zeros(1, 8), torch.zeros(1, 64), torch.zeros(1, 4, dtype=torch.int32)

            def decode_surface(self, features, logits, query):
                ids = query["face_ids"]
                return None, torch.nn.functional.one_hot(ids % 64, 64).float()

        with patch("vae_v2.geometry.SurfaceQueries", Queries):
            labels, fallback = paired.decode_new_faces(Vae(), Template([[0, 0, 0, 0]]),
                                                       torch.zeros(1, 32), geometry, "cpu")
        np.testing.assert_array_equal(labels, np.arange(total) % 64)
        self.assertEqual(max(calls), 1024)
        self.assertEqual(sum(calls), total)
        self.assertEqual(fallback, 0.)

    def test_export_roundtrip_shared_palette_transform_and_input_preview(self):
        import trimesh
        from PIL import Image
        from vae_v2.visuals import class_colors, render_palette

        out = self.root / "objects"
        transform = paired.export_pair(out, self.geometry, self.maps, np.array([1, 2]), np.array([63, 0]))
        expected = (self.geometry["mesh_vertices"]-self.geometry["center"]) / float(self.geometry["extent"])
        matrix = np.asarray(transform["source_to_web"])
        np.testing.assert_allclose(expected, self.geometry["mesh_vertices"] @ matrix[:3, :3].T + matrix[:3, 3])
        for name, labels in (("old", [1, 2]), ("new", [63, 0])):
            mesh = trimesh.load(out / f"{name}.glb", force="scene", process=False).to_geometry()
            np.testing.assert_allclose(mesh.triangles, expected[self.geometry["mesh_faces"]])
            np.testing.assert_array_equal(mesh.visual.vertex_colors[mesh.faces, :3],
                                          np.repeat(class_colors(64)[labels, None], 3, axis=1))
        with Image.open(out / "inputcondition.png") as image:
            self.assertEqual(image.size, (1536, 1024))
            np.testing.assert_array_equal(np.asarray(image)[:512, :512], render_palette(64)[self.maps[0]])
        self.assertTrue((out / "comparison.png").is_file())

    def test_runner_writes_hashes_predictions_and_condition_only_metrics(self):
        # Exercise orchestration with fake networks; real geometry I/O and outputs.
        runner = paired.Runner.__new__(paired.Runner)
        runner.plan = {"samples": [self.row], "output_dir": str(self.root / "output"),
                       "prediction_dir": str(self.root / "predictions")}
        runner.plan_digest = digest(runner.plan)
        runner.device = torch.device("cpu")
        runner.seed, runner.steps, runner.max_tokens = 42, 20, 16384
        runner.old_stats = runner.new_stats = stats()
        runner.hashes, runner.runtime = {}, {}
        coords = np.column_stack([np.zeros(len(self.latent_coords), np.int32), self.latent_coords])
        template = Template(coords)
        runner._old_template = lambda coords: template
        runner.new_vae = SimpleNamespace(geometry_template=lambda coords: template, encoder=object())
        runner.old_encoder = object()
        runner.old_dit, runner.new_dit = object(), object()
        runner.old_decoder = lambda latent: SimpleNamespace(
            coords=torch.tensor(np.column_stack([np.zeros(len(self.geometry["coords"])), self.geometry["coords"]])),
            feats=torch.zeros(len(self.geometry["coords"]), 20))
        mv = {"mv_coords": np.zeros((6, 3), np.int32), "mv_feats": np.zeros((6, 32), np.float32),
              "view_offsets": np.arange(7)}
        noise_seen = []

        def generate(model, template, noise, *args):
            noise_seen.append(noise.clone())
            return noise

        with patch.object(paired, "encode_views", return_value=mv), \
             patch.object(paired, "generate_latent", side_effect=generate), \
             patch.object(paired, "bake_legacy_faces", return_value=np.array([1, 2], np.int32)), \
             patch.object(paired, "decode_new_faces", return_value=(np.array([2, 1], np.int32), 0.)):
            result = runner.run(self.row)
        torch.testing.assert_close(*noise_seen)
        self.assertEqual(result["old"]["condition_agreement"]["accuracy"], 1.)
        self.assertEqual(result["new"]["condition_agreement"]["accuracy"], 0.)
        self.assertFalse(result["postprocessing"])
        self.assertTrue(result["initial_noise"]["identical"])
        self.assertEqual(result["shape_valid_fraction"], 1.)
        self.assertEqual(json.loads((self.root / "output/metrics/example.json").read_text())["plan_digest"], digest(runner.plan))
        with np.load(self.root / "predictions/example.npz", allow_pickle=False) as prediction:
            np.testing.assert_array_equal(prediction["old_faces"], [1, 2])
            np.testing.assert_array_equal(prediction["new_faces"], [2, 1])
            self.assertEqual(str(prediction["source_geometry_hash"]), paired.file_sha256(self.row["geometry_path"]))
            self.assertFalse(any("gt" in key for key in prediction.files))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
