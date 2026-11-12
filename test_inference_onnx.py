"""Checks that inference_onnx.py prepares a chip the way the training loader does.
Run: python3 test_inference_onnx.py   (needs numpy only)"""
import ast
import pathlib
import unittest

import numpy as np

import inference_onnx as io

ROOT = pathlib.Path(__file__).parent


def training_hillshade():
    """MBESDataset.compute_hillshade taken from util/data.py without importing torch/cv2."""
    tree = ast.parse((ROOT / "util" / "data.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_hillshade":
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "util/data.py", "exec"), ns)
            return lambda elev: ns["compute_hillshade"](None, elev)
    raise AssertionError("compute_hillshade not found in util/data.py")


class Preprocessing(unittest.TestCase):
    def test_hillshade_matches_the_training_function(self):
        rng = np.random.default_rng(0)
        y, x = np.mgrid[0:64, 0:80].astype(np.float32)
        elev = -100 + 0.4 * x - 0.9 * y + rng.normal(0, 0.5, x.shape).astype(np.float32)
        want = training_hillshade()(elev).astype(np.float32) / 255.0
        np.testing.assert_array_equal(io.compute_hillshade(elev), want)

    def test_a_small_chip_is_resampled_whole_not_padded(self):
        chip = np.full((200, 200), -50.0, np.float32)
        chip[150:, 150:] = -40.0  # feature in the far corner
        out = io.prepare_chip(chip)
        self.assertEqual(out.shape, (1, 2, 416, 416))
        depth = out[0, 0]
        self.assertEqual(depth[415, 415], 1.0)       # the corner feature fills the output corner
        self.assertEqual(depth[0, 0], 0.0)
        self.assertAlmostEqual(float((depth == 1.0).mean()), 1 / 16, places=2)

    def test_a_large_chip_keeps_what_is_outside_the_top_left(self):
        chip = np.full((800, 800), -50.0, np.float32)
        chip[700:, 700:] = -40.0
        depth = io.prepare_chip(chip)[0, 0]
        self.assertEqual(depth[415, 415], 1.0)

    def test_zeros_stay_zero_and_valid_depths_span_0_to_1(self):
        chip = np.zeros((416, 416), np.float32)
        chip[:, :100] = -101.0
        chip[:, 100:200] = -99.0
        depth = io.prepare_chip(chip)[0, 0]
        self.assertEqual(depth[0, 0], 0.0)      # deepest valid sample
        self.assertEqual(depth[0, 150], 1.0)    # shallowest valid sample
        self.assertTrue((depth[:, 200:] == 0.0).all())  # no data stays 0

    def test_nan_is_treated_as_no_data(self):
        chip = np.full((416, 416), np.nan, np.float32)
        chip[:, :208] = np.linspace(-120, -80, 208, dtype=np.float32)
        out = io.prepare_chip(chip)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out[0, 0][:, 208:] == 0.0).all())

    def test_a_chip_with_no_usable_depth_is_refused(self):
        for chip in (np.full((64, 64), np.nan, np.float32), np.zeros((64, 64), np.float32)):
            with self.assertRaises(ValueError):
                io.prepare_chip(chip)


if __name__ == "__main__":
    unittest.main()
