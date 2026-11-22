import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/ik2_compare_decode.py"


class DecodeComparison(unittest.TestCase):
    def test_whole_prompt_includes_its_first_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            raw, ref = root / "prompt.f32", root / "ref.npy"
            np.log([[.1, .9], [.7, .3]]).astype("<f4").tofile(raw)
            np.save(ref, np.log([[.9, .1], [.7, .3]]))
            result = subprocess.check_output([sys.executable, str(SCRIPT), str(raw), "--ref", str(ref),
                                              "--prefill", "--prompt-length", "2", "--vocab", "2"], text=True)
            self.assertIn("top-1 1/2", result)
            self.assertIn("input position 0 ", result)

    def test_forced_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            raw, ref = root / "engine.f32", root / "ref.npy"
            # Equal logits for the first forced input, opposite odds for the second.
            engine = np.log([[.8, .2], [.75, .25]]).astype("<f4")
            engine.tofile(raw)
            np.save(ref, np.log([[.1, .9], [.8, .2], [.25, .75]]))
            cmd = [sys.executable, str(SCRIPT), str(raw), "--gen", "3", "--prompt-length", "1", "--vocab", "2"]
            result = subprocess.check_output([*cmd, "--ref", str(ref)], text=True)
            self.assertIn("top-1 1/2", result)
            self.assertIn("at input position 2", result)
            mean = float(result.split("mean ")[1].split(",")[0])
            self.assertAlmostEqual(mean, np.log(3) / 4)
            equal = subprocess.check_output([*cmd, "--baseline", str(raw)], text=True)
            self.assertIn("top-1 2/2", equal)
            self.assertIn("mean 0,", equal)
            for bad in (engine[:1], np.full_like(engine, np.nan)):
                bad.tofile(raw)
                result = subprocess.run([*cmd, "--ref", str(ref)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
