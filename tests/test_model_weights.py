"""本地模型预检应拒绝下载中断后留下的分片。"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.check_model_weights import check_model_weights


class ModelWeightsTest(unittest.TestCase):
    def test_truncated_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            shard = "model-00001-of-00001.safetensors"
            (model_dir / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 4},
                "weight_map": {"weight": shard},
            }))
            header = json.dumps({
                "__metadata__": {"format": "pt"},
                "weight": {"dtype": "F32", "shape": [1],
                           "data_offsets": [0, 4]},
            }).encode()
            path = model_dir / shard
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            problems, actual, expected = check_model_weights(model_dir)
            self.assertEqual(problems, [])
            self.assertEqual(actual, expected)

            path.write_bytes(path.read_bytes()[:-1])
            problems, actual, expected = check_model_weights(model_dir)
            self.assertEqual(len(problems), 1)
            self.assertIn(shard, problems[0])
            self.assertEqual(expected - actual, 1)


if __name__ == "__main__":
    unittest.main()
