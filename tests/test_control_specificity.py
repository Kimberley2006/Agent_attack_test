"""对照臂的阈值必须在留出位置之外拟合。"""

import unittest

import torch

from analysis.signals import control_specificity_lopo


class ControlSpecificityTest(unittest.TestCase):
    def test_held_position_does_not_set_its_control_threshold(self):
        feat = {
            "p0_clean": torch.tensor([0.0]),
            "p0_attack": torch.tensor([10.0]),
            "p1_clean": torch.tensor([0.0]),
            "p1_attack": torch.tensor([10.0]),
            "p2_clean": torch.tensor([0.0]),
            "p2_attack": torch.tensor([10.0]),
        }
        labels = {key: int(key.endswith("attack")) for key in feat}
        groups = {key: key.split("_")[0] for key in feat}
        controls = [
            {"run_id": "p0_n1", "position_id": "p0", "payload_id": "N1"},
            {"run_id": "p0_n2", "position_id": "p0", "payload_id": "N2"},
        ]
        control_feat = {"p0_n1": torch.tensor([0.0]),
                        "p0_n2": torch.tensor([10.0])}
        first = control_specificity_lopo(feat, labels, groups,
                                         controls, control_feat)
        self.assertEqual(first["control_clean"],
                         {"correct": 1, "total": 2, "rate": 0.5})
        self.assertEqual(first["control_by_payload"]["N1"]["correct"], 1)
        self.assertEqual(first["control_by_payload"]["N2"]["correct"], 0)

        # 改动整个留出位置的 clean/attack，不应改变该位置 control 的判定。
        feat["p0_clean"] = torch.tensor([1000000.0])
        feat["p0_attack"] = torch.tensor([1000010.0])
        second = control_specificity_lopo(feat, labels, groups,
                                          controls, control_feat)
        self.assertEqual(first["control_predictions"],
                         second["control_predictions"])


if __name__ == "__main__":
    unittest.main()
