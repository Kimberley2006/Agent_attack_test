"""多 GPU 权重预算应根据当前空闲显存预留生成和采集空间。"""

import unittest

from agent.model_server import gpu_shard_budgets


GIB = 2**30


class _Cuda:
    def __init__(self, free_gib):
        self.free_gib = free_gib

    def device_count(self):
        return len(self.free_gib)

    def mem_get_info(self, gpu):
        return int(self.free_gib[gpu] * GIB), 32 * GIB


class _Torch:
    def __init__(self, free_gib):
        self.cuda = _Cuda(free_gib)


class ModelShardingTest(unittest.TestCase):
    def test_budget_excludes_busy_gpu_and_reserves_memory(self):
        budgets = gpu_shard_budgets(_Torch([31, 5, 22]), reserve_gb=6)
        self.assertEqual(budgets, {0: 25 * GIB, 2: 16 * GIB})

    def test_no_gpu_with_headroom_fails_early(self):
        with self.assertRaisesRegex(RuntimeError, "没有 GPU"):
            gpu_shard_budgets(_Torch([2, 6]), reserve_gb=6)


if __name__ == "__main__":
    unittest.main()
