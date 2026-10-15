"""冻结任务集合（方案④）：调试 10 / 开发 20 / baseline 100。

- 任务列表在运行前确定，不依据模型成败筛选；
- asin 级别互斥，避免同一商品（高度相似任务）跨集合泄漏；
- 记录数据指纹（文件 sha256、goal 数量、seed），保证可复现。
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_attack.config import load_config  # noqa: E402
from agent_attack.env_client import EnvClient  # noqa: E402


def file_sha256_fast(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_asin_disjoint(tasks, sizes, seed):
    """按 goal 的 asin 分组做互斥抽样；同 asin 的多个 goal 不会跨集合。"""
    by_asin = {}
    for t in tasks:
        by_asin.setdefault(t["asin"], []).append(t)
    asins = sorted(by_asin.keys())
    rng = random.Random(seed)
    rng.shuffle(asins)

    splits, cursor = {name: [] for name in sizes}, 0
    for name, size in sizes.items():
        while len(splits[name]) < size and cursor < len(asins):
            splits[name].extend(by_asin[asins[cursor]])
            cursor += 1
        # 同 asin 可能带多个 goal，超出部分裁剪（保持顺序即 asin 组连续）
        if len(splits[name]) > size:
            splits[name] = splits[name][:size]
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    env = EnvClient(cfg.get("environment.service_url"))
    info = env.health()
    print(f"service: {info}")
    data = env.list_tasks()
    tasks = data["tasks"]
    print(f"total goals: {len(tasks)}")

    sizes = {"debug": cfg.get("tasks.debug_size"),
             "dev": cfg.get("tasks.dev_size"),
             "baseline": cfg.get("tasks.baseline_size")}
    splits = sample_asin_disjoint(tasks, sizes, cfg.get("tasks.split_seed"))
    for name, lst in splits.items():
        print(f"{name}: {len(lst)} tasks")

    registry = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service_info": info,
        "n_total_goals": len(tasks),
        "split_seed": cfg.get("tasks.split_seed"),
        "asin_disjoint": cfg.get("tasks.asin_disjoint"),
        "data_fingerprint": {
            "items_shuffle_sha256": file_sha256_fast(
                cfg.get("environment.paths.items_shuffle")),
            "items_ins_v2_sha256": file_sha256_fast(
                cfg.get("environment.paths.items_ins_v2")),
            "items_human_ins_sha256": file_sha256_fast(
                cfg.get("environment.paths.items_human_ins")),
        },
        "splits": splits,
    }
    out = Path(args.out or cfg.get("tasks.registry"))
    out.write_text(json.dumps(registry, indent=1, ensure_ascii=False))
    print(f"registry -> {out}")

    # 互斥校验
    asin_sets = {k: {t["asin"] for t in v} for k, v in splits.items()}
    names = list(asin_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = asin_sets[names[i]] & asin_sets[names[j]]
            assert not inter, f"asin leak between {names[i]} and {names[j]}: {inter}"
    print("asin disjointness: OK")


if __name__ == "__main__":
    main()
