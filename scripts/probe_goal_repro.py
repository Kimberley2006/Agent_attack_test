"""探针：完全复刻旧 env_service 的构建顺序，检验 goal 列表可否复现冻结 registry。

关键顺序：random.seed(233) 必须发生在 import webshop 模块**之前**
（import 期 spacy/gym 等可能消耗随机流），随后只传官方默认等价格式参数。
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webshop_repo"))

random.seed(233)                      # 必须在 import 之前（服务顺序）
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402

FIELDS = ["instruction_text", "asin", "price_upper", "goal_options",
          "attributes", "query", "name"]

env = WebAgentTextEnv(observation_mode="text", human_goals=1, num_products=None)
goals = env.server.goals
print(f"n_goals={len(goals)}")

reg = json.load(open(ROOT / "task_registry.json"))
print(f"registry n_total_goals={reg['n_total_goals']}")

prices = env.server.product_prices
pv = list(prices.values())
print(f"product_prices: n={len(pv)}, sample={[round(x,2) for x in pv[:8]]}, "
      f"const50={sum(1 for x in pv if abs(x-50.0)<1e-9)}")

hist = Counter(g["price_upper"] for g in goals)
print("price_upper histogram (top10):", hist.most_common(10))

n_ok = n_bad = 0
mism = []
for split in ("debug", "dev", "baseline"):
    for t in reg["splits"][split]:
        g = goals[t["goal_idx"]]
        diffs = [f for f in FIELDS if t.get(f) != g.get(f)]
        if diffs:
            n_bad += 1
            mism.append({"split": split, "goal_idx": t["goal_idx"],
                         "diff_fields": diffs,
                         "registry_price": t.get("price_upper"),
                         "env_price": g.get("price_upper")})
        else:
            n_ok += 1
print(f"校验: {n_ok} 一致, {n_bad} 不一致")
json.dump(mism, open("/tmp/goal_probe_mismatches.json", "w"),
          ensure_ascii=False, indent=1)
print("mismatch 明细 -> /tmp/goal_probe_mismatches.json")
