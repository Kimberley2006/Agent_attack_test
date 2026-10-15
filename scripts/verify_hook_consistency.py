"""⑤ hooks 一致性验证：相同输入 + 相同配置，对比 开/关采集 与 sdpa/eager
的生成结果是否完全一致（token ids 层面）。每个配置在独立子进程中运行，
避免同进程重复加载大模型的显存残留。

用法：CUDA_VISIBLE_DEVICES=1 python scripts/verify_hook_consistency.py [--max-new 128]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_attack import prompts  # noqa: E402

PROBE_MESSAGES = [
    [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content":
         prompts.build_task_message(
             "I need a long-lasting 6 oz bottle of almond-scented lotion "
             "with a pump, and price lower than 50.00 dollars")
         + "\n\n" + prompts.format_observation(
             "[SEP] Instruction [SEP] i need a long lasting 6 oz bottle ... "
             "[SEP] Page [SEP] Back to Search [SEP] Page 1 of 10 "
             "[SEP] Buy Now [SEP] Almond Body Lotion, 6 Ounce ($9.99) "
             "[SEP] A rich almond lotion with pump dispenser. [SEP] "
             "Description Features Reviews [SEP] Next >", first=True)},
    ],
    [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content":
         prompts.build_task_message("cheap blue toothbrush, under $30")
         + "\n\n" + prompts.format_observation(
             "[SEP] WebShop [SEP] Instruction [SEP] cheap blue toothbrush "
             "[SEP] Search", first=True)},
    ],
]

CHILD_CODE = r'''
import json, sys
sys.path.insert(0, ".")
from agent_attack.config import load_config
from agent_attack.model_backend import ModelBackend

collect, max_new, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = load_config("configs/baseline_v1.yaml")
cfg._data["generation"]["max_new_tokens"] = max_new
backend = ModelBackend(cfg, collect=collect)
import agent_attack.hooks_collector as hc
msgs = json.load(open("/tmp/probe_messages.json"))
outs = []
for i, m in enumerate(msgs):
    res, cap = backend.generate(m, f"probe_{i}")
    backend.finalize(res, cap, None)
    outs.append(res.output_ids)
    print(f"  probe {i}: {len(res.output_ids)} tokens finish={res.finish_reason} "
          f"tail={res.output_text[-80:]!r}", flush=True)
    if res.reps_payload is not None:
        p = res.reps_payload
        print(f"    reps: anchors={p['anchors_gen_pos']} "
              f"prefill_last layers={len(p['prefill_last']['layer_in'])} "
              f"anchor_states={len(p['anchor_states'])} "
              f"detailed_avail={p['detailed_available']}", flush=True)
json.dump(outs, open(out_path, "w"))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v1.yaml")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    json.dump(PROBE_MESSAGES, open("/tmp/probe_messages.json", "w"))

    configs = [("off", None), ("basic", None), ("detailed", "eager")]
    outs = {}
    for collect, attn in configs:
        out_path = f"/tmp/probe_out_{collect}.json"
        print(f"\n=== config: collect={collect} attn={attn or 'default'} ===",
              flush=True)
        cmd = [sys.executable, "-c", CHILD_CODE, collect, str(args.max_new),
               out_path]
        env = dict(os.environ)
        if attn:
            env["FORCE_ATTN"] = attn
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"FAILED for collect={collect}")
            sys.exit(2)
        outs[collect] = json.load(open(out_path))

    ref = outs["off"]
    report = {}
    for label in ("basic", "detailed"):
        same = [outs[label][i] == ref[i] for i in range(len(ref))]
        report[label] = {
            "identical_to_off": all(same), "per_probe": same,
            "note": "detailed 使用 eager 内核；不一致属预期，须记录且正式"
                    " clean/attack 不得混用内核",
        }
        print(f"{label}: identical_to_off={all(same)} per_probe={same}")

    from agent_attack.config import load_config
    cfg = load_config(args.config)
    out = Path(cfg.get("run.output_root")) / "hook_consistency.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")
    sys.exit(0 if report["basic"]["identical_to_off"] else 1)


if __name__ == "__main__":
    main()
