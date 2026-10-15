# -*- coding: utf-8 -*-
"""信号定位与模型内泛化的离线补强分析（Stage 3 + Stage 4，不占 GPU）。

Stage 3 —— 信号在哪里（读取位置维度）：
  同一批 episode 的首调用 capture 中，比较 P（prefill 末）、R（</think> 后
  首个可见 token）、A*（动作 token 区均值）三种读取位置 × 各层的
  信号一 AUC（clean vs attack，LOPO）。回答"信号是否依赖 P 位"。

Stage 4 —— 未见攻击家族泛化（模型内）：
  LOFO：用 3 个家族的 attack + 非测试位置的 clean 训练，在留出家族的
  attack + 测试位置的 clean 上测试。注意：当前 5 位置 × 4 家族网格中，
  其他家族的 attack 仍与测试家族共享位置，任务级泄漏无法完全排除
  ——结果标记为初步，严格 task×family 留出需 E1 扩样网格。

用法（webshop 环境，CPU）:
  python -m analysis.signals_v2 --branch-dirs data_out/branch_v11/clean \
      data_out/branch_v11/attack --captures /cache/wuhan/captures/g0 \
      --out data_out/branch_v11/signals_locate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.signals import probe_auc_lopo, rank_auc, scan_episodes  # noqa: E402

STREAMS = ("layer_out", "attn_out")


def load_first_call(capture_dir: Path, fname: str) -> dict | None:
    """返回 {pos_kind: {stream: tensor[n_layers, H]}}，
    pos_kind ∈ {P, R, A}；A 为动作区锚点均值。"""
    p = torch.load(capture_dir / fname, map_location="cpu", weights_only=False)
    n_layers = p["n_layers"]
    out = {}

    def stack_prefill(stream):
        d = p["prefill_last"].get(stream) or {}
        if len(d) != n_layers:
            return None
        return torch.stack([d[i].float().reshape(-1) for i in range(n_layers)])

    def stack_anchors(stream, sel):
        mats = [a[stream].float() for a in p["anchor_states"] if a["gen_pos"] in sel]
        if not mats:
            return None
        return torch.stack(mats).mean(0)

    info = p.get("anchor_info") or {}
    r_tok, span = info.get("r_tok"), info.get("action_span_tok")
    r_sel = {r_tok} if r_tok is not None else set()
    a_sel = set(range(span[0], span[1] + 1)) if span else set()

    for stream in STREAMS:
        pf = stack_prefill(stream)
        if pf is not None:
            out.setdefault("P", {})[stream] = pf
        if r_sel:
            rf = stack_anchors(stream, r_sel)
            if rf is not None:
                out.setdefault("R", {})[stream] = rf
        if a_sel:
            af = stack_anchors(stream, a_sel)
            if af is not None:
                out.setdefault("A", {})[stream] = af
    return out if out else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data_out/branch_v11/signals_locate.json")
    args = ap.parse_args()

    eps = scan_episodes(args.branch_dirs)
    clean_eps = [e for e in eps if e["arm"] == "clean"]
    attack_eps = [e for e in eps if e["arm"] == "attack"]
    print(f"episodes: clean={len(clean_eps)} attack={len(attack_eps)}")

    feats = {}          # run_id -> {pos_kind: {stream: [n_layers,H]}}
    for e in eps:
        try:
            f = load_first_call(args.captures, e["first_capture"])
        except Exception as exc:  # noqa: BLE001
            print(f"载入失败 {e['first_capture']}: {exc}", file=sys.stderr)
            continue
        if f:
            feats[e["run_id"]] = f
    n_layers = 64
    print(f"载入 {len(feats)} 个 episode")

    report = {"stage3_position": {}, "stage4_family_generalization": {}}

    # ---------------- Stage 3：P / R / A* × 层 × 流 ----------------
    labels = {e["run_id"]: 0 if e["arm"] == "clean" else 1 for e in eps}
    groups = {e["run_id"]: e["position_id"] for e in eps}
    for pos_kind in ("P", "R", "A"):
        for stream in STREAMS:
            X = {rid: f[pos_kind][stream] for rid, f in feats.items()
                 if pos_kind in f and stream in f[pos_kind]}
            if len(X) < 10:
                continue
            per_layer = {}
            for layer in range(n_layers):
                feat_l = {k: v[layer] for k, v in X.items()}
                per_layer[layer] = round(probe_auc_lopo(feat_l, labels, groups), 4)
            report["stage3_position"][f"{pos_kind}:{stream}"] = {
                "n": len(X),
                "best_layers": sorted(per_layer.items(), key=lambda kv: -kv[1])[:5],
                "auc_by_layer_LOPO": per_layer,
            }

    # ---------------- Stage 4：LOFO 未见家族（P 位） ----------------
    families = sorted({e["payload_id"][0] for e in attack_eps})
    test_positions = {e["position_id"] for e in attack_eps}
    for stream in STREAMS:
        per_family = {}
        for fam in families:
            tr = {}
            test = {}
            for e in eps:
                if e["run_id"] not in feats or "P" not in feats[e["run_id"]]:
                    continue
                v = feats[e["run_id"]]["P"].get(stream)
                if v is None:
                    continue
                is_test_attack = (e["arm"] == "attack"
                                  and e["payload_id"][0] == fam)
                is_test_clean = (e["arm"] == "clean"
                                 and e["position_id"] in test_positions)
                if is_test_attack or is_test_clean:
                    test[e["run_id"]] = (v, 1 if is_test_attack else 0)
                else:
                    # 训练集：其他家族的 attack（含测试位置——泄漏点，已声明）
                    # + 非测试位置的 clean（与测试 clean 位置不相交）
                    tr[e["run_id"]] = (v, 1 if e["arm"] == "attack" else 0)
            if not tr or not test:
                continue
            import numpy as np
            xs1 = [v.numpy() for v, y in tr.values() if y == 1]
            xs0 = [v.numpy() for v, y in tr.values() if y == 0]
            if not xs1 or not xs0:
                continue
            layer_auc = {}
            for layer in range(n_layers):
                w = np.mean([x[layer] for x in xs1], 0) - \
                    np.mean([x[layer] for x in xs0], 0)
                nrm = np.linalg.norm(w)
                if nrm == 0:
                    continue
                w = w / nrm
                pos = [float(v.numpy()[layer] @ w) for v, y in test.values() if y == 1]
                neg = [float(v.numpy()[layer] @ w) for v, y in test.values() if y == 0]
                layer_auc[layer] = round(rank_auc(pos, neg), 4)
            per_family[fam] = {
                "n_test_attack": sum(1 for _v, y in test.values() if y == 1),
                "n_test_clean": sum(1 for _v, y in test.values() if y == 0),
                "best_layers": sorted(layer_auc.items(), key=lambda kv: -kv[1])[:5],
            }
        report["stage4_family_generalization"][stream] = per_family

    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("\n=== Stage 3：读取位置对比（信号一 LOPO AUC，每位置取最佳 5 层）===")
    for k, v in report["stage3_position"].items():
        print(f"  {k}: n={v['n']} 最佳层 {v['best_layers'][:3]}")
    print("\n=== Stage 4：LOFO 未见家族（P 位，测试=留出家族 attack + 测试位置 clean）===")
    for stream, fams in report["stage4_family_generalization"].items():
        for fam, v in fams.items():
            print(f"  {stream} 留出家族 {fam}: "
                  f"test {v['n_test_attack']}attack+{v['n_test_clean']}clean "
                  f"最佳层 {v['best_layers'][:3]}")
    print(f"\n完整报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
