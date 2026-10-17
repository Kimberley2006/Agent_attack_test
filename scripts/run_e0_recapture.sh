#!/usr/bin/env bash
# 原始 capture 丢失时，在宿主机重新采集 clean/attack 并完成 E0 对照臂。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_PATH=${AA_MODEL_PATH:-/data/wangxiang/models/Qwen3.8-27B}
MODEL_PYTHON=${AA_MODEL_PYTHON:-$PWD/.conda-envs/model-cu128/bin/python}
WEBSHOP_PYTHON=${AA_WEBSHOP_PYTHON:-$PWD/.conda-envs/webshop/bin/python}
OUT_ROOT=${AA_E0_OUT_ROOT:-$PWD/data_out/branch_v11_recapture}
CAPTURE_DIR="$OUT_ROOT/captures"
SOCKET=${AA_E0_SOCKET:-/tmp/aa_e0_model.sock}
GPU=${AA_GPU:-0}
DEVICE_MAP=${AA_DEVICE_MAP:-balanced}
GPU_RESERVE_GB=${AA_GPU_RESERVE_GB:-6}
POSITIONS_FILE=${AA_POSITIONS_FILE:-$PWD/data_out/attack/positions_20/positions.jsonl}

export WEBSHOP_ITEM_PATH=${WEBSHOP_ITEM_PATH:-$PWD/webshop_repo/data/items_shuffle.json}
export WEBSHOP_ATTR_PATH=${WEBSHOP_ATTR_PATH:-$PWD/webshop_repo/data/items_ins_v2.json}
export WEBSHOP_HUMAN_INS_PATH=${WEBSHOP_HUMAN_INS_PATH:-$PWD/webshop_repo/data/items_human_ins.json}

"$MODEL_PYTHON" -m scripts.check_model_weights "$MODEL_PATH"
"$MODEL_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("模型 Python 环境无法访问 CUDA；请在有 GPU 的宿主环境运行")
print(f"CUDA 可用，GPU 数：{torch.cuda.device_count()}")
PY
"$WEBSHOP_PYTHON" - <<'PY'
from agent.env_wrapper import ensure_webshop_importable
ensure_webshop_importable()
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
print("WebShop 导入成功")
PY

mkdir -p "$CAPTURE_DIR" "$OUT_ROOT/clean" "$OUT_ROOT/attack" "$OUT_ROOT/control"
if [ -e "$SOCKET" ]; then
  echo "模型 socket 已存在：$SOCKET。请停止旧服务或设置 AA_E0_SOCKET。" >&2
  exit 2
fi

HF_HUB_OFFLINE=1 "$MODEL_PYTHON" -m agent.model_server \
  --model-path "$MODEL_PATH" --gpu "$GPU" --socket "$SOCKET" \
  --device-map "$DEVICE_MAP" --gpu-reserve-gb "$GPU_RESERVE_GB" \
  --collect basic --capture-dir "$CAPTURE_DIR" \
  > "$OUT_ROOT/model_server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$SOCKET"
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "模型服务提前退出；见 $OUT_ROOT/model_server.log" >&2
    exit 3
  fi
  if [ -S "$SOCKET" ] && AA_E0_SOCKET="$SOCKET" "$WEBSHOP_PYTHON" - <<'PY' >/dev/null 2>&1
import os
from agent.model_client import ModelClient
assert ModelClient(os.environ["AA_E0_SOCKET"], timeout_s=5).ping()
PY
  then
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "模型服务 20 分钟内未就绪；见 $OUT_ROOT/model_server.log" >&2
  exit 3
fi

common=(--positions "$POSITIONS_FILE"
  --clean-dirs data_out/clean_debug10_frozen data_out/clean_baseline_100a
               data_out/clean_baseline_100b
  --lang zh --socket "$SOCKET")

"$WEBSHOP_PYTHON" -m attack.run_branch "${common[@]}" \
  --arm clean --out "$OUT_ROOT/clean" --max-positions 12 \
  > "$OUT_ROOT/clean.log" 2>&1
"$WEBSHOP_PYTHON" -m attack.run_branch "${common[@]}" \
  --arm attack --out "$OUT_ROOT/attack" --max-positions 5 \
  --only-payloads A1,B1,C1,D1 > "$OUT_ROOT/attack.log" 2>&1

# 缺失文件和行为不一致都意味着旧结果不能直接沿用。
AA_E0_OUT_ROOT="$OUT_ROOT" AA_CAPTURE_DIR="$CAPTURE_DIR" AA_POSITIONS_FILE="$POSITIONS_FILE" \
  "$WEBSHOP_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["AA_E0_OUT_ROOT"])
captures = Path(os.environ["AA_CAPTURE_DIR"])
positions = [json.loads(line)["position_id"] for line in
             Path(os.environ["AA_POSITIONS_FILE"]).read_text().splitlines()
             if line.strip()]
expected = {
    "clean": {f"{p}__CLEAN.json" for p in positions[:12]},
    "attack": {f"{p}__{a}.json" for p in positions[:5]
               for a in ("A1", "B1", "C1", "D1")},
}
for arm, names in expected.items():
    present = {p.name for p in (root / arm).glob("*.json")}
    if present != names:
        raise SystemExit(f"{arm} 数量/位置不符：缺 {sorted(names-present)}；多 {sorted(present-names)}")
    for name in names:
        new = json.loads((root / arm / name).read_text())
        old = json.loads((Path("data_out/branch_v11") / arm / name).read_text())
        steps = new["outcome"]["steps"]
        first = next((s["capture"]["file"] for s in steps if s.get("capture")), None)
        if first is None or not (captures / first).is_file():
            raise SystemExit(f"缺少首调用 capture：{arm}/{name}")
        old_actions = [s.get("action") for s in old["outcome"]["steps"]]
        new_actions = [s.get("action") for s in steps]
        fields = ("done_reason", "purchased_asin")
        if old_actions != new_actions or any(
            old["outcome"][field] != new["outcome"][field] for field in fields
        ) or old["attack_effective"] != new["attack_effective"]:
            raise SystemExit(f"重采行为与原实验不一致：{arm}/{name}")
    print(f"{arm} 重采与原实验行为一致：{len(names)}/{len(names)}")
PY

"$WEBSHOP_PYTHON" -m attack.run_branch "${common[@]}" \
  --arm control --out "$OUT_ROOT/control" --max-positions 5 \
  > "$OUT_ROOT/control.log" 2>&1
"$MODEL_PYTHON" -m analysis.signals \
  --branch-dirs "$OUT_ROOT/clean" "$OUT_ROOT/attack" "$OUT_ROOT/control" \
  --captures "$CAPTURE_DIR" --out "$OUT_ROOT/signals_full.json" \
  > "$OUT_ROOT/signals.log" 2>&1

AA_E0_OUT_ROOT="$OUT_ROOT" "$MODEL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["AA_E0_OUT_ROOT"])
report = json.loads((root / "signals_full.json").read_text())
e0 = report["e0_primary"]
if e0["status"] != "complete":
    raise SystemExit(f"E0 数据不完整：{e0['status']}；见 {root/'signals.log'}")
print(json.dumps({
    "status": e0["status"],
    "control_clean": e0["metrics"]["control_clean"],
    "control_by_payload": e0["metrics"]["control_by_payload"],
    "specificity_pass_0_8": e0["specificity_pass_0_8"],
}, ensure_ascii=False, indent=2))
print(f"完整报告：{root/'signals_full.json'}")
PY
