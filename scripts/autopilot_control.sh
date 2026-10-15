#!/bin/bash
# 自动链：等待空闲 GPU → 起模型服务 → 跑对照组 → 跑全量信号分析。
# 完全脱离会话运行（setsid），所有输出写 /tmp/aa_autopilot.log。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=${AA_AUTOPILOT_LOG:-/tmp/aa_autopilot.log}
export WEBSHOP_ITEM_PATH=${WEBSHOP_ITEM_PATH:-$PWD/webshop_repo/data/items_shuffle.json}
export WEBSHOP_ATTR_PATH=${WEBSHOP_ATTR_PATH:-$PWD/webshop_repo/data/items_ins_v2.json}
export WEBSHOP_HUMAN_INS_PATH=${WEBSHOP_HUMAN_INS_PATH:-$PWD/webshop_repo/data/items_human_ins.json}
WS=${AA_WEBSHOP_PYTHON:-python}
PY=${AA_MODEL_PYTHON:-python}
CAPTURE_DIR=${AA_CAPTURE_DIR:-/cache/wuhan/captures/g0}

for path in "$WEBSHOP_ITEM_PATH" "$WEBSHOP_ATTR_PATH" \
            "$WEBSHOP_HUMAN_INS_PATH" data_out/attack/positions_ip/positions.jsonl; do
  if [ ! -f "$path" ]; then
    echo "$(date '+%F %T') [autopilot] 缺少文件: $path" >> "$LOG"
    exit 2
  fi
done
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "$(date '+%F %T') [autopilot] 无可用 NVIDIA 驱动/GPU" >> "$LOG"
  exit 2
fi
if [ ! -d "$CAPTURE_DIR" ]; then
  echo "$(date '+%F %T') [autopilot] 缺少原始 capture 目录: $CAPTURE_DIR" >> "$LOG"
  exit 2
fi
if ! AA_CAPTURE_DIR="$CAPTURE_DIR" "$WS" - <<'PY' >> "$LOG" 2>&1
import json
import os
from pathlib import Path

capture_dir = Path(os.environ["AA_CAPTURE_DIR"])
missing = []
for arm in ("clean", "attack"):
    for path in Path(f"data_out/branch_v11/{arm}").glob("*.json"):
        episode = json.loads(path.read_text())
        captures = [s["capture"]["file"] for s in episode["outcome"]["steps"]
                    if s.get("capture")]
        if not captures or not (capture_dir / captures[0]).is_file():
            missing.append(path.name)
if missing:
    print(f"缺少 {len(missing)} 条 clean/attack 的首调用 capture: {missing[:5]}")
    raise SystemExit(1)
PY
then
  echo "$(date '+%F %T') [autopilot] 原始 capture 不完整，停止" >> "$LOG"
  exit 2
fi

echo "$(date '+%F %T') [autopilot] 开始等待空闲 GPU（需 ≥62GB）" >> $LOG
GPU=""
START=$(date +%s)
while true; do
  for i in 0 1; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i" 2>/dev/null || true)
    if [ -n "$free" ] && [ "$free" -ge 63000 ]; then GPU=$i; break; fi
  done
  [ -n "$GPU" ] && break
  if [ $(( $(date +%s) - START )) -gt 172800 ]; then
    echo "$(date '+%F %T') [autopilot] 48 小时无空闲 GPU，放弃" >> "$LOG"; exit 2
  fi
  sleep 120
done
echo "$(date '+%F %T') [autopilot] GPU $GPU 空闲（${free}MiB），启动模型服务" >> $LOG

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
rm -f /tmp/aa_model.sock
$PY -m agent.model_server --gpu "$GPU" --socket /tmp/aa_model.sock \
    --collect basic --capture-dir "$CAPTURE_DIR" >> /tmp/aa_server_g0.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

for i in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "$(date '+%F %T') [autopilot] 模型服务提前退出" >> "$LOG"
    exit 3
  fi
  if [ -S /tmp/aa_model.sock ] && "$WS" -c \
    'from agent.model_client import ModelClient; import sys; sys.exit(0 if ModelClient("/tmp/aa_model.sock").ping() else 1)' \
    >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
if [ ! -S /tmp/aa_model.sock ] || ! "$WS" -c \
  'from agent.model_client import ModelClient; import sys; sys.exit(0 if ModelClient("/tmp/aa_model.sock").ping() else 1)' \
  >/dev/null 2>&1; then
  echo "$(date '+%F %T') [autopilot] 服务启动失败" >> $LOG; kill $SERVER_PID 2>/dev/null; exit 3
fi
echo "$(date '+%F %T') [autopilot] 服务就绪 (pid $SERVER_PID)，启动对照组" >> $LOG

$WS -m attack.run_branch \
  --positions data_out/attack/positions_ip/positions.jsonl \
  --clean-dirs data_out/clean_debug10_frozen data_out/clean_baseline_100a data_out/clean_baseline_100b \
  --lang zh --arm control --socket /tmp/aa_model.sock \
  --out data_out/branch_v11/control --max-positions 5 >> /tmp/aa_control_run.log 2>&1
echo "$(date '+%F %T') [autopilot] 对照组退出，episode 数: $(ls data_out/branch_v11/control 2>/dev/null | grep -c json)" >> $LOG

$WS -m analysis.signals \
  --branch-dirs data_out/branch_v11/clean data_out/branch_v11/attack data_out/branch_v11/control \
  --captures "$CAPTURE_DIR" \
  --out data_out/branch_v11/signals_full.json >> $LOG 2>&1
if ! "$WS" - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('data_out/branch_v11/signals_full.json').read_text())
assert report['e0_primary']['status'] == 'complete', report['e0_primary']['status']
PY
then
  echo "$(date '+%F %T') [autopilot] E0 不完整，请检查 control episode 和 capture" >> "$LOG"
  exit 4
fi
echo "$(date '+%F %T') [autopilot] E0 完成 -> data_out/branch_v11/signals_full.json" >> "$LOG"

kill "$SERVER_PID" 2>/dev/null || true
trap - EXIT
echo "$(date '+%F %T') [autopilot] 全部完成，模型服务已关闭" >> $LOG
