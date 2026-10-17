"""后台执行 20 任务、33 位置的配对实验：恢复数据、重放、等待 3 GPU、三臂采集。

python3 scripts/run_positions20_background.py --detach
进度见 data_out/branch_positions20/status.json 和各阶段日志。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data_out/branch_positions20"
MODEL_PYTHON = os.environ.get("AA_MODEL_PYTHON", str(ROOT / ".conda-envs/model-cu128/bin/python"))
WEB_PYTHON = str(ROOT / ".conda-envs/webshop/bin/python")
MODEL = "/data/wangxiang/models/Qwen3.8-27B"
SOCKET = "/tmp/aa_positions20.sock"
PAYLOADS = {"clean": ["CLEAN"], "attack": ["A1", "B1", "C1", "D1"],
            "control": ["N1", "N2", "N3"]}
children = []


def status(stage, **details):
    data = {"stage": stage, "pid": os.getpid(), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "planned_episodes": {"clean": 33, "attack": 132, "control": 99}, **details}
    temp = OUT / "status.json.tmp"
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(OUT / "status.json")
    print(json.dumps(data, ensure_ascii=False), flush=True)


def run(command, log_name, env, check=True):
    with (OUT / log_name).open("a") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(proc)
        code = proc.wait()
        children.remove(proc)
    if check and code:
        raise RuntimeError(f"{log_name}: exit {code}")
    return code


def choose_gpus(rows):
    # 三卡各留 6 GiB；25 GiB 空闲/卡足以容纳冻结 BF16 权重和生成缓冲。
    ready = sorted((r for r in rows if r["free_mib"] >= 25 * 1024),
                   key=lambda r: (-r["free_mib"], r["index"]))
    return [r["index"] for r in ready[:3]] if len(ready) >= 3 else []


def verified_positions_unchanged():
    """复用已完成的重放验收，核对配置、位置清单及所有源轨迹的指纹。"""
    directory = ROOT / "data_out/attack/positions_20"
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        config = json.loads((ROOT / "configs/position_comparison20.json").read_text())
        if manifest["config"] != config or manifest["validation"] != "all selected goals, messages and replay observations matched":
            return False
        paths = [(ROOT / config["source_positions"], config["source_sha256"])]
        paths += [(directory / name, sha) for name, sha in manifest["files_sha256"].items()]
        paths += [(ROOT / name, sha) for name, sha in manifest["trajectory_sha256"].items()]
        return all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in paths)
    except (OSError, KeyError, ValueError):
        return False


def wait_gpus():
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        rows = [{"index": int(line.split(",")[0]), "free_mib": int(line.split(",")[1])}
                for line in result.stdout.splitlines() if line.strip()]
        selected = choose_gpus(rows)
        if selected:
            return selected
        status("waiting_for_3_gpus", gpu_free=rows, minimum_free_mib_per_gpu=25 * 1024)
        time.sleep(60)


def validate_arm(arm, positions):
    expected = {f"{p['position_id']}__{payload}" for p in positions for payload in PAYLOADS[arm]}
    rows = [json.loads(p.read_text()) for p in (OUT / arm).glob("*.json")]
    if {r["run_id"] for r in rows} != expected:
        raise RuntimeError(f"{arm} episodes 不完整，见 {arm}.log")
    for row in rows:
        steps = row["outcome"]["steps"]
        if row["outcome"]["done_reason"] == "error" or not steps:
            raise RuntimeError(f"episode 错误: {row['run_id']}")
        for step in steps:
            capture = step.get("capture", {}).get("file") if step.get("capture") else None
            if not capture or not (OUT / "captures" / capture).is_file():
                raise RuntimeError(f"capture 缺失: {row['run_id']}")
    return rows


def stop_children():
    for proc in reversed(children):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("已有相同实验的后台任务运行")
    (OUT / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    for key, name in (("WEBSHOP_ITEM_PATH", "items_shuffle"), ("WEBSHOP_ATTR_PATH", "items_ins_v2"),
                      ("WEBSHOP_HUMAN_INS_PATH", "items_human_ins")):
        env[key] = str(ROOT / "webshop_repo/data" / f"{name}.json")
    try:
        status("restoring_data")
        run([sys.executable, "scripts/restore_webshop_data.py"], "data_restore.log", env)
        status("checking_model_and_index")
        run([MODEL_PYTHON, "scripts/check_model_weights.py", MODEL], "model_check.log", env)
        index_check = "import pickle; d=pickle.load(open('webshop_repo/search_engine/indexes.pkl','rb')); assert len(d['asins'])==1181430; assert d['bm25'].corpus_size==1181430; print('index verified')"
        if run([WEB_PYTHON, "-c", index_check], "index_check.log", env, check=False):
            status("rebuilding_index")
            run([WEB_PYTHON, "scripts/build_bm25_index.py"], "index_build.log", env)
            run([WEB_PYTHON, "-c", index_check], "index_check.log", env)
        if os.environ.get("AA_REUSE_VERIFIED_POSITIONS") == "1" and verified_positions_unchanged():
            status("reusing_verified_positions", reason="configuration, position and trajectory hashes unchanged; data hashes verified")
        else:
            status("verifying_positions")
            run([WEB_PYTHON, "scripts/prepare_position_comparison.py"], "positions_verify.log", env)
        source = ROOT / "data_out/attack/positions_20/all_positions.jsonl"
        positions = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(positions) != 33 or len({p["task_id"] for p in positions}) != 20 or not all(p["replay_verified"] for p in positions):
            raise RuntimeError("选点数量或重放验证状态发生变化")
        (OUT / "positions.jsonl").write_bytes(source.read_bytes())
        for arm in PAYLOADS:
            (OUT / arm).mkdir(exist_ok=True)
        (OUT / "captures").mkdir(exist_ok=True)
        selected = wait_gpus()
        model_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(map(str, selected)), HF_HUB_OFFLINE="1")
        status("checking_cuda_runtime", gpu_indices=selected)
        preflight = (
            "import torch, transformers; from transformers import Qwen3_5ForCausalLM; "
            "print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'transformers', transformers.__version__, flush=True); "
            "assert torch.cuda.is_available() and torch.cuda.device_count() == 3; "
            "[(print(i, torch.cuda.get_device_name(i), float((torch.ones((16,16), device=f'cuda:{i}', dtype=torch.bfloat16) @ "
            "torch.ones((16,16), device=f'cuda:{i}', dtype=torch.bfloat16)).sum()), flush=True)) for i in range(3)]"
        )
        run([MODEL_PYTHON, "-c", preflight], "cuda_preflight.log", model_env)
        if Path(SOCKET).exists():
            raise RuntimeError(f"实验 socket 已存在: {SOCKET}")
        command = [MODEL_PYTHON, "-m", "agent.model_server", "--model-path", MODEL,
                   "--device-map", "balanced", "--gpu-reserve-gb", "6", "--gpu", "0",
                   "--socket", SOCKET, "--collect", "basic", "--capture-dir", str(OUT / "captures")]
        (OUT / "run_config.json").write_text(json.dumps({"gpu_indices": selected, "model_command": command,
            "payloads": PAYLOADS, "n_positions": 33, "n_tasks": 20,
            "config": json.loads((ROOT / "configs/position_comparison20.json").read_text())}, indent=2))
        run([MODEL_PYTHON, "-m", "pip", "freeze"], "model_environment.txt", model_env)
        status("loading_model", gpu_indices=selected)
        log = (OUT / "model_server.log").open("a")
        server = subprocess.Popen(command, cwd=ROOT, env=model_env, stdout=log, stderr=subprocess.STDOUT)
        children.append(server)
        from agent.model_client import ModelClient
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("模型服务退出，见 model_server.log")
            client = ModelClient(SOCKET, timeout_s=5)
            try:
                if client.ping():
                    break
            except (OSError, ConnectionError):
                pass
            finally:
                client.close()
            time.sleep(10)
        else:
            raise RuntimeError("模型服务 30 分钟内未就绪")
        rows = []
        for arm in PAYLOADS:
            status(f"running_{arm}", gpu_indices=selected)
            command = [WEB_PYTHON, "-m", "attack.run_branch", "--positions", str(OUT / "positions.jsonl"),
                       "--clean-dirs", "data_out/clean_debug10_frozen", "data_out/clean_baseline_100a",
                       "data_out/clean_baseline_100b", "--entry", "auto", "--lang", "zh", "--arm", arm,
                       "--socket", SOCKET, "--only-payloads", ",".join(PAYLOADS[arm]), "--out", str(OUT / arm)]
            run(command, f"{arm}.log", env)
            rows.extend(validate_arm(arm, positions))
        # 每个位置的行为结果，后续 hidden-state 分析必须按 task_id 留出。
        by_position = {}
        for pos in positions:
            subset = [r for r in rows if r["position"]["position_id"] == pos["position_id"]]
            by_position[pos["position_id"]] = {
                "group": pos["position_group"], "task_id": pos["task_id"],
                "results": [{"arm": r["arm"], "payload": r["payload"]["payload_id"],
                             "bought_injected_item": r["outcome"]["purchased_asin"] == pos["asin"],
                             "reward": r["outcome"]["final_reward"], "done_reason": r["outcome"]["done_reason"]} for r in subset]}
        (OUT / "behavior_summary.json").write_text(json.dumps(by_position, ensure_ascii=False, indent=2))
        status("complete", counts=dict(Counter(r["arm"] for r in rows)), gpu_indices=selected)
    except BaseException as exc:
        status("stopped" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_children()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reuse-verified-positions", action="store_true",
                        help="数据及配置指纹一致时复用此前成功的环境重放验证")
    args = parser.parse_args()
    if args.reuse_verified_positions:
        os.environ["AA_REUSE_VERIFIED_POSITIONS"] = "1"
    if args.status:
        data = json.loads((OUT / "status.json").read_text())
        try:
            os.kill(data["pid"], 0)
            data["supervisor_alive"] = True
        except ProcessLookupError:
            data["supervisor_alive"] = False
        print(json.dumps(data, ensure_ascii=False, indent=2))
    elif args.detach:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "supervisor.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        print(json.dumps({"pid": proc.pid, "status_file": str(OUT / "status.json")}))
    else:
        main()
