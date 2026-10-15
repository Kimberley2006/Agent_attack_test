"""多 GPU 并行运行：每个 worker 独立 GPU + 独立环境服务（WebShop 浏览器状态
不可共享），任务按 round-robin 分片（贪心+冻结配置下每个任务独立确定，
分片不改变任何轨迹），完成后合并指标。

用法：
  python scripts/launch_parallel.py --split baseline --gpus 1,2,3,4,5,6 \
      --collect basic --tag frozen-v1p
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/home/ma-user/miniconda3/bin/python"
WS_PY = "/home/ma-user/conda-envs/webshop/bin/python"
ENVVARS = {
    "WEBSHOP_ITEM_PATH": "/cache/wuhan/webshop_data/items_shuffle.json",
    "WEBSHOP_ATTR_PATH": "/cache/wuhan/webshop_data/items_ins_v2.json",
    "WEBSHOP_HUMAN_INS_PATH": str(ROOT / "webshop_repo/data/items_human_ins.json"),
}


def wait_health(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=10) as r:
                if json.loads(r.read())["ok"]:
                    return True
        except Exception:
            time.sleep(5)
    return False


def ensure_service(port, log):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=5) as r:
            if json.loads(r.read())["ok"]:
                print(f"service :{port} already up")
                return None
    except Exception:
        pass
    cmd = (f"cd {ROOT}/webshop_repo && env "
           + " ".join(f"{k}={shlex.quote(v)}" for k, v in ENVVARS.items())
           + f" {WS_PY} -c \"import sys; sys.path.insert(0,'.'); "
             "exec(open('../scripts/env_service.py').read())\" "
           f"--port {port}")
    # 注：env_service.py 自身以项目根运行；直接用绝对路径启动
    cmd = (f"env {' '.join(f'{k}={shlex.quote(v)}' for k, v in ENVVARS.items())} "
           f"{WS_PY} {ROOT}/scripts/env_service.py --port {port}")
    return subprocess.Popen(["bash", "-c", cmd],
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="baseline")
    ap.add_argument("--gpus", default="1,2,3,4,5,6")
    ap.add_argument("--collect", default="basic")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-port", type=int, default=5001)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inject-file", default=None)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    n = len(gpus)

    registry = json.loads((ROOT / "task_registry.json").read_text())
    tasks = registry["splits"][args.split]
    if args.limit:
        tasks = tasks[:args.limit]
    shards = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        shards[i % n].append(t["goal_idx"])
    print(f"{len(tasks)} tasks over {n} GPUs: "
          + ", ".join(f"{len(s)}" for s in shards))

    # 1. 服务（交错启动，避免同时全量建库抢 CPU）
    procs = []
    for w in range(n):
        port = args.base_port + w
        p = ensure_service(port, f"/tmp/env_service_p{port}.log")
        procs.append(p)
        print(f"service :{port} {'starting' if p else 'reused'}")
    for w in range(n):
        port = args.base_port + w
        ok = wait_health(port)
        print(f"service :{port} {'READY' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)

    # 2. workers
    logs = []
    for w, gpu in enumerate(gpus):
        port = args.base_port + w
        shard_ids = ",".join(str(i) for i in shards[w])
        log = f"/tmp/worker_{args.tag}_g{gpu}.log"
        cmd = ([PY, f"{ROOT}/scripts/run_baseline.py",
                "--task-ids", shard_ids, "--collect", args.collect,
                "--service-url", f"http://127.0.0.1:{port}",
                "--tag", f"{args.tag}-w{w}"]
               + (["--inject-file", args.inject_file] if args.inject_file else []))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                         stderr=subprocess.STDOUT)
        logs.append(log)
        print(f"worker {w} -> GPU {gpu}, port {port}, {len(shards[w])} tasks, log {log}")

    print("\nAll workers launched. Monitor with:")
    for log in logs:
        print(f"  tail -f {log}")


if __name__ == "__main__":
    main()
