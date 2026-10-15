# -*- coding: utf-8 -*-
"""模型推理服务进程（base conda 环境，GPU）。

加载 Qwen3.8-27B 一次，循环服务 generate 请求。逐调用返回
input_ids/output_ids —— 这是攻击阶段"注入送达核对"的数据来源。

v2: 接入内部表征采集（agent_attack/hooks_collector，basic 档）。
- --collect basic 时每次 generate 在 P（prefill 末 token）、R（思考结束
  后首个可见 token，如有）、A*（动作 token 区，因果对齐）位置捕获每层
  residual-in/out、attn 输出、MLP 输出，落盘 {capture_dir}/cap_*.pt。
- 请求可携带 meta（task_id/step/arm 等），原样写入 capture 与响应。
- 采集只读不改计算；--collect off 时行为与 v1 完全一致。

用法:
    HF_HUB_OFFLINE=1 python -m agent.model_server --gpu 0 \
        --socket /tmp/aa_model.sock --collect basic \
        --capture-dir /cache/wuhan/caps/run_x
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import MODEL_ID, MODEL_REVISION  # noqa: E402
from agent.model_client import recv_frame, send_frame  # noqa: E402

_ACTION_RE = re.compile(r"(search|click)\[[^\[\]]*\]", re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)


def build_generation_kwargs(gen: dict) -> dict:
    kwargs = {"max_new_tokens": int(gen.get("max_new_tokens", 512))}
    if gen.get("do_sample"):
        kwargs.update(do_sample=True, temperature=gen.get("temperature", 1.0),
                      top_p=gen.get("top_p", 1.0))
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kwargs


def pick_gpu(preferred: int, min_free_gb: float = 62.0) -> int:
    """优先用 preferred，显存不足则换第一个空闲 GPU（共享集群存在抢卡竞争）。"""
    import torch

    order = [preferred] + [i for i in range(torch.cuda.device_count()) if i != preferred]
    for i in order:
        free, _total = torch.cuda.mem_get_info(i)
        if free >= min_free_gb * 2**30:
            return i
    return preferred


def gpu_shard_budgets(torch, reserve_gb: float) -> dict[int, int]:
    """按各卡当前空闲显存限制权重分配，给 KV cache 和采集缓冲留空间。"""
    reserve = int(reserve_gb * 2**30)
    budgets = {}
    for gpu in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(gpu)
        usable = free - reserve
        print(f"[server] cuda:{gpu} free={free / 2**30:.1f} GiB, "
              f"total={total / 2**30:.1f} GiB, "
              f"weight_budget={max(usable, 0) / 2**30:.1f} GiB", flush=True)
        if usable > 0:
            budgets[gpu] = usable
    if not budgets:
        raise RuntimeError(f"没有 GPU 留出 {reserve_gb:g} GiB 后仍有可用显存")
    return budgets


class _Ticker:
    """generate 每生成一个 token 后步进一次（供 hooks 定位 decode 步号）。"""

    def __init__(self, collector):
        self.collector = collector
        self.n = 0

    def __call__(self, input_ids, scores, **kwargs):
        if self.collector is not None and self.collector.active is not None:
            self.collector.active._gen_pos += 1
        self.n += 1
        return False


def token_char_offsets(tokenizer, output_ids: list[int]) -> list[int]:
    """prefix 解码长度：offs[j] = decode(ids[:j+1]) 的可见字符数。O(n^2)，n<=768 可接受。"""
    offs = []
    for j in range(1, len(output_ids) + 1):
        offs.append(len(tokenizer.decode(output_ids[:j], skip_special_tokens=True)))
    return offs


def char_to_token(ci: int, offs: list[int]) -> int:
    """字符下标 -> 覆盖它的 token 下标。"""
    for j, e in enumerate(offs):
        if e > ci:
            return j
    return len(offs) - 1


def compute_anchors(tokenizer, output_ids: list[int], text: str) -> dict:
    """返回 anchor 方案：动作 token 区（因果对齐 buffer 下标 a0..a1）、R 位置。

    buffer[gen_pos p] 保存的是"预测 token p"的层状态（该步 forward 的末位
    输入是 token p-1，因果对齐）。因此预测动作 token a0..a1 的状态 =
    buffer[a0..a1]。R = 思考结束后首个可见 token 的下标（无思考块则 None）。
    """
    if not output_ids:
        return {"action_span_tok": None, "r_tok": None, "anchors": []}
    offs = token_char_offsets(tokenizer, output_ids)
    matches = list(_ACTION_RE.finditer(text))
    r_tok = None
    m = _THINK_END_RE.search(text)
    if m:
        r_tok = char_to_token(m.end(), offs)
    if not matches:
        anchors = [r_tok] if r_tok is not None else []
        return {"action_span_tok": None, "r_tok": r_tok, "anchors": anchors}
    a = matches[-1]
    t0 = char_to_token(a.start(), offs)
    t1 = min(char_to_token(a.end() - 1, offs), len(output_ids) - 1)
    anchors = list(range(t0, t1 + 1))
    if r_tok is not None and r_tok not in anchors:
        anchors.append(r_tok)
    return {"action_span_tok": [t0, t1], "r_tok": r_tok,
            "anchors": sorted(set(anchors))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2,
                        help="首选 GPU；显存不足时自动换卡")
    parser.add_argument("--socket", default="/tmp/aa_model.sock")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--collect", default="off", choices=["off", "basic"])
    parser.add_argument("--capture-dir", default=None,
                        help="collect=basic 时的落盘目录（默认 /cache/wuhan/captures/{socket名}）")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="本地模型目录；缺省从冻结的 Hugging Face revision 加载")
    parser.add_argument("--device-map", default="single",
                        choices=["single", "balanced"],
                        help="single=单卡；balanced=多卡按层分配 BF16 权重")
    parser.add_argument("--gpu-reserve-gb", type=float, default=6.0,
                        help="多卡模式下每张 GPU 为生成和采集预留的显存")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    model_source = str(args.model_path) if args.model_path else MODEL_ID
    revision_kw = {} if args.model_path else {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **revision_kw)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    model, device = None, None
    if args.device_map == "balanced":
        budgets = gpu_shard_budgets(torch, args.gpu_reserve_gb)
        if args.model_path:
            index_path = args.model_path / "model.safetensors.index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text())
                weight_bytes = int(index.get("metadata", {}).get("total_size", 0))
                if weight_bytes and sum(budgets.values()) < weight_bytes:
                    raise RuntimeError(
                        f"GPU 权重预算合计 {sum(budgets.values()) / 2**30:.1f} GiB，"
                        f"模型权重需 {weight_bytes / 2**30:.1f} GiB；"
                        "请释放 GPU 显存或增加可用 GPU")
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_source, **revision_kw, torch_dtype=dtype,
            attn_implementation=args.attn, device_map="balanced",
            max_memory=budgets,
        )
        placements = getattr(model, "hf_device_map", {})
        if not placements or any(
            not (isinstance(value, int) or str(value).startswith("cuda:"))
            for value in placements.values()
        ):
            raise RuntimeError(f"模型未完全分配到 GPU：{placements}")
        device = model.get_input_embeddings().weight.device
        print(f"[server] 权重分配：{placements}", flush=True)
    else:
        for attempt in range(3):
            device = f"cuda:{pick_gpu(args.gpu)}"
            try:
                print(f"[server] 尝试加载到 {device} (第 {attempt+1} 次)", flush=True)
                model = Qwen3_5ForCausalLM.from_pretrained(
                    model_source, **revision_kw, torch_dtype=dtype,
                    attn_implementation=args.attn,
                ).to(device)
                break
            except torch.OutOfMemoryError as e:
                print(f"[server] {device} OOM: {e}", flush=True)
                args.gpu = (args.gpu + 1) % torch.cuda.device_count()
                model = None
        if model is None:
            print("[server] 三次加载均失败", flush=True)
            return 1
    model.eval()
    print(f"[server] 模型就绪 ({device}, {args.attn}, {args.dtype})", flush=True)

    collector = None
    capture_dir = None
    if args.collect == "basic":
        from agent_attack.hooks_collector import HooksCollector
        collector = HooksCollector(model, collect="basic")
        capture_dir = Path(args.capture_dir) if args.capture_dir else \
            Path("/cache/wuhan/captures") / Path(args.socket).name
        capture_dir.mkdir(parents=True, exist_ok=True)
        print(f"[server] 采集开启 basic -> {capture_dir}", flush=True)

    sock_path = Path(args.socket)
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)

    import socket as pysock

    server = pysock.socket(pysock.AF_UNIX, pysock.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    print(f"[server] 监听 {sock_path}", flush=True)

    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # 断点续跑时保留已有 cap 文件；episode JSON 中的文件名不能被新服务覆盖。
    n_calls = 1 + max(
        (int(path.stem[4:]) for path in capture_dir.glob("cap_*.pt")
         if re.fullmatch(r"cap_\d+", path.stem)),
        default=-1,
    ) if capture_dir else 0
    start_call_index = n_calls
    while not stop["flag"]:
        try:
            conn, _ = server.accept()
        except OSError:
            break
        with conn:
            while not stop["flag"]:
                try:
                    req = recv_frame(conn)
                except (ConnectionError, json.JSONDecodeError):
                    break
                op = req.get("op")
                if op == "ping":
                    send_frame(conn, {"id": req.get("id"), "ok": True})
                    continue
                if op != "generate":
                    send_frame(conn, {"id": req.get("id"), "ok": False,
                                      "error": f"unknown op {op}"})
                    continue
                t0 = time.time()
                try:
                    tk = req.get("template_kwargs") or {}

                    def _template(messages):
                        # transformers 5.x: tokenize=True 返回 BatchEncoding
                        enc = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True,
                            tokenize=True, return_tensors="pt", **tk,
                        )
                        return enc["input_ids"].to(device)

                    try:
                        input_ids = _template(req["messages"])
                    except TypeError:
                        # 模板不支持额外 kwargs（如 enable_thinking）时回退
                        tk.pop("enable_thinking", None)
                        input_ids = _template(req["messages"])
                    prompt_ids = input_ids[0].tolist()
                    gen_kwargs = build_generation_kwargs(req.get("gen", {}))
                    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
                    ticker = None
                    if collector is not None:
                        ticker = collector.begin_call(
                            f"call_{n_calls:06d}", len(prompt_ids),
                            gen_kwargs["max_new_tokens"])
                        gen_kwargs["stopping_criteria"] = [ticker]
                    try:
                        with torch.no_grad():
                            out = model.generate(input_ids=input_ids, **gen_kwargs)
                    finally:
                        if collector is not None:
                            cap = collector.end_call()
                    all_ids = out[0].tolist()
                    output_ids = all_ids[len(prompt_ids):]
                    n_new = len(output_ids)
                    truncated = n_new >= gen_kwargs["max_new_tokens"] and \
                        output_ids[-1] != tokenizer.eos_token_id
                    output_text = tokenizer.decode(output_ids,
                                                    skip_special_tokens=True)
                    cap_info = None
                    if collector is not None and cap is not None and n_new > 0:
                        anchor_info = compute_anchors(tokenizer, output_ids,
                                                      output_text)
                        payload = collector.extract(
                            cap, anchor_info["anchors"], n_new)
                        payload["meta"] = req.get("meta") or {}
                        payload["anchor_info"] = anchor_info
                        cpath = capture_dir / f"cap_{n_calls:06d}.pt"
                        torch.save(payload, cpath)
                        cap_info = {"file": cpath.name,
                                    "n_anchors": len(anchor_info["anchors"]),
                                    "action_span_tok": anchor_info["action_span_tok"],
                                    "r_tok": anchor_info["r_tok"]}
                    resp = {
                        "id": req["id"], "ok": True,
                        "output_text": output_text,
                        "input_ids": prompt_ids,
                        "n_prompt_tokens": len(prompt_ids),
                        "n_new_tokens": n_new,
                        "truncated": bool(truncated),
                        "wall_s": round(time.time() - t0, 2),
                        "call_index": n_calls,
                        "capture": cap_info,
                    }
                    n_calls += 1
                except Exception:  # noqa: BLE001
                    resp = {"id": req.get("id"), "ok": False,
                            "error": traceback.format_exc(limit=3)}
                try:
                    send_frame(conn, resp)
                except (BrokenPipeError, ConnectionError, OSError):
                    break  # 客户端已断开（如运行器被 kill），服务继续存活
    print(f"[server] 退出，共服务 {n_calls - start_call_index} 次调用", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
