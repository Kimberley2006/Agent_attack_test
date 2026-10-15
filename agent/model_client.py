# -*- coding: utf-8 -*-
"""模型推理服务的客户端（纯标准库，base/webshop 两个环境通用）。

协议：Unix socket 上的 JSON 帧，8 字节小端长度前缀 + UTF-8 JSON。
请求 {"op": "generate", "id", "messages", "gen", "template_kwargs"}
响应 {"id", "ok", "output_text", "input_ids", "output_ids",
      "n_prompt_tokens", "n_new_tokens", "truncated", "wall_s"}
"""
from __future__ import annotations

import json
import socket
import time
import uuid


def send_frame(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(len(data).to_bytes(8, "little") + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被服务端关闭")
        buf += chunk
    return buf


def recv_frame(sock: socket.socket) -> dict:
    (length,) = int.from_bytes(recv_exact(sock, 8), "little"),
    return json.loads(recv_exact(sock, length).decode("utf-8"))


class ModelClient:
    """向 agent.model_server 发送生成请求。"""

    def __init__(self, socket_path: str, timeout_s: float = 1800.0):
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def _connect(self) -> socket.socket:
        if self._sock is None:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout_s)
            s.connect(self.socket_path)
            self._sock = s
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def ping(self) -> bool:
        sock = self._connect()
        send_frame(sock, {"op": "ping", "id": "ping"})
        return recv_frame(sock).get("ok", False)

    def generate(self, messages: list[dict], max_new_tokens: int,
                 do_sample: bool = False,
                 template_kwargs: dict | None = None,
                 meta: dict | None = None) -> dict:
        request = {
            "op": "generate",
            "id": uuid.uuid4().hex,
            "messages": messages,
            "gen": {"max_new_tokens": max_new_tokens, "do_sample": do_sample},
            "template_kwargs": template_kwargs or {},
            "meta": meta or {},
        }
        t0 = time.time()
        sock = self._connect()
        send_frame(sock, request)
        resp = recv_frame(sock)
        if not resp.get("ok"):
            raise RuntimeError(f"模型服务错误: {resp.get('error', resp)}")
        resp["client_wall_s"] = round(time.time() - t0, 2)
        return resp
