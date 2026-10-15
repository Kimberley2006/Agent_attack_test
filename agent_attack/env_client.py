"""HTTP 客户端：与独立部署的 WebShop 环境服务通信（webshop conda 环境）。"""

import json
import urllib.request


class EnvServiceError(RuntimeError):
    pass


def _post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


class EnvClient:
    def __init__(self, service_url: str):
        self.base = service_url.rstrip("/")
        self.service_info = None

    def health(self) -> dict:
        return _get(f"{self.base}/health")

    def list_tasks(self) -> dict:
        return _get(f"{self.base}/tasks", timeout=1800)

    def reset(self, session: int) -> dict:
        """重置到指定任务（goal index）。返回 obs / instruction / available_actions。"""
        return _post(f"{self.base}/reset", {"session": int(session)})

    def step(self, session: int, action: str) -> dict:
        """执行动作。返回 obs/reward/done/available_actions/invalid/url/purchased。"""
        return _post(f"{self.base}/step", {"session": int(session), "action": action})

    def close_session(self, session: int) -> dict:
        return _post(f"{self.base}/close_session", {"session": int(session)})
