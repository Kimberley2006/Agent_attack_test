# -*- coding: utf-8 -*-
"""WebShop 环境包装：确定性创建、逐步记录、按动作序列重放。

随机性契约（分支重放的根基，scripts/smoke_env.py 已验证）：
- env 创建期间有三处未播种随机数：区间定价 uniform、goal price_upper 抽样、
  goal 顺序（SimServer 内部的 seed(233) 只覆盖洗牌之后的流）。
  因此本包装器在创建 WebAgentTextEnv 之前必须 random.seed(config.seed)，
  之后两个独立实例的 goals / 价格表 / 观测逐字节一致。
- 交互期无随机源：session 必须传 int（goals[session_int] 确定性选任务），
  string session 会走 random_idx 随机选 goal，禁止使用。
- 同一实例内重放前必须清掉 server.user_sessions 中的旧 session
  （否则 asins/actions 计数跨重放累积，观测不再一致），见 reset_fresh。
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WEBSHOP_REPO = Path(__file__).resolve().parents[1] / "webshop_repo"


def ensure_webshop_importable(webshop_repo: Path = DEFAULT_WEBSHOP_REPO) -> None:
    repo = str(Path(webshop_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


@dataclass
class EnvConfig:
    seed: int = 42
    num_products: int = 1000
    human_goals: bool = True          # True: items_human_ins.json（13 个）；
                                      # False: 属性合成 goals（量大，用于扩规模）
    observation_mode: str = "text"
    session_prefix: str = "task"
    get_image: bool = False
    num_prev_obs: int = 0             # 历史由 agent loop 自管，env 不拼接
    num_prev_actions: int = 0
    items_file: str | None = None     # None → WebShop 默认（支持环境变量覆盖）
    show_attrs: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class StepResult:
    obs: str
    url: str
    action: str | None = None
    reward: float | None = None
    done: bool = False


class WebShopEnvWrapper:
    """以确定性方式驱动 WebAgentTextEnv 并支持同实例隔离重放。"""

    def __init__(self, config: EnvConfig | None = None,
                 webshop_repo: Path = DEFAULT_WEBSHOP_REPO):
        self.config = config or EnvConfig()
        ensure_webshop_importable(webshop_repo)
        # 播种必须在首次 import webshop 模块之前：import 期（spacy/gym 等）
        # 会消耗随机流，先导入后播种会使价格/goal 整体漂移。
        # 服务顺序复现已验证：seed(233)+先播种后导入 → 130/130 与冻结
        # task_registry 一致（scripts/probe_goal_repro.py）。
        random.seed(self.config.seed)
        from web_agent_site.envs import WebAgentTextEnv  # 延迟导入，需 webshop 环境

        kwargs = dict(
            filter_goals=None,
            limit_goals=-1,
            num_products=self.config.num_products,
            human_goals=1 if self.config.human_goals else 0,
            get_image=self.config.get_image,
            num_prev_obs=self.config.num_prev_obs,
            num_prev_actions=self.config.num_prev_actions,
            session_prefix=self.config.session_prefix,
            show_attrs=self.config.show_attrs,
        )
        if self.config.items_file:
            kwargs["file_path"] = self.config.items_file

        self.env = WebAgentTextEnv(
            observation_mode=self.config.observation_mode, **kwargs
        )
        self.goals = self.env.server.goals
        self._products = self.env.server.product_item_dict
        self._prices = self.env.server.product_prices

    # ------------------------------------------------------------------ 查询

    @property
    def n_goals(self) -> int:
        return len(self.goals)

    def goal(self, goal_idx: int) -> dict:
        return self.goals[goal_idx]

    def product(self, asin: str) -> dict:
        return self._products[asin]

    def price(self, asin: str) -> float:
        return self._prices[asin]

    def product_table(self) -> dict:
        """{asin: {title, price, available}}，价格取自本 env 实例的播种抽样结果。

        位置挖掘必须用这张表（而不是 positions.build_product_table 的
        独立重算），保证与产生轨迹的 env 价格完全一致。
        """
        return {
            asin: {
                "title": p.get("Title", ""),
                "price": self._prices[asin],
                "available": True,
            }
            for asin, p in self._products.items()
        }

    def available_actions(self) -> list[str]:
        return list(self.env.get_available_actions()["clickables"])

    # ------------------------------------------------------------- 交互与重放

    def _session_key(self, goal_idx: int) -> str:
        return f"{self.config.session_prefix}{goal_idx}"

    def reset_fresh(self, goal_idx: int) -> StepResult:
        """清掉旧 session 后 reset，保证同实例内多次重放互不污染。"""
        self.env.server.user_sessions.pop(self._session_key(goal_idx), None)
        obs, _ = self.env.reset(session=goal_idx)   # reset 返回 (obs, None)
        return StepResult(obs=obs, url=self.env.browser.current_url)

    def step(self, action: str) -> StepResult:
        obs, reward, done, _ = self.env.step(action)
        return StepResult(
            obs=obs, url=self.env.browser.current_url,
            action=action, reward=reward, done=done,
        )

    def replay(self, goal_idx: int, actions: list[str]) -> list[StepResult]:
        """重放动作序列，返回逐步结果（含 reset，长度 = len(actions)+1）。

        分支运行器用它重建共同状态：第 k 步观测对上后停止，
        后续动作由分支自己的模型生成。StepResult.url 用于 asin/页面识别。
        """
        results = [self.reset_fresh(goal_idx)]
        for a in actions:
            results.append(self.step(a))
        return results
