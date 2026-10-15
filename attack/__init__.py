# -*- coding: utf-8 -*-
"""attack 包：配对注入实验（WebShop + Qwen3.8-27B）。

模块规划（按第一版协议）：
- payloads : 攻击载荷与困难负例的构造、校验、冻结      （已完成）
- inject   : 观测拦截器，仅在描述区段追加文本          （已完成）
- positions: 从 clean 轨迹挖掘配对分支点               （已完成）
- branch_runner / delivery / labeling / probes / analysis （待实现）
"""
from .payloads import (
    ATTACK_FAMILIES,
    HARD_NEGATIVES,
    PayloadRecord,
    PayloadValidationError,
    build_payloads,
    default_new_budget,
    freeze_manifest,
    length_stats,
    load_manifest,
    save_manifest,
    validate_text,
    verify_manifest,
)
from .inject import (
    InjectionResult,
    assert_description_page,
    classify_page,
    default_pad_target,
    find_description_region,
    inject_into_description,
    neutral_pad,
    parse_segments,
    render_branch_observation,
    verify_only_append,
)
from .positions import (
    BranchPosition,
    TaskMineResult,
    build_product_table,
    mine_all,
    mine_trajectory,
    parse_price_ceiling,
)

__all__ = [
    # payloads
    "ATTACK_FAMILIES",
    "HARD_NEGATIVES",
    "PayloadRecord",
    "PayloadValidationError",
    "build_payloads",
    "default_new_budget",
    "freeze_manifest",
    "length_stats",
    "load_manifest",
    "save_manifest",
    "validate_text",
    "verify_manifest",
    # inject
    "InjectionResult",
    "assert_description_page",
    "classify_page",
    "default_pad_target",
    "find_description_region",
    "inject_into_description",
    "neutral_pad",
    "parse_segments",
    "render_branch_observation",
    "verify_only_append",
    # positions
    "BranchPosition",
    "TaskMineResult",
    "build_product_table",
    "mine_all",
    "mine_trajectory",
    "parse_price_ceiling",
]
