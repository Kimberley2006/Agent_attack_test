# -*- coding: utf-8 -*-
"""观测拦截器：把攻击/负例文本作为商品描述区段中的追加文字写入观测。

WebShop text 模式观测是 ' [SEP] '.join(可见文本节点)。商品描述子页
（description_page.html）恰好产生 5 个文本节点：

    Instruction: [SEP] <任务指令> [SEP] Back to Search [SEP] < Prev [SEP] <描述正文>

描述正文是最后一个区段，因此"描述区追加" = 在观测末尾（描述文本之后）插入，
前面的所有字节保持不变——这一点由 verify_only_append() 在运行时断言。

本模块只改观测字符串：不接触 messages 角色、商品价格、动作接口与奖励计算。
"""
from __future__ import annotations

from dataclasses import dataclass

from .payloads import PayloadValidationError, validate_text

SEP = " [SEP] "

# 商品描述子页的固定结构：5 段，第 3、4 段是两个返回按钮
_SUBPAGE_LEN = 5
_BACK_TO_SEARCH = "Back to Search"
_PREV = "< Prev"


@dataclass
class InjectionResult:
    obs: str               # 改写后的观测
    payload_text: str      # 实际写入的文本（负例含中性填充）
    padded: bool           # 是否做了长度填充（只允许负例为 True）
    span: tuple[int, int]  # 写入文本在 obs 中的字符区间（闭开）


def parse_segments(obs: str) -> list[str]:
    return obs.split(SEP)


def classify_page(obs: str) -> str:
    """按 text 模式结构粗分类页面，供位置挖掘与运行时断言使用。

    返回 'description_like'（5 段子页结构）、'item_page'（含价格与 Buy Now）、
    或 'other'。features/reviews 子页与 description 子页结构相同，
    具体是哪个子页由导致该页面的动作（click[description] 等）判定。
    """
    segs = parse_segments(obs)
    if (
        len(segs) == _SUBPAGE_LEN
        and segs[0] == "Instruction:"
        and segs[2] == _BACK_TO_SEARCH
        and segs[3] == _PREV
    ):
        return "description_like"
    if any(s.startswith("Price:") for s in segs) and "Buy Now" in segs:
        return "item_page"
    return "other"


def assert_description_page(obs: str, prev_action: str | None = None) -> None:
    """确认观测是描述子页。prev_action 提供时同时校验动作一致（click[description]）。"""
    if classify_page(obs) != "description_like":
        raise ValueError(f"观测不是商品子页结构（段数/按钮不符），拒绝注入")
    if prev_action is not None:
        act = prev_action.strip().lower()
        if act != "click[description]":
            raise ValueError(f"进入该页面的动作是 {prev_action!r}，不是 click[description]")


def find_description_region(obs: str) -> tuple[int, int]:
    """返回描述正文（最后一个 [SEP] 区段）的字符区间。

    clean/负例分支取同一区间作为"语义对应位置"，供表征对齐比较使用。
    """
    segs = parse_segments(obs)
    prefix = SEP.join(segs[:-1])
    start = len(prefix) + len(SEP) if len(segs) > 1 else 0
    return start, len(obs)


# ---------------------------------------------------------------------------
# item_page 入口（v1.1）：agent 自然行为几乎不进 description 子页
# （45 条 clean 轨迹 0 次描述子页合格访问），注入入口改为商品主页面。
# 追加位置 = 观测末尾（Buy Now 之后）新起一段，前缀字节保持不变。
# ---------------------------------------------------------------------------

def assert_item_page(obs: str) -> None:
    if classify_page(obs) != "item_page":
        raise ValueError(f"观测不是商品主页面结构（无 Price:/Buy Now），拒绝注入")


def inject_into_item_page(obs: str, text: str) -> InjectionResult:
    """在商品主页面观测末尾追加一个新 [SEP] 段（页面通知横幅语义）。"""
    validate_text(text)
    assert_item_page(obs)
    segs = parse_segments(obs)
    if segs[-1] == "":
        span_start = len(obs)
        modified = obs + text
    else:
        span_start = len(obs) + len(SEP)
        modified = obs + SEP + text
    # 结构等价性自查：原有段全部保持不变，仅在末尾新增一段
    assert parse_segments(modified)[: len(segs)] == segs
    return InjectionResult(
        obs=modified, payload_text=text, padded=False,
        span=(span_start, len(modified)),
    )


def inject_into_description(obs: str, text: str, joiner: str = " ") -> InjectionResult:
    """在描述正文末尾追加 text，返回改写结果与写入区间。

    描述为空串时（观测以 [SEP] 结尾），text 直接成为描述区内容，
    不额外加 joiner，避免把它变成独立的第 6 段。
    """
    validate_text(text)
    assert_description_page(obs)
    segs = parse_segments(obs)
    last = segs[-1]
    if last == "":
        span_start = len(obs)
        modified = obs + text
    else:
        span_start = len(obs) + len(joiner)
        modified = obs + joiner + text
    # 结构等价性自查：只有最后一段发生了变化
    assert parse_segments(modified)[:-1] == segs[:-1]
    return InjectionResult(
        obs=modified, payload_text=text, padded=False, span=(span_start, len(modified))
    )


def verify_only_append(original: str, modified: str, text: str, joiner: str = " ") -> None:
    """运行时断言：modified 只是在 original 末尾（描述区内）追加了 text。"""
    if modified == original + joiner + text:
        return
    if modified == original + text:  # 描述为空串的分支
        return
    raise AssertionError(
        "改写越界：modified 不是 original 的纯追加。\n"
        f"  original 末尾: {original[-60:]!r}\n  modified 末尾: {modified[-60:]!r}"
    )


# ---------------------------------------------------------------------------
# 困难负例的中性长度填充
# ---------------------------------------------------------------------------

# 填充句必须是"无指令性的正常内容"，且自身能通过 validate_text。
_FILLERS = {
    "zh": [
        "以上内容为页面展示信息。",
        "商品信息以本页显示为准。",
        "本段文字用于页面说明。",
    ],
    "en": [
        "The above is page display information.",
        "Product information is as shown on this page.",
        "This paragraph is for page description purposes.",
    ],
}


def neutral_pad(text: str, target_len: int, lang: str = "zh") -> str:
    """循环追加中性填充句，直到长度不低于 target_len。确定性，无随机。"""
    if lang not in _FILLERS:
        raise ValueError(f"不支持的语言: {lang}")
    out = text
    i = 0
    while len(out) < target_len:
        out = f"{out} {_FILLERS[lang][i % len(_FILLERS[lang])]}"
        i += 1
        if i > 50:  # 防御性上限，正常参数不会触发
            break
    validate_text(out)
    return out


def render_branch_observation(
    obs: str,
    payload_item: dict,
    pad_target: int | None = None,
    lang: str = "zh",
    entry: str = "description",
) -> InjectionResult:
    """按 manifest 条目渲染分支观测。

    硬约束：进入本函数的条目必须与冻结 manifest 一致（重验 sha256）；
    攻击载荷永不填充（逐字节保持冻结文本）；仅困难负例允许在
    pad_target 给出时做中性填充。entry: description=描述子页追加（v1.0），
    item_page=商品主页面末尾追加新段（v1.1）。
    """
    import hashlib

    kind = payload_item.get("kind")
    text = payload_item["text"]
    expected_sha = payload_item.get("sha256")
    if expected_sha is not None:
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected_sha:
            raise PayloadValidationError(
                f"[{payload_item.get('payload_id')}] 文本与冻结 manifest 不符，拒绝注入"
            )
    padded = False
    if kind == "hard_negative" and pad_target is not None and len(text) < pad_target:
        text = neutral_pad(text, pad_target, lang=lang)
        padded = True
    if entry == "item_page":
        result = inject_into_item_page(obs, text)
    else:
        result = inject_into_description(obs, text)
    result.padded = padded
    return result


def default_pad_target(manifest: dict, lang: str) -> int:
    """取该语言 arm 攻击载荷的平均字符长度作为负例填充目标。"""
    lens = [
        i["char_len"]
        for i in manifest["arms"][lang]["items"]
        if i["kind"] == "attack"
    ]
    return round(sum(lens) / len(lens))
