# 决策动作解析：只针对最终动作输出区域（思考块之后），取最后一个匹配。
# 设计约束（对应方案③）：
#   - 动作格式固定为 Action: search[...] / Action: click[...]
#   - 点击目标必须来自当前页面（环境侧会拒绝不在 clickable 集合内的点击）
#   - 解析失败不猜测、不修复，返回 None 由主循环记日志并给固定错误反馈

import re

ACTION_RE = re.compile(r"^\s*(?:Action\s*:\s*)?(search|click)\[([^\[\]]*)\]",
                       re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
# 生成可能带尾部特殊标记（<|im_end|> 等），解析时剥掉
SPECIAL_TAIL_RE = re.compile(r"(<\|[^|>]+\|>)+\s*$")


def strip_thinking(text: str) -> str:
    """去掉思考区域，只保留最终输出区。 tolerate 未闭合的 think 块。"""
    text = THINK_BLOCK_RE.sub("", text)
    text = OPEN_THINK_RE.sub("", text)
    # Qwen3.5 可能使用不同思考标记；若存在 </think> 但无开头，取其后内容
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_action(model_output: str):
    """返回 (action_str, parse_info)。

    action_str: 'search[keywords]' 或 'click[name]'，参数保持原样（由环境 lower-case）；
                解析失败返回 None。
    parse_info: dict(candidates=[...], chosen_index=..., thinking_detected=bool)
    """
    final_zone = strip_thinking(model_output)
    candidates = []
    for lineno, raw_line in enumerate(final_zone.splitlines()):
        line = SPECIAL_TAIL_RE.sub("", raw_line).rstrip()
        m = ACTION_RE.match(line)
        if m:
            name, arg = m.group(1).lower(), m.group(2).strip()
            candidates.append((lineno, f"{name}[{arg}]"))
    parse_info = {
        "candidates": [c[1] for c in candidates],
        "n_candidates": len(candidates),
        "thinking_detected": len(THINK_BLOCK_RE.findall(model_output)) > 0
                             or "<think>" in model_output or "</think>" in model_output,
    }
    if not candidates:
        return None, parse_info
    # 多个候选时取最后一个：模型被要求只输出一个动作，末尾的是最终决定
    lineno, action = candidates[-1]
    parse_info["chosen_line"] = lineno
    return action, parse_info
