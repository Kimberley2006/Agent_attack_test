"""Qwen3.8-27B 推理后端（base 环境，transformers 5.19）。

- greedy 生成（实验控制），原生思考模式（enable_thinking）可配置；
- 每次调用返回：文本、prompt/output token ids、思考/动作 token 区间、
  基础档表征 payload（若开启采集）；
- 动作 token 定位采用逐前缀解码匹配，保证 span 与 token 一一对齐。
"""

import dataclasses
import time
from typing import List, Optional

import torch


@dataclasses.dataclass
class GenerationResult:
    call_id: str
    prompt_text: str
    output_text: str
    prompt_ids: List[int]
    output_ids: List[int]
    finish_reason: str            # stop | length
    n_prompt_tokens: int
    n_output_tokens: int
    gen_seconds: float
    think_span: Optional[list]    # [start, end) 生成序列内 token 下标
    action_span: Optional[list]   # 动作区（Action: ... 行）token 下标区间
    anchors: List[int]            # 表征对齐用 anchor 位置（见 hooks_collector）
    reps_payload: Optional[dict] = None


class ModelBackend:
    def __init__(self, cfg, collect: str = "off"):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        m = cfg.model
        self.model_path = m.model_path
        dtype = getattr(torch, m.dtype)
        kwargs = {}
        if m.attn_implementation and collect != "detailed":
            kwargs["attn_implementation"] = m.attn_implementation
        elif collect == "detailed":
            # detailed 注意力采集需要 eager；输出一致性须单独验证（方案⑤）
            kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForImageTextToText.from_pretrained(
            m.model_path, dtype=dtype, device_map=m.device,
            trust_remote_code=m.trust_remote_code, **kwargs)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(m.model_path)

        g = cfg.generation
        self.max_new_tokens = g.max_new_tokens
        self.enable_thinking = g.enable_thinking
        self.reasoning_effort = g.reasoning_effort
        self.gen_kwargs = dict(do_sample=False, max_new_tokens=g.max_new_tokens)
        self._eos = set(self.model.generation_config.eos_token_id
                        if isinstance(self.model.generation_config.eos_token_id, list)
                        else [self.model.generation_config.eos_token_id])

        from agent_attack.hooks_collector import HooksCollector
        self.collector = HooksCollector(self.model, collect=collect) \
            if collect != "off" else None
        self.collect = collect

    # ---------------- token 区间定位 ----------------
    def _locate_span_by_prefix_decode(self, output_ids, substr: str):
        """在生成序列中定位 substr 的 token 区间 [s, e)。
        逐前缀解码：找到最小 s 使 decode(ids[:s]) 已包含 substr 开头匹配。
        返回 None 表示未找到。"""
        if not substr:
            return None
        text = self.tokenizer.decode(output_ids)
        char_start = text.find(substr)
        if char_start < 0:
            return None
        target = text[:char_start + len(substr)]
        # 二分：decode(ids[:m]) 的长度随 m 单调不减
        lo, hi = 1, len(output_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(self.tokenizer.decode(output_ids[:mid])) >= len(target):
                hi = mid
            else:
                lo = mid + 1
        end = lo  # 最小前缀使解码文本覆盖 substr 结尾
        # 回找 start：最小 s 使 decode(ids[:s]) 文本长度 > char_start
        lo2, hi2 = 0, end
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            if len(self.tokenizer.decode(output_ids[:mid])) > char_start:
                hi2 = mid
            else:
                lo2 = mid + 1
        return [lo2, end]

    def _think_span(self, output_ids):
        # Qwen3.5 模板会在生成提示里自动开启 <think>，因此思考从输出 token 0
        # 开始；输出中通常只有闭合标记 </think>。
        close = self._locate_span_by_prefix_decode(output_ids, "</think>")
        if close is not None:
            return [0, close[1]]
        open_ = self._locate_span_by_prefix_decode(output_ids, "<think>")
        if open_ is not None:
            return [open_[0], len(output_ids)]   # 未闭合的思考
        return None                              # 无思考标记

    # ---------------- 生成 ----------------
    def generate(self, messages: List[dict], call_id: str):
        """生成并返回 (GenerationResult, capture)。动作 span 与表征提取在
        动作解析之后由 finalize() 完成（解析发生在生成之后）。"""
        tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not self.enable_thinking:
            tpl_kwargs["enable_thinking"] = False
        elif self.reasoning_effort != "native":
            tpl_kwargs["reasoning_effort"] = self.reasoning_effort
        prompt_text = self.tokenizer.apply_chat_template(messages, **tpl_kwargs)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

        ticker = None
        if self.collector is not None:
            ticker = self.collector.begin_call(call_id, len(prompt_ids),
                                               self.max_new_tokens)
        t0 = time.time()
        with torch.inference_mode():
            gen_kwargs = dict(self.gen_kwargs)
            if ticker is not None:
                gen_kwargs["stopping_criteria"] = [ticker]
            out = self.model.generate(
                input_ids=torch.tensor([prompt_ids], device=self.model.device),
                pad_token_id=self.tokenizer.pad_token_id
                             or self.model.generation_config.pad_token_id,
                **gen_kwargs)
        gen_seconds = time.time() - t0
        output_ids = out[0][len(prompt_ids):].tolist()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        n_out = len(output_ids)
        finish_reason = "stop" if (n_out and output_ids[-1] in self._eos) else "length"

        res = GenerationResult(
            call_id=call_id, prompt_text=prompt_text, output_text=output_text,
            prompt_ids=prompt_ids, output_ids=output_ids,
            finish_reason=finish_reason, n_prompt_tokens=len(prompt_ids),
            n_output_tokens=n_out, gen_seconds=gen_seconds,
            think_span=self._think_span(output_ids),
            action_span=None, anchors=[])
        cap = self.collector.end_call() if self.collector is not None else None
        return res, cap

    def finalize(self, res: GenerationResult, cap, action_substring: Optional[str]):
        """动作解析后调用：定位动作 token 区间、计算 anchors、提取表征。"""
        output_ids = res.output_ids
        if action_substring:
            final_zone_start = res.think_span[1] if res.think_span else 0
            final_ids = output_ids[final_zone_start:]
            span = self._locate_span_by_prefix_decode(final_ids, action_substring)
            if span:
                res.action_span = [final_zone_start + span[0],
                                   final_zone_start + span[1]]
        if cap is not None:
            if res.action_span and res.action_span[0] > 0:
                # 预测动作 token j 的状态位于 j-1：覆盖 [a0-1, a_last-1]；
                # 再往前多取一个位置，覆盖动作名 token 可能与前导空格合并的情况
                # （如 ' search'），保证动作名自身的预测状态也在 anchors 内。
                lo = max(0, res.action_span[0] - 2)
                res.anchors = list(range(lo, res.action_span[1]))
            else:
                res.anchors = [len(output_ids) - 1] if output_ids else []
            res.reps_payload = self.collector.extract(
                cap, res.anchors, len(output_ids))
        return res
