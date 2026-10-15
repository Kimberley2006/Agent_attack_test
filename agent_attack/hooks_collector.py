"""内部表征采集（hooks 版）。

设计对应方案⑥：
- 区分三个关键位置：
    P  = prefill 末 token（完整输入读完、尚未生成本回合输出）
    R  = 推理结束、动作开始前的位置（思考模式下的 </think> 附近；无思考时 == P）
    A* = 动作名及参数的 token 位置（search[/click[ 及其参数）
- 因果对齐原则：预测 token j 的状态位于位置 j-1（即该 token 输入之前），
  anchor 集合 = {P} ∪ {a_0-1 .. a_last-1}（动作首 token 前一个位置起的各状态）。
- basic 档：每层 residual（层输入/层输出）、self_attn 模块输出、MLP(FFN) 输出。
- detailed 档（实验性）：full-attention 层注意力矩阵（需 eager kernel，单独做
  输出一致性验证）、linear-attention(DeltaNet) 层的内部状态（尽力捕获模块
  暴露的状态张量；融合内核未暴露的部分记录为不可用）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


def find_decoder_layers(model):
    """自动定位解码层 ModuleList（兼容多模态包装下的不同路径）。"""
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers:
            candidates.append((name, mod))
    if not candidates:
        raise RuntimeError(f"cannot locate decoder layers (n={n_layers}); "
                           f"top modules: {[n for n, _ in model.named_children()]}")
    # 优先路径中含 language_model 的（避开 vision tower 等）
    candidates.sort(key=lambda x: (0 if "language_model" in x[0] else 1, len(x[0])))
    name, layers = candidates[0]
    return name, layers


def layer_types(model):
    """从 config 读取每层的类型（full_attention / linear_attention）。"""
    text_cfg = getattr(model.config, "text_config", model.config)
    return list(getattr(text_cfg, "layer_types", ["full_attention"] * text_cfg.num_hidden_layers))


def _submodule(layer, key):
    """key: 'attn' | 'mlp'——按类名匹配，兼容实现命名差异。"""
    want = "ATTN" if key == "attn" else ("MLP", "FEEDFORWARD")
    for cname, child in layer.named_children():
        upper = cname.upper() + "|" + type(child).__name__.upper()
        if key == "attn" and "ATTN" in upper and "NORM" not in upper:
            return cname, child
        if key == "mlp" and any(w in upper for w in want):
            return cname, child
    return None, None


@dataclass
class CallCapture:
    call_id: str
    prompt_len: int
    max_new: int
    hidden: int
    n_layers: int
    device: str
    dtype: torch.dtype = torch.float16
    # GPU 缓冲：[n_layers, max_new, H]
    layer_in: Optional[torch.Tensor] = None
    layer_out: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    mlp_out: Optional[torch.Tensor] = None
    # prefill 末 token：{层号: tensor[H]}
    prefill_layer_in: Dict[int, torch.Tensor] = field(default_factory=dict)
    prefill_layer_out: Dict[int, torch.Tensor] = field(default_factory=dict)
    prefill_attn_out: Dict[int, torch.Tensor] = field(default_factory=dict)
    prefill_mlp_out: Dict[int, torch.Tensor] = field(default_factory=dict)
    # detailed 档（尽力捕获）
    attention_weights: Dict = field(default_factory=dict)   # {(layer, step): tensor}
    linear_states: Dict = field(default_factory=dict)       # {(layer, step): [tensor]}
    _gen_pos: int = 0

    def alloc(self):
        n, m, h = self.n_layers, self.max_new, self.hidden
        self.layer_in = torch.zeros((n, m, h), dtype=self.dtype, device=self.device)
        self.layer_out = torch.zeros((n, m, h), dtype=self.dtype, device=self.device)
        self.attn_out = torch.zeros((n, m, h), dtype=self.dtype, device=self.device)
        self.mlp_out = torch.zeros((n, m, h), dtype=self.dtype, device=self.device)

    def free(self):
        for k in ("layer_in", "layer_out", "attn_out", "mlp_out"):
            setattr(self, k, None)
        torch.cuda.empty_cache()


class HooksCollector:
    """在 generate 前后 bracket 一次调用，采集指定档位的内部状态。

    用法：
        collector.begin_call(call_id, prompt_len, max_new)
        out = model.generate(...)
        cap = collector.end_call()          # CallCapture（缓冲仍占 GPU）
        payload = collector.extract(cap, anchors_gen_pos)  # 选 anchor 后释放
    """

    def __init__(self, model, collect: str = "basic", layer_ids=None):
        self.collect = collect
        self.layers_name, self.layers = find_decoder_layers(model)
        self.types = layer_types(model)
        self.n = len(self.layers)
        self.layer_ids = list(range(self.n)) if layer_ids is None else layer_ids
        self.handles = []
        self.model = model
        self.active: Optional[CallCapture] = None
        if collect != "off":
            self._install()

    # ---------------- 安装 ----------------
    def _install(self):
        for i in self.layer_ids:
            layer = self.layers[i]
            self.handles.append(layer.register_forward_hook(
                self._make_layer_hook(i), with_kwargs=False))
            for key in ("attn", "mlp"):
                cname, child = _submodule(layer, key)
                if child is None:
                    continue
                self.handles.append(child.register_forward_hook(
                    self._make_module_hook(i, key), with_kwargs=False))
            if self.collect == "detailed":
                if self.types[i] == "full_attention":
                    self.handles.append(
                        self.layers[i].register_forward_hook(
                            self._make_attn_weights_hook(i)))
                else:
                    self.handles.append(
                        self.layers[i].register_forward_hook(
                            self._make_linear_state_hook(i)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---------------- hooks ----------------
    def _make_layer_hook(self, i):
        def hook(module, inputs, output):
            cap = self.active
            if cap is None:
                return
            hs = inputs[0]           # [B, L, H] residual 进入该层
            out = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:      # prefill（batch=1）
                cap.prefill_layer_in[i] = hs[:, -1].detach().to(cap.dtype).cpu()
                cap.prefill_layer_out[i] = out[:, -1].detach().to(cap.dtype).cpu()
            else:                    # decode step（步号由 StepTicker 维护）
                p = cap._gen_pos
                if p < cap.max_new:
                    cap.layer_in[i, p] = hs[0, 0].to(device=cap.device, dtype=cap.dtype)
                    cap.layer_out[i, p] = out[0, 0].to(device=cap.device, dtype=cap.dtype)
        return hook

    def _make_module_hook(self, i, key):
        buf_attr = "attn_out" if key == "attn" else "mlp_out"
        prefill_buf = f"prefill_{buf_attr}"

        def hook(module, inputs, output):
            cap = self.active
            if cap is None:
                return
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() == 3 and out.shape[1] > 1:
                getattr(cap, prefill_buf)[i] = out[:, -1].detach().to(cap.dtype).cpu()
            elif out.dim() == 3 and out.shape[1] == 1:
                p = cap._gen_pos
                if p < cap.max_new:
                    getattr(cap, buf_attr)[i, p] = out[0, 0].to(
                        device=cap.device, dtype=cap.dtype)
        return hook

    def _make_attn_weights_hook(self, i):
        """detailed：捕获 full-attention 层输出元组中的注意力矩阵（如有）。
        sdpa/flash 内核不返回权重——需要 attn_implementation='eager'，且必须
        单独验证 kernel 更换后的输出一致性（方案⑤）。"""
        def hook(module, inputs, output):
            cap = self.active
            if cap is None:
                return
            tensors = output if isinstance(output, tuple) else (output,)
            for t in tensors:
                if (torch.is_tensor(t) and t.dim() == 4
                        and t.shape[0] == 1 and t.shape[-1] == t.shape[-2]):
                    cap.attention_weights[(i, cap._gen_pos)] = \
                        t.detach().to(cap.dtype).cpu()
                    break
        return hook

    def _make_linear_state_hook(self, i):
        """detailed：尽力捕获 linear-attention(DeltaNet) 层暴露的状态张量。"""
        def hook(module, inputs, output):
            cap = self.active
            if cap is None:
                return
            tensors = output if isinstance(output, tuple) else (output,)
            states = []
            for t in tensors:
                if torch.is_tensor(t) and t.dim() >= 2 and not (
                        t.dim() == 3 and t.shape[-1] == cap.hidden):
                    # 粗过滤：不是 [B, L, H] 形状的 hidden 输出
                    states.append(t.detach().to(cap.dtype).cpu())
            if states:
                cap.linear_states[(i, cap._gen_pos)] = states
        return hook

    # ---------------- 调用生命周期 ----------------
    def begin_call(self, call_id, prompt_len, max_new):
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.active = CallCapture(
            call_id=call_id, prompt_len=prompt_len, max_new=max_new,
            hidden=text_cfg.hidden_size, n_layers=self.n,
            device=self.model.get_input_embeddings().weight.device)
        if self.collect in ("basic", "detailed"):
            self.active.alloc()
        return self._StepTicker(self)

    class _StepTicker:
        """decode 步计数器：generate 每生成一个 token 后调用一次。
        hook 读到的 _gen_pos 恰为当前步序号（步 p 的 forward 在该步 tick 之前）。"""

        def __init__(self, collector):
            self.collector = collector

        def __call__(self, input_ids, scores, **kwargs):
            cap = self.collector.active
            if cap is not None:
                cap._gen_pos += 1
            return False

    def end_call(self) -> CallCapture:
        cap = self.active
        self.active = None
        return cap

    def extract(self, cap: CallCapture, anchors: List[int],
                actual_gen_len: int) -> dict:
        """anchors: 生成序列中的绝对位置（0-based，生成 token 下标）。
        返回可序列化 payload（CPU fp16），随后释放 GPU 缓冲。"""
        payload = {
            "call_id": cap.call_id,
            "n_layers": cap.n_layers,
            "layer_types": self.types,
            "layers_module": self.layers_name,
            "prompt_len": cap.prompt_len,
            "gen_len": actual_gen_len,
            "anchors_gen_pos": anchors,
            "prefill_last": {
                "layer_in": {i: v for i, v in cap.prefill_layer_in.items()},
                "layer_out": {i: v for i, v in cap.prefill_layer_out.items()},
                "attn_out": {i: v for i, v in cap.prefill_attn_out.items()},
                "mlp_out": {i: v for i, v in cap.prefill_mlp_out.items()},
            },
            "anchor_states": [],
        }
        if cap.layer_in is not None:
            for p in anchors:
                if p >= actual_gen_len:
                    continue
                payload["anchor_states"].append({
                    "gen_pos": p,
                    "layer_in": cap.layer_in[:, p].cpu(),
                    "layer_out": cap.layer_out[:, p].cpu(),
                    "attn_out": cap.attn_out[:, p].cpu(),
                    "mlp_out": cap.mlp_out[:, p].cpu(),
                })
        payload["attention_weights"] = {
            f"L{k[0]}_p{k[1]}": v for k, v in cap.attention_weights.items()} \
            if cap.attention_weights else {}
        payload["linear_states"] = {
            f"L{k[0]}_p{k[1]}": v for k, v in cap.linear_states.items()} \
            if cap.linear_states else {}
        payload["detailed_available"] = {
            "attention_weights": bool(cap.attention_weights),
            "linear_states": bool(cap.linear_states),
            "note": "空 = 当前内核未暴露相应内部状态，需更换/扩展内核（见方案⑥）",
        }
        cap.free()
        return payload
