# Agent_Attack — Clean Baseline 实验框架

目标：建立 WebShop 环境下 Qwen3.8-27B Agent 的 clean baseline
（正常行为 + 对应内部表征），供后续 prompt-injection 攻击实验做配对对比。

## 布局

```
configs/baseline_v1.yaml     冻结实验配置（①）
agent_attack/                实验框架包
  config.py                  配置加载/快照（每次运行复制进 run 目录）
  prompts.py                 提示模板（版本化，react-webshop-v1）
  action_parser.py           动作解析（只看思考块之后的最终输出区）
  env_client.py              环境服务 HTTP 客户端
  model_backend.py           模型加载 + 生成 + token 对齐
  hooks_collector.py         内部表征采集（⑥ basic/detailed 两档）
  episode_logger.py          JSONL 日志（⑦ 全字段）
  agent.py                   ReAct 决策循环（③）
scripts/
  env_service.py             WebShop 环境服务（webshop 环境，独立进程）
  build_bm25_index.py        构建商品检索索引（一次性）
  manual_check.py            无模型购物流程验证（②）
  freeze_tasks.py            冻结任务集 debug/dev/baseline（④）
  run_baseline.py            主入口
  verify_hook_consistency.py 采集不影响行为的验证（⑤）
  compute_metrics.py         指标计算（⑧）
  verify_replay.py           重放验证（⑨）
webshop_repo/                官方 WebShop 克隆 + 已记录补丁（见 PATCHES.md）
task_registry.json           冻结的任务集合
/cache/wuhan/                大文件：模型权重 / 商品数据 / 运行输出
```

## 启动流程

```bash
# 0. 环境服务（webshop conda 环境，独立进程）
WEBSHOP_ITEM_PATH=/cache/wuhan/webshop_data/items_shuffle.json \
WEBSHOP_ATTR_PATH=/cache/wuhan/webshop_data/items_ins_v2.json \
WEBSHOP_HUMAN_INS_PATH=/temp/wuhan/Agent_Attack/webshop_repo/data/items_human_ins.json \
/home/ma-user/conda-envs/webshop/bin/python scripts/env_service.py --port 5001 &

# 1. 无模型验证购物流程
python scripts/manual_check.py

# 2. 冻结任务集
python scripts/freeze_tasks.py

# 3. 调试集（先关采集确认交互，再开 basic 采集对比）
CUDA_VISIBLE_DEVICES=1 python scripts/run_baseline.py --split debug --collect off
CUDA_VISIBLE_DEVICES=1 python scripts/run_baseline.py --split debug --collect basic

# 4. 指标与重放
python scripts/compute_metrics.py /cache/wuhan/runs/<run_id>
python scripts/verify_replay.py /cache/wuhan/runs/<run_id>
```

## 关键设计决定

- **环境与推理分离**：WebShop（旧依赖）在 webshop conda 环境（python3.10）
  中以 HTTP 服务运行；模型在 base 环境（transformers 5.19）本地加载装 hooks。
- **检索后端**：rank_bm25 替换 pyserini/Lucene（无 Java），文档字段与官方一致，
  见 webshop_repo/PATCHES.md。
- **任务确定性**：服务启动即 `random.seed(233)`（补丁 3），goal 列表跨重启稳定；
  任务集冻结含数据文件 sha256；agent 每次 reset 核对 instruction 漂移。
- **采集两档**：basic=全部任务（各层 prefill 末 token + 动作 token 区的
  residual/attn/FFN 输出）；detailed=少量任务（attention 权重需 eager 内核，
  DeltaNet 状态尽力捕获内核暴露部分，未暴露的记录为不可用）。
- **token 对齐**：预测动作 token j 的状态取位置 j-1（因果对齐），anchor 区间
  = [action_first-1, action_last-1]，另存 prefill 末位置（与推理结束位置区分）。
- **指标口径**：WebShop Score=平均 reward（未购买记 0）；SR=reward==1.0 比例
  （论文口径）；greedy 解码为实验控制配置，不得表述为官方推荐性能。
# Agent_attack_test
