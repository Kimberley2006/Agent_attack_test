"""WebShop 环境服务（在 webshop conda 环境中运行）。

用法：
  WEBSHOP_ITEM_PATH=... WEBSHOP_ATTR_PATH=... WEBSHOP_HUMAN_INS_PATH=... \
  python scripts/env_service.py --port 5001 [--num-products N]

- 单进程串行服务：SimBrowser 共享单一浏览器状态，同一时刻只处理一个任务；
  客户端协议为 reset → step* → close_session。
- 关键可复现性补丁：构造环境前先 random.seed(233)，使 get_human_goals 中的
  价格上限采样确定化（官方代码在 goal 构建后才 seed(233)，进程重启会漂移）。
"""

import argparse
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webshop_repo"))

from flask import Flask, jsonify, request  # noqa: E402

app = Flask(__name__)

STATE = {"env": None, "info": {}}

PAGE_TYPES = ["search_results", "item_page", "item_sub_page", "done"]


def get_env():
    return STATE["env"]


def classify_page(url):
    if url is None:
        return "index"
    for t in PAGE_TYPES:
        if t in url:
            if t == "item_sub_page":
                # URL 布局 .../item_sub_page/{session}/{asin}/{kw}/{page}/{name}/{options}
                # 子页名取倒数第二段（末段是 options JSON）
                segs = url.rstrip("/").split("/")
                return f"item_sub_page_{segs[-2].lower()}"
            return t
    return "index"


def build_env(num_products, human_goals):
    # 必须在导入 engine/goal 并构建 env 之前固定种子（见文件头说明）
    random.seed(233)
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(
        observation_mode="text",
        human_goals=human_goals,
        num_products=num_products,
    )
    return env


@app.route("/health", methods=["GET"])
def health():
    env = get_env()
    return jsonify({
        "ok": True,
        "n_goals": len(env.server.goals),
        "num_products": STATE["info"].get("num_products"),
        "product_library": STATE["info"].get("item_path"),
        "search_backend": type(env.server.search_engine).__name__,
        "bm25_doc_n": getattr(env.server.search_engine, "_asins", None)
                      and len(env.server.search_engine._asins),
        "server_started_at": STATE["info"].get("started_at"),
    })


@app.route("/tasks", methods=["GET"])
def tasks():
    """枚举全部 goal（评估端数据：真实商品与约束，不提供给模型）。"""
    env = get_env()
    out = []
    for i, g in enumerate(env.server.goals):
        out.append({
            "goal_idx": i,
            "instruction_text": g["instruction_text"],
            "asin": g["asin"],
            "attributes": g["attributes"],
            "goal_options": g["goal_options"],
            "price_upper": g["price_upper"],
            "query": g["query"],
            "name": g["name"],
        })
    return jsonify({"n": len(out), "tasks": out})


def _session_payload(env, session_key):
    sess = env.server.user_sessions.get(session_key, {})
    purchased = None
    if sess.get("done"):
        purchased = {
            "asin": sess.get("asin"),
            "options": {k: v for k, v in (sess.get("options") or {}).items()},
            "price": env.server.product_prices.get(sess.get("asin")),
            "price_tag": env.server.product_item_dict[sess["asin"]]["Price"]
                         if sess.get("asin") else None,
            "reward": sess.get("reward"),
            "verbose_info": sess.get("verbose_info"),
        }
    return purchased


@app.route("/reset", methods=["POST"])
def reset():
    env = get_env()
    session = int(request.json["session"])
    obs, _ = env.reset(session=session)
    avail = env.get_available_actions()
    # HTML 模板会给指令加 "Instruction:" 前缀，剥掉以与 /tasks 的 goal 文本一致
    instr = env.instruction_text
    if instr.lower().startswith("instruction:"):
        instr = instr.split(":", 1)[1].strip()
    sess_key = env.session
    STATE["info"]["last_reset"] = time.time()
    return jsonify({
        "session": sess_key,
        "goal_idx": session,
        "instruction_text": instr,
        "obs": obs,
        "available_actions": avail,
        "url": env.browser.current_url,
        "page_type": classify_page(env.browser.current_url),
    })


@app.route("/step", methods=["POST"])
def step():
    env = get_env()
    session = int(request.json["session"])
    action = request.json["action"]

    if env.session != str(session):
        return jsonify({"error": f"env.session={env.session} != {session}; "
                                 "must reset first (sequential single-task use)"}), 409

    # 官方语义下的可执行性判定（env.step 对无效动作静默 no-op）：
    #   search: 参数非空即可执行（官方不检查 has_search_bar）
    #   click : lower(arg) 必须在当前页 clickable 集合内
    import re
    m = re.match(r"(.+)\[(.+)\]", action)
    action_name, action_arg = (m.groups() if m else (action, None))
    if action_arg is not None:
        action_arg = action_arg.lower()
    avail = env.get_available_actions()
    if action_name == "search":
        invalid = (action_arg is None or action_arg == "")
    elif action_name == "click":
        invalid = not (action_arg in env.text_to_clickable and action_arg != "search")
    else:
        invalid = True

    obs, reward, done, info = env.step(action)
    avail = env.get_available_actions()
    purchased = _session_payload(env, str(session))
    return jsonify({
        "session": str(session),
        "action": action,
        "invalid": invalid,
        "obs": obs,
        "reward": reward,
        "done": done,
        "available_actions": avail,
        "url": env.browser.current_url,
        "page_type": classify_page(env.browser.current_url),
        "purchased": purchased,
    })


@app.route("/close_session", methods=["POST"])
def close_session():
    env = get_env()
    session = str(request.json["session"])
    removed = env.server.user_sessions.pop(session, None) is not None
    return jsonify({"closed": removed})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--num-products", type=int, default=None)
    ap.add_argument("--human-goals", type=int, default=1)
    args = ap.parse_args()

    print(f"[env_service] item_path={os.environ.get('WEBSHOP_ITEM_PATH')}")
    print(f"[env_service] attr_path={os.environ.get('WEBSHOP_ATTR_PATH')}")
    print(f"[env_service] human_ins_path={os.environ.get('WEBSHOP_HUMAN_INS_PATH')}")
    STATE["info"] = {
        "num_products": args.num_products,
        "item_path": os.environ.get("WEBSHOP_ITEM_PATH"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.time()
    STATE["env"] = build_env(args.num_products, args.human_goals)
    print(f"[env_service] env built in {time.time()-t0:.1f}s, "
          f"goals={len(STATE['env'].server.goals)}")
    # threaded=True：避免单个 keep-alive 连接阻塞整个单线程服务
    # （环境串行语义由客户端协议保证：reset → step* → close_session）
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
