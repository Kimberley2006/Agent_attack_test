"""② 无模型手动验证：脚本化走通完整购物流程。

重置任务 → 搜索 → 点击商品 → 查看描述 → 选择规格 → 购买 → 读取终局。
核对：页面观测与状态一致、动作可执行、终局可读出商品ID/规格/真实价格、
奖励与终止状态正常返回。在 webshop 环境进程之外通过服务接口执行。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_attack.config import load_config  # noqa: E402
from agent_attack.env_client import EnvClient  # noqa: E402


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--session", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = EnvClient(cfg.get("environment.service_url"))

    ok = True
    h = env.health()
    ok &= check("service healthy", h.get("ok") and h.get("n_goals", 0) > 0,
                f"goals={h.get('n_goals')} search={h.get('search_backend')} "
                f"docs={h.get('bm25_doc_n')}")

    tasks = env.list_tasks()["tasks"]
    t = tasks[args.session]
    print(f"task {t['goal_idx']}: {t['instruction_text'][:120]}")
    print(f"  truth: asin={t['asin']} attrs={t['attributes']} "
          f"opts={t['goal_options']} price_upper={t['price_upper']}")

    # 1. 重置
    r = env.reset(args.session)
    ok &= check("reset returns instruction",
                r["instruction_text"].strip() == t["instruction_text"].strip())
    ok &= check("reset page is search/index",
                r["page_type"] == "index", r["page_type"])
    ok &= check("search bar available",
                r["available_actions"]["has_search_bar"])

    # 2. 搜索（用任务 query 的关键词）
    kw = " ".join(t["query"].split()[:4]) or t["name"][:30]
    s = env.step(args.session, f"search[{kw}]")
    ok &= check("search executes", not s["invalid"] and
                s["page_type"] == "search_results", f"kw={kw!r}")
    ok &= check("results page has clickables",
                len(s["available_actions"]["clickables"]) > 0)

    # 3. 点击第一个商品链接（clickable 中非按钮的长文本通常是商品标题）
    links = [c for c in s["available_actions"]["clickables"]
             if c not in ("buy now", "next >", "< prev", "back to search",
                          "description", "features", "reviews")]
    ok &= check("product links present", len(links) > 0)
    if links:
        pick = links[0]
        c = env.step(args.session, f"click[{pick}]")
        ok &= check("click product -> item_page",
                    c["page_type"] == "item_page" and not c["invalid"],
                    f"pick={pick[:50]!r}")
        # 4. 描述子页（官方行为：子页无 Buy Now，需 < prev 返回）
        if "description" in c["available_actions"]["clickables"]:
            d = env.step(args.session, "click[description]")
            ok &= check("description sub-page reachable",
                        "description" in (d.get("url") or "").lower() and
                        d["page_type"] == "item_sub_page")
            c = env.step(args.session, "click[< prev]")
            ok &= check("back to item page",
                        c["page_type"] == "item_page" and not c["invalid"])
        # 5. 选规格（商品有 radio 选项时；此商品可能没有）
        std = {"buy now", "next >", "< prev", "back to search",
               "description", "features", "reviews"}
        opts = [x for x in c["available_actions"]["clickables"]
                if x not in std and x not in links]
        chosen = []
        for opt in opts[:2]:
            o = env.step(args.session, f"click[{opt}]")
            if not o["invalid"]:
                chosen.append(opt)
        print(f"  options clicked: {chosen}")
        # 6. 购买（在商品页点击 Buy Now）
        b = env.step(args.session, "click[buy now]")
        ok &= check("buy now terminates", b["done"] and b["page_type"] == "done")
        p = b.get("purchased") or {}
        ok &= check("purchase readable: asin", bool(p.get("asin")), str(p.get("asin")))
        ok &= check("purchase readable: price",
                    p.get("price") is not None, str(p.get("price")))
        ok &= check("reward returned", p.get("reward") is not None,
                    f"reward={p.get('reward')}")
        ok &= check("verbose reward breakdown present",
                    isinstance(p.get("verbose_info"), dict))
        print(f"  purchased: asin={p.get('asin')} price={p.get('price')} "
              f"reward={p.get('reward')}")
        print(f"  verbose_info: {json.dumps(p.get('verbose_info'), default=str)[:300]}")

    # 7. 无效动作不被执行
    env.close_session(args.session)
    r2 = env.reset(args.session)
    bad = env.step(args.session, "click[this button does not exist]")
    ok &= check("invalid click rejected, page unchanged",
                bad["invalid"] and bad["page_type"] == "index")
    bad2 = env.step(args.session, "dance[around]")
    ok &= check("unknown action type rejected", bad2["invalid"])
    env.close_session(args.session)

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
