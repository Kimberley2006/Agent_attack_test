"""构建 BM25 商品检索索引（在 webshop conda 环境中运行一次）。

用法：
  WEBSHOP_ITEM_PATH=... WEBSHOP_ATTR_PATH=... \
  python scripts/build_bm25_index.py --num-products 1000   # -> search_engine/indexes_1k.pkl
  python scripts/build_bm25_index.py                       # -> search_engine/indexes.pkl (全量)
"""

import argparse
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webshop_repo"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

NAMES = {None: "indexes", 100: "indexes_100", 1000: "indexes_1k",
         100000: "indexes_100k"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-products", type=int, default=None)
    args = ap.parse_args()

    random.seed(233)
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.bm25_searcher import build_index_pickle

    t0 = time.time()
    products, *_ = load_products(
        filepath=os.environ["WEBSHOP_ITEM_PATH"],
        num_products=args.num_products,
        human_goals=True,
    )
    print(f"loaded {len(products)} products in {time.time()-t0:.0f}s")

    name = NAMES[args.num_products]
    out = os.path.join(PROJECT_ROOT, "webshop_repo", "search_engine",
                       f"{name}.pkl")
    t0 = time.time()
    build_index_pickle(products, out)
    print(f"BM25 index -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
