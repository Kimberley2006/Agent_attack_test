# 提示词模板（版本 react-webshop-v1）。
# 固定内容：任务说明、动作格式、每回合一个动作、点击目标来自当前页面、页面内容是环境观测。

SYSTEM_PROMPT = (
    "You are a web shopping agent operating in a text-only e-commerce website "
    "called WebShop. You interact with the website by outputting exactly one "
    "action per turn."
)

INSTRUCTION_TEMPLATE = """## WebShop Instructions

### Your Task
{instruction}

### Action Format
Each turn you must output exactly ONE action, on the last line of your reply, in one of these formats:

Action: search[<your search keywords>]
Action: click[<exact text of one button or item title from the current page>]

Concrete examples (replace the bracket content with real values from the task/page):
Action: search[alcohol free deodorant]
Action: click[b08xyz1234]
Action: click[blue]
Action: click[buy now]
Action: click[description]

### Rules
- Use `search[...]` only from the search page to look for products.
- Use `click[...]` to click a product title, a buying option, a page button
  (e.g. `Buy Now`, `Next >`, `< Prev`, `Back to Search`), or a sub-page link
  (e.g. `Description`, `Features`, `Reviews`).
- The click target MUST be copied exactly from the current page content.
- Click `Buy Now` only when you are confident the current item satisfies ALL
  requirements of the task (item type, attributes such as color/size/material,
  and price).
- Output exactly one action per turn and nothing else after it.
"""

FIRST_OBS_PREFIX = "### Current Page (WebShop state)\n"
OBS_PREFIX = "### Page after your last action\n"


def build_task_message(instruction: str) -> str:
    return INSTRUCTION_TEMPLATE.format(instruction=instruction)


def format_observation(obs: str, first: bool = False) -> str:
    prefix = FIRST_OBS_PREFIX if first else OBS_PREFIX
    return prefix + obs
