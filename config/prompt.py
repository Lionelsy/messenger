"""
提示词模板
"""

from __future__ import annotations
import json


SYSTEM_CN_JSON = """你是一个严谨的学术论文助手。
你必须基于输入文本回答，不要编造不存在的信息。
输出必须是严格合法的 JSON（不要 markdown，不要多余解释文字）。"""

SYSTEM_CN_RELEVANCE = """你是一个判断论文是否与研究方向相关的科研助手。
你必须只输出“是”或“否”，不要输出其它任何内容。"""


def build_user_prompt_step03_summary_cn(summary: str) -> str:
    """
    参考 Step03_query_GPT.py：把摘要结构化为中文 JSON（单层 key-value）。
    """
    s = (summary or "").replace("\n", " ").strip()
    return f"""
请将以下论文摘要分点用中文总结，避免使用数学符号，并以 JSON 格式输出，每个要点对应一个键值对。

摘要：{s}

输出 JSON 的 key 必须严格为：
{{
  "总结": "内容",
  "背景": "内容",
  "目的": "内容",
  "方法": "内容",
  "主要发现": "内容",
  "结论": "内容",
  "翻译": "内容"
}}

规则：
- 必须输出严格合法 JSON（不要 markdown、不要代码块）。
- 如果摘要里没有提到某项，请填 "unknown"。
""".strip()


def build_user_prompt_step03_relevance_cn(summary: str, interest_description: str) -> str:
    """
    参考 Step03_query_GPT.py：只判断是否高度相关，输出“是/否”。
    """
    s = (summary or "").replace("\n", " ").strip()
    interest = (interest_description or "").strip() or "3D场景表示、理解、智能"
    return f"""
你正在筛选论文是否与你的研究方向高度相关。

研究方向：{interest}

请判断以下论文摘要是否与你的研究方向高度相关，仅回复“是”或“否”：

摘要：{s}
""".strip()


def build_user_prompt_step03_deep_cn(paper_title: str, paper_text: str) -> str:
    """
    参考 Step03_query_GPT.py：对正文进行“深度解读”，输出 5 个问题的 JSON。
    """
    title = (paper_title or "").strip()
    text = (paper_text or "").strip()
    return f"""
你是一个优秀的学术论文解读助手，请通读并分析以下论文的原始内容，帮我回答以下问题。

要求：
- 请使用简洁、准确、通俗的语言解释，并尽量避免使用公式、符号或缩写。
- 请以 JSON 格式输出，其中每个问题作为 key，每个回答作为对应的 value。
- JSON 的 key 必须严格等于下面 5 个问题文本，且不能有任何额外的 key。
- 如果文中没有明确说明，请对应字段写 "unknown"，不要编造。
- 每个回答尽量简短（建议 2-4 句），不要列长清单，不要输出数组/嵌套对象。

论文标题：《{title}》

论文内容（部分或全部）如下：
{text}

请根据内容，回答以下问题：

1. 这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？
2. 作者是如何思考并设计出这个方法的？是否有借鉴现有工作？
3. 这个方法的核心思想是什么？整体实现流程是怎样的？
4. 论文的关键创新点有哪些？相比之前的工作，有什么不同？
5. 如果要用一句话总结这篇论文的贡献，你会怎么说？

请用标准的 JSON 格式输出，如：
{{
  "这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？": "回答1内容",
  "作者是如何思考并设计出这个方法的？是否有借鉴现有工作？": "回答2内容",
  "这个方法的核心思想是什么？整体实现流程是怎样的？": "回答3内容",
  "论文的关键创新点有哪些？相比之前的工作，有什么不同？": "回答4内容",
  "如果要用一句话总结这篇论文的贡献，你会怎么说？": "回答5内容"
}}
""".strip()


def build_user_prompt_step03_deep_fix_cn(required_q_keys: list[str], bad_output_text: str) -> str:
    """
    当模型没有按要求输出 key 时，用于“格式修正”：强制转换成 required_q_keys 对应的 JSON。
    """
    schema = "{\n" + "\n".join([f'  \"{k}\": \"...\"' for k in required_q_keys]) + "\n}"
    return f"""
把下面的内容转换成严格合法 JSON 对象，并满足：
- JSON 的 key 必须且只能是下面这 5 个问题文本（不允许新增/改写 key）
- value 用中文回答；若原内容缺信息则写 "unknown"
- 每个回答尽量简短（建议 2-4 句），不要列长清单，不要输出数组/嵌套对象。
- 只输出 JSON（不要任何额外文字）

5 个 key（必须逐字一致）：
{json.dumps(required_q_keys, ensure_ascii=False, indent=2)}

输出格式示例：
{schema}

待转换内容：
{bad_output_text}
""".strip()
