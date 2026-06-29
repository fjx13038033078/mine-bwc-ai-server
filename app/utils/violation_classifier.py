# -*- coding: utf-8 -*-
"""
违规事件判定工具

视觉模型返回的事件列表中，可能同时包含：
- 真实违规描述（如「未佩戴安全帽」）
- 合规/无违规描述（如「未发现违规…正确佩戴安全帽」）

旧逻辑把「任意非空事件」都视为违规，且用「安全帽」关键词误判「未佩戴安全帽」。
本模块用于在后处理阶段区分真实违规与合规描述。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

# 明确无违规 / 合规（优先匹配）
_NO_VIOLATION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"未发现违规",
        r"无违规",
        r"没有违规",
        r"不存在违规",
        r"未检测到违规",
        r"未发现任何违规",
        r"未出现违规",
        r"符合(?:安全)?规范",
        r"操作规范",
        r"作业规范",
        r"正确佩戴",
        r"规范佩戴",
        r"已佩戴",
        r"正确穿戴",
        r"规范穿戴",
        r"有序操作",
        r"在安全(?:护栏|范围内)",
    )
]

# 明确违规关键词
_VIOLATION_KEYWORD_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"未佩戴",
        r"未戴",
        r"没戴",
        r"没带",
        r"未带",
        r"未穿",
        r"未系",
        r"睡岗",
        r"脱岗",
        r"违规操作",
        r"违章",
        r"吸烟",
        r"烟火",
        r"存在违规",
        r"发现违规",
        r"涉嫌违规",
        r"有违规",
        r"违规行为",
    )
]


def is_violation_event(description: Optional[str]) -> bool:
    """根据事件描述判断是否为真实违规（而非「未发现违规/合规」类描述）。"""
    if not description or not str(description).strip():
        return False

    text = str(description).strip()

    for pattern in _NO_VIOLATION_PATTERNS:
        if pattern.search(text):
            return False

    for pattern in _VIOLATION_KEYWORD_PATTERNS:
        if pattern.search(text):
            return True

    # 单独出现「违规」且未被上面的无违规规则排除
    if "违规" in text:
        return True

    return False


def extract_violation_type(description: Optional[str]) -> Optional[str]:
    """从已确认违规的事件描述中提取违规类型。"""
    if not description or not str(description).strip():
        return None

    text = str(description).strip()

    if re.search(r"睡岗|脱岗", text):
        return "睡岗"
    if "安全帽" in text and re.search(r"未佩戴|未戴|没戴|没带|未带", text):
        return "未佩戴安全帽"
    if "护目镜" in text and re.search(r"未佩戴|未戴|没戴|没带|未带", text):
        return "未佩戴护目镜"
    if re.search(r"反光衣|工作服|防护服", text) and re.search(r"未穿|未戴|未佩戴", text):
        return "未穿戴防护服"
    if re.search(r"违规操作|违章操作", text):
        return "违规操作"
    if re.search(r"吸烟|烟火", text):
        return "吸烟/烟火"

    return "安全违规"


def filter_violation_events(
    events: Optional[List[Union[Dict[str, Any], Any]]],
    description_key: str = "event_description",
) -> List[Any]:
    """从事件列表中筛出真实违规事件。支持 dict 或带 event_description 属性的对象。"""
    if not events:
        return []

    result: List[Any] = []
    for event in events:
        if isinstance(event, dict):
            desc = event.get(description_key, "")
        else:
            desc = getattr(event, "event_description", "") or ""
        if is_violation_event(desc):
            result.append(event)
    return result
