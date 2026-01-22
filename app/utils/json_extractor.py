# -*- coding: utf-8 -*-
"""
JSON提取工具
"""
import json
import re


def extract_safety_report(text: str) -> str:
    """
    从AI响应中提取安全报告内容
    
    Args:
        text: AI响应文本
        
    Returns:
        提取的安全报告内容
    """
    text = text.strip()
    
    # 尝试查找JSON对象
    brace_count = 0
    start_idx = -1
    end_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i
                break
    
    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx + 1]
        try:
            data = json.loads(json_str)
            if "安全报告" in data:
                return data["安全报告"]
        except json.JSONDecodeError:
            pass
    
    # 使用正则表达式提取
    patterns = [
        r'"安全报告"\s*:\s*"([\s\S]*?)"\s*[,\}]',
        r'"安全报告"\s*:\s*"([\s\S]*?)(?:"\s*[,\}]|\s*\}$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace('\\n', '\n')
    
    return text
