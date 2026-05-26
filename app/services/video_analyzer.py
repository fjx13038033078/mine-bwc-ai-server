# -*- coding: utf-8 -*-
"""
视频分析服务
"""
import json
import logging
from typing import Dict, Any, List

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# 视频分析提示词
ANALYSIS_PROMPT = '''
    你是一位高度专业的铜矿生产视频行为分析专家，同时具备严谨的数据处理能力。
    你的任务是检查视频中是否有违规行为，同时读取叠加在帧图像上的元数据（日期、时间、用户编号、单位编号、序列号），并在违规行为开始到结束期间记录这些元数据。
    任务目标：
    1.视频帧与元数据解析：
        视频提取：处理每一帧视频图像，并同时读取叠加在图像上的文本信息。
        元数据提取：精准提取以下五项元数据。
            日期：视频拍摄的日期。
            时间：视频拍摄的具体时间。
            用户编号：视频中的用户编号。
            单位编号：视频中的单位编号。
            序列号：视频中的序列号。
    2.事件记录与关联元数据：
        对每个违规行为，记录事件的「开始时间」和「结束时间」,同时关联该行为中涉及的用户编号、单位编号、序列号以及日期。
    3.输出要求：
        将违规行为记录为可被机器直接解析的 JSON 数组，无需任何 Markdown 代码块标记。
        若视频中未识别到任何事件，则每个字段都返回无。
        每个 JSON 对象代表一个事件，必须包含以下字段：
            event_description： 对违规行为的详细说明。
            date： 事件发生日期（YYYY-MM-DD）。
            start_time： 事件发生时间（HH:MM:SS）。
            end_time： 事件结束时间（HH:MM:SS）。
            user_number： 涉事用户的编号。
            unit_number： 涉事单位的编号。
            serial_number： 相关的设备或视频序列号。
            start_second: 事件在视频第几秒开始。
            end_second: 事件在视频第几秒结束。
        JSON 对象结构示例如下：
            [
                {
                "event_description": "",
                "date": "",
                "start_time": "",
                "end_time": "",
                "user_number": "",
                "unit_number": "",
                "serial_number": ""，
                "start_second": "",
                "end_second": ""
                }
            ]
    '''


class VideoAnalyzer:
    """视频分析器"""

    def __init__(self):
        self.settings = get_settings()
        self._init_models()

    def _init_models(self):
        """初始化视觉模型"""
        self.vision_model = openAI(
            base_url=self.settings.vision_model_url,
            api_key=self.settings.model_api_key,
            timeout=self.settings.model_timeout
        )

    async def analyze(self, video_url: str) -> Dict[str, Any]:
        """
        分析视频

        Args:
            video_url: 视频URL（用于AI模型访问）

        Returns:
            分析结果字典
        """
        logger.info(f"开始分析视频: {video_url}")

        vision_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": ANALYSIS_PROMPT}
                ]
            }
        ]

        try:
            response = self.vision_model.chat.completions.create(
                model=self.settings.vision_model_name,
                messages=vision_messages,
                max_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                },
            )
            events_data = self._parse_events(response.choices[0].message.content)

            return {
                "success": True,
                "events": events_data,
                "total_events": len(events_data),
            }

        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise Exception(f"调用AI模型失败: {str(e)}")

    def _strip_thinking_content(self, content: str) -> str:
        """去掉模型思考内容，只保留 </think> 之后的输出结果"""
        for end_tag in ("</think>", ""):
            if end_tag in content:
                content = content.split(end_tag, 1)[1]
                break
        return content.strip()

    def _parse_events(self, content: str) -> List[Dict]:
        """解析事件JSON"""
        cleaned = _strip_thinking_content(content)
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            data = json.loads(cleaned)
            return data if isinstance(data, list) else [data] if data else []
        except json.JSONDecodeError:
            logger.warning("无法解析JSON，返回空列表")
            return []
