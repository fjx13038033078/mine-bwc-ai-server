# -*- coding: utf-8 -*-
"""
视频分析服务
"""
import os
import json
import logging
import cv2
from typing import Dict, Any, List, Optional

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from ultralytics import YOLO

from app.config import get_settings
from app.utils.json_extractor import extract_safety_report

logger = logging.getLogger(__name__)

# 视频分析提示词
ANALYSIS_PROMPT = '''
你是一位高度专业的铜矿生产视频行为分析专家，同时具备严谨的数据处理能力。
你的任务是逐帧分析视频流，同时识别出视频中的生产行为，每一个事件由一个或多个相关帧组成，代表一个较完整的行为过程。
你必须同时读取叠加在帧图像上的元数据（日期、时间、用户编号、单位编号、序列号），并在事件开始到结束期间记录这些元数据。

任务目标：
1.视频帧与元数据解析：
    视频提取：处理每一帧视频图像，并同时读取叠加在图像上的文本信息。
    元数据提取：精准提取以下五项元数据：日期、时间、用户编号、单位编号、序列号。

2.事件识别与描述：
    事件定义：一个"事件"指的是一个明确开始、过程、结束的行为片段。
    描述：对每一个识别出的事件，生成详细的文字描述。

3.事件记录与关联元数据：
    对每个事件，记录事件的「开始时间」和「结束时间」。
    同时关联该事件中涉及的用户编号、单位编号、序列号以及日期。

4.输出要求：
    将所有事件记录汇总为一个纯净、规范、可被机器直接解析的 JSON 数组。
    若视频中未识别到任何事件，则输出 []。
    每个 JSON 对象必须包含以下字段：
        event_description, date, start_time, end_time, user_number, unit_number, serial_number
'''

# 安全分析提示词
SAFETY_ANALYSIS_PROMPT = '''
你是一名矿山安全生产领域的合规审查专家。请根据输入的作业行为或事件描述，自动识别其中涉及的安全违规行为，匹配对应的矿山安全生产法律法规或管理规定，并按照标准矿山安全违规报告的规范格式输出完整报告。

以JSON格式输出结果：
{
    "安全报告": ""
}
'''


class VideoAnalyzer:
    """视频分析器"""
    
    def __init__(self):
        self.settings = get_settings()
        self._init_models()
    
    def _init_models(self):
        """初始化AI模型"""
        self.vision_model = init_chat_model(
            model=self.settings.vision_model_name,
            model_provider="openai",
            base_url=self.settings.vision_model_url,
            api_key=self.settings.model_api_key,
            timeout=self.settings.model_timeout
        )
        
        self.thinking_model = init_chat_model(
            model=self.settings.thinking_model_name,
            model_provider="openai",
            base_url=self.settings.thinking_model_url,
            api_key=self.settings.model_api_key,
            timeout=self.settings.model_timeout
        )
    
    def _create_yolo_tool(self, video_path: str):
        """创建YOLO检测工具"""
        settings = self.settings
        
        @tool(description='detection')
        def yolo_detection():
            """使用YOLO检测视频中的物体"""
            if not os.path.exists(video_path):
                logger.warning(f"视频文件不存在: {video_path}")
                return ["视频文件不存在，跳过YOLO检测"]
            
            if not os.path.exists(settings.yolo_model_path):
                logger.warning(f"YOLO模型不存在: {settings.yolo_model_path}")
                return ["YOLO模型不存在，跳过检测"]
            
            model = YOLO(settings.yolo_model_path)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps) if fps > 0 else 30
            
            detection_results = []
            frame_count = 0
            second_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                if frame_count % frame_interval == 0:
                    second_count += 1
                    results = model(frame)
                    
                    objects = []
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id]
                        objects.append(class_name)
                    
                    if objects:
                        detection_results.append(
                            f"第{second_count}秒检测到: {', '.join(objects)}"
                        )
                
                frame_count += 1
            
            cap.release()
            return detection_results if detection_results else ["未检测到物体"]
        
        return yolo_detection
    
    async def analyze(
        self, 
        video_url: str, 
        remote_file_path: str,
        use_yolo: bool = False,
        use_safety_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        分析视频
        
        Args:
            video_url: 视频URL（用于AI模型访问）
            remote_file_path: 远程文件路径（用于YOLO检测）
            use_yolo: 是否使用YOLO检测
            use_safety_analysis: 是否进行安全分析
            
        Returns:
            分析结果字典
        """
        logger.info(f"开始分析视频: {video_url}")
        
        # 构建消息
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
            # 创建Agent并分析
            yolo_tool = self._create_yolo_tool(remote_file_path)
            agent = create_agent(
                self.vision_model, 
                [yolo_tool], 
                system_prompt="您可以使用工具来辅助进行视频理解。"
            )
            
            result = agent.invoke({"messages": vision_messages})
            content = result["messages"][-1].content
            
            # 解析JSON结果
            events_data = self._parse_events(content)
            
            # 安全分析
            unsafe_events = []
            if use_safety_analysis and events_data:
                unsafe_events = await self._analyze_safety(events_data)
            
            return {
                "success": True,
                "events": events_data,
                "total_events": len(events_data),
                "unsafe_events": unsafe_events,
                "total_unsafe_events": len(unsafe_events)
            }
            
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise Exception(f"调用AI模型失败: {str(e)}")
    
    def _parse_events(self, content: str) -> List[Dict]:
        """解析事件JSON"""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1]) if len(lines) > 2 else cleaned
        
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, list) else [data] if data else []
        except json.JSONDecodeError:
            logger.warning("无法解析JSON，返回空列表")
            return []
    
    async def _analyze_safety(self, events: List[Dict]) -> List[Dict]:
        """安全分析"""
        unsafe_events = []
        
        for event in events:
            description = event.get('event_description', event.get('事件描述', ''))
            if not description:
                continue
            
            try:
                safety_result = self.thinking_model.invoke([
                    {"role": "user", "content": SAFETY_ANALYSIS_PROMPT + description}
                ])
                
                unsafe_events.append({
                    "event_description": extract_safety_report(safety_result.content),
                    "date": event.get('date', event.get('日期', '')),
                    "start_time": event.get('start_time', event.get('开始时间', '')),
                    "end_time": event.get('end_time', event.get('结束时间', '')),
                    "user_number": event.get('user_number', event.get('用户编号', '')),
                    "unit_number": event.get('unit_number', event.get('单位编号', '')),
                    "serial_number": event.get('serial_number', event.get('序列号', ''))
                })
            except Exception as e:
                logger.warning(f"安全分析失败: {e}")
        
        return unsafe_events
