# -*- coding: utf-8 -*-
"""
视频流分析服务
直接从预签名URL流式读取视频，不下载到本地
"""
import os
import json
import logging
import cv2
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

from app.config import get_settings
from app.utils.json_extractor import extract_safety_report
from app.models.schemas import VideoTaskMessage, VideoTaskResult, EventInfo

logger = logging.getLogger(__name__)

# 视频分析提示词
ANALYSIS_PROMPT = '''
你是一位高度专业的铜矿生产视频行为分析专家，同时具备严谨的数据处理能力。
你的任务是逐帧分析视频流，同时识别出视频中的生产行为，每一个事件由一个或多个相关帧组成，代表一个较完整的行为过程。

任务目标：
1.视频帧与元数据解析：处理每一帧视频图像，精准提取元数据（日期、时间、用户编号、单位编号、序列号）。
2.事件识别与描述：识别并描述视频中的行为事件。
3.事件记录与关联元数据：记录事件的开始时间和结束时间，关联相关元数据。
4.输出要求：将所有事件记录汇总为JSON数组，若未识别到任何事件则输出[]。

每个JSON对象必须包含以下字段：
event_description, date, start_time, end_time, user_number, unit_number, serial_number
'''

SAFETY_ANALYSIS_PROMPT = '''
你是一名矿山安全生产领域的合规审查专家。请根据输入的作业行为或事件描述，自动识别其中涉及的安全违规行为，匹配对应的矿山安全生产法律法规或管理规定，并按照标准矿山安全违规报告的规范格式输出完整报告。

以JSON格式输出结果：
{
    "安全报告": ""
}
'''

# 线程池：用于执行阻塞的CPU/GPU密集型操作
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_analyzer")


class StreamVideoAnalyzer:
    """
    流式视频分析器
    直接从预签名URL读取视频流，不下载到本地磁盘
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._vision_model = None
        self._thinking_model = None
        self._yolo_model = None
    
    @property
    def vision_model(self):
        """懒加载视觉模型"""
        if self._vision_model is None:
            self._vision_model = init_chat_model(
                model=self.settings.vision_model_name,
                model_provider="openai",
                base_url=self.settings.vision_model_url,
                api_key=self.settings.model_api_key,
                timeout=self.settings.model_timeout
            )
        return self._vision_model
    
    @property
    def thinking_model(self):
        """懒加载思考模型"""
        if self._thinking_model is None:
            self._thinking_model = init_chat_model(
                model=self.settings.thinking_model_name,
                model_provider="openai",
                base_url=self.settings.thinking_model_url,
                api_key=self.settings.model_api_key,
                timeout=self.settings.model_timeout
            )
        return self._thinking_model
    
    def _create_stream_yolo_tool(self, video_url: str):
        """
        创建流式YOLO检测工具
        直接从URL读取视频流，不下载到本地
        """
        settings = self.settings
        
        @tool(description='detection')
        def yolo_detection():
            """使用YOLO检测视频中的物体（流式读取）"""
            logger.info(f"[YOLO] 开始流式读取视频: {video_url[:100]}...")
            
            # 检查YOLO模型
            if not os.path.exists(settings.yolo_model_path):
                logger.warning(f"YOLO模型不存在: {settings.yolo_model_path}")
                return ["YOLO模型不存在，跳过检测"]
            
            try:
                from ultralytics import YOLO
                model = YOLO(settings.yolo_model_path)
                
                # 直接从URL流式读取视频（核心：不下载到本地）
                cap = cv2.VideoCapture(video_url)
                
                if not cap.isOpened():
                    logger.error(f"无法打开视频流: {video_url[:100]}...")
                    return ["无法打开视频流"]
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps) if fps > 0 else 30
                
                detection_results = []
                frame_count = 0
                second_count = 0
                max_seconds = 60  # 最多处理60秒
                
                while cap.isOpened() and second_count < max_seconds:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    if frame_count % frame_interval == 0:
                        second_count += 1
                        results = model(frame, verbose=False)
                        
                        objects = []
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = results[0].names[class_id]
                            objects.append(class_name)
                        
                        if objects:
                            detection_results.append(
                                f"第{second_count}秒检测到: {', '.join(set(objects))}"
                            )
                    
                    frame_count += 1
                
                cap.release()
                logger.info(f"[YOLO] 流式检测完成，处理了{second_count}秒视频")
                
                return detection_results if detection_results else ["未检测到特定物体"]
                
            except Exception as e:
                logger.error(f"YOLO检测失败: {e}")
                return [f"YOLO检测失败: {str(e)}"]
        
        return yolo_detection
    
    def analyze_from_stream(self, task: VideoTaskMessage) -> VideoTaskResult:
        """
        从预签名URL流式分析视频
        
        Args:
            task: 视频任务消息
            
        Returns:
            VideoTaskResult: 处理结果
        """
        import time
        start_time = time.time()
        
        logger.info(f"[分析] 开始处理任务: taskId={task.task_id}, videoId={task.video_id}")
        logger.info(f"[分析] 预签名URL: {task.presigned_url[:80]}...")
        
        try:
            # 构建消息
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": task.presigned_url}},
                        {"type": "text", "text": ANALYSIS_PROMPT}
                    ]
                }
            ]
            
            # 创建YOLO工具（使用预签名URL）
            yolo_tool = self._create_stream_yolo_tool(task.presigned_url)
            
            # 创建Agent并分析
            agent = create_agent(
                self.vision_model,
                [yolo_tool],
                system_prompt="您可以使用工具来辅助进行视频理解。"
            )
            
            result = agent.invoke({"messages": vision_messages})
            content = result["messages"][-1].content
            
            # 解析事件
            events_data = self._parse_events(content)
            
            # 安全分析
            unsafe_events = self._analyze_safety_sync(events_data)
            
            process_time = time.time() - start_time
            logger.info(f"[分析] 任务完成: taskId={task.task_id}, 耗时={process_time:.2f}s, 事件数={len(events_data)}")
            
            # 如果有违规事件，尝试捕获违规帧
            violation_frame = None
            violation_timestamp = None
            if unsafe_events:
                # 尝试从第一个违规事件中提取时间戳
                first_unsafe = unsafe_events[0]
                start_time_str = first_unsafe.get('start_time', first_unsafe.get('开始时间', ''))
                violation_timestamp = self._parse_time_to_seconds(start_time_str)
                
                # 尝试捕获违规帧
                try:
                    cap = cv2.VideoCapture(task.presigned_url)
                    if cap.isOpened() and violation_timestamp is not None:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            frame_number = int(violation_timestamp * fps)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        success_read, violation_frame = cap.read()
                        if not success_read:
                            violation_frame = None
                    cap.release()
                except Exception as e:
                    logger.warning(f"捕获违规帧失败: {e}")
            
            return VideoTaskResult(
                task_id=task.task_id,
                video_id=task.video_id,
                success=True,
                events=[EventInfo(**e) for e in events_data] if events_data else None,
                unsafe_events=[EventInfo(**e) for e in unsafe_events] if unsafe_events else None,
                process_time=process_time,
                violation_frame=violation_frame,
                violation_timestamp=violation_timestamp
            )
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"[分析] 任务失败: taskId={task.task_id}, error={e}")
            
            return VideoTaskResult(
                task_id=task.task_id,
                video_id=task.video_id,
                success=False,
                error_message=str(e),
                process_time=process_time
            )
    
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
    
    def _analyze_safety_sync(self, events: List[Dict]) -> List[Dict]:
        """同步安全分析"""
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
    
    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        将时间字符串解析为秒数
        支持格式: HH:MM:SS, MM:SS, SS
        """
        if not time_str:
            return None
        
        try:
            parts = time_str.strip().split(':')
            if len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 1:  # SS
                return float(parts[0])
            return None
        except (ValueError, IndexError):
            return None


# 全局分析器实例
_analyzer: Optional[StreamVideoAnalyzer] = None


def get_stream_analyzer() -> StreamVideoAnalyzer:
    """获取流式分析器单例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = StreamVideoAnalyzer()
    return _analyzer


def get_executor() -> ThreadPoolExecutor:
    """获取线程池"""
    return _executor
