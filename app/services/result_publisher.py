# -*- coding: utf-8 -*-
"""
结果消息发布服务
将AI检测结果发送到 RabbitMQ，供 Java 端消费
"""
import json
import logging
from typing import Optional, List

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractConnection, AbstractChannel, AbstractExchange

from app.config import get_settings
from app.models.schemas import VideoTaskResult, VideoAnalysisResultMessage, EventInfo
from app.services.minio_client import upload_screenshot, capture_violation_frame
from app.utils.violation_classifier import extract_violation_type, filter_violation_events, is_violation_event

logger = logging.getLogger(__name__)


class ResultPublisher:
    """
    结果消息发布者
    负责将AI检测结果发送到 video.result.exchange
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._connection: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._exchange: Optional[AbstractExchange] = None
    
    async def connect(self) -> None:
        """建立 RabbitMQ 连接"""
        try:
            url = (
                f"amqp://{self.settings.rabbitmq_user}:{self.settings.rabbitmq_password}"
                f"@{self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}"
                f"{self.settings.rabbitmq_vhost}"
            )
            
            logger.info(f"结果发布者正在连接 RabbitMQ: {self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}")
            
            self._connection = await aio_pika.connect_robust(url)
            self._channel = await self._connection.channel()
            
            # 声明结果交换机
            self._exchange = await self._channel.declare_exchange(
                self.settings.result_exchange,
                ExchangeType.TOPIC,
                durable=True
            )
            logger.info(f"结果发布者连接成功，交换机: {self.settings.result_exchange}")
            
        except Exception as e:
            logger.error(f"结果发布者连接失败: {e}")
            raise
    
    async def ensure_connected(self) -> None:
        """确保连接有效"""
        if self._connection is None or self._connection.is_closed:
            await self.connect()
    
    async def publish_result(
        self,
        task_result: VideoTaskResult,
        video_url: Optional[str] = None
    ) -> bool:
        """
        发布检测结果到 RabbitMQ
        
        Args:
            task_result: 视频任务处理结果
            video_url: 原始视频URL（用于截图）
        
        Returns:
            是否发送成功
        """
        try:
            await self.ensure_connected()
            
            # 处理截图上传
            screenshot_url = None
            # 真实违规事件（过滤「未发现违规/合规」类描述）
            violation_events = filter_violation_events(task_result.unsafe_events)
            if not violation_events and task_result.events:
                violation_events = filter_violation_events(task_result.events)

            has_violation = bool(violation_events)
            violation_type = None
            ai_description = None

            violation_start_second = None
            violation_end_second = None
            if has_violation:
                # 提取违规类型、描述及起止时间点
                violation_type, ai_description = self._extract_violation_info(violation_events)
                first_unsafe = violation_events[0]
                violation_start_second = getattr(first_unsafe, 'start_second', None)
                violation_end_second = getattr(first_unsafe, 'end_second', None)
                
                # 上传截图
                if task_result.violation_frame is not None:
                    success, screenshot_url = upload_screenshot(
                        frame=task_result.violation_frame,
                        video_id=task_result.video_id,
                        task_id=task_result.task_id,
                        timestamp=task_result.violation_timestamp
                    )
                    if not success and video_url:
                        # 如果没有预存的帧，尝试从视频URL捕获
                        frame = capture_violation_frame(video_url, task_result.violation_timestamp or 0)
                        if frame is not None:
                            _, screenshot_url = upload_screenshot(
                                frame=frame,
                                video_id=task_result.video_id,
                                task_id=task_result.task_id,
                                timestamp=task_result.violation_timestamp
                            )
            
            # 无论有无违规，都尽量填 ai_description，让前端能展示分析报告
            if not ai_description:
                if task_result.events:
                    # 用 vision 模型解析出的事件描述
                    descs = [e.event_description for e in task_result.events if e.event_description]
                    if descs:
                        ai_description = "\n".join(descs)
                if not ai_description and task_result.raw_analysis:
                    # 最终兜底：用原始分析文本（截取前 2000 字）
                    ai_description = task_result.raw_analysis[:2000]

            # 构建结果消息
            result_message = VideoAnalysisResultMessage(
                task_id=task_result.task_id,
                video_id=task_result.video_id,
                clip_id=task_result.clip_id,
                status="SUCCESS" if task_result.success else "FAILED",
                has_violation=has_violation,
                violation_type=violation_type,
                ai_description=ai_description,
                violation_start_second=violation_start_second,
                violation_end_second=violation_end_second,
                screenshot_url=screenshot_url,
                events_json=self._events_to_json(task_result.events or task_result.unsafe_events),
                process_time=task_result.process_time,
                error_message=task_result.error_message
            )
            
            # 转换为Java端期望的格式
            message_body = json.dumps(result_message.to_java_dict(), ensure_ascii=False)
            
            # 发送消息
            message = Message(
                body=message_body.encode('utf-8'),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type="application/json"
            )
            
            await self._exchange.publish(
                message,
                routing_key=self.settings.result_routing_key
            )
            
            logger.info(
                f"结果消息发送成功: taskId={task_result.task_id}, "
                f"videoId={task_result.video_id}, clipId={task_result.clip_id}, "
                f"status={result_message.status}, hasViolation={has_violation}"
            )
            return True
            
        except Exception as e:
            logger.error(f"发送结果消息失败: taskId={task_result.task_id}, error={e}", exc_info=True)
            return False
    
    def _extract_violation_info(self, unsafe_events: Optional[List[EventInfo]]) -> tuple:
        """从违规事件中提取违规类型和描述"""
        if not unsafe_events:
            return None, None

        # 取第一个真实违规事件作为主要违规
        first_event = unsafe_events[0]
        description = first_event.event_description or ""
        violation_type = extract_violation_type(description) or "安全违规"

        # 合并所有违规描述及对应规章制度
        parts = []
        for e in unsafe_events:
            if not e.event_description or not is_violation_event(e.event_description):
                continue
            part = e.event_description
            if e.regulations:
                part = f"{part}\n【相关规章制度】\n{e.regulations}"
            parts.append(part)
        ai_description = "\n\n".join(parts) if parts else description

        return violation_type, ai_description
    
    def _events_to_json(self, events: Optional[List[EventInfo]]) -> Optional[str]:
        """将事件列表转换为JSON字符串"""
        if not events:
            return None
        try:
            events_data = [e.model_dump() for e in events]
            return json.dumps(events_data, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"事件序列化失败: {e}")
            return None
    
    async def close(self) -> None:
        """关闭连接"""
        if self._channel:
            await self._channel.close()
            self._channel = None
        if self._connection:
            await self._connection.close()
            self._connection = None
        logger.info("结果发布者连接已关闭")


# 全局实例
_publisher: Optional[ResultPublisher] = None


def get_result_publisher() -> ResultPublisher:
    """获取结果发布者单例"""
    global _publisher
    if _publisher is None:
        _publisher = ResultPublisher()
    return _publisher


async def publish_analysis_result(
    task_result: VideoTaskResult,
    video_url: Optional[str] = None
) -> bool:
    """
    便捷函数：发布分析结果
    
    Args:
        task_result: 任务结果
        video_url: 视频URL（用于截图）
    
    Returns:
        是否成功
    """
    publisher = get_result_publisher()
    return await publisher.publish_result(task_result, video_url)
