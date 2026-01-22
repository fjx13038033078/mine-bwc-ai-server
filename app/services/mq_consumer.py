# -*- coding: utf-8 -*-
"""
RabbitMQ 消费者服务
使用 aio_pika 实现异步消息消费
"""
import asyncio
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

import aio_pika
from aio_pika import ExchangeType, Message
from aio_pika.abc import AbstractIncomingMessage, AbstractConnection, AbstractChannel

from app.config import get_settings
from app.models.schemas import VideoTaskMessage, VideoTaskResult
from app.services.stream_analyzer import get_stream_analyzer, get_executor
from app.services.result_publisher import publish_analysis_result

logger = logging.getLogger(__name__)


class MQConsumer:
    """
    RabbitMQ 消费者
    负责从队列中消费视频检测任务并处理
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._connection: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._is_running = False
    
    async def connect(self) -> None:
        """建立 RabbitMQ 连接"""
        try:
            # 构建连接URL
            url = (
                f"amqp://{self.settings.rabbitmq_user}:{self.settings.rabbitmq_password}"
                f"@{self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}"
                f"{self.settings.rabbitmq_vhost}"
            )
            
            logger.info(f"正在连接 RabbitMQ: {self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}")
            
            # 建立连接
            self._connection = await aio_pika.connect_robust(url)
            
            # 创建通道
            self._channel = await self._connection.channel()
            
            # 设置 QoS (prefetch_count=1: 同时只处理一个任务，防止GPU显存溢出)
            await self._channel.set_qos(prefetch_count=self.settings.mq_prefetch_count)
            
            logger.info(f"RabbitMQ 连接成功，QoS prefetch_count={self.settings.mq_prefetch_count}")
            
        except Exception as e:
            logger.error(f"RabbitMQ 连接失败: {e}")
            raise
    
    async def setup_queue(self) -> aio_pika.Queue:
        """声明交换机和队列，并完成绑定"""
        if not self._channel:
            raise RuntimeError("未建立 RabbitMQ 连接")
        
        # 声明 Topic 类型的持久化交换机
        exchange = await self._channel.declare_exchange(
            self.settings.mq_exchange,
            ExchangeType.TOPIC,
            durable=True
        )
        logger.info(f"声明交换机: {self.settings.mq_exchange} (Topic, Durable)")
        
        # 声明持久化队列（参数必须与Java端一致，否则会报PRECONDITION_FAILED）
        queue = await self._channel.declare_queue(
            self.settings.mq_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",  # 与Java端一致
                "x-dead-letter-routing-key": f"{self.settings.mq_queue}.dlq",  # 与Java端一致
                "x-max-length": 10000,  # 队列最大长度
            }
        )
        logger.info(f"声明队列: {self.settings.mq_queue} (Durable, with DLQ)")
        
        # 绑定队列到交换机
        await queue.bind(exchange, routing_key=self.settings.mq_routing_key)
        logger.info(f"队列绑定完成: {self.settings.mq_queue} -> {self.settings.mq_exchange} (routing_key={self.settings.mq_routing_key})")
        
        return queue
    
    async def process_message(self, message: AbstractIncomingMessage) -> None:
        """
        处理单条消息
        
        核心逻辑：
        1. 解析消息为 VideoTaskMessage
        2. 使用线程池执行阻塞的AI推理
        3. 根据结果发送 ACK/NACK
        """
        task_id = "unknown"
        
        try:
            # 1. 解析消息
            body = message.body.decode('utf-8')
            logger.info(f"收到消息: {body[:200]}...")
            
            data = json.loads(body)
            task = VideoTaskMessage(**data)
            task_id = task.task_id
            
            logger.info(f"解析任务成功: taskId={task.task_id}, videoId={task.video_id}")
            
            # 2. 使用线程池执行阻塞的AI推理（防止阻塞事件循环）
            loop = asyncio.get_event_loop()
            analyzer = get_stream_analyzer()
            executor = get_executor()
            
            result: VideoTaskResult = await loop.run_in_executor(
                executor,
                analyzer.analyze_from_stream,
                task
            )
            
            # 3. 处理结果
            if result.success:
                logger.info(
                    f"任务处理成功: taskId={task_id}, "
                    f"events={len(result.events) if result.events else 0}, "
                    f"unsafe_events={len(result.unsafe_events) if result.unsafe_events else 0}, "
                    f"耗时={result.process_time:.2f}s"
                )
                
                # 发送结果到 Java 端
                await publish_analysis_result(result, task.presigned_url)
                
                # 发送 ACK
                await message.ack()
                logger.info(f"消息已确认(ACK): taskId={task_id}")
                
            else:
                logger.error(f"任务处理失败: taskId={task_id}, error={result.error_message}")
                
                # 即使失败也发送结果（通知Java端任务失败）
                await publish_analysis_result(result, None)
                
                # 发送 NACK，不重新入队（避免死循环）
                await message.nack(requeue=False)
                logger.warning(f"消息已拒绝(NACK): taskId={task_id}, requeue=False")
            
        except json.JSONDecodeError as e:
            logger.error(f"消息解析失败(JSON格式错误): {e}")
            # JSON格式错误，不重新入队
            await message.nack(requeue=False)
            
        except Exception as e:
            logger.error(f"消息处理异常: taskId={task_id}, error={e}", exc_info=True)
            # 发送 NACK，不重新入队
            await message.nack(requeue=False)
    
    async def start_consuming(self) -> None:
        """开始消费消息"""
        if not self._channel:
            await self.connect()
        
        queue = await self.setup_queue()
        
        self._is_running = True
        logger.info("开始消费消息...")
        
        # 开始消费
        await queue.consume(self.process_message)
    
    async def stop(self) -> None:
        """停止消费并关闭连接"""
        self._is_running = False
        
        if self._channel:
            await self._channel.close()
            self._channel = None
            
        if self._connection:
            await self._connection.close()
            self._connection = None
            
        logger.info("RabbitMQ 连接已关闭")
    
    @property
    def is_running(self) -> bool:
        return self._is_running


# 全局消费者实例
_consumer: Optional[MQConsumer] = None


def get_mq_consumer() -> MQConsumer:
    """获取MQ消费者单例"""
    global _consumer
    if _consumer is None:
        _consumer = MQConsumer()
    return _consumer


async def start_mq_consumer() -> None:
    """启动MQ消费者（用于FastAPI启动事件）"""
    consumer = get_mq_consumer()
    try:
        await consumer.start_consuming()
    except Exception as e:
        logger.error(f"MQ消费者启动失败: {e}")
        # 不抛出异常，允许HTTP服务继续运行


async def stop_mq_consumer() -> None:
    """停止MQ消费者（用于FastAPI关闭事件）"""
    consumer = get_mq_consumer()
    await consumer.stop()
