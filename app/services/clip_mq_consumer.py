# -*- coding: utf-8 -*-
"""
视频切割任务 MQ 消费者
监听 video.clip.queue，调用 video_clipper 完成切割，结果回传 Java。
"""
import asyncio
import json
import logging
from typing import Optional

import aio_pika
from aio_pika import ExchangeType
from aio_pika.abc import AbstractIncomingMessage, AbstractConnection, AbstractChannel

from app.config import get_settings
from app.models.schemas import VideoClipTaskMessage, VideoClipResultMessage, ClipInfo
from app.services.clip_result_publisher import publish_clip_result
from app.services.stream_analyzer import get_executor

logger = logging.getLogger(__name__)


class ClipMQConsumer:
    def __init__(self):
        self.settings = get_settings()
        self._connection: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._is_running = False

    async def connect(self) -> None:
        url = (
            f"amqp://{self.settings.rabbitmq_user}:{self.settings.rabbitmq_password}"
            f"@{self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}"
            f"{self.settings.rabbitmq_vhost}"
        )
        logger.info(f"[ClipConsumer] 连接 RabbitMQ: {self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}")
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)  # 串行，防止资源竞争
        logger.info("[ClipConsumer] 连接成功")

    async def setup_queue(self) -> aio_pika.Queue:
        exchange = await self._channel.declare_exchange(
            self.settings.clip_mq_exchange, ExchangeType.TOPIC, durable=True
        )
        queue = await self._channel.declare_queue(
            self.settings.clip_mq_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": f"{self.settings.clip_mq_queue}.dlq",
                "x-max-length": 1000,
            },
        )
        await queue.bind(exchange, routing_key=self.settings.clip_mq_routing_key)
        logger.info(f"[ClipConsumer] 队列绑定完成: {self.settings.clip_mq_queue}")
        return queue

    async def process_message(self, message: AbstractIncomingMessage) -> None:
        task_id = "unknown"
        try:
            body = message.body.decode("utf-8")
            logger.info(f"[ClipConsumer] 收到消息: {body[:200]}...")
            data = json.loads(body)
            task = VideoClipTaskMessage(**data)
            task_id = task.task_id
            logger.info(f"[ClipConsumer] 任务解析成功: taskId={task_id}, videoId={task.video_id}")

            from app.services.video_clipper import process_clip_task

            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                get_executor(),
                lambda: process_clip_task(
                    video_id=task.video_id,
                    task_id=task.task_id,
                    presigned_url=task.presigned_url,
                    min_segment_duration=task.min_segment_duration,
                    vid_stride=task.vid_stride,
                    conf=task.conf,
                ),
            )

            clips = [ClipInfo(**{
                "clipIndex": c["clip_index"],
                "objectName": c["object_name"],
                "url": c["url"],
                "startSecond": c["start_second"],
                "endSecond": c["end_second"],
                "durationSeconds": c["duration_seconds"],
                "fileSize": c["file_size"],
            }) for c in raw.get("clips", [])]

            result = VideoClipResultMessage(
                taskId=task_id,
                videoId=task.video_id,
                status="SUCCESS" if raw["success"] else "FAILED",
                clips=clips,
                errorMessage=raw.get("error_message"),
            )

            await publish_clip_result(result)

            if raw["success"]:
                await message.ack()
                logger.info(f"[ClipConsumer] ACK: taskId={task_id}, 切片数={len(clips)}")
            else:
                await message.nack(requeue=False)
                logger.warning(f"[ClipConsumer] NACK: taskId={task_id}")

        except json.JSONDecodeError as e:
            logger.error(f"[ClipConsumer] JSON 解析失败: {e}")
            await message.nack(requeue=False)
        except Exception as e:
            logger.error(f"[ClipConsumer] 处理异常: taskId={task_id}, {e}", exc_info=True)
            await message.nack(requeue=False)

    async def start_consuming(self) -> None:
        if not self._channel:
            await self.connect()
        queue = await self.setup_queue()
        self._is_running = True
        await queue.consume(self.process_message)
        logger.info("[ClipConsumer] 开始消费切割任务...")

    async def stop(self) -> None:
        self._is_running = False
        if self._channel:
            await self._channel.close()
        if self._connection:
            await self._connection.close()
        logger.info("[ClipConsumer] 连接已关闭")

    @property
    def is_running(self) -> bool:
        return self._is_running


_consumer: Optional[ClipMQConsumer] = None


def get_clip_mq_consumer() -> ClipMQConsumer:
    global _consumer
    if _consumer is None:
        _consumer = ClipMQConsumer()
    return _consumer


async def start_clip_mq_consumer() -> None:
    consumer = get_clip_mq_consumer()
    try:
        await consumer.start_consuming()
    except Exception as e:
        logger.error(f"[ClipConsumer] 启动失败: {e}")


async def stop_clip_mq_consumer() -> None:
    consumer = get_clip_mq_consumer()
    await consumer.stop()
