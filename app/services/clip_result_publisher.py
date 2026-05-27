# -*- coding: utf-8 -*-
"""
切割结果发布者
将视频切割结果回传到 video.clip.result.exchange，供 Java 端消费落库。
"""
import json
import logging
from typing import Optional

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractConnection, AbstractChannel, AbstractExchange

from app.config import get_settings
from app.models.schemas import VideoClipResultMessage, ClipInfo

logger = logging.getLogger(__name__)


class ClipResultPublisher:
    def __init__(self):
        self.settings = get_settings()
        self._connection: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._exchange: Optional[AbstractExchange] = None

    async def _ensure_connected(self) -> None:
        if self._connection is None or self._connection.is_closed:
            url = (
                f"amqp://{self.settings.rabbitmq_user}:{self.settings.rabbitmq_password}"
                f"@{self.settings.rabbitmq_host}:{self.settings.rabbitmq_port}"
                f"{self.settings.rabbitmq_vhost}"
            )
            self._connection = await aio_pika.connect_robust(url)
            self._channel = await self._connection.channel()
            self._exchange = await self._channel.declare_exchange(
                self.settings.clip_result_exchange,
                ExchangeType.TOPIC,
                durable=True,
            )
            logger.info(f"[ClipResultPublisher] 连接成功，交换机: {self.settings.clip_result_exchange}")

    async def publish(self, result: VideoClipResultMessage) -> bool:
        try:
            await self._ensure_connected()
            body = json.dumps(result.to_java_dict(), ensure_ascii=False).encode("utf-8")
            msg = Message(body=body, delivery_mode=DeliveryMode.PERSISTENT, content_type="application/json")
            await self._exchange.publish(msg, routing_key=self.settings.clip_result_routing_key)
            logger.info(
                f"[ClipResultPublisher] 发送成功: taskId={result.task_id}, "
                f"videoId={result.video_id}, status={result.status}, clips={len(result.clips)}"
            )
            return True
        except Exception as e:
            logger.error(f"[ClipResultPublisher] 发送失败: {e}", exc_info=True)
            return False

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()
        if self._connection:
            await self._connection.close()


_publisher: Optional[ClipResultPublisher] = None


def get_clip_result_publisher() -> ClipResultPublisher:
    global _publisher
    if _publisher is None:
        _publisher = ClipResultPublisher()
    return _publisher


async def publish_clip_result(result: VideoClipResultMessage) -> bool:
    return await get_clip_result_publisher().publish(result)
