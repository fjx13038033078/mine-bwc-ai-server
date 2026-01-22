# -*- coding: utf-8 -*-
"""
FastAPI 应用主入口
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import get_settings
from app.api import router
from app.services import start_mq_consumer, stop_mq_consumer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    - startup: 启动MQ消费者
    - shutdown: 停止MQ消费者
    """
    settings = get_settings()
    
    # ==================== Startup ====================
    logger.info("=" * 60)
    logger.info(f"{settings.app_name} v{settings.app_version} 启动中...")
    logger.info("=" * 60)
    
    # 启动 MQ 消费者（在后台任务中运行，不阻塞HTTP服务）
    mq_task = asyncio.create_task(start_mq_consumer())
    logger.info("MQ消费者任务已创建")
    
    logger.info(f"HTTP服务: http://localhost:8000")
    logger.info(f"API文档: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    yield  # 应用运行中
    
    # ==================== Shutdown ====================
    logger.info("正在关闭服务...")
    
    # 取消MQ任务
    mq_task.cancel()
    try:
        await mq_task
    except asyncio.CancelledError:
        pass
    
    # 停止MQ消费者
    await stop_mq_consumer()
    
    logger.info("服务已关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="视频文件上传到远程服务器并进行AI智能分析，支持MQ异步任务消费",
        lifespan=lifespan
    )
    
    # 注册路由
    app.include_router(router)
    
    return app


# 创建应用实例
app = create_app()
