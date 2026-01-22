# -*- coding: utf-8 -*-
"""
MinIO 客户端服务
用于上传违规截图到 MinIO
"""
import io
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

# 延迟导入 minio，避免启动时报错
_minio_client = None


def get_minio_client():
    """获取MinIO客户端单例"""
    global _minio_client
    if _minio_client is None:
        try:
            from minio import Minio
            settings = get_settings()
            _minio_client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure
            )
            logger.info(f"MinIO客户端初始化成功: {settings.minio_endpoint}")
        except ImportError:
            logger.warning("minio 库未安装，截图上传功能将不可用")
            return None
        except Exception as e:
            logger.error(f"MinIO客户端初始化失败: {e}")
            return None
    return _minio_client


def ensure_bucket_exists(bucket_name: str) -> bool:
    """确保存储桶存在"""
    client = get_minio_client()
    if client is None:
        return False
    
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"创建存储桶: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"检查/创建存储桶失败: {e}")
        return False


def upload_screenshot(
    frame: np.ndarray,
    video_id: int,
    task_id: str,
    timestamp: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    上传违规截图到 MinIO
    
    Args:
        frame: OpenCV 格式的图像帧 (BGR)
        video_id: 视频ID
        task_id: 任务ID
        timestamp: 视频时间戳（秒）
    
    Returns:
        (success, url): 成功标志和截图URL
    """
    settings = get_settings()
    client = get_minio_client()
    
    if client is None:
        logger.warning("MinIO客户端不可用，跳过截图上传")
        return False, None
    
    if frame is None or frame.size == 0:
        logger.warning("截图帧为空，跳过上传")
        return False, None
    
    try:
        # 确保存储桶存在
        if not ensure_bucket_exists(settings.minio_bucket):
            return False, None
        
        # 编码图像为 JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("图像编码失败")
            return False, None
        
        # 生成文件名
        date_path = datetime.now().strftime("%Y/%m/%d")
        timestamp_str = f"_{int(timestamp)}s" if timestamp else ""
        filename = f"{date_path}/violation_{video_id}_{task_id[:8]}{timestamp_str}_{uuid.uuid4().hex[:8]}.jpg"
        
        # 上传到 MinIO
        image_data = io.BytesIO(buffer.tobytes())
        image_size = len(buffer.tobytes())
        
        client.put_object(
            bucket_name=settings.minio_bucket,
            object_name=filename,
            data=image_data,
            length=image_size,
            content_type="image/jpeg"
        )
        
        # 生成预签名URL（24小时有效）
        url = client.presigned_get_object(
            bucket_name=settings.minio_bucket,
            object_name=filename,
            expires=timedelta(hours=24)
        )
        
        logger.info(f"截图上传成功: {filename}")
        return True, url
        
    except Exception as e:
        logger.error(f"截图上传失败: {e}", exc_info=True)
        return False, None


def capture_violation_frame(video_url: str, timestamp: float = 0) -> Optional[np.ndarray]:
    """
    从视频URL中捕获指定时间戳的帧
    
    Args:
        video_url: 视频URL（预签名URL）
        timestamp: 时间戳（秒），默认为0（第一帧）
    
    Returns:
        OpenCV格式的帧，失败返回None
    """
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_url[:80]}...")
            return None
        
        # 跳转到指定时间戳
        if timestamp > 0:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # 读取帧
        success, frame = cap.read()
        cap.release()
        
        if success:
            logger.info(f"成功捕获视频帧: timestamp={timestamp}s")
            return frame
        else:
            logger.warning(f"读取视频帧失败: timestamp={timestamp}s")
            return None
            
    except Exception as e:
        logger.error(f"捕获视频帧异常: {e}")
        return None
