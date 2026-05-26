# -*- coding: utf-8 -*-
"""
API路由定义
"""
import asyncio
import os
import tempfile
import time
import uuid
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.config import get_settings
from app.services import RemoteUploader, VideoAnalyzer, get_person_segment_detector
from app.services.person_segment_detector import resolve_video_path
from app.services.stream_analyzer import get_executor
from app.models import VideoAnalysisResponse, UploadInfo, PersonSegmentResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# 服务实例
uploader = RemoteUploader()
analyzer = VideoAnalyzer()


def _safe_filename(original: str) -> str:
    """
    生成安全的存储文件名，避免跨端编码问题。
    若原文件名含非ASCII或路径/URL非法字符（如?），则使用时间戳+UUID生成。
    """
    if not original or not original.strip():
        return f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
    ext = ".mp4"
    if "." in original:
        ext = "." + original.rsplit(".", 1)[-1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        ext = ".mp4"
    unsafe_chars = set('?\\/:*"<>| \t\n\r')
    if any(ord(c) > 127 or c in unsafe_chars for c in original):
        return f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    return original


def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许"""
    settings = get_settings()
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in settings.allowed_extensions


@router.get("/")
async def root():
    """根路径"""
    settings = get_settings()
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "/upload-video": "POST - 上传视频并分析",
            "/detect-person-segments": "POST - 检测视频中有人出现的时间段",
            "/health": "GET - 健康检查"
        }
    }


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "视频分析API"}


@router.post("/upload-video", response_model=VideoAnalysisResponse)
async def upload_video(
    file: UploadFile = File(...),
    max_tokens: int = 2048,
):
    """
    上传视频并进行分析

    Args:
        file: 视频文件（MP4格式）
        max_tokens: 最大token数
    """
    settings = get_settings()
    
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="仅支持MP4格式文件")
    
    logger.info(f"收到视频上传请求: {file.filename}")

    # 生成安全存储文件名，避免跨端编码导致路径被截断（如?被当作URL查询符）
    safe_name = _safe_filename(file.filename)
    if safe_name != file.filename:
        logger.info(f"文件名含非常规字符，使用安全名: {safe_name}")
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # 使用安全文件名上传到远程服务器
        upload_result = uploader.upload(temp_file_path, safe_name)
        
        if not upload_result["success"]:
            raise HTTPException(status_code=500, detail=upload_result["message"])
        
        remote_file_path = upload_result["remote_path"]
        video_url = f"file://{remote_file_path}"
        
        # 视频分析
        try:
            analysis_result = await analyzer.analyze(video_url=video_url)
            
            logger.info(f"视频分析完成: {file.filename}")
            
            return VideoAnalysisResponse(
                success=True,
                message="视频上传并分析成功",
                upload_info=UploadInfo(
                    remote_path=remote_file_path,
                    filename=file.filename
                ),
                analysis_result=analysis_result
            )
            
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            return VideoAnalysisResponse(
                success=True,
                message="视频上传成功，但分析失败",
                upload_info=UploadInfo(
                    remote_path=remote_file_path,
                    filename=file.filename
                ),
                analysis_error=str(e)
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频上传失败: {str(e)}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.post("/detect-person-segments", response_model=PersonSegmentResponse)
async def detect_person_segments_api(
    video_path: str,
    min_segment_duration: float = Query(
        default=3.0, ge=0, description="最短有效片段时长（秒）"
    ),
    vid_stride: int = Query(default=3, ge=1, description="每 N 帧推理一次"),
    imgsz: int = Query(default=640, ge=320, le=1280, description="推理输入尺寸"),
    conf: float = Query(default=0.5, ge=0, le=1, description="检测置信度阈值"),
):
    """
    人体片段检测接口

    参数:
        video_path: 视频文件路径（服务器本地路径）
        min_segment_duration: 最短有效片段时长（秒），默认 3.0
        vid_stride: 每 N 帧推理一次，默认 3
        imgsz: 推理输入尺寸，默认 640
        conf: 检测置信度阈值，默认 0.5
    """
    resolved_path = resolve_video_path(video_path)
    detector = get_person_segment_detector()

    logger.info("开始人体片段检测: %s", resolved_path)
    try:
        segments = await asyncio.get_event_loop().run_in_executor(
            get_executor(),
            lambda: detector.detect(
                resolved_path,
                min_segment_duration=min_segment_duration,
                vid_stride=vid_stride,
                imgsz=imgsz,
                conf=conf,
            ),
        )
        logger.info("人体片段检测完成: %s, 共 %d 个片段", resolved_path, len(segments))
        return PersonSegmentResponse(segments=segments, total_segments=len(segments))
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("人体片段检测失败: %s", e)
        raise HTTPException(status_code=500, detail=f"视频检测失败: {str(e)}") from e
