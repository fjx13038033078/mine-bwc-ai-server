# -*- coding: utf-8 -*-
"""
API路由定义
"""
import os
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config import get_settings
from app.services import RemoteUploader, VideoAnalyzer
from app.models import VideoAnalysisResponse, UploadInfo

logger = logging.getLogger(__name__)
router = APIRouter()

# 服务实例
uploader = RemoteUploader()
analyzer = VideoAnalyzer()


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
    use_yolo: bool = False,
    use_safety_analysis: bool = True
):
    """
    上传视频并进行分析
    
    Args:
        file: 视频文件（MP4格式）
        max_tokens: 最大token数
        use_yolo: 是否使用YOLO检测
        use_safety_analysis: 是否进行安全分析
    """
    settings = get_settings()
    
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="仅支持MP4格式文件")
    
    logger.info(f"收到视频上传请求: {file.filename}")
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        # 上传到远程服务器
        upload_result = uploader.upload(temp_file_path, file.filename)
        
        if not upload_result["success"]:
            raise HTTPException(status_code=500, detail=upload_result["message"])
        
        remote_file_path = upload_result["remote_path"]
        video_url = f"file://{remote_file_path}"
        
        # 视频分析
        try:
            analysis_result = await analyzer.analyze(
                video_url=video_url,
                remote_file_path=remote_file_path,
                use_yolo=use_yolo,
                use_safety_analysis=use_safety_analysis
            )
            
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
