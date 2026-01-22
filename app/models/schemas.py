# -*- coding: utf-8 -*-
"""
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime


class EventInfo(BaseModel):
    """事件信息"""
    event_description: str = ""
    date: str = ""
    start_time: str = ""
    end_time: str = ""
    user_number: str = ""
    unit_number: str = ""
    serial_number: str = ""


class UploadInfo(BaseModel):
    """上传信息"""
    remote_path: str
    filename: str


class VideoAnalysisResponse(BaseModel):
    """视频分析响应"""
    success: bool
    message: str
    upload_info: Optional[UploadInfo] = None
    analysis_result: Optional[dict] = None
    analysis_error: Optional[str] = None


# ==================== MQ 消息模型 ====================

class VideoTaskMetadata(BaseModel):
    """视频任务元数据"""
    video_code: Optional[str] = Field(None, alias="videoCode", description="视频编码/设备号")
    user_name: Optional[str] = Field(None, alias="userName", description="用户名称")
    record_time: Optional[str] = Field(None, alias="recordTime", description="录制时间")
    file_name: Optional[str] = Field(None, alias="fileName", description="文件名")
    file_size: Optional[int] = Field(None, alias="fileSize", description="文件大小(字节)")
    content_type: Optional[str] = Field(None, alias="contentType", description="内容类型")
    original_path: Optional[str] = Field(None, alias="originalPath", description="原始文件路径")
    ext_fields: Optional[Dict[str, Any]] = Field(None, alias="extFields", description="扩展字段")
    
    class Config:
        populate_by_name = True


class VideoTaskMessage(BaseModel):
    """
    视频检测任务消息（MQ消息体）
    对应 Java 端的 VideoUploadMessage
    """
    task_id: str = Field(..., alias="taskId", description="任务ID（业务追踪ID）")
    video_id: int = Field(..., alias="videoId", description="视频ID（数据库主键）")
    presigned_url: str = Field(..., alias="presignedUrl", description="预签名URL（核心字段，24小时有效）")
    bucket_name: Optional[str] = Field(None, alias="bucketName", description="存储桶名称")
    object_name: Optional[str] = Field(None, alias="objectName", description="对象名称（MinIO中的文件路径）")
    original_url: Optional[str] = Field(None, alias="originalUrl", description="原始文件URL")
    create_time: Optional[datetime] = Field(None, alias="createTime", description="消息创建时间")
    metadata: Optional[VideoTaskMetadata] = Field(None, description="元数据")
    
    class Config:
        populate_by_name = True


class VideoTaskResult(BaseModel):
    """视频任务处理结果（内部使用）"""
    task_id: str
    video_id: int
    success: bool
    events: Optional[List[EventInfo]] = None
    unsafe_events: Optional[List[EventInfo]] = None
    error_message: Optional[str] = None
    process_time: Optional[float] = None  # 处理耗时（秒）
    violation_frame: Optional[Any] = None  # 违规帧图像（OpenCV格式，不序列化）
    violation_timestamp: Optional[float] = None  # 违规发生的视频时间戳


# ==================== 结果回传消息模型 ====================

class VideoAnalysisResultMessage(BaseModel):
    """
    视频分析结果消息（发送给Java端）
    对应 Java 端的 VideoAnalysisResult
    """
    task_id: str = Field(..., alias="taskId", description="任务ID（与请求一致）")
    video_id: int = Field(..., alias="videoId", description="视频ID（数据库主键）")
    status: str = Field(..., description="状态: SUCCESS/FAILED")
    has_violation: bool = Field(False, alias="hasViolation", description="是否有违规")
    violation_type: Optional[str] = Field(None, alias="violationType", description="违规类型")
    ai_description: Optional[str] = Field(None, alias="aiDescription", description="AI分析描述")
    screenshot_url: Optional[str] = Field(None, alias="screenshotUrl", description="违规截图URL")
    events_json: Optional[str] = Field(None, alias="eventsJson", description="事件列表JSON")
    process_time: Optional[float] = Field(None, alias="processTime", description="处理耗时(秒)")
    error_message: Optional[str] = Field(None, alias="errorMessage", description="错误信息")
    
    class Config:
        populate_by_name = True
        
    def to_java_dict(self) -> Dict[str, Any]:
        """转换为Java端期望的格式（驼峰命名）"""
        return {
            "taskId": self.task_id,
            "videoId": self.video_id,
            "status": self.status,
            "hasViolation": self.has_violation,
            "violationType": self.violation_type,
            "aiDescription": self.ai_description,
            "screenshotUrl": self.screenshot_url,
            "eventsJson": self.events_json,
            "processTime": self.process_time,
            "errorMessage": self.error_message
        }