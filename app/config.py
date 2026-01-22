# -*- coding: utf-8 -*-
"""
应用配置
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用信息
    app_name: str = "视频上传与分析API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # 远程服务器配置
    remote_host: str = "172.18.1.1"
    remote_port: int = 22
    remote_user: str = "user"
    remote_password: str = "User@1234"
    remote_base_path: str = "/home/user/zby/inference/"
    
    # AI模型服务配置
    vision_model_url: str = "http://172.18.1.1:22002/v1"
    vision_model_name: str = "Qwen3-VL-4B-Instruct"
    thinking_model_url: str = "http://172.18.1.1:22000/v1"
    thinking_model_name: str = "Qwen3-1.7B-Thinking"
    model_api_key: str = "EMPTY"
    model_timeout: int = 60
    
    # YOLO配置
    yolo_model_path: str = "yolo/yolo11n.pt"
    
    # 文件配置
    allowed_extensions: set = {"mp4"}
    max_tokens: int = 2048
    
    # RabbitMQ 配置
    rabbitmq_host: str = "127.0.0.1"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_vhost: str = "/"
    
    # MQ 任务队列配置（消费Java端发来的任务）
    mq_exchange: str = "video.upload.exchange"
    mq_queue: str = "video.upload.queue"
    mq_routing_key: str = "video.upload.#"
    mq_prefetch_count: int = 1  # 同时只处理一个任务，防止GPU显存溢出
    
    # MQ 结果队列配置（发送检测结果给Java端）
    result_exchange: str = "video.result.exchange"
    result_queue: str = "video.result.queue"
    result_routing_key: str = "video.result.finish"
    
    # MinIO 配置（用于上传违规截图）
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = "ruoyi"
    minio_secret_key: str = "ruoyi123"
    minio_bucket: str = "violation-images"
    minio_secure: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
