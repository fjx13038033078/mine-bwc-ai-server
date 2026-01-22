# 视频上传与分析API

基于 FastAPI 的视频分析服务，支持两种模式：
1. **HTTP 同步模式**：视频上传至远程服务器并立即返回分析结果
2. **MQ 异步模式**：从 RabbitMQ 消费视频检测任务，流式读取 MinIO 预签名URL

## 功能特性

- 视频文件上传至远程服务器（SFTP）
- AI 视觉模型分析视频内容（Qwen3-VL）
- 矿山安全违规行为检测（Qwen3-Thinking）
- 可选的 YOLO 物体检测
- **RabbitMQ 异步任务消费**（流式读取，不下载到本地）

## 项目结构

```
cameraAi/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口（含生命周期管理）
│   ├── config.py            # 配置管理
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # HTTP API 路由
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # 数据模型（含MQ消息模型）
│   ├── services/
│   │   ├── __init__.py
│   │   ├── remote_uploader.py   # 远程上传服务
│   │   ├── video_analyzer.py    # HTTP模式视频分析
│   │   ├── stream_analyzer.py   # MQ模式流式分析
│   │   └── mq_consumer.py       # RabbitMQ 消费者
│   └── utils/
│       ├── __init__.py
│       └── json_extractor.py    # JSON 提取工具
├── yolo/                    # YOLO 模型文件目录
├── run.py                   # 启动脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

可通过环境变量或 `.env` 文件配置：

```env
# 远程服务器（HTTP上传模式）
REMOTE_HOST=172.18.1.1
REMOTE_PORT=22
REMOTE_USER=user
REMOTE_PASSWORD=your_password

# AI 模型服务
VISION_MODEL_URL=http://172.18.1.1:22002/v1
VISION_MODEL_NAME=Qwen3-VL-4B-Instruct
THINKING_MODEL_URL=http://172.18.1.1:22000/v1
THINKING_MODEL_NAME=Qwen3-1.7B-Thinking

# RabbitMQ（MQ异步模式）
RABBITMQ_HOST=127.0.0.1
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
MQ_EXCHANGE=video.upload.exchange
MQ_QUEUE=video.upload.queue
MQ_ROUTING_KEY=video.upload.#
MQ_PREFETCH_COUNT=1
```

### 3. 启动服务

```bash
python run.py
```

服务启动后：
- HTTP API: http://localhost:8000/docs
- MQ消费者自动开始监听队列

## 运行模式

### 模式一：HTTP 同步模式

前端直接上传视频，服务端处理后立即返回结果。

```
前端 -> POST /upload-video -> FastAPI -> AI分析 -> 返回结果
```

### 模式二：MQ 异步模式

Java后端将任务发送到RabbitMQ，Python端消费处理。

```
Java后端 -> RabbitMQ(预签名URL) -> FastAPI消费者 -> 流式读取视频 -> AI分析
```

**关键特性：**
- `prefetch_count=1`：同时只处理一个任务，防止GPU显存溢出
- 流式读取：直接从预签名URL读取视频流，不下载到本地磁盘
- 线程池执行：阻塞的AI推理在线程池中执行，不阻塞事件循环

## MQ 消息格式

Java端发送的消息格式：

```json
{
  "taskId": "VID-123-1234567890",
  "videoId": 123,
  "presignedUrl": "http://minio:9000/bucket/video.mp4?X-Amz-...",
  "bucketName": "zhifajiluyi",
  "objectName": "2026/01/21/xxx.mp4",
  "originalUrl": "http://minio:9000/bucket/video.mp4",
  "createTime": "2026-01-21T10:00:00",
  "metadata": {
    "userName": "张三",
    "fileName": "video.mp4"
  }
}
```

## API 接口

### POST /upload-video

上传视频并进行分析（同步模式）

### GET /health

健康检查接口

## 依赖说明

| 依赖 | 用途 |
|------|------|
| FastAPI | Web 框架 |
| aio-pika | RabbitMQ 异步客户端 |
| LangChain | AI 模型调用 |
| OpenCV | 视频流处理 |
| Ultralytics | YOLO 物体检测 |
