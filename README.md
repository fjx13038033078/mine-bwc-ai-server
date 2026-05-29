# cameraAi — 执法视频 AI 检测与视频切割服务（FastAPI）

`cameraAi` 是执法记录仪智能分析系统的 **AI 推理服务**，基于 **FastAPI** 构建。它通过 **RabbitMQ** 与 Java 后端（`RuoYi-Cloud-Plus / ruoyi-camera`）解耦协作，承担两条核心业务链路的计算任务：

- **AI 违规检测**：用视觉大模型识别执法视频中的安全违规行为，并通过 RAG 匹配对应的规章制度；
- **视频人体切割**：用 YOLO 检测视频中有人出现的时段，再用 FFmpeg 切出有效片段。

> 配套仓库：`RuoYi-Cloud-Plus / ruoyi-camera`（业务编排）、`plus-ui`（Vue3 前端）。

---

## 一、整体架构

```
                         RabbitMQ
ruoyi-camera ───── video.upload.queue ─────►┐
(Java 后端)  ◄──── video.result.queue ──────┤   cameraAi (FastAPI)
             ───── video.clip.queue ───────►│   ├─ mq_consumer          消费 AI 检测任务
             ◄──── video.clip.result.queue ─┘   ├─ clip_mq_consumer     消费切割任务
                                                ├─ stream_analyzer      视觉模型 + YOLO + RAG
                                                ├─ person_segment_detector  YOLO 人体检测
                                                ├─ video_clipper        FFmpeg 切割
                                                └─ result/clip_result_publisher  回传结果
        外部依赖：MinIO（对象存储）、视觉模型、思考模型、Embedding（ModelScope）
```

应用启动时（`app/main.py` 的 lifespan）会同时拉起 **AI 检测消费者** 与 **切割消费者**，并后台预热 YOLO 模型。

---

## 二、AI 违规检测链路

入口：`stream_analyzer.StreamVideoAnalyzer.analyze_from_stream`（MQ）/ `analyze_url`（HTTP）。

1. **YOLO 辅助检测**：本地流式跑 YOLO，逐秒输出辅助信息拼入提示词；
2. **视觉模型识别**：调用视觉模型（`hrylora`）分析视频，输出违规事件 JSON（描述、起止时间、用户/单位/序列号等）；
3. **RAG 规章制度匹配**：
   - 从 `knowledge/*.docx` 加载规章制度，**按单条规章拆分**后用 Embedding（`Qwen3-Embedding-8B` @ ModelScope）建向量库（`InMemoryVectorStore`，懒加载缓存）；
   - 对每条违规行为做相似度检索，再由思考模型（`Qwen3-1.7B-Thinking`）基于检索结果给出对应规章；思考模型不可用时**回退使用检索到的规章原文**，保证 `regulations` 不为空；
4. **关键帧截图**：用 OpenCV 截取违规起始帧，上传 MinIO；
5. **结果回传**：`result_publisher` 发送 `video.result.queue`，含 `aiDescription`、`eventsJson`（每条事件含 `regulations`）、`screenshotUrl` 等。

---

## 三、视频人体切割链路

入口：`clip_mq_consumer` 消费 `video.clip.queue`。

1. **下载原视频**：通过预签名 URL 下载到临时文件；
2. **人体片段检测**：`person_segment_detector` 用 YOLO 识别有人出现的时间段（支持最短片段时长、抽帧步长、置信度等参数）；
3. **切割**：`video_clipper` 用 FFmpeg（`-c copy` 无损快速切割）逐段切出片段；
4. **上传与回传**：切片上传 MinIO（`clips/` 前缀），`clip_result_publisher` 发送 `video.clip.result.queue`，含每个切片的索引、起止秒、时长、文件大小等。

---

## 四、HTTP 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 服务信息与接口列表 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/upload-video` | 上传视频并**同步**分析（含 RAG），返回 `events` 与 `unsafe_events`（含 `regulations`），供前端演示页使用 |
| `POST` | `/detect-person-segments` | 检测视频中有人出现的时间段 |

---

## 五、目录结构

```
cameraAi/
├── app/
│   ├── main.py                      # FastAPI 入口，启动双 MQ 消费者 + YOLO 预热
│   ├── config.py                    # 配置（pydantic-settings，自动读取项目根 .env）
│   ├── api/routes.py                # HTTP 路由
│   ├── models/schemas.py            # Pydantic 模型与 MQ 消息体（含 EventInfo.regulations）
│   ├── services/
│   │   ├── stream_analyzer.py       # 核心分析器：视觉模型 + YOLO + RAG（唯一分析入口）
│   │   ├── mq_consumer.py           # AI 检测任务消费者
│   │   ├── clip_mq_consumer.py      # 切割任务消费者
│   │   ├── person_segment_detector.py  # YOLO 人体片段检测
│   │   ├── video_clipper.py         # FFmpeg 切割
│   │   ├── result_publisher.py      # AI 结果回传
│   │   ├── clip_result_publisher.py # 切割结果回传
│   │   ├── remote_uploader.py       # 远程上传
│   │   └── minio_client.py          # MinIO 客户端
│   └── utils/json_extractor.py      # JSON 提取工具
├── knowledge/                       # RAG 规章制度文档（.docx）
├── yolo/                            # YOLO 模型权重（yolo11n.pt）
├── requirements.txt
└── .env                             # 敏感配置（embedding key 等，已 .gitignore）
```

---

## 六、依赖模型与外部服务

| 角色 | 默认配置项 | 说明 |
|------|-----------|------|
| 视觉模型 | `vision_model_url` / `vision_model_name=hrylora` | 识别视频违规行为 |
| 思考模型 | `thinking_model_url` / `thinking_model_name=Qwen3-1.7B-Thinking` | RAG 规章制度精炼 |
| Embedding | `embedding_model_name=Qwen/Qwen3-Embedding-8B` @ ModelScope | 规章制度向量化 |
| 对象存储 | `minio_*` | 视频、截图、切片存储 |
| 消息队列 | `rabbitmq_*` | 与 Java 端一致 |
| FFmpeg | `ffmpeg_binary` | 视频切割可执行文件路径 |

> **敏感配置（如 `EMBEDDING_API_KEY`）放在项目根目录 `.env` 文件**，由 `app/config.py` 以绝对路径加载，不受启动工作目录影响；`.env` 已加入 `.gitignore`，禁止提交。

---

## 七、本地运行

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\activate        # Windows PowerShell

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备 .env（在项目根目录）
#    EMBEDDING_API_KEY=你的ModelScope密钥

# 4. 启动服务（默认 8000 端口）
uvicorn app.main:app --host 0.0.0.0 --port 8000
#    API 文档： http://localhost:8000/docs
```

> 运行前请确认：RabbitMQ、MinIO 已就绪且与 Java 端配置一致；视觉模型/思考模型服务可访问；`knowledge/` 下存在规章制度 docx；`yolo/` 下存在模型权重；FFmpeg 路径正确。

---

## 八、关键说明

- **唯一分析入口**：所有分析统一走 `stream_analyzer.StreamVideoAnalyzer`，旧的 `video_analyzer.py` 已废弃删除。
- **RAG 健壮性**：向量库初始化失败或思考模型不可用时自动降级，日志中以 `[RAG]` 前缀输出向量库初始化、命中规章长度等信息，便于排查。
- **第三方 Embedding 端点**：`OpenAIEmbeddings` 已设置 `check_embedding_ctx_length=False`，发送原始文本而非 token 数组，适配 ModelScope 等第三方端点。
