# -*- coding: utf-8 -*-
"""
视频流分析服务（整合版）
- MQ 主流程：analyze_from_stream，消费 RabbitMQ 任务，调用视觉模型分析
- HTTP 接口：analyze_url，供 /upload-video 直接调用
原 video_analyzer.py 已废弃，本文件为唯一分析入口。
"""
import json
import logging
import cv2
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from app.config import get_settings
from app.models.schemas import VideoTaskMessage, VideoTaskResult, EventInfo

logger = logging.getLogger(__name__)

# ─────────────────────────── Prompts ───────────────────────────

ANALYSIS_PROMPT = '''
    你是一位铜矿生产视频安全分析专家。请客观分析视频画面，判断是否存在违反安全生产规定的行为，并读取画面叠加的元数据。

    【分析原则】
    - 仅依据画面中可直接观察到的证据判断，不推测、不编造。
    - 大多数视频为合规作业，未发现违规是正常且常见的结果。
    - 只有能明确认定违规时才填写违规相关字段；证据不足或看不清时，判定为未发现违规。

    【输出格式】
    仅输出可被机器直接解析的 JSON 数组，不要 Markdown 代码块，不要其他任何说明文字。
    每个 JSON 对象代表一个事件，必须包含以下字段：
    - event_description：违规类型及可见证据
    - date：YYYY-MM-DD；
    - start_time：违规开始时刻 HH:MM:SS
    - end_time：违规结束时刻 HH:MM:SS
    - user_number：画面叠加的用户编号
    - unit_number：画面叠加的单位编号
    - serial_number：画面叠加的序列号
    - start_second：违规开始秒数
    - end_second：违规结束秒数
    - regulations：违反的相关条例

    【情况一：未发现任何违规行为】
    仍输出包含一个对象的 JSON 数组，填写对应的真实值

    【情况二：发现违规行为】
    输出 JSON 数组，每个违规事件一个对象，填写对应的真实值

    【情况三：同一视频存在多个违规事件】
    输出多个对象，每个对象对应一个独立违规事件。

    JSON 对象结构示例如下：将违规行为记录为可被机器直接解析的 JSON 数组，无需任何 Markdown 代码块标记。
    [
        {
            "event_description": "",
            "date": "",
            "start_time": "",
            "end_time": "",
            "user_number": "",
            "unit_number": "",
            "serial_number": ""，
            "start_second": "",
            "end_second": ""，
            "regulations": ""
        }
    ]
'''

# 线程池：用于执行阻塞的CPU/GPU密集型操作
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_analyzer")


class StreamVideoAnalyzer:
    """
    流式视频分析器（整合版，唯一分析入口）
    - MQ 流程：analyze_from_stream
    - HTTP 流程：analyze_url
    """

    def __init__(self):
        self.settings = get_settings()
        self._vision_client: Optional[OpenAI] = None

    @property
    def vision_client(self) -> OpenAI:
        """懒加载视觉模型客户端（原生 openai，可传 temperature/top_p/presence_penalty 等）"""
        if self._vision_client is None:
            self._vision_client = OpenAI(
                base_url=self.settings.vision_model_url,
                api_key=self.settings.model_api_key,
                timeout=self.settings.model_timeout,
            )
        return self._vision_client

    def _strip_thinking_content(self, content: str) -> str:
        """去掉模型思考内容，只保留最终输出"""
        if not content:
            return content
        think_end = "</" + "think" + ">"
        for end_tag in (think_end, "</think>"):
            if end_tag in content:
                content = content.split(end_tag, 1)[1]
                break
        return content.strip()

    def _call_vision_model(self, video_url: str) -> str:
        """调用视觉模型，返回原始文本。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ]
        response = self.vision_client.chat.completions.create(
            model=self.settings.vision_model_name,
            messages=messages,
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={"top_k": 20},
        )
        return self._strip_thinking_content(response.choices[0].message.content) or ""

    def analyze_from_stream(self, task: VideoTaskMessage) -> VideoTaskResult:
        """
        从预签名URL流式分析视频

        Args:
            task: 视频任务消息

        Returns:
            VideoTaskResult: 处理结果
        """
        import time
        start_time = time.time()

        logger.info(
            f"[分析] 开始处理任务: taskId={task.task_id}, videoId={task.video_id}, "
            f"clipId={task.clip_id}, clipStartSecond={task.clip_start_second}"
        )
        logger.info(f"[分析] 预签名URL: {task.presigned_url[:80]}...")

        try:
            content = self._call_vision_model(task.presigned_url)
            events_data = self._parse_events(content)
            unsafe_events = events_data

            process_time = time.time() - start_time
            logger.info(f"[分析] 任务完成: taskId={task.task_id}, 耗时={process_time:.2f}s, 事件数={len(events_data)}")

            # 如果有违规事件，尝试捕获违规帧，并根据视频时长校验违规时间
            violation_frame = None
            violation_timestamp = None
            if unsafe_events:
                try:
                    cap = cv2.VideoCapture(task.presigned_url)
                    if cap.isOpened():
                        # 获取视频时长，用于校验违规时间是否合理（防止模型误用画面上的时钟时间）
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        video_duration = (frame_count / fps) if fps and fps > 0 and frame_count > 0 else 0.0
                        if video_duration > 0:
                            self._sanitize_violation_times(unsafe_events, video_duration)

                        # 尝试从第一个违规事件中提取时间戳并捕获违规帧
                        first_unsafe = unsafe_events[0]
                        violation_timestamp = first_unsafe.get('start_second')
                        if violation_timestamp is not None and fps and fps > 0:
                            frame_number = int(violation_timestamp * fps)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            success_read, violation_frame = cap.read()
                            if not success_read:
                                violation_frame = None
                    cap.release()
                except Exception as e:
                    logger.warning(f"捕获违规帧失败: {e}")

            # 切分预处理链路：将事件时间偏移回原视频时间轴（须在违规帧捕获之后执行，
            # 因为帧捕获使用的是切片内相对时间）
            if task.clip_start_second:
                self._apply_clip_offset(events_data, task.clip_start_second)
                # unsafe_events 与 events_data 可能是同一列表对象，避免重复偏移
                if unsafe_events is not events_data:
                    self._apply_clip_offset(unsafe_events, task.clip_start_second)
                logger.info(
                    f"[分析] 事件时间已偏移到原视频时间轴: clipId={task.clip_id}, "
                    f"offset={task.clip_start_second}s"
                )

            return VideoTaskResult(
                task_id=task.task_id,
                video_id=task.video_id,
                clip_id=task.clip_id,
                success=True,
                events=[EventInfo(**e) for e in events_data] if events_data else None,
                unsafe_events=[EventInfo(**e) for e in unsafe_events] if unsafe_events else None,
                process_time=process_time,
                violation_frame=violation_frame,
                violation_timestamp=violation_timestamp,
                raw_analysis=content
            )

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"[分析] 任务失败: taskId={task.task_id}, error={e}")

            return VideoTaskResult(
                task_id=task.task_id,
                video_id=task.video_id,
                clip_id=task.clip_id,
                success=False,
                error_message=str(e),
                process_time=process_time
            )

    def analyze_url(self, video_url: str) -> Dict[str, Any]:
        """
        HTTP 接口入口（替代已废弃的 VideoAnalyzer.analyze）。
        调用视觉模型分析指定 URL 的视频，返回识别出的违规事件列表。
        """
        import time as _time
        logger.info(f"[analyze_url] 开始分析: {video_url[:80]}...")
        t0 = _time.time()
        content = self._call_vision_model(video_url)
        events_data = self._parse_events(content)
        unsafe_events = events_data
        elapsed = _time.time() - t0
        logger.info(
            f"[analyze_url] 分析完成，耗时={elapsed:.2f}s，事件数={len(events_data)}，违规事件数={len(unsafe_events)}"
        )
        return {
            "success": True,
            "events": events_data,
            "total_events": len(events_data),
            "unsafe_events": unsafe_events,
            "total_unsafe_events": len(unsafe_events),
        }

    def _parse_events(self, content: str) -> List[Dict]:
        """
        从模型回复中提取事件 JSON 列表。
        依次尝试：
          1. 去掉 markdown 代码块后直接 json.loads
          2. 从文本中定位第一个 '[' 到最后一个 ']' 的片段
          3. 从文本中定位第一个 '{' 到最后一个 '}' 的片段（单对象包成列表）
        全部失败时记录原始内容供排查，返回空列表。
        """
        preview = content[:500].replace('\n', ' ') if content else ''
        logger.info(f"[parse_events] 模型原始回复(前500字): {preview}")

        if not content or not content.strip():
            logger.warning("[parse_events] 模型返回空内容")
            return []

        cleaned = content.strip()

        # 剥离思维链：推理模型会在 </think> 前输出推理过程，真正结果在其后
        for end_tag in ("</think>", "<|im_end|>"):
            if end_tag in cleaned:
                cleaned = cleaned.split(end_tag, 1)[1].strip()
                logger.info(f"[parse_events] 检测到思维链标记 {end_tag!r}，已剥离，剩余内容: {cleaned[:200]}")
                break

        # 去掉 markdown 代码块标记
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            cleaned = '\n'.join(inner_lines).strip()

        try:
            data = json.loads(cleaned)
            return self._normalize_events(data)
        except json.JSONDecodeError:
            pass

        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
                return self._normalize_events(data)
            except json.JSONDecodeError:
                pass

        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
                return self._normalize_events(data)
            except json.JSONDecodeError:
                pass

        logger.warning(f"[parse_events] 无法解析JSON，原始回复: {content[:300]}")
        return []

    def _normalize_events(self, data) -> List[Dict]:
        """将解析结果统一为列表格式，并归一化秒数字段类型"""
        if isinstance(data, list):
            events = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            events = None
            for key in ("events", "结果", "事件列表", "event_list"):
                if key in data and isinstance(data[key], list):
                    events = [item for item in data[key] if isinstance(item, dict)]
                    break
            if events is None:
                events = [data]
        else:
            return []
        # 模型有时把 start_second/end_second 返回成字符串（如 "0"、"49"），统一转 float
        for ev in events:
            for sec_key in ("start_second", "end_second"):
                if sec_key in ev:
                    ev[sec_key] = self._to_float(ev.get(sec_key))
        return events

    @staticmethod
    def _to_float(value) -> Optional[float]:
        """安全地将任意值转为 float，无法转换时返回 None。"""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return None

    def _apply_clip_offset(self, events: List[Dict], offset: float) -> None:
        """
        切分预处理链路：将切片内相对时间偏移回原视频时间轴。
        同步调整 start_second/end_second（数值）与 start_time/end_time（HH:MM:SS 字符串）。
        """
        if not events or not offset:
            return
        for ev in events:
            for sec_key, time_key in (("start_second", "start_time"), ("end_second", "end_time")):
                sec = self._to_float(ev.get(sec_key))
                if sec is None:
                    sec = self._parse_time_to_seconds(ev.get(time_key, ""))
                if sec is None:
                    continue
                new_sec = sec + offset
                ev[sec_key] = new_sec
                ev[time_key] = self._seconds_to_time_str(new_sec)

    @staticmethod
    def _seconds_to_time_str(seconds: float) -> str:
        """秒数转 HH:MM:SS 字符串"""
        total = int(round(seconds))
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    def _sanitize_violation_times(self, unsafe_events: List[Dict], video_duration: float) -> None:
        """
        根据视频时长校验违规时间，超出范围则置为 None。
        防止模型误用画面上的录制时钟时间（如 18:18:42）导致错误。
        """
        for ev in unsafe_events:
            start_sec = self._to_float(ev.get('start_second'))
            end_sec = self._to_float(ev.get('end_second'))
            # 回写归一化后的数值，保证后续使用一致
            ev['start_second'] = start_sec
            ev['end_second'] = end_sec
            if start_sec is not None and start_sec > video_duration:
                logger.warning(f"违规时间 start_second={start_sec}s 超出视频时长 {video_duration:.1f}s，已置空")
                ev['start_second'] = None
            if end_sec is not None and end_sec > video_duration:
                logger.warning(f"违规时间 end_second={end_sec}s 超出视频时长 {video_duration:.1f}s，已置空")
                ev['end_second'] = None

    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        将时间字符串解析为秒数
        支持格式: HH:MM:SS, MM:SS, SS
        """
        if not time_str:
            return None

        try:
            parts = time_str.strip().split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 1:
                return float(parts[0])
            return None
        except (ValueError, IndexError):
            return None


# 全局分析器实例
_analyzer: Optional[StreamVideoAnalyzer] = None


def get_stream_analyzer() -> StreamVideoAnalyzer:
    """获取流式分析器单例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = StreamVideoAnalyzer()
    return _analyzer


def get_executor() -> ThreadPoolExecutor:
    """获取线程池"""
    return _executor
