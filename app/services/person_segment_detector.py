# -*- coding: utf-8 -*-
"""
YOLO 视频人体片段检测服务
检测视频中有人出现的时间段
"""
import logging
from pathlib import Path

import cv2
from fastapi import HTTPException
from ultralytics import YOLO

from app.config import get_settings

logger = logging.getLogger(__name__)


def has_person(result) -> bool:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return False
    return any(int(cls) == 0 for cls in boxes.cls.tolist())


def try_append_segment(
    segments: list[tuple[float, float]],
    start_time: float,
    end_time: float,
    min_duration: float,
) -> None:
    if end_time - start_time >= min_duration:
        segments.append((start_time, end_time))


def close_segment(
    segments: list[tuple[float, float]],
    segment_start_frame: int,
    last_person_frame: int,
    fps: float,
    min_segment_duration: float,
) -> None:
    start_time = segment_start_frame / fps
    end_time = (last_person_frame + 1) / fps
    try_append_segment(segments, start_time, end_time, min_segment_duration)


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def resolve_video_path(video_path: str) -> str:
    """解析并校验视频路径。"""
    if not video_path or not video_path.strip():
        raise HTTPException(status_code=400, detail="视频路径不能为空")
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"视频文件不存在: {video_path}")
    return str(path)


class PersonSegmentDetector:
    """YOLO 人体片段检测器"""

    def __init__(self):
        self.settings = get_settings()
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            logger.info("加载 YOLO 人体片段检测模型: %s", self.settings.person_yolo_model_path)
            self._model = YOLO(self.settings.person_yolo_model_path)
        return self._model

    def detect(
        self,
        video_path: str,
        *,
        min_segment_duration: float | None = None,
        vid_stride: int | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        conf: float | None = None,
        gap_tolerance_seconds: float | None = None,
    ) -> list[dict[str, float]]:
        """检测视频中有人出现的时间段，返回 [{"start": 秒, "end": 秒}, ...]。"""
        settings = self.settings
        min_segment_duration = (
            min_segment_duration if min_segment_duration is not None else settings.yolo_min_segment_duration
        )
        vid_stride = vid_stride if vid_stride is not None else settings.yolo_vid_stride
        imgsz = imgsz if imgsz is not None else settings.yolo_imgsz
        device = device if device is not None else settings.yolo_device
        conf = conf if conf is not None else settings.yolo_conf
        gap_tolerance_seconds = (
            gap_tolerance_seconds
            if gap_tolerance_seconds is not None
            else settings.yolo_gap_tolerance_seconds
        )

        fps = get_video_fps(video_path)
        gap_tolerance_frames = max(1, int(fps * gap_tolerance_seconds))

        segments: list[tuple[float, float]] = []
        in_segment = False
        segment_start_frame = 0
        last_person_frame = 0
        infer_count = 0

        for result in self.model.predict(
            source=video_path,
            stream=True,
            vid_stride=vid_stride,
            conf=conf,
            classes=[0],
            device=device,
            half=True,
            imgsz=imgsz,
            verbose=False,
        ):
            infer_count += 1
            frame_idx = (infer_count - 1) * vid_stride

            if has_person(result):
                if not in_segment:
                    in_segment = True
                    segment_start_frame = frame_idx
                last_person_frame = frame_idx
            elif in_segment and frame_idx - last_person_frame > gap_tolerance_frames:
                close_segment(
                    segments, segment_start_frame, last_person_frame, fps, min_segment_duration
                )
                in_segment = False

        if in_segment:
            close_segment(
                segments, segment_start_frame, last_person_frame, fps, min_segment_duration
            )

        return [{"start": round(s, 3), "end": round(e, 3)} for s, e in segments]


_detector: PersonSegmentDetector | None = None


def get_person_segment_detector() -> PersonSegmentDetector:
    """获取人体片段检测器单例"""
    global _detector
    if _detector is None:
        _detector = PersonSegmentDetector()
    return _detector
