# -*- coding: utf-8 -*-
"""
视频切割服务
1. 用预签名 URL 下载原视频到本地临时文件
2. 调用 PersonSegmentDetector 找出有人片段
3. 用 ffmpeg -c copy 快速切割（无重编码，关键帧精度 ±1s）
4. 每段上传到 MinIO，收集元数据
"""
import logging
import os
import subprocess
import tempfile
from typing import List

import requests

from app.config import get_settings
from app.services.minio_client import upload_video_clip
from app.services.person_segment_detector import get_person_segment_detector

logger = logging.getLogger(__name__)


def _download_to_temp(presigned_url: str) -> str:
    """将预签名 URL 下载到本地临时 MP4 文件，返回临时文件路径。"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        logger.info(f"[Clipper] 开始下载视频: {presigned_url[:80]}...")
        with requests.get(presigned_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp.write(chunk)
        tmp.flush()
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        logger.info(f"[Clipper] 视频下载完成: {tmp.name} ({size_mb:.1f} MB)")
        return tmp.name
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise
    finally:
        tmp.close()


def _cut_segment(
    input_path: str,
    start: float,
    end: float,
    output_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> bool:
    """
    使用 ffmpeg -c copy 快速切割一个片段。
    返回是否成功。
    """
    cmd = [
        ffmpeg_bin, "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-c", "copy",
        "-avoid_negative_ts", "1",
        output_path,
    ]
    logger.debug(f"[Clipper] ffmpeg: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if result.returncode != 0:
            logger.error(f"[Clipper] ffmpeg 失败: {result.stderr.decode(errors='replace')[-400:]}")
            return False
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error(f"[Clipper] ffmpeg 输出为空: {output_path}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[Clipper] ffmpeg 超时: start={start} end={end}")
        return False
    except FileNotFoundError:
        logger.error(f"[Clipper] 未找到 ffmpeg: {ffmpeg_bin}")
        return False


def process_clip_task(
    video_id: int,
    task_id: str,
    presigned_url: str,
    min_segment_duration: float = 3.0,
    vid_stride: int = 3,
    conf: float = 0.5,
) -> dict:
    """
    执行完整切割流程，返回结果字典：
      success / clips / failed_clips / error_message
    """
    settings = get_settings()
    local_input = None
    clip_files: List[str] = []

    try:
        local_input = _download_to_temp(presigned_url)

        detector = get_person_segment_detector()
        segments = detector.detect(
            local_input,
            min_segment_duration=min_segment_duration,
            vid_stride=vid_stride,
            conf=conf,
        )
        logger.info(f"[Clipper] videoId={video_id} 检测到 {len(segments)} 个有人片段")

        if not segments:
            return {"success": True, "clips": [], "failed_clips": [], "error_message": None}

        clips = []
        failed_clips = []

        for idx, seg in enumerate(segments):
            start = seg["start"]
            end = seg["end"]
            duration = round(end - start, 3)

            clip_tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4", prefix=f"clip_{video_id}_{idx}_"
            )
            clip_tmp.close()
            clip_path = clip_tmp.name
            clip_files.append(clip_path)

            try:
                if not _cut_segment(local_input, start, end, clip_path, settings.ffmpeg_binary):
                    raise RuntimeError("ffmpeg 切割失败")

                object_name, url, file_size = upload_video_clip(
                    local_path=clip_path,
                    video_id=video_id,
                    clip_index=idx,
                )
                clips.append({
                    "clip_index": idx,
                    "object_name": object_name,
                    "url": url,
                    "start_second": start,
                    "end_second": end,
                    "duration_seconds": duration,
                    "file_size": file_size,
                })
                logger.info(
                    f"[Clipper] 片段 {idx} 成功: {start:.1f}s-{end:.1f}s "
                    f"({file_size / 1024 / 1024:.1f} MB)"
                )
            except Exception as e:
                logger.error(f"[Clipper] 片段 {idx} 失败: {e}", exc_info=True)
                failed_clips.append({
                    "clip_index": idx,
                    "start_second": start,
                    "end_second": end,
                    "error": str(e),
                })

        return {
            "success": True,
            "clips": clips,
            "failed_clips": failed_clips,
            "error_message": None,
        }

    except Exception as e:
        logger.error(f"[Clipper] taskId={task_id} 整体失败: {e}", exc_info=True)
        return {"success": False, "clips": [], "failed_clips": [], "error_message": str(e)}

    finally:
        if local_input and os.path.exists(local_input):
            try:
                os.unlink(local_input)
            except Exception:
                pass
        for f in clip_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
