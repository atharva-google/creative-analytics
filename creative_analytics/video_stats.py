"""Fields measured from the video file in code (no Gemini involved).

Metadata: duration, fps, aspect_ratio.
Attract:  shot_count_0-5s, first_cut_timestamp_sec (PySceneDetect).

These need the actual video file. Local files are read directly; YouTube
links are downloaded at low resolution into a temporary folder that is deleted
afterwards.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .inputs import is_youtube_url

FIRST_SECONDS_WINDOW = 5.0

_COMMON_RATIOS = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "21:9": 21 / 9,
}


def aspect_ratio_label(width: int, height: int) -> str:
    """Nearest common aspect ratio (within 2%), else 'width:height'."""
    ratio = width / height
    name, target = min(_COMMON_RATIOS.items(), key=lambda kv: abs(kv[1] - ratio))
    if abs(target - ratio) / target <= 0.02:
        return name
    return f"{width}:{height}"


def shot_fields(cut_times_sec: List[float]) -> Dict[str, Any]:
    """Attract fields from the list of cut timestamps (seconds).

    shot_count_0-5s is the number of shots seen in the first 5 seconds: one
    for the opening shot plus one per cut before the 5-second mark.
    first_cut_timestamp_sec is the first cut anywhere in the video, or None
    if the video is a single shot.
    """
    cuts = sorted(cut_times_sec)
    return {
        "shot_count_0-5s": 1 + sum(1 for t in cuts if t < FIRST_SECONDS_WINDOW),
        "first_cut_timestamp_sec": round(cuts[0], 2) if cuts else None,
    }


def detect_cut_times(video_path: str) -> List[float]:
    """Timestamps of every hard cut, detected with PySceneDetect."""
    from scenedetect import ContentDetector, detect

    def seconds(timecode) -> float:
        # `.seconds` in newer PySceneDetect releases, `.get_seconds()` in older ones.
        value = getattr(timecode, "seconds", None)
        return float(value if value is not None else timecode.get_seconds())

    scenes = detect(video_path, ContentDetector())
    return [seconds(start) for start, _ in scenes[1:]]


def probe_video(video_path: str) -> Dict[str, Any]:
    """Duration (seconds), fps and aspect ratio read from the file."""
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not fps or not width or not height:
        raise ValueError(f"Could not read video properties: {video_path}")
    return {
        "duration": round(frames / fps, 2),
        "fps": round(fps, 2),
        "aspect_ratio": aspect_ratio_label(width, height),
    }


def _download_youtube(link: str, folder: str) -> Dict[str, Any]:
    """Download a small video-only copy. Returns {'path', 'source_fps'}."""
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "format": "bv*[height<=480][ext=mp4]/bv*[height<=480]/b[height<=480]/b",
        "outtmpl": str(Path(folder) / "%(id)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(link, download=True)
        path = ydl.prepare_filename(info)
    # The small copy can have a lower frame rate than the published video,
    # so report the frame rate of the highest-resolution format instead.
    formats = [f for f in info.get("formats") or [] if f.get("fps") and f.get("height")]
    best = max(formats, key=lambda f: f["height"], default=None)
    return {"path": path, "source_fps": float(best["fps"]) if best else None}


def measure_video(link: str) -> Dict[str, Any]:
    """Measure one video. Returns {'metadata': {...}, 'attract': {...}}.

    Raises if the video cannot be fetched or read.
    """
    with tempfile.TemporaryDirectory(prefix="creative_analytics_") as folder:
        source_fps: Optional[float] = None
        if is_youtube_url(link):
            downloaded = _download_youtube(link, folder)
            path, source_fps = downloaded["path"], downloaded["source_fps"]
        elif Path(link).exists():
            path = link
        else:
            raise ValueError(f"Need a YouTube URL or an existing local file: {link}")
        metadata = probe_video(path)
        if source_fps:
            metadata["fps"] = round(source_fps, 2)
        return {"metadata": metadata, "attract": shot_fields(detect_cut_times(path))}


EMPTY_STATS: Dict[str, Any] = {
    "metadata": {"duration": None, "fps": None, "aspect_ratio": None},
    "attract": {"shot_count_0-5s": None, "first_cut_timestamp_sec": None},
}
