"""Input parsing: brand / product / video links from JSON or CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_YOUTUBE_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


@dataclass
class BrandJob:
    """One brand + product and the videos to analyse for it."""

    brand: str
    product: str
    video_links: List[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slugify(f"{self.brand}-{self.product}")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def is_youtube_url(link: str) -> bool:
    return bool(_YOUTUBE_ID.search(link))


def video_id_for(link: str) -> str:
    """Stable ID for a video: the YouTube ID, or a short hash of the path/URL.

    This is the key Phase 3 will use to join extracted features to metrics.
    """
    match = _YOUTUBE_ID.search(link)
    if match:
        return match.group(1)
    return "v_" + hashlib.sha1(link.strip().encode("utf-8")).hexdigest()[:12]


def _merge(jobs: List[BrandJob]) -> List[BrandJob]:
    """Merge jobs with the same brand + product and drop duplicate links."""
    merged: dict = {}
    for job in jobs:
        brand, product = job.brand.strip(), job.product.strip()
        if not brand or not product:
            raise ValueError("Every entry needs a non-empty brand and product.")
        key = (brand.lower(), product.lower())
        target = merged.setdefault(key, BrandJob(brand, product))
        seen = {video_id_for(link) for link in target.video_links}
        for link in job.video_links:
            link = link.strip()
            if link and video_id_for(link) not in seen:
                target.video_links.append(link)
                seen.add(video_id_for(link))
    return list(merged.values())


def _load_json(path: Path) -> List[BrandJob]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else [data]
    jobs = []
    for entry in entries:
        missing = {"brand", "product"} - set(entry)
        if missing:
            raise ValueError(f"{path}: entry is missing {sorted(missing)}")
        jobs.append(
            BrandJob(entry["brand"], entry["product"], list(entry.get("video_links", [])))
        )
    return jobs


def _load_csv(path: Path) -> List[BrandJob]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"brand", "product", "video_link"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: CSV is missing columns {sorted(missing)}")
        return [
            BrandJob(row["brand"], row["product"], [row["video_link"]]) for row in reader
        ]


def load_jobs(path: str) -> List[BrandJob]:
    """Load jobs from a .json or .csv file.

    JSON: one object, or a list of objects, shaped like
        {"brand": "Nestle", "product": "KitKat", "video_links": ["https://..."]}
    CSV: columns brand, product, video_link (one row per video).
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        jobs = _load_json(file_path)
    elif suffix == ".csv":
        jobs = _load_csv(file_path)
    else:
        raise ValueError(f"Unsupported input type '{suffix}'. Use .json or .csv.")
    return _merge(jobs)
