"""Phase 2: video data collection.

For each video:
1. Code measures the file: duration, fps, aspect ratio, shot cuts.
2. One Gemini call returns the structured fields. For the three brand-data
   fields, the Phase 1 brand profile (if any) supplies the preferred names.
3. Both are merged into one row and appended to a JSONL file.

Re-running skips videos that already have a row.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from .brand_profile import BrandProfile
from .client import DEFAULT_MODEL, with_retries
from .custom import (
    CustomFeature,
    build_custom_model,
    load_state,
    read_rows,
    stale_features,
    state_line,
    state_path,
)
from .inputs import BrandJob, is_youtube_url, video_id_for
from .video_schema import CUSTOM_BLOCK_DESCRIPTION, build_output_model
from .video_stats import EMPTY_STATS, FIRST_SECONDS_WINDOW, measure_video

SYSTEM_INSTRUCTION = """You are an Expert Ad Video Analyser.
Your primary function is to act as a meticulous and objective creative expert.
Your goal is to analyze video ad content and return a structured JSON output.
Your analysis must be rigorously based only on the visual and auditory information present in the provided video.

## CORE DIRECTIVES
- Absolute Objectivity: Your analysis must be based exclusively on concrete evidence from the video. Do not infer, assume, or use any external knowledge. If you cannot see or hear it in the video, it did not happen.
- No Hallucination: Your primary directive is to avoid making up information. If a feature is ambiguous, not clearly shown, or impossible to verify from the video, skip that feature.
- Brand Reference Lists: Some fields list assets this brand is known to use. The list only tells you what to look and listen for; it is not evidence that an asset appears. Report an asset only if it is actually present in this video. If it matches a listed asset, use that exact name; if it is present but not listed, describe it in your own words; if nothing is present, return an empty list.
- Strict Adherence to Format: The output format is non-negotiable. Any deviation will result in failure."""

_USER_PROMPT = "Analyse this video advertisement for the brand '{brand}', product '{product}'."


def _video_part(client: genai.Client, link: str) -> types.Part:
    if is_youtube_url(link):
        return types.Part.from_uri(file_uri=link, mime_type="video/mp4")
    if Path(link).exists():
        uploaded = client.files.upload(file=link)
        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name == "FAILED":
            raise ValueError(f"Video processing failed for {link}: {uploaded.error}")
        return types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
    raise ValueError(f"Unsupported video link (need a YouTube URL or a local file): {link}")


def _canonical_name(value: str, assets) -> str:
    """Use the profile's spelling when the label matches a known asset name."""
    def key(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    for asset in assets:
        if key(asset.name) == key(value):
            return asset.name
    return value.strip()


def _distinct_assets(entries, key: str, assets) -> List[Dict[str, Any]]:
    """One entry per distinct asset, at its earliest timestamp, in time order."""
    earliest: Dict[str, Dict[str, Any]] = {}
    for entry in entries or []:
        name = _canonical_name(entry[key], assets)
        if not name:
            continue
        seen = earliest.get(name.lower())
        if seen is None or entry["timestamp"] < seen["timestamp"]:
            earliest[name.lower()] = {**entry, key: name}
    return sorted(earliest.values(), key=lambda e: e["timestamp"])


def build_row(
    link: str,
    profile: BrandProfile,
    model: str,
    gemini: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge Gemini's output and the code-measured stats into one output row."""
    attract = gemini["attract_features"]
    brand = dict(gemini["brand_features"])
    for block, key, assets in (
        ("brand_sonic_asset", "sonic_asset", profile.sonic_assets),
        ("hero_product_hook", "hook_type", profile.key_visuals),
        ("distinctive_brand_asset", "brand_asset", profile.distinctive_brand_assets),
    ):
        brand[block] = _distinct_assets(brand[block], key, assets)
    brand["brand_in_first_5s"] = brand["first_brand_touchpoint_sec"] <= FIRST_SECONDS_WINDOW
    return {
        "video_link": link,
        "brand": profile.brand,
        "product": profile.product,
        "model": model,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": {"video_id": video_id_for(link), **stats["metadata"]},
        "talent_and_star_power": gemini["talent_and_star_power"],
        "setting_analysis": gemini["setting_analysis"],
        "attract_features": {
            **stats["attract"],
            "framing_density": attract["framing_density"],
            "product_interaction_0-5s": attract["product_interaction_0_5s"],
            "supers_present_0-5s": attract["supers_present_0_5s"],
        },
        "brand_features": brand,
        "connect_features": gemini["connect_features"],
        "direct_features": gemini["direct_features"],
        "custom": gemini.get("custom") or {},
    }


def extract_video(
    client: genai.Client,
    link: str,
    profile: BrandProfile,
    output_model: Type[BaseModel],
    model: str = DEFAULT_MODEL,
    skip_video_stats: bool = False,
) -> Dict[str, Any]:
    """Extract one video and return its output row.

    The file is measured first, so a video that cannot be fetched fails before
    any Gemini call is made. With `skip_video_stats`, the measured fields are
    saved as null instead.
    """
    stats = EMPTY_STATS if skip_video_stats else measure_video(link)
    part = _video_part(client, link)

    def call() -> BaseModel:
        response = client.models.generate_content(
            model=model,
            contents=[
                part,
                _USER_PROMPT.format(brand=profile.brand, product=profile.product),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=output_model,
                temperature=0,
            ),
        )
        if not response.text:
            raise RuntimeError("Model returned an empty response.")
        return output_model.model_validate_json(response.text)

    gemini = with_retries(call).model_dump(mode="json")
    return build_row(link, profile, model, gemini, stats)


def features_path(output_dir: str, job: BrandJob) -> Path:
    return Path(output_dir) / "video_features" / f"{job.slug}.jsonl"


def _done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            json.loads(line)["metadata"]["video_id"] for line in handle if line.strip()
        }


def extract_job(
    client: genai.Client,
    job: BrandJob,
    profile: BrandProfile,
    output_dir: str,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
    skip_video_stats: bool = False,
    custom_features: Sequence[CustomFeature] = (),
) -> Dict[str, int]:
    """Extract every video in a job that has no row yet.

    Videos that already have a row are never touched here, even if their
    custom answers are missing or out of date; see `reevaluate_custom`.
    Returns counts of done / skipped / failed.
    """
    results = features_path(output_dir, job)
    errors = results.with_suffix(".errors.jsonl")
    results.parent.mkdir(parents=True, exist_ok=True)

    done = _done_ids(results)
    pending = [link for link in job.video_links if video_id_for(link) not in done]
    counts = {"done": 0, "skipped": len(job.video_links) - len(pending), "failed": 0}
    if not pending:
        return counts

    output_model = build_output_model(profile, custom_features)
    lock = threading.Lock()

    def append(path: Path, row: Dict[str, Any]) -> None:
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                extract_video, client, link, profile, output_model, model, skip_video_stats
            ): link
            for link in pending
        }
        for future in as_completed(futures):
            link = futures[future]
            try:
                row = future.result()
                append(results, row)
                if custom_features:
                    with lock, state_path(results).open("a", encoding="utf-8") as handle:
                        handle.write(
                            state_line(row["metadata"]["video_id"], custom_features) + "\n"
                        )
                counts["done"] += 1
                print(f"  ok      {link}")
            except Exception as exc:  # keep the batch going; record the failure
                append(
                    errors,
                    {
                        "video_id": video_id_for(link),
                        "video_link": link,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                )
                counts["failed"] += 1
                print(f"  FAILED  {link}: {exc}")
    return counts


_CUSTOM_PROMPT = (
    "This is a video advertisement for the brand '{brand}', product '{product}'. "
    + CUSTOM_BLOCK_DESCRIPTION
)


def reevaluate_custom(
    client: genai.Client,
    job: BrandJob,
    custom_features: Sequence[CustomFeature],
    output_dir: str,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
) -> Dict[str, int]:
    """Bring custom answers up to date for already-extracted videos.

    Only runs when the user starts it. For each video whose custom answers are
    missing or were given under an older definition, one Gemini call asks for
    just those features and the answers are merged into the existing row. All
    other fields in the row are left as they are.
    """
    results = features_path(output_dir, job)
    rows = read_rows(results)
    state = load_state(results)

    todo: Dict[int, List[CustomFeature]] = {}
    for index, row in enumerate(rows):
        stale = stale_features(row, custom_features, state)
        if stale["missing"] or stale["changed"]:
            todo[index] = stale["missing"] + stale["changed"]
    counts = {"updated": 0, "up_to_date": len(rows) - len(todo), "failed": 0}
    if not todo:
        return counts

    def evaluate(index: int) -> Dict[str, Any]:
        row, features = rows[index], todo[index]
        schema = build_custom_model(features)
        part = _video_part(client, row["video_link"])

        def call() -> BaseModel:
            response = client.models.generate_content(
                model=model,
                contents=[
                    part,
                    _CUSTOM_PROMPT.format(brand=row["brand"], product=row["product"]),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                ),
            )
            if not response.text:
                raise RuntimeError("Model returned an empty response.")
            return schema.model_validate_json(response.text)

        return with_retries(call).model_dump(mode="json")

    state_lines: List[str] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(evaluate, index): index for index in todo}
            for future in as_completed(futures):
                index = futures[future]
                row = rows[index]
                try:
                    answers = future.result()
                    row["custom"] = {**(row.get("custom") or {}), **answers}
                    state_lines.append(state_line(row["metadata"]["video_id"], todo[index]))
                    counts["updated"] += 1
                    print(f"  ok      {row['video_link']}")
                except Exception as exc:  # keep going; this video stays out of date
                    counts["failed"] += 1
                    print(f"  FAILED  {row['video_link']}: {exc}")
    finally:
        # Save whatever finished, even if the run was interrupted.
        if state_lines:
            temp = results.with_suffix(".jsonl.tmp")
            with temp.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            temp.replace(results)
            with state_path(results).open("a", encoding="utf-8") as handle:
                handle.write("\n".join(state_lines) + "\n")
    return counts
