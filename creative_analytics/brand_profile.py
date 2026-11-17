"""Phase 1: brand data collection.

Given a brand and product, gather what the brand is *already known for* in its
advertising: sonic assets, key visuals and distinctive brand assets. The result
is a small closed list per field. Phase 2 makes the model choose from these
lists instead of inventing labels per video.

This phase only gathers data. It does no analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .client import DEFAULT_MODEL, with_retries
from .inputs import BrandJob

MAX_ITEMS_PER_FIELD = 10


class BrandAsset(BaseModel):
    name: str = Field(
        ...,
        description="Short, unique 2-5 word label, e.g. 'KitKat Wafer Snap'.",
    )
    description: str = Field(
        ...,
        description="One sentence on how to recognise this asset in a video ad.",
    )


class BrandResearch(BaseModel):
    """What the model fills in. Kept free of defaults for the response schema."""

    sonic_assets: List[BrandAsset] = Field(
        ...,
        description=(
            "Audio the brand is known for: jingles, sound mnemonics, signature "
            "product sounds (e.g. KitKat snap). Empty list if none are documented."
        ),
    )
    key_visuals: List[BrandAsset] = Field(
        ...,
        description=(
            "Signature sensory or tactile product hooks the brand repeatedly uses "
            "(e.g. KitKat finger snap, Nescafe coffee stirring, Maggi steam rise)."
        ),
    )
    distinctive_brand_assets: List[BrandAsset] = Field(
        ...,
        description=(
            "Recognisable brand devices other than a plain text logo: mascots, "
            "proprietary characters, signature visual devices, unique packaging "
            "shapes, signature brand colours."
        ),
    )


class BrandProfile(BrandResearch):
    """Saved brand profile: the research plus identity and cited sources."""

    brand: str
    product: str
    sources: List[str] = Field(
        default_factory=list, description="Web sources the research step cited."
    )


ASSET_FIELDS = ("sonic_assets", "key_visuals", "distinctive_brand_assets")

_RESEARCH_PROMPT = """You are researching the advertising of one product so that video ads can later be tagged consistently.

Brand: {brand}
Product: {product}

Use Google Search. Report only what this brand is documented to use in its advertising for this product. Do not guess what it might use. If you cannot find evidence for an item, leave it out; a short list is better than a padded one.

Cover each of these, with at most {max_items} items each:
1. Sonic assets: jingles, sound mnemonics, signature product sounds.
2. Key visuals: signature sensory or tactile product shots the ads repeat (a snap, a stir, steam rising, a pour).
3. Distinctive brand assets: mascots, recurring characters, signature visual devices, signature colours, packaging shapes.

For every item give a short name (2-5 words) and one sentence on how to recognise it in a video. Items within a list must be clearly different from each other; merge near-duplicates. Return an empty list where nothing is documented."""

def _grounding_sources(response) -> List[str]:
    sources: List[str] = []
    for candidate in response.candidates or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and web.uri:
                label = f"{web.title} - {web.uri}" if web.title else web.uri
                if label not in sources:
                    sources.append(label)
    return sources


def dedupe_assets(assets: List[BrandAsset], limit: int = MAX_ITEMS_PER_FIELD) -> List[BrandAsset]:
    """Drop items whose names match ignoring case/punctuation, then cap the list."""
    seen, unique = set(), []
    for asset in assets:
        key = "".join(ch for ch in asset.name.lower() if ch.isalnum())
        if key and key not in seen:
            seen.add(key)
            unique.append(asset)
    return unique[:limit]


def collect_brand_profile(
    client: genai.Client, brand: str, product: str, model: str = DEFAULT_MODEL
) -> BrandProfile:
    """One Gemini call: research with Google Search and return the profile JSON.

    Combining Search grounding with a JSON response schema in one request
    needs a Gemini 3 series model or newer.
    """
    response = with_retries(
        lambda: client.models.generate_content(
            model=model,
            contents=_RESEARCH_PROMPT.format(
                brand=brand, product=product, max_items=MAX_ITEMS_PER_FIELD
            ),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=BrandResearch,
                temperature=0,
            ),
        )
    )
    if not response.text:
        raise RuntimeError(f"Brand research returned no text for {brand} / {product}.")
    draft = BrandResearch.model_validate_json(response.text)
    profile = BrandProfile(
        brand=brand,
        product=product,
        sources=_grounding_sources(response),
        **draft.model_dump(),
    )
    for name in ASSET_FIELDS:
        setattr(profile, name, dedupe_assets(getattr(profile, name)))
    return profile


def empty_profile(brand: str, product: str) -> BrandProfile:
    """A profile with no brand data: Phase 2 then uses free text for all three fields."""
    return BrandProfile(
        brand=brand, product=product,
        sonic_assets=[], key_visuals=[], distinctive_brand_assets=[],
    )


def profile_path(output_dir: str, job: BrandJob) -> Path:
    return Path(output_dir) / "brand_profiles" / f"{job.slug}.json"


def load_profile(path: Path) -> BrandProfile:
    return BrandProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profile: BrandProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_or_create_profile(
    client: Optional[genai.Client],
    job: BrandJob,
    output_dir: str,
    model: str = DEFAULT_MODEL,
    refresh: bool = False,
) -> BrandProfile:
    """Reuse the saved profile if present, so hand edits to the JSON are kept."""
    path = profile_path(output_dir, job)
    if path.exists() and not refresh:
        return load_profile(path)
    if client is None:
        raise RuntimeError(f"No saved profile at {path} and no client to create one.")
    profile = collect_brand_profile(client, job.brand, job.product, model)
    save_profile(profile, path)
    return profile
