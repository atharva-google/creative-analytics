"""Custom features: user-defined fields that Gemini fills in per video.

Definitions live in their own file (custom.json), separate from the video
list. A feature can sit on a brand (inherited by all its products) or on one
product:

    {
      "Nestle": {
        "custom": [ {"name": "...", "type": "...", "description": "..."} ],
        "products": {
          "Maggi": { "custom": [ ... ] }
        }
      }
    }

Types: boolean, number, text, choice (choice needs "options").

Saving or editing definitions never triggers any Gemini call. New videos are
extracted with the current definitions; already-extracted videos are only
updated when the user starts a re-evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, Field, create_model, field_validator, model_validator

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class CustomFeature(BaseModel):
    name: str
    type: Literal["boolean", "number", "text", "choice"]
    description: str
    options: Optional[List[str]] = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError(
                f"'{value}' is not a valid name. Use lowercase letters, digits and "
                "underscores, starting with a letter (e.g. festival_theme)."
            )
        return value

    @field_validator("description")
    @classmethod
    def _has_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty; it is what Gemini follows.")
        return value.strip()

    @model_validator(mode="after")
    def _options_match_type(self) -> "CustomFeature":
        if self.type == "choice":
            options = [o.strip() for o in self.options or []]
            if len(options) < 2 or any(not o for o in options):
                raise ValueError(f"'{self.name}': a choice needs at least 2 non-empty options.")
            if len(set(options)) != len(options):
                raise ValueError(f"'{self.name}': options must be unique.")
            self.options = options
        elif self.options:
            raise ValueError(f"'{self.name}': only a choice can have options.")
        return self

    @property
    def fingerprint(self) -> str:
        """Changes whenever the question asked of Gemini changes."""
        payload = json.dumps([self.type, self.description, self.options], ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _key(text: str) -> str:
    return text.strip().lower()


class CustomConfig:
    """Validated custom feature definitions for all brands and products."""

    def __init__(
        self,
        brand_features: Dict[str, List[CustomFeature]],
        product_features: Dict[Tuple[str, str], List[CustomFeature]],
        labels: Dict[Any, str],
    ) -> None:
        self._brand = brand_features
        self._product = product_features
        self._labels = labels

    def features_for(self, brand: str, product: str) -> List[CustomFeature]:
        """Brand-level features followed by this product's own features."""
        return self._brand.get(_key(brand), []) + self._product.get(
            (_key(brand), _key(product)), []
        )

    def unmatched(self, jobs: Iterable[Tuple[str, str]]) -> List[str]:
        """Brands/products defined here that are not in the video list (likely typos)."""
        brands = {_key(b) for b, _ in jobs}
        pairs = {(_key(b), _key(p)) for b, p in jobs}
        missing = [self._labels[b] for b in self._brand if b not in brands]
        missing += [self._labels[pair] for pair in self._product if pair not in pairs]
        return missing


def _parse_features(raw: Any, where: str) -> List[CustomFeature]:
    if not isinstance(raw, list):
        raise ValueError(f"{where}: 'custom' must be a list of feature definitions.")
    features = []
    for item in raw:
        try:
            features.append(CustomFeature.model_validate(item))
        except Exception as exc:
            raise ValueError(f"{where}: {exc}") from exc
    names = [f.name for f in features]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{where}: duplicate custom feature name(s) {duplicates}.")
    return features


def parse_custom(data: Any) -> CustomConfig:
    """Validate custom definitions given as a dict (what the frontend form saves).

    Duplicate names are rejected: within a brand, within a product, and
    between a product and its brand.
    """
    if not isinstance(data, dict):
        raise ValueError("Custom features must be a JSON object keyed by brand name.")
    brand_features: Dict[str, List[CustomFeature]] = {}
    product_features: Dict[Tuple[str, str], List[CustomFeature]] = {}
    labels: Dict[Any, str] = {}

    for brand, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{brand}: expected an object with 'custom' and/or 'products'.")
        unknown = set(entry) - {"custom", "products"}
        if unknown:
            raise ValueError(f"{brand}: unknown key(s) {sorted(unknown)}.")
        if _key(brand) in brand_features:
            raise ValueError(f"Brand '{brand}' is defined more than once.")
        brand_list = _parse_features(entry.get("custom", []), brand)
        brand_features[_key(brand)] = brand_list
        labels[_key(brand)] = brand
        brand_names = {f.name for f in brand_list}

        products = entry.get("products", {})
        if not isinstance(products, dict):
            raise ValueError(f"{brand}: 'products' must be an object keyed by product name.")
        for product, product_entry in products.items():
            where = f"{brand} / {product}"
            if not isinstance(product_entry, dict) or set(product_entry) - {"custom"}:
                raise ValueError(f"{where}: expected an object with only 'custom'.")
            pair = (_key(brand), _key(product))
            if pair in product_features:
                raise ValueError(f"Product '{where}' is defined more than once.")
            product_list = _parse_features(product_entry.get("custom", []), where)
            clashes = sorted(brand_names & {f.name for f in product_list})
            if clashes:
                raise ValueError(
                    f"{where}: custom feature name(s) {clashes} are already defined "
                    f"on the brand '{brand}'. Duplicate names are not allowed."
                )
            product_features[pair] = product_list
            labels[pair] = where
    return CustomConfig(brand_features, product_features, labels)


def load_custom(path: Optional[str]) -> CustomConfig:
    """Load custom.json. No path means no custom features."""
    if not path:
        return parse_custom({})
    return parse_custom(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------- schema
def build_custom_model(features: Sequence[CustomFeature]) -> Type[BaseModel]:
    """Response schema for a set of custom features."""
    fields: Dict[str, Any] = {}
    for feature in features:
        if feature.type == "boolean":
            kind: Any = bool
        elif feature.type == "number":
            kind = Optional[float]
        elif feature.type == "text":
            kind = Optional[str]
        else:
            kind = Enum(
                f"{feature.name}_choice",
                {f"OPTION_{i}": option for i, option in enumerate(feature.options)},
                type=str,
            )
        fields[feature.name] = (kind, Field(..., description=feature.description))
    return create_model("Custom", **fields)


# -------------------------------------------- state: what each answer was for
def state_path(features_file: Path) -> Path:
    """Sidecar recording which definition each video's custom answers used."""
    return features_file.with_suffix(".custom_state.jsonl")


def load_state(features_file: Path) -> Dict[str, Dict[str, str]]:
    """{video_id: {feature_name: fingerprint}}; later lines update earlier ones."""
    state: Dict[str, Dict[str, str]] = {}
    path = state_path(features_file)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    state.setdefault(entry["video_id"], {}).update(entry["fingerprints"])
    return state


def state_line(video_id: str, features: Sequence[CustomFeature]) -> str:
    return json.dumps(
        {"video_id": video_id, "fingerprints": {f.name: f.fingerprint for f in features}}
    )


def stale_features(
    row: Dict[str, Any],
    features: Sequence[CustomFeature],
    state: Dict[str, Dict[str, str]],
) -> Dict[str, List[CustomFeature]]:
    """Which current features this row lacks ('missing') or answered under an
    older definition ('changed')."""
    answers = row.get("custom") or {}
    used = state.get(row["metadata"]["video_id"], {})
    result: Dict[str, List[CustomFeature]] = {"missing": [], "changed": []}
    for feature in features:
        if feature.name not in answers:
            result["missing"].append(feature)
        elif used.get(feature.name) != feature.fingerprint:
            result["changed"].append(feature)
    return result


def custom_status(
    features_file: Path, features: Sequence[CustomFeature]
) -> Dict[str, Any]:
    """Counts of out-of-date custom answers. Reads local files only; no Gemini."""
    rows = read_rows(features_file)
    state = load_state(features_file)
    per_feature = {f.name: {"missing": 0, "changed": 0} for f in features}
    videos_to_update = 0
    for row in rows:
        stale = stale_features(row, features, state)
        if stale["missing"] or stale["changed"]:
            videos_to_update += 1
        for kind in ("missing", "changed"):
            for feature in stale[kind]:
                per_feature[feature.name][kind] += 1
    return {
        "videos_extracted": len(rows),
        "videos_to_update": videos_to_update,
        "features": per_feature,
    }


def read_rows(features_file: Path) -> List[Dict[str, Any]]:
    if not features_file.exists():
        return []
    with features_file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
