# creative-analytics

Extracts structured creative features from video ads with Gemini, so they can
later be joined to performance metrics (VTR, CTR).

| Phase | What it does | Status |
|---|---|---|
| 1. Brand data collection | For a brand + product, gathers the assets the brand is known for into small closed lists | Built |
| 2. Video data collection | Tags each video with Gemini (brand assets chosen only from the Phase 1 lists) and measures shots and file metadata in code | Built |
| 3. Connect with metrics | Join features to VTR / CTR on `video_id` | Not started |
| 4. Generate analysis | Find what works for the brand | Not started |

## Use in Colab

Open `notebooks/quickstart.ipynb`, or run:

```python
!git clone https://github.com/atharva-google/creative-analytics.git
%cd creative-analytics
%pip install -q -e .

import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")  # add it under Colab Secrets first

!python -m creative_analytics run --input videos.csv
```

The API key is read from the `GEMINI_API_KEY` environment variable only. Do not
put it in code or in the input file.

## Input

JSON, one object or a list of them:

```json
[
  {
    "brand": "Nestle",
    "product": "KitKat",
    "video_links": ["https://www.youtube.com/watch?v=-WOLE1oSn5k"]
  }
]
```

Or CSV with columns `brand,product,video_link`, one row per video. A link is a
YouTube URL or a path to a local video file.

## Commands

```bash
python -m creative_analytics profile --input in.json   # Phase 1 only
python -m creative_analytics extract --input in.json   # Phase 2 (runs Phase 1 if no profile is saved)
python -m creative_analytics run     --input in.json   # both
```

Options: `--custom`, `--brand`, `--product`, `--output-dir` (default `output`),
`--model`, `--max-workers` (default 4), `--refresh-profile`,
`--skip-brand-research`, `--skip-video-stats`.

## Phase 1: brand profile

Output: `output/brand_profiles/<brand>-<product>.json`

The profile holds up to 10 unique items for each of `sonic_assets`,
`key_visuals` and `distinctive_brand_assets`, plus the web sources cited. Each
item has a short `name` and a one-line `description` of how to recognise it.

It is produced by one Gemini call per brand and product, which researches the
brand with Google Search grounding and returns the profile as JSON. This needs
a Gemini 3 series model or newer. Phase 1 only gathers data; it does no
analysis.

**Review this file before running Phase 2.** Its lists are the names Gemini is
told to prefer for the matching brand-data fields. You can edit it by hand (rename, remove or
add items); a saved profile is reused as is unless you pass `--refresh-profile`.

## Phase 2: video features

Output: `output/video_features/<brand>-<product>.jsonl`, one row per video.
Each row has `video_link`, `brand`, `product`, `model`, `extracted_at` and the
blocks below. `metadata.video_id` is the YouTube ID (or a hash of the path for
local files) and is the key for joining metrics in Phase 3.

Re-running skips videos that already have a row, so a batch can be resumed.
Failures go to `<brand>-<product>.errors.jsonl` and are retried on the next run.

Source: **G** = Gemini, **C** = computed in code, **G + brand** = Gemini,
guided by the Phase 1 brand profile when one is available.

| Block | Field | Source |
|---|---|---|
| `metadata` | `video_id` | C |
| | `duration` (seconds) | C |
| | `fps` | C |
| | `aspect_ratio` (e.g. `16:9`) | C |
| `talent_and_star_power` | `celebrity_cast` (name, first_appearance_sec, role) | G |
| | `supporting_cast` | G |
| `setting_analysis` | `primary_setting` (timestamp, location_type, description, ambient_mood) | G |
| | `setting_transition_count` | G |
| `attract_features` | `shot_count_0-5s` | C |
| | `first_cut_timestamp_sec` | C |
| | `framing_density` | G |
| | `product_interaction_0-5s` | G |
| | `supers_present_0-5s` | G |
| `brand_features` | `first_brand_touchpoint_sec` | G |
| | `brand_in_first_5s` | C |
| | `logo_present` | G |
| | `spoken_brand_name` | G |
| | `see_and_say_branding` | G |
| | `brand_total_screen_time_sec` | G |
| | `brand_sonic_asset` (timestamp, sonic_asset) | G + brand |
| | `hero_product_hook` (timestamp, hook_type) | G + brand |
| | `distinctive_brand_asset` (timestamp, brand_asset) | G + brand |
| `connect_features` | `narrative_arc`, `linguistic_register`, `audio_mix`, `music_tempo_bpm` | G |
| | `dialogue_wpm`, `jargon_intensity` | G |
| | `humor_details` (integration, archetype) | G |
| | `product_utilisation_present` | G |
| `direct_features` | `cta_type`, `cta_delivery` | G |
| `custom` | your own fields, see Custom features | G |

### Brand-data fields

`brand_sonic_asset`, `hero_product_hook` and `distinctive_brand_asset` each
hold one asset and its timestamp, or null if the video has none.

- **With brand research:** Gemini is given the profile's list for the field. If
  the asset in the video matches a listed one, it records that exact name. If
  an asset is present but not on the list, it is still recorded, described in
  3-4 words of free text.
- **Without it:** where the profile's list is empty, Gemini always describes the
  asset in 3-4 words of free text.

Brand research is optional. Pass `--skip-brand-research` to skip Phase 1
entirely; all three fields are then free text.

`spoken_brand_name` is true if the brand name is spoken anywhere.
`see_and_say_branding` is true only if it is spoken while branding is on screen.

### Computed fields

These are measured from the video file. Local files are read directly; YouTube
links are downloaded at up to 480p into a temporary folder and deleted after.

- `duration`, `aspect_ratio`: read from the file. `fps` is the frame rate of the
  highest-resolution YouTube format (or of the file, for local videos).
- `shot_count_0-5s`: shots seen in the first 5 seconds, i.e. 1 + the number of
  cuts before 5.0 s. Cuts are found with PySceneDetect's content detector.
- `first_cut_timestamp_sec`: the first cut anywhere in the video, null if the
  video is one continuous shot.
- `brand_in_first_5s`: `first_brand_touchpoint_sec` <= 5.

A video that cannot be downloaded or read fails before any Gemini call. Pass
`--skip-video-stats` to tag videos without downloading them; the measured
fields are then saved as null.

## Custom features

Custom features are extra fields you define yourself. They live in their own
file, separate from the video list, and are passed with `--custom custom.json`.
The file is optional.

```json
{
  "Nestle": {
    "custom": [
      {"name": "festival_theme", "type": "choice",
       "description": "Festival the ad is themed around, if any.",
       "options": ["Diwali", "Holi", "Christmas", "None"]}
    ],
    "products": {
      "Maggi": {
        "custom": [
          {"name": "cooking_time_claim", "type": "boolean",
           "description": "True if a cooking time such as '2 minutes' is shown or spoken."}
        ]
      }
    }
  }
}
```

- **Levels:** a feature under a brand's `custom` applies to all of that brand's
  products. A feature under a product applies to that product only.
- **Types:** `boolean`, `number`, `text`, `choice` (a choice needs `options`).
  `number` and `text` can come back as null.
- **Names:** lowercase letters, digits and underscores. Duplicate names are
  rejected within a brand, within a product, and between a product and its brand.
- **Description:** this is the question Gemini answers, so write it precisely.
- **Output:** each row has a `custom` block, e.g.
  `"custom": {"festival_theme": "Diwali", "cooking_time_claim": true}`.

Custom features can only be things Gemini judges from the video.

### When definitions change

Adding or editing a custom feature never triggers Gemini calls on its own.

- `extract` / `run` only process videos that have no row yet. Those videos get
  the current custom features in their one Gemini call. Videos already
  extracted are left exactly as they are.
- `custom-status` shows, per feature, how many extracted videos are **missing**
  an answer (feature added later) or have an **outdated** one (definition
  edited later). It reads local files only and needs no API key.
- `reevaluate-custom` updates them. It makes one Gemini call per affected
  video, asking only for the missing or outdated custom features, and merges
  the answers into the existing row. No other field is changed.

```bash
python -m creative_analytics custom-status     --input videos.csv --custom custom.json
python -m creative_analytics reevaluate-custom --input videos.csv --custom custom.json --brand Nestle --product Maggi
```

`--brand` and `--product` limit any command to one brand or one product.

Until re-evaluated, an edited feature keeps its old answer. A feature removed
from `custom.json` stays in rows that already have it. Which definition each
answer was given under is tracked in `<brand>-<product>.custom_state.jsonl`
next to the output; do not delete it.

## Changing the schema

- Gemini fields and enums: `creative_analytics/video_schema.py`
- Brand profile fields and the research prompt: `creative_analytics/brand_profile.py`
- Fields measured from the video file: `creative_analytics/video_stats.py`
- Custom feature definitions and rules: `creative_analytics/custom.py`
