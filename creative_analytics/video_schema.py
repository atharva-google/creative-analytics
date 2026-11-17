"""Phase 2 schema: what Gemini extracts from one video.

These are the fields of the original notebook. Three of them can take brand
data from the Phase 1 profile:

- brand_sonic_asset.sonic_asset         -> profile.sonic_assets
- hero_product_hook.hook_type           -> profile.key_visuals
- distinctive_brand_asset.brand_asset   -> profile.distinctive_brand_assets

Each field is a list (one entry per distinct asset found in the video, empty
if none). Each value is a short text label. Where the profile has a list, Gemini uses
the exact name of a known asset when it matches and describes anything else in
its own words. Where the list is empty (or brand research was skipped), Gemini
always describes the asset in its own words.

Fields measured in code (metadata, shot counts) are not here; see
`video_stats.py`.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, Type

from pydantic import BaseModel, Field, create_model

from .brand_profile import BrandAsset, BrandProfile
from .custom import CustomFeature, build_custom_model

CUSTOM_BLOCK_DESCRIPTION = (
    "User-defined features. Answer each one by following its own description, "
    "based only on this video."
)


# ------------------------------------------------------------------- talent
class CelebrityRole(str, Enum):
    HOST = "Direct Host / Presenter (Talking to Camera)"
    USER = "Active User / Beneficiary (Demonstrating/Using Product)"
    OBSERVER_REACTOR = "Observer / Reactor"


class CelebrityInstance(BaseModel):
    name: str = Field(..., description="Name of the celebrity.")
    first_appearance_sec: float = Field(
        ...,
        description="Exact timestamp in seconds when this celebrity first appears on screen.",
    )
    role: CelebrityRole = Field(
        ..., description="The primary role this celebrity plays in the ad."
    )


class NonCelebArchetype(str, Enum):
    UGC_CREATOR = "UGC Creator / Social Influencer"
    ACTOR = "Everyday Actor (Relatable / Scripted)"
    CGI_MASCOT = "Animated Character / Brand Mascot"


class TalentAndStarPower(BaseModel):
    celebrity_cast: Optional[List[CelebrityInstance]] = Field(
        ...,
        description="List of details for each celebrity detected. null/None if no celebrity appears.",
    )
    supporting_cast: Optional[List[NonCelebArchetype]] = Field(
        ...,
        description=(
            "List of non-celebrity character archetypes present in the ad. null/None if no archetypes appear."
        ),
    )


# ------------------------------------------------------------------ setting
class LocationType(str, Enum):
    INDOOR = "Indoor"
    OUTDOOR = "Outdoor"


class Setting(BaseModel):
    timestamp: float = Field(
        ...,
        description="The exact timestamp in seconds when the ad transitions into this setting.",
    )
    location_type: LocationType = Field(
        ..., description="Classify the scene location backdrop."
    )
    description: str = Field(
        ..., description="3-4 word description of the backdrop setting."
    )
    ambient_mood: str = Field(
        ...,
        description="3-4 word description of the overall emotional mood of the backdrop setting.",
    )


class SettingAnalysis(BaseModel):
    primary_setting: Setting = Field(
        ..., description="Main environment where the story takes place."
    )
    setting_transition_count: int = Field(
        ...,
        description="Number of times the video clearly transitions to a completely different physical environment.",
    )


# ------------------------------------------------------- attract (first 5s)
class FramingDensity(str, Enum):
    MACRO = "Tight Macro / Close-Up (Mobile-Optimized)"
    MEDIUM = "Medium Shot (Waist-Up / Balanced)"
    WIDE = "Wide / Establishing Shot"
    UI = "Screen Recording / UI Viewport"


class AttractFeatures(BaseModel):
    """Gemini's part of the attract block. Saved as `..._0-5s` in the output;
    shot_count_0-5s and first_cut_timestamp_sec are added by code."""

    framing_density: FramingDensity = Field(
        ...,
        description="The dominant camera framing scale utilized in the first 5 seconds of the video.",
    )
    product_interaction_0_5s: bool = Field(
        ...,
        description=(
            "Judge only the first 5 seconds of the video (0.0s to 5.0s). "
            "Set to True if a human character is physically holding, touching, tasting, consuming, or actively demonstrating the product. "
            "Must be False if the product is only shown on its own (e.g., table-top / pack-only shot)."
        ),
    )
    supers_present_0_5s: bool = Field(
        ...,
        description=(
            "Judge only the first 5 seconds of the video (0.0s to 5.0s). "
            "Set to True if any text overlays, titles, or graphical supers are present on-screen. Exclude text found physically on the product's packaging."
        ),
    )


# ------------------------------------------------------------------ connect
class NarrativeArc(str, Enum):
    PROBLEM_SOLUTION = "Brand Saves the Day (Problem - Solution)"
    SLICE_OF_LIFE = "Slice-of-Life / Social Moment"
    DEMO = "Direct Product Demo / Walkthrough"
    TESTIMONIAL = "Testimonial / Social Proof"
    MONTAGE = "High-Energy Montage / Hype Reel"
    TUTORIAL = "Educational / How-To / Tutorial"


class LinguisticRegister(str, Enum):
    SCRIPTED_VO = "Commercial / Scripted Voiceover"
    CONVERSATIONAL = "Casual Conversational (Creator/Peer)"
    HIGH_ENERGY = "High-Energy / Urgent Delivery"
    EMOTIONAL = "Inspirational / Emotional Storytelling"
    SILENT = "No Spoken Script / Instrumental Only"


class AudioTrackElement(str, Enum):
    MUSIC = "Background Music / Soundtrack"
    VO = "Voiceover (Omniscient/off-screen narrator)"
    DIALOGUE = "On-Screen Dialogue (Characters speaking on camera)"
    SOUND_EFFECTS = "Sound Effects (Signature swooshes, crunches, pops, diegetic sound)"


class MusicTempo(str, Enum):
    VERY_FAST = "Very Fast (>140 BPM)"
    UPBEAT = "Upbeat / Moderate (100-140 BPM)"
    SLOW = "Slow / Ambient (<100 BPM)"
    NO_MUSIC = "No Background Music"


class BrandHumorIntegration(str, Enum):
    INTRINSIC = "Intrinsic (Brand/product is essential to the punchline)"
    EXTRINSIC = "Extrinsic (Humor is an unrelated setup/backdrop)"


class HumorDetails(BaseModel):
    integration: BrandHumorIntegration = Field(
        ...,
        description="Evaluate how deeply the brand or product is integrated into the comedy.",
    )
    archetype: str = Field(
        ...,
        description="3-4 word description of primary stylistic archetype of comedic relief utilized.",
    )


class ConnectFeatures(BaseModel):
    narrative_arc: NarrativeArc = Field(
        ...,
        description="The overarching storytelling structure governing the ad's progression.",
    )
    linguistic_register: LinguisticRegister = Field(
        ...,
        description="The dominant delivery style and tone of the spoken voiceover or dialogue.",
    )
    audio_mix: List[AudioTrackElement] = Field(
        ...,
        description="A list of all distinct audio elements present in the video's sound mix. Select all that apply. If the video has absolutely no audio (completely silent), this list MUST be empty [].",
    )
    music_tempo_bpm: MusicTempo = Field(
        ...,
        description="The speed bracket of the background music soundtrack. Output a realistic estimate.",
    )
    dialogue_wpm: int = Field(
        ...,
        description="The spoken dialogue or voiceover pacing, calculated roughly as (total words spoken / video duration in minutes). Output a realistic estimate.",
    )
    jargon_intensity: int = Field(
        ...,
        description=(
            "The density and intensity of specialized, clinical, or highly technical terminology used in the script. "
            "Scale: "
            "0 (No jargon / entirely everyday conversational language), "
            "1 (Low / casual product claims or simple technical names), "
            "2 (Medium / clear functional terminology or active ingredients), "
            "3 (High / heavy scientific, clinical, or corporate terminology)."
        ),
    )
    humor_details: Optional[HumorDetails] = Field(
        ...,
        description="Detailed metrics about the comedy. null/None if no humor is present.",
    )
    product_utilisation_present: bool = Field(
        ...,
        description="Set to True is product is actively shown being handled, held, prepared, or consumed on-screen.",
    )


# ------------------------------------------------------------------- direct
class CtaType(str, Enum):
    TRANSACTIONAL = "Transactional / Direct Conversion (Buy, Shop, Order)"
    LEAD_GEN = "Lead Gen / Sign-Up (Register, Apply, Book Demo)"
    OCCASION = "Occasion / Contextual (Try for Dinner, Switch Today)"
    DISCOVERY = "Discovery / Learn More (Explore, Visit Site)"
    SLOGAN = "Brand Statement / Slogan Only"
    NONE = "None"


class CtaDelivery(str, Enum):
    DUAL = "Dual Delivery (Super + Voiceover)"
    VISUAL = "Visual Super Only"
    AUDIO = "Audio Voiceover Only"
    NONE = "None"


class DirectFeatures(BaseModel):
    cta_type: CtaType
    cta_delivery: CtaDelivery


# ---------------------------------------------- brand (uses the brand profile)
def _known(assets: List[BrandAsset]) -> str:
    return "; ".join(f"'{a.name}' = {a.description}" for a in assets)


def _asset_model(
    model_name: str,
    value_field: str,
    assets: List[BrandAsset],
    timestamp_description: str,
    known_description: str,
    free_text_description: str,
) -> Type[BaseModel]:
    """A {timestamp, <value_field>} model for one brand-data field.

    The value is always a short text label. With brand research, the model is
    told to use the exact name of a known asset when it matches, and to
    describe the asset in its own words when it is not on the list. Without
    research (empty list), it always describes the asset in its own words.
    """
    if assets:
        description = (
            f"{known_description} If what is present matches one of the brand's known assets, "
            f"output that asset's name exactly as written here: {_known(assets)}. "
            f"If it is clearly present but matches none of them, describe it instead: {free_text_description}"
        )
    else:
        description = free_text_description
    return create_model(
        model_name,
        timestamp=(float, Field(..., description=timestamp_description)),
        **{value_field: (str, Field(..., description=description))},
    )


def build_brand_features(profile: BrandProfile) -> Type[BaseModel]:
    """BrandFeatures for one brand.

    The three brand-data fields are text labels. The profile's lists, where
    present, are given to the model as the preferred names to use.
    """
    SonicAsset = _asset_model(
        "SonicAsset",
        "sonic_asset",
        profile.sonic_assets,
        "The exact timestamp in seconds when this sonic asset is first heard.",
        "The brand sonic asset heard in the audio track.",
        "3-4 words highly descriptive custom string identifying any official brand sonic assets, sound mnemonics, or signature product interaction sounds detected in the audio track. "
        "Examples: 'KitKat Acoustic Snap', 'Nescafé Shaker / Stir Rhythm', 'Intel Bong', 'Xbox Startup Chime'.",
    )
    ProductHook = _asset_model(
        "ProductHook",
        "hook_type",
        profile.key_visuals,
        "The exact timestamp in seconds when this product hook first begins.",
        "The sensory or tactile product hook shown.",
        "A short, highly descriptive 3-4 word phrase capturing the sensory, tactile, or physical trigger used to make the product look appealing (ex: Steam Rise, Condensation Drip, Sauce Sizzle, Cheese Pull, Metallic Shine).",
    )
    DistinctiveBrandAsset = _asset_model(
        "DistinctiveBrandAsset",
        "brand_asset",
        profile.distinctive_brand_assets,
        "The exact timestamp in seconds when this distinctive brand asset first appears prominently.",
        "The distinctive brand asset featured.",
        "A short, highly descriptive 3-4 word phrase identifying the distinctive brand asset (DBA) featured, other than a basic text logo. This includes official brand mascots, proprietary animated characters, signature visual devices, unique product packaging shapes, or the strict usage of signature brand colors as a dominant visual theme.",
    )
    return create_model(
        "BrandFeatures",
        first_brand_touchpoint_sec=(
            float,
            Field(
                ...,
                description="The exact timestamp in seconds when the brand or target offering is first seen or heard. This is triggered by the first appearance of a brand logo, branded product packaging, visual sub-brand text, or the first spoken mention of the brand name.",
            ),
        ),
        logo_present=(
            bool,
            Field(
                ...,
                description="Set to True if the official brand logo is clearly shown on-screen at any point in the video.",
            ),
        ),
        spoken_brand_name=(
            bool,
            Field(
                ...,
                description="Set to True if the brand name is audibly spoken at any point in the video, whether or not any branding is on screen at that moment. Judge this from the audio alone.",
            ),
        ),
        see_and_say_branding=(
            bool,
            Field(
                ...,
                description="Set to True only if the ad displays visual brand assets (logo, packaging, or text) at the exact same moment the brand name is spoken aloud. This 'see-and-say' alignment must occur at least once in the video. A spoken brand name with no branding on screen at that moment does not count.",
            ),
        ),
        brand_total_screen_time_sec=(
            float,
            Field(
                ...,
                description="The cumulative, total duration in seconds that the brand's visual identity (logo, branded packaging, or clearly branded product) is prominently visible on screen. CRITICAL: Do not count background clutter; only count instances where the branding is a primary or highly legible focal point.",
            ),
        ),
        brand_sonic_asset=(
            List[SonicAsset],
            Field(
                ...,
                description="Timing and classification of every distinct brand sonic signature heard in the ad. One entry per distinct sonic asset, with the timestamp of when it is first heard; do not repeat the same asset. Empty list [] if the ad only uses standard background music, stock sound effects, or basic voiceover.",
            ),
        ),
        hero_product_hook=(
            List[ProductHook],
            Field(
                ...,
                description="Every distinct visually or audibly compelling sensory or tactile product hook present in the ad. One entry per distinct hook, with the timestamp of when it first begins; do not repeat the same hook. Empty list [] if no distinct sensory product hooks are shown.",
            ),
        ),
        distinctive_brand_asset=(
            List[DistinctiveBrandAsset],
            Field(
                ...,
                description="Every distinct distinctive brand asset prominently featured in the creative. One entry per distinct asset, with the timestamp of when it first appears; do not repeat the same asset. Empty list [] if the creative features no distinctive brand asset beyond a basic text logo.",
            ),
        ),
    )


def build_output_model(
    profile: BrandProfile, custom_features: Sequence[CustomFeature] = ()
) -> Type[BaseModel]:
    """The Phase 2 response schema for one brand profile.

    Custom features, if any, are added as a `custom` block so they are
    answered in the same Gemini call.
    """
    extra = {}
    if custom_features:
        extra["custom"] = (
            build_custom_model(custom_features),
            Field(..., description=CUSTOM_BLOCK_DESCRIPTION),
        )
    return create_model(
        "Output",
        talent_and_star_power=(
            TalentAndStarPower,
            Field(..., description="Celebrity and non-celebrity character casting."),
        ),
        setting_analysis=(
            SettingAnalysis,
            Field(..., description="Environmental backdrop and transition counts."),
        ),
        attract_features=(
            AttractFeatures,
            Field(..., description="What happens in the first 5 seconds of the video."),
        ),
        brand_features=(
            build_brand_features(profile),
            Field(
                ...,
                description="Brand prominence, touchpoints, logo, spoken name, see-and-say alignment, and sonic assets.",
            ),
        ),
        connect_features=(
            ConnectFeatures,
            Field(
                ...,
                description="Narrative arc, audio mix, tempo, register, and comedy integration.",
            ),
        ),
        direct_features=(
            DirectFeatures,
            Field(..., description="Call to action type and delivery."),
        ),
        **extra,
    )
