"""Command line entry point.

    python -m creative_analytics profile --input videos.csv
    python -m creative_analytics extract --input videos.csv --custom custom.json
    python -m creative_analytics run     --input videos.csv --custom custom.json

    python -m creative_analytics custom-status     --input videos.csv --custom custom.json
    python -m creative_analytics reevaluate-custom --input videos.csv --custom custom.json
"""

from __future__ import annotations

import argparse
import sys

from .brand_profile import ASSET_FIELDS, empty_profile, get_or_create_profile, profile_path
from .client import DEFAULT_MODEL, make_client
from .custom import custom_status, load_custom
from .inputs import load_jobs
from .video_extraction import extract_job, features_path, reevaluate_custom


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="creative_analytics")
    parser.add_argument(
        "command",
        choices=["profile", "extract", "run", "custom-status", "reevaluate-custom"],
        help="profile = Phase 1 only; extract = Phase 2 for videos not yet extracted "
        "(creates a profile if missing); run = both; custom-status = show which "
        "extracted videos have missing or outdated custom answers (no Gemini calls); "
        "reevaluate-custom = update those custom answers.",
    )
    parser.add_argument("--input", required=True, help="Path to a .csv or .json video list.")
    parser.add_argument(
        "--custom", help="Path to custom.json with custom feature definitions (optional)."
    )
    parser.add_argument("--brand", help="Only this brand.")
    parser.add_argument("--product", help="Only this product.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--refresh-profile",
        action="store_true",
        help="Re-run Phase 1 even if a saved brand profile exists (overwrites edits).",
    )
    parser.add_argument(
        "--skip-brand-research",
        action="store_true",
        help="Skip Phase 1 and ignore any saved brand profile. Gemini then "
        "identifies sonic assets, product hooks and brand assets as free text.",
    )
    parser.add_argument(
        "--skip-video-stats",
        action="store_true",
        help="Do not download/measure the video file. Duration, fps, aspect ratio "
        "and the shot fields are saved as null.",
    )
    return parser


def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _print_status(status: dict) -> None:
    print(
        f"  custom: {status['videos_to_update']} of {status['videos_extracted']} "
        "extracted videos need re-evaluation"
    )
    for name, counts in status["features"].items():
        print(f"    {name}: {counts['missing']} missing, {counts['changed']} outdated")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    all_jobs = load_jobs(args.input)
    custom = load_custom(args.custom)
    for label in custom.unmatched([(j.brand, j.product) for j in all_jobs]):
        print(f"Warning: '{label}' in {args.custom} is not in {args.input}; ignored.")

    jobs = [
        job
        for job in all_jobs
        if (not args.brand or _same(job.brand, args.brand))
        and (not args.product or _same(job.product, args.product))
    ]
    if not jobs:
        print("No brand/product in the input matches --brand/--product.")
        return 1

    client = None if args.command == "custom-status" else make_client()
    failed = 0

    for job in jobs:
        print(f"\n{job.brand} / {job.product}")
        features = custom.features_for(job.brand, job.product)
        results = features_path(args.output_dir, job)

        if args.command == "custom-status":
            _print_status(custom_status(results, features))
            continue

        if args.command == "reevaluate-custom":
            counts = reevaluate_custom(
                client, job, features, args.output_dir, args.model, args.max_workers
            )
            failed += counts["failed"]
            print(
                f"  custom: {counts['updated']} videos updated, "
                f"{counts['up_to_date']} already up to date, {counts['failed']} failed"
            )
            continue

        if args.skip_brand_research:
            profile = empty_profile(job.brand, job.product)
            print("  profile: skipped (brand-data fields will be free text)")
        else:
            profile = get_or_create_profile(
                client, job, args.output_dir, args.model, refresh=args.refresh_profile
            )
            summary = ", ".join(f"{len(getattr(profile, f))} {f}" for f in ASSET_FIELDS)
            print(f"  profile: {profile_path(args.output_dir, job)} ({summary})")
        if args.command == "profile":
            continue

        counts = extract_job(
            client,
            job,
            profile,
            args.output_dir,
            args.model,
            args.max_workers,
            args.skip_video_stats,
            features,
        )
        failed += counts["failed"]
        print(
            f"  videos: {counts['done']} extracted, {counts['skipped']} already done, "
            f"{counts['failed']} failed -> {results}"
        )
        status = custom_status(results, features)
        if status["videos_to_update"]:
            _print_status(status)
            print("    (not updated automatically; run reevaluate-custom to update)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
