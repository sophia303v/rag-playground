"""
Download OpenI chest X-ray images from HuggingFace and update reports.json.

Usage:
    python scripts/download_openi_images.py                # Download all (~3826 reports)
    python scripts/download_openi_images.py --max-samples 100  # Download first 100

Images are saved to data/openi/images/{uid}_frontal.png and {uid}_lateral.png.
reports.json image_paths field is updated to point to local paths.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def download_images(max_samples: int | None = None):
    from datasets import load_dataset

    openi_dir = config.DATA_DIR / "openi"
    images_dir = openi_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    reports_path = openi_dir / "reports.json"

    if not reports_path.exists():
        print(f"Error: {reports_path} not found. Run the OpenI report download first.")
        sys.exit(1)

    # Load existing reports for UID lookup
    with open(reports_path) as f:
        reports = json.load(f)
    uid_to_report = {r["uid"]: r for r in reports}

    # Load HF dataset
    split = "train" if max_samples is None else f"train[:{max_samples}]"
    print(f"Loading HuggingFace ykumards/open-i dataset (split={split})...")
    ds = load_dataset("ykumards/open-i", split=split)

    saved_count = 0
    skipped_count = 0

    for i, item in enumerate(ds):
        uid = str(item["uid"])

        if uid not in uid_to_report:
            skipped_count += 1
            continue

        image_paths = []

        # Save frontal image
        frontal_bytes = item.get("img_frontal")
        if frontal_bytes and len(frontal_bytes) > 0:
            frontal_path = images_dir / f"{uid}_frontal.png"
            frontal_path.write_bytes(frontal_bytes)
            image_paths.append(f"images/{uid}_frontal.png")

        # Save lateral image
        lateral_bytes = item.get("img_lateral")
        if lateral_bytes and len(lateral_bytes) > 0:
            lateral_path = images_dir / f"{uid}_lateral.png"
            lateral_path.write_bytes(lateral_bytes)
            image_paths.append(f"images/{uid}_lateral.png")

        uid_to_report[uid]["image_paths"] = image_paths
        saved_count += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} items ({saved_count} saved, {skipped_count} skipped)...")

    # Write updated reports.json
    updated_reports = list(uid_to_report.values())
    with open(reports_path, "w") as f:
        json.dump(updated_reports, f, indent=2)

    # Summary
    total_images = len(list(images_dir.glob("*.png")))
    print(f"\nDone!")
    print(f"  Reports processed: {saved_count} (skipped {skipped_count})")
    print(f"  Total images saved: {total_images}")
    print(f"  Images directory: {images_dir}")
    print(f"  Updated: {reports_path}")


def main():
    parser = argparse.ArgumentParser(description="Download OpenI X-ray images from HuggingFace")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of HF dataset items to process")
    args = parser.parse_args()
    download_images(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
