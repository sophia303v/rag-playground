"""Load and parse Indiana Chest X-ray dataset from Hugging Face."""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class MedicalReport:
    """A single radiology report with associated images."""
    uid: str
    indication: str = ""
    findings: str = ""
    impression: str = ""
    image_paths: list[str] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list, repr=False)

    @property
    def full_text(self) -> str:
        """Combine all report sections into one text."""
        parts = []
        if self.indication:
            parts.append(f"Indication: {self.indication}")
        if self.findings:
            parts.append(f"Findings: {self.findings}")
        if self.impression:
            parts.append(f"Impression: {self.impression}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "indication": self.indication,
            "findings": self.findings,
            "impression": self.impression,
            "image_paths": self.image_paths,
        }


def load_openi_from_huggingface(max_samples: int = 500) -> list[MedicalReport]:
    """
    Load Indiana Chest X-ray dataset from Hugging Face.
    Uses the ykumards/open-i dataset.

    Args:
        max_samples: Maximum number of reports to load (for MVP speed)

    Returns:
        List of MedicalReport objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading OpenI dataset from Hugging Face (max {max_samples} samples)...")
    dataset = load_dataset("ykumards/open-i", split=f"train[:{max_samples}]")

    reports = []
    for i, item in enumerate(dataset):
        report = MedicalReport(
            uid=str(item.get("uid", f"report_{i}")),
            indication=item.get("indication", "") or "",
            findings=item.get("findings", "") or "",
            impression=item.get("impression", "") or "",
        )

        # Handle images if present in the dataset
        if "image" in item and item["image"] is not None:
            report.images = [item["image"]] if not isinstance(item["image"], list) else item["image"]

        # Only keep reports that have meaningful text
        if report.findings or report.impression:
            reports.append(report)

    print(f"Loaded {len(reports)} reports with content.")
    return reports


def load_reports_from_json(json_path: str | Path) -> list[MedicalReport]:
    """Load previously saved reports from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return [MedicalReport(**item) for item in data]


def save_reports_to_json(reports: list[MedicalReport], json_path: str | Path):
    """Save reports to JSON for caching."""
    data = [r.to_dict() for r in reports]
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(reports)} reports to {json_path}")
