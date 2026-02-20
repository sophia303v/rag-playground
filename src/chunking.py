"""Chunking strategies for medical reports."""
from dataclasses import dataclass
from src.data_loader import MedicalReport


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    chunk_id: str
    report_uid: str
    section: str  # 'indication', 'findings', 'impression', or 'full'
    metadata: dict


def chunk_by_section(report: MedicalReport) -> list[Chunk]:
    """
    Chunk a medical report by its natural sections.

    Radiology reports have a natural structure:
    - Indication: Why the exam was ordered
    - Findings: What was observed
    - Impression: Summary/conclusion

    Each section is semantically complete, making it an ideal chunk boundary.
    """
    chunks = []

    sections = {
        "indication": report.indication,
        "findings": report.findings,
        "impression": report.impression,
    }

    for section_name, text in sections.items():
        if text and text.strip():
            chunk = Chunk(
                text=f"[Report {report.uid}] {section_name.capitalize()}: {text.strip()}",
                chunk_id=f"{report.uid}_{section_name}",
                report_uid=report.uid,
                section=section_name,
                metadata={
                    "uid": report.uid,
                    "section": section_name,
                    "has_images": len(report.image_paths) > 0 or len(report.images) > 0,
                },
            )
            chunks.append(chunk)

    # Also create a full-text chunk for broader context retrieval
    if report.full_text.strip():
        chunks.append(
            Chunk(
                text=f"[Report {report.uid}] {report.full_text}",
                chunk_id=f"{report.uid}_full",
                report_uid=report.uid,
                section="full",
                metadata={
                    "uid": report.uid,
                    "section": "full",
                    "has_images": len(report.image_paths) > 0 or len(report.images) > 0,
                },
            )
        )

    return chunks


def chunk_reports(reports: list[MedicalReport]) -> list[Chunk]:
    """Chunk all reports."""
    all_chunks = []
    for report in reports:
        all_chunks.extend(chunk_by_section(report))
    print(f"Created {len(all_chunks)} chunks from {len(reports)} reports.")
    return all_chunks
