"""One-time data ingestion script. Run this before using the RAG system."""
import sys
sys.path.insert(0, ".")

from src.rag_pipeline import MedicalImagingRAG


def main():
    print("=" * 60)
    print("Medical Imaging RAG - Data Ingestion")
    print("=" * 60)

    rag = MedicalImagingRAG()

    if rag._is_ingested:
        print("\nIndex already exists. Delete data/chroma_db/ to re-index.")
        response = input("Re-index anyway? (y/N): ").strip().lower()
        if response != "y":
            print("Skipping ingestion.")
            return
        # Clear existing index
        import shutil
        import config
        if config.CHROMA_DIR.exists():
            shutil.rmtree(config.CHROMA_DIR)
            print("Cleared existing index.")

    rag.ingest(max_samples=300)

    # Quick test
    print("\n" + "=" * 60)
    print("Quick test query...")
    result = rag.query("What are the most common findings in chest X-rays?")
    print(f"\nAnswer:\n{result.answer[:500]}...")
    print(f"\nSources used: {len(result.retrieval.documents)}")


if __name__ == "__main__":
    main()
