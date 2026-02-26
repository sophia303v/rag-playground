"""BM25 sparse retrieval index over chunked documents."""
from rank_bm25 import BM25Okapi


class BM25Index:
    """Wrapper around BM25Okapi for sparse keyword retrieval."""

    def __init__(self):
        self._index = None
        self._docs = []
        self._metadatas = []

    def build(self, docs: list[str], metadatas: list[dict]):
        """Build BM25 index from documents and metadata."""
        self._docs = docs
        self._metadatas = metadatas
        tokenized = [doc.lower().split() for doc in docs]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, n_results: int = 20) -> tuple[list[str], list[dict], list[float]]:
        """Search the BM25 index.

        Returns:
            (documents, metadatas, scores) — top n_results by BM25 score.
        """
        if self._index is None:
            return [], [], []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        # Get top-n indices sorted by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_results]

        docs = [self._docs[idx] for idx, _ in ranked]
        metas = [self._metadatas[idx] for idx, _ in ranked]
        top_scores = [score for _, score in ranked]

        return docs, metas, top_scores


# Module-level cache (same pattern as _cross_encoder in retriever.py)
_bm25_index: BM25Index | None = None


def get_or_build_index(collection) -> BM25Index:
    """Get cached BM25 index, building from ChromaDB collection on first call."""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    print("Building BM25 index from ChromaDB collection...")
    # Pull all documents from the collection
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"] or []
    metas = all_data["metadatas"] or []

    _bm25_index = BM25Index()
    _bm25_index.build(docs, metas)
    print(f"BM25 index built with {len(docs)} documents.")
    return _bm25_index
