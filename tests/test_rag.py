import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.rag import (
    EmptyCorpusError,
    IndexDimensionError,
    LiteratureIndex,
    build_literature_index_from_pages,
    split_pages,
)


class KeywordEmbedder:
    model_name = "test-keyword-embedder"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("autonomic" in lowered or "scopa" in lowered),
                    float("cognition" in lowered or "moca" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class ThreeDimensionQueryEmbedder(KeywordEmbedder):
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class LiteratureIndexTests(unittest.TestCase):
    def test_split_pages_preserves_page_metadata_and_overlap(self):
        pages = [
            {
                "document_id": "paper-1",
                "title": "Autonomic Study",
                "source": "autonomic-study.pdf",
                "page": 3,
                "text": "ABCDEFGHIJ",
            }
        ]

        chunks = split_pages(pages, chunk_size=6, overlap=2)

        self.assertEqual([row["text"] for row in chunks], ["ABCDEF", "EFGHIJ"])
        self.assertEqual(chunks[0]["page"], 3)
        self.assertEqual(chunks[0]["source_id"], "paper-1-p3-c1")
        self.assertEqual(chunks[1]["source_id"], "paper-1-p3-c2")

    def test_build_and_load_index_returns_most_similar_evidence_first(self):
        pages = [
            {
                "document_id": "paper-a",
                "title": "Autonomic dysfunction in Parkinson disease",
                "source": "autonomic.pdf",
                "page": 1,
                "text": "SCOPA autonomic symptoms were evaluated in Parkinson disease.",
            },
            {
                "document_id": "paper-b",
                "title": "Cognition in Parkinson disease",
                "source": "cognition.pdf",
                "page": 7,
                "text": "MoCA cognition scores were followed over time.",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            summary = build_literature_index_from_pages(
                pages,
                index_dir,
                KeywordEmbedder(),
                chunk_size=200,
                overlap=20,
            )
            index = LiteratureIndex.load(index_dir, KeywordEmbedder())

            results = index.search("autonomic progression", top_k=2)

            self.assertEqual(summary["document_count"], 2)
            self.assertEqual(summary["chunk_count"], 2)
            self.assertEqual(results[0]["title"], "Autonomic dysfunction in Parkinson disease")
            self.assertEqual(results[0]["page"], 1)
            self.assertGreater(results[0]["score"], results[1]["score"])
            metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["embedding_model"], "test-keyword-embedder")
            self.assertEqual(metadata["dimension"], 2)
            self.assertTrue((index_dir / "embeddings.npy").exists())

    def test_build_rejects_pages_without_usable_text(self):
        pages = [
            {
                "document_id": "empty",
                "title": "Image only",
                "source": "image-only.pdf",
                "page": 1,
                "text": "   ",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EmptyCorpusError):
                build_literature_index_from_pages(
                    pages,
                    Path(directory),
                    KeywordEmbedder(),
                )

    def test_search_rejects_query_embedding_with_wrong_dimension(self):
        pages = [
            {
                "document_id": "paper-a",
                "title": "Autonomic Study",
                "source": "autonomic.pdf",
                "page": 1,
                "text": "autonomic symptoms",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            build_literature_index_from_pages(pages, index_dir, KeywordEmbedder())
            index = LiteratureIndex.load(index_dir, ThreeDimensionQueryEmbedder())

            with self.assertRaises(IndexDimensionError):
                index.search("autonomic", top_k=1)

    def test_split_pages_rejects_overlap_that_cannot_advance(self):
        with self.assertRaises(ValueError):
            split_pages([], chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
