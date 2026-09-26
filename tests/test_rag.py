import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.rag import (
    EmptyCorpusError,
    IndexDimensionError,
    LiteratureIndex,
    LiteratureRAG,
    RAGError,
    SentenceTransformerEmbedder,
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
    def test_sentence_transformer_loading_failure_becomes_expected_rag_error(self):
        class FailingSentenceTransformer:
            def __init__(self, model_name):
                raise OSError("cached model is unavailable")

        fake_module = types.SimpleNamespace(
            SentenceTransformer=FailingSentenceTransformer
        )

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with self.assertRaises(RAGError):
                SentenceTransformerEmbedder().encode(["Parkinson disease"])

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

    def test_literature_rag_maps_validated_source_ids_to_citations(self):
        pages = [
            {
                "document_id": "paper-a",
                "title": "Autonomic Study",
                "source": "autonomic.pdf",
                "page": 2,
                "text": "SCOPA autonomic symptoms and Parkinson progression.",
            },
            {
                "document_id": "paper-b",
                "title": "Cognition Study",
                "source": "cognition.pdf",
                "page": 4,
                "text": "MoCA cognition scores and Parkinson progression.",
            },
        ]
        captured = {}

        def fake_answerer(question, evidence):
            captured["question"] = question
            captured["evidence"] = evidence
            return {
                "answer": "自主神经症状与群体层面的纵向结局有关。",
                "cited_source_ids": [evidence[0]["source_id"]],
                "evidence_limitations": "不能推断个体因果关系。",
                "provider": "DeepSeek",
                "model": "test-model",
            }

        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            build_literature_index_from_pages(pages, index_dir, KeywordEmbedder())
            service = LiteratureRAG(
                LiteratureIndex.load(index_dir, KeywordEmbedder()),
                answerer=fake_answerer,
            )

            result = service.ask("autonomic progression", top_k=2)

        self.assertEqual(captured["question"], "autonomic progression")
        self.assertEqual(captured["evidence"][0]["title"], "Autonomic Study")
        self.assertEqual(result["citations"][0]["page"], 2)
        self.assertEqual(result["citations"][0]["title"], "Autonomic Study")
        self.assertIn("科研", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()
