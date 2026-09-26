"""Build the local page-aware medical literature vector index."""

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.rag import build_literature_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_DIR / "knowledge_base" / "source_documents",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_DIR / "knowledge_base" / "index",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    summary = build_literature_index(
        args.source_dir,
        args.index_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(
        "Literature index saved to: {path} ({documents} documents, {chunks} chunks)".format(
            path=args.index_dir.resolve(),
            documents=summary["document_count"],
            chunks=summary["chunk_count"],
        )
    )


if __name__ == "__main__":
    main()
