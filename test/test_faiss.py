import api_keys
from embedding.faiss_embedder import build_faiss_db


def test_faiss_creation():
    faiss_path = build_faiss_db(
        "storage/content_units_cleaned_chunked.jsonl"
    )

    assert faiss_path.exists()


if __name__ == "__main__":
    test_faiss_creation()
    print("FAISS creation test passed")