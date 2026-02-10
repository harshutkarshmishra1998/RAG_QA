from chunk.chunk import chunk_content_units

def test_chunking_runs():
    out = chunk_content_units("storage/content_units_cleaned.jsonl")
    assert out.exists()
    assert out.name.endswith("_chunked.jsonl")

if __name__ == "__main__":
    test_chunking_runs()
    print("Chunking test passed")