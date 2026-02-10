from pathlib import Path
import faiss


def inspect_faiss(path: str) -> None:
    path = Path(path) #type: ignore
    if not path.exists(): #type: ignore
        raise FileNotFoundError(path)

    index = faiss.read_index(str(path))

    print("FAISS index loaded")
    print(f"Index type   : {type(index).__name__}")
    print(f"Vector dim  : {index.d}")
    print(f"Total vecs  : {index.ntotal}")
    print(f"Is trained  : {index.is_trained}")


if __name__ == "__main__":
    inspect_faiss("storage/content_units.faiss")