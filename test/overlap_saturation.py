import faiss
import numpy as np


MAX_ACCEPTABLE_SIM = 0.92


def detect_overlap():
    index = faiss.read_index("storage/content_units.faiss")

    overlaps = []

    for i in range(index.ntotal - 1):
        v1 = index.reconstruct(i)
        v2 = index.reconstruct(i + 1)
        sim = float(np.dot(v1, v2))
        overlaps.append(sim)

    avg = sum(overlaps) / len(overlaps)
    high = sum(1 for s in overlaps if s > MAX_ACCEPTABLE_SIM)

    print(f"Avg adjacent similarity: {avg:.4f}")
    print(f"Over-saturated pairs (> {MAX_ACCEPTABLE_SIM}): {high}")
    print(f"Total pairs checked: {len(overlaps)}")


if __name__ == "__main__":
    detect_overlap()