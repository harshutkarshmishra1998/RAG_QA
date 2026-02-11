import os
from pathlib import Path

from retrieval.retrieval_pipeline import retrieve_latest_query_chunks


def test_retrieve_latest_query():

    print("\n=== Running Retrieval Test ===\n")

    result = retrieve_latest_query_chunks()

    print(f"Query ID: {result['query_id']}")
    print(f"Original Query: {result['original_query']}\n")

    for i, sub in enumerate(result["subqueries"], start=1):
        print(f"--- Subquery {i} ---")
        print(f"Subquery ID: {sub['subquery_id']}")
        print(f"Total Queries Used: {len(sub['queries_used'])}")
        print(f"Retrieved Chunks: {len(sub['retrieved_chunks'])}")

        if sub["retrieved_chunks"]:
            top_chunk = sub["retrieved_chunks"][0]
            print(f"Top Chunk ID: {top_chunk['chunk_id']}")
            print(f"Top RRF Score: {top_chunk['rrf_score']}")
            print(f"Top Source Queries: {top_chunk['source_queries']}")
        else:
            print("No chunks retrieved.")

        print()

    print("=== Retrieval Test Completed ===\n")


if __name__ == "__main__":
    test_retrieve_latest_query()