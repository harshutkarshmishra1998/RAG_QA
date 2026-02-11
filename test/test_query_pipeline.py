from query.query_pipeline_v2 import process_user_query

def test_query_processing():

    query = "What are the high-level process steps performed after a covered institution fails under Part 370 and moreover what is described in the High-Level Process at Failure section?"

    result = process_user_query(query)

    print("\nProcessed Query Output:\n")
    print(result)


if __name__ == "__main__":
    test_query_processing()