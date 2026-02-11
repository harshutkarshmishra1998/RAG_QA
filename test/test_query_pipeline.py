from query.query_pipeline import process_user_query

def test_query_processing():

    query = "What are the high-level process steps performed after a covered institution fails under Part 370?"

    result = process_user_query(query)

    print("\nProcessed Query Output:\n")
    print(result)


if __name__ == "__main__":
    test_query_processing()