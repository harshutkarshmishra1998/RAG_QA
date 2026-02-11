from query.query_pipeline_v3 import process_user_query

def test_query_processing():

    query = "What are the high-level process steps performed after a covered institution fails under Part 370?"
    # query = "What are the high-level process steps performed after a covered institution fails under Part 370 and moreover what is described in the High-Level Process at Failure section?"
    # query = "What are the high-level process steps performed after a covered institution fails under Part 370? What activities must be completed within 24 hours after a bank failure according to Part 370?"
    # query = "What are the high-level process steps performed after a covered institution fails under Part 370? what is described in the High-Level Process at Failure section? What activities must be completed within 24 hours after a bank failure according to Part 370?"


    result = process_user_query(query)

    print("\nProcessed Query Output:\n")
    print(result)


if __name__ == "__main__":
    test_query_processing()