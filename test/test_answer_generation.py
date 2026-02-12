from answer_generation.answer_generation_v3 import generate_answer_from_last_entry
import json

# =====================================================
# TEST RUNNER
# =====================================================

if __name__ == "__main__":
    result = generate_answer_from_last_entry()

    print("\nGenerated Result:\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))