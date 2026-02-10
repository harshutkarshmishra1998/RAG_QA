from clean_normalize.clean_normalize import clean_content_units_file

if __name__ == "__main__":
    output_path = clean_content_units_file(
        "storage/content_units_dummy.jsonl"
    )
    print(f"Cleaned file written to: {output_path}")