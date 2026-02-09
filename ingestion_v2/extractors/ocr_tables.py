from PIL import Image
import pytesseract


def extract_ocr_table_from_image(path):
    data = pytesseract.image_to_data(
        Image.open(path), output_type=pytesseract.Output.DICT
    )
    rows = {}
    for i, txt in enumerate(data["text"]):
        if txt.strip():
            rows.setdefault(data["block_num"][i], []).append(txt)
    return list(rows.values())
