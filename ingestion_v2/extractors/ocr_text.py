from PIL import Image
import pytesseract


def extract_ocr_text_from_image(path):
    text = pytesseract.image_to_string(Image.open(path)).strip()
    return text