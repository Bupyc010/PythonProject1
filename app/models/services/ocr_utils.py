import easyocr

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ru", "en"], gpu=False)
    return _reader

def read_text(image):
    reader = get_reader()
    result = reader.readtext(image, detail=0, paragraph=True)
    if not result:
        return ""
    return " ".join([r.strip() for r in result if r.strip()])