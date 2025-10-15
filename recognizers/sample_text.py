import pytesseract
from pytesseract import Output

class Recognizer:
    def recognize(self, image_np):
        data = pytesseract.image_to_data(image_np, output_type=Output.DICT)
        results = []
        n = len(data["level"])
        for i in range(n):
            text = data["text"][i].strip()
            if text:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                conf = data["conf"][i]
                try:
                    score = float(conf) / 100
                except (ValueError, TypeError):
                    score = None
                results.append({
                    "label": text,
                    "score": score,
                    "box": [x, y, x + w, y + h]
                })
        return results