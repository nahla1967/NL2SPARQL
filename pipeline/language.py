from langdetect import detect

def detect_language(question):
    lang = detect(question)
    if lang in ["ar", "fr", "en"]:
        return lang
    return "en"