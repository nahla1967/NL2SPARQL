from langdetect import detect, DetectorFactory

# Setting a fixed seed makes langdetect deterministic.
# Without this, the same input can produce different outputs
# across runs because langdetect uses a probabilistic model internally.
# Reproducibility is required for a valid thesis evaluation.
DetectorFactory.seed = 0

def detect_language(question):
    """
    Detects the language of the input question.
    Supported languages: Arabic (ar), French (fr), English (en).
    Falls back to English for any unrecognized language.
    """
    lang = detect(question)
    if lang in ["ar", "fr", "en"]:
        return lang
    return "en"