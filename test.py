from router import route

tests = [
    "ما هي السرعة العمودية للرحلة VF535؟",
    "What airline operates flight MAE107?",
    "Quelle compagnie aérienne opère le vol MAE107?",
    "ما هي شركة الطيران التي تشغل الرحلة MAE107؟",
]

for q in tests:
    r = route(q)
    print(f"{q!r} -> {r['query_type']}")