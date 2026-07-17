from router import route

TEST_QUESTIONS = [
    # the 3 real out_of_scope rows — should all now say out_of_scope
    ("What's the weather like at VIE airport today?", "out_of_scope"),
    ("Can I bring a pet on flight FR947?", "out_of_scope"),
    ("What is the history of the airline industry?", "out_of_scope"),
    # controls — make sure we didn't break anything real
    ("What is the elevation of VIE?", "single_kg2"),
    ("What country does flight LO225 land in?", "cross_kg"),
    ("What is the gate of flight OS529?", "single_kg1"),
]

for question, expected in TEST_QUESTIONS:
    result = route(question)
    actual = result["query_type"]
    status = "OK " if actual == expected else "FAIL"
    print(f"[{status}] expected={expected:12s} got={actual:12s} | {question}")