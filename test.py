from pipeline.mapper import load_lexicon, map_property_cascade
from kg_registry import get_lexicon

lexicon_path = get_lexicon("university")
lexicon = load_lexicon(lexicon_path)

test_cases = [
    ("enrolled in which class",              "takesCourse"),
    ("where did they get their masters",     "mastersDegreeFrom"),
    ("is she tenured",                       "tenured"),
    ("what is his name",                     "name"),
    ("what is her job title",                "title"),
    ("which office is he in",                "officeNumber"),
    ("who is the dean of the college",       "headOfCollege"),
    ("qui a écrit cet article",              "publicationAuthor"),
    ("appartient à quelle université",       "subOrganizationOf"),
    ("مساعد تدريس لأي مادة",                  "teachingAssistantOf"),
]

correct = 0
for text, expected in test_cases:
    result = map_property_cascade(text, lexicon, lexicon_path)
    got = result[0]
    status = "OK" if got == expected else "WRONG"
    if got == expected:
        correct += 1
    print(f"{status:6} input='{text}' expected={expected} got={got} tier={result[1]}")

print(f"\n{correct}/{len(test_cases)} correct")