import json

with open("lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

two_hop = {
    k: v for k, v in lexicon["properties"].items()
    if isinstance(v, list)
}

print(f"Two-hop entries found: {len(two_hop)}")
for phrase, chain in two_hop.items():
    print(f"  '{phrase}' → {chain}")