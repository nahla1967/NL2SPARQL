from template_resolver import _sanitize_params, _build_filter_string_kg2

# The exact corrupted params from your log (filter_string_kg2_001, ar)
broken_params = {"property": "countryName", "value": '"Italy"', "limit": 10}

clean_params = _sanitize_params(broken_params)
print("before:", broken_params)
print("after: ", clean_params)

sparql, label = _build_filter_string_kg2(clean_params)
print(sparql)