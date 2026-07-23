# fix_classification_prompt.py
with open("router.py", encoding="utf-8") as f:
    src = f.read()

start = src.index('_CLASSIFICATION_PROMPT = """')
end = src.index('"""', start + len('_CLASSIFICATION_PROMPT = """')) + 3
block = src[start:end]

fixed_block = block.replace("{{", "{").replace("}}", "}")
assert "{question}" in fixed_block, "placeholder got mangled — abort, don't save"

new_src = src[:start] + fixed_block + src[end:]
with open("router.py", "w", encoding="utf-8") as f:
    f.write(new_src)

print("Done. Doubled braces remaining:", fixed_block.count("{{"))