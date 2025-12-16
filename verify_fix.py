
from utils.parsing import safe_parse_list

# The specific failure pattern from logs + a markdown case
cases = [
    "1. This is tweet one\n2. This is tweet two",
    "```json\n[\"Hook 1\", \"Hook 2\"]\n```"
]

print("START_TEST")
for c in cases:
    res = safe_parse_list(c, fix_with_llm=False)
    if res and len(res) == 2:
        print("SUCCESS")
    else:
        print(f"FAILURE: {res}")
print("END_TEST")
