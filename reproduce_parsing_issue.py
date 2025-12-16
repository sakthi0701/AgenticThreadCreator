
from utils.parsing import safe_parse_list

test_cases = [
    # Case 1: Code block wrapper (Common LLM failure)
    """```json
    ["Tweet 1", "Tweet 2"]
    ```""",
    
    # Case 2: Text before/after
    "Here is your thread:\n['Tweet 1', 'Tweet 2']\nHope you like it!",
    
    # Case 3: Loose formatting (No quotes on keys if dict, or just bad quotes)
    "[Tweet 1, Tweet 2]", # Invalid JSON/Python
    
    # Case 4: Numbered list (Drafter often does this)
    "1. Tweet one\n2. Tweet two",
    
    # Case 5: The "DrafterAgent" failure seen in logs
    # "It's an incredible full-circle moment when the thr..." (This looks like raw text, not a list)
    "It's an incredible full-circle moment when the thr..."
]

print("--- Testing safe_parse_list ---")
for i, case in enumerate(test_cases):
    print(f"\nCase {i+1}:")
    try:
        # Disable LLM repair to see if static parsing works
        result = safe_parse_list(case, fix_with_llm=False)
        if result:
            print(f"✅ Parsed: {result}")
        else:
            print(f"❌ Failed (Would trigger LLM repair)")
    except Exception as e:
        print(f"❌ Error: {e}")
