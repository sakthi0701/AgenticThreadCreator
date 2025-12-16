
import ast
import json
import re
import dspy

# We might need to import configure_llm if we want to use dspy for fixing
# But strict circular imports might be an issue if configure_llm is in dspy_module
# For now, we assume dspy is already configured globally by the main app

class ListRepair(dspy.Signature):
    """Repair a broken string representation of a Python list or JSON array into a valid JSON string."""
    broken_string = dspy.InputField(desc="The malformed string that should be a list.")
    repaired_json = dspy.OutputField(desc="A valid JSON string string representing the list. e.g. '[\"item1\", \"item2\"]'")

def safe_parse_list(text: str, fix_with_llm=False) -> list:
    """
    Attempts to parse a string into a python list using multiple strategies.
    1. ast.literal_eval (Python style)
    2. json.loads (Standard JSON)
    3. Custom parsing for numbered lists (e.g. "[1] Tweet text")
    4. Regex extraction of [ ... ] (JSON/List blocks)
    5. (Optional) LLM Repair
    """
    if not isinstance(text, str):
        if isinstance(text, list):
            return text
        return []

    text = text.strip()

    # Strategy 1: AST (Best for Python-like lists with single quotes)
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            return val
    except:
        pass

    # Strategy 2: JSON (Strict double quotes)
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except:
        pass

    # Strategy 3: Handle "Markdown Code Block" wrapping explicitly
    if "```" in text:
        clean_text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()
        try:
             return json.loads(clean_text)
        except:
             try:
                 return ast.literal_eval(clean_text)
             except:
                 pass

    # Strategy 4: Custom Parser for "[1] Text" or "1. Text" formats
    # This must come BEFORE the generic regex `\[.*?\]` because `[1]` matches that regex
    # and returns a list [1], which is NOT what we want.
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_lines = []
    
    for line in lines:
        # Patter: Matches "[1]", "1.", "1)", "- " at start
        # strict check to avoid matching random text
        match = re.match(r'^(\[\d+\]|\d+\.|\d+\)|\-|\*)\s*(.*)', line)
        if match:
            content = match.group(2).strip()
            # Remove wrapping quotes/guillemets if present
            if (content.startswith('"') and content.endswith('"')) or \
               (content.startswith("'") and content.endswith("'")) or \
               (content.startswith("«") and content.endswith("»")):
                content = content[1:-1]
            if content:
                cleaned_lines.append(content)
                
    if len(cleaned_lines) > 1:
        return cleaned_lines

    # Strategy 5: Generic Regex extraction (Find the first [...] block)
    # Only if we haven't found a multi-item list structure yet
    try:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                val = ast.literal_eval(candidate)
                if isinstance(val, list) and len(val) > 1: # Only accept if it's a real list, not just `[1]`
                     return val
            except:
                try:
                    val = json.loads(candidate)
                    if isinstance(val, list):
                        return val
                except:
                    pass
    except:
        pass
        
    # Strategy 6: LLM Repair (Last Resort)
    if fix_with_llm:
        print(f"🔧 Parsing failed. Attempting LLM Repair for: {text[:50]}...")
        try:
            # We assume dspy is configured
            repair_bot = dspy.Predict(ListRepair)
            result = repair_bot(broken_string=text)
            
            # Try parsing the LLM's fix
            clean_json = result.repaired_json.strip()
            clean_json = clean_json.replace("```json", "").replace("```", "").strip()
            
            return json.loads(clean_json)
        except Exception as e:
            print(f"⚠️ LLM Repair failed: {e}")
            # Final fallback: return the raw text as a single item list (better than empty)
            return [text]

    # Total Failure
    print("❌ Critical: Parsing failed completely.")
    return []
