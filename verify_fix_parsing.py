
import sys
import os

# Add parent directory to path
sys.path.append(os.getcwd())

from utils.parsing import safe_parse_list

# The raw string from the user's log
raw_output = """[1] «IndiGo's public struggles masked a deeper issue, a cascade of decisions that led to the brink. What if I told you the *real* cause wasn't unique to them, but a pattern every business repeats? The 3 crucial lessons are waiting...»
[2] «IndiGo, a titan of Indian aviation, recently faced a massive operational meltdown. It wasn't just turbulence; it was a near-disaster costing millions in trust and revenue.

Could your business be unknowingly heading down a similar path, *losing* big?»
[3] «Their core issue? A failure to adapt to new pilot rest regulations, leading to widespread flight cancellations and operational chaos. This highlights the immediate, tangible *losses* incurred when things go wrong.»
[4] «This isn't just an airline story. It's a high-stakes masterclass in what *not* to do, offering 3 critical lessons your business *must* learn to prevent similar, crippling mistakes and *avoid massive losses*.»
[5] «**Lesson 1: The Peril of Hyper-Efficiency (Don't lose resilience)**
    *   *IndiGo's Mistake:* Pushing efficiency to the absolute limit without buffers. A small change caused system-wide collapse.
    *   *Your Takeaway:* Efficiency is good, but *resilience is essential*. Are you ignoring vulnerabilities for the sake of "lean" and *risking* a total system collapse?»
"""

print(f"Testing Parser with raw input:\n{raw_output[:100]}...\n")

parsed = safe_parse_list(raw_output)

print(f"Parsed Result Type: {type(parsed)}")
print(f"Parsed Result Length: {len(parsed)}")
print(f"Parsed Result Content: {parsed}")

if len(parsed) >= 5 and isinstance(parsed, list) and isinstance(parsed[0], str):
    print("\n✅ SUCCESS: Parser correctly handled the numbered block format.")
else:
    print("\n❌ FAILURE: Parser failed to extract the tweets correctly.")
