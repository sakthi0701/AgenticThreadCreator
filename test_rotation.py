import sys
import os
sys.path.append(os.getcwd())
from dspy_module import configure_llm

print("--- Testing Key Rotation ---")
for i in range(4):
    print(f"Call {i+1}:")
    configure_llm()
print("--- Test Complete ---")
