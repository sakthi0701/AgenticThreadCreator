import os
import sys

print("Checking imports...")
try:
    import agno
    print("✅ agno imported")
except ImportError as e:
    print(f"❌ agno failed: {e}")

try:
    import dspy
    print("✅ dspy imported")
except ImportError as e:
    print(f"❌ dspy failed: {e}")

try:
    import google.generativeai as genai
    print("✅ google.generativeai imported")
except ImportError as e:
    print(f"❌ google.generativeai failed: {e}")

try:
    import config
    print("✅ config imported")
    print(f"API Key present: {bool(config.settings.gemini_api_key)}")
except ImportError as e:
    print(f"❌ config failed: {e}")
except Exception as e:
    print(f"❌ config error: {e}")

print("Done checking imports.")
