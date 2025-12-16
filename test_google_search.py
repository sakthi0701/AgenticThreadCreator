import os
from serpapi import GoogleSearch
import pprint # For pretty printing the results
import json

# --- Test Parameters ---
# Using your last simplified query:
TEST_QUERY = "How to learn Subjects of EEE (Electrical and Electronics Engineering) 10x faster"

print("--- Starting SerpApi AI Mode Debug Test ---")
try:
    # IMPORTANT: Ensure your SERPAPI_API_KEY is set in your environment
    serpapi_key = "042f0719648784de19ab1c5cc9a31f3fd2e6f64b8191125f3dd15114b83f4adb"
    if not serpapi_key:
        print("❌ ERROR: SERPAPI_API_KEY environment variable not set. Set this before running.")
        # Exit or use a placeholder key (will fail API call)
        serpapi_key = "YOUR_MISSING_KEY" 
    
    # 1. Define the parameters for the search
    params = {
        "engine": "google_ai_mode", # Use the specialized AI Mode engine
        "q": TEST_QUERY,
        "api_key": serpapi_key,
        "hl": "en"
    }
    
    # 2. Execute the search
    print(f"🔎 Calling engine '{params['engine']}' with query: '{TEST_QUERY}'")
    search = GoogleSearch(params)
    results = search.get_dict()

    # 3. Print the ENTIRE result dictionary for inspection
    print("\n" + "="*80)
    print("RAW SERPAPI OUTPUT DICTIONARY (Look for 'ai_summary', 'answer_box', or 'ai_overview')")
    print("="*80)
    
    # Use pprint for a structured, readable output
    pprint.pprint(results)

    print("\n" + "="*80)
    print("--- Debug Test Finished ---")
    
except Exception as e:
    print(f"\n❌ A critical error occurred during the API call: {e}")