
import requests
import json
import time

URL = "http://localhost:8080/generate-thread"
DATA = {
    "topic": "The future of AI agents for developers",
    "platform": "LinkedIn"
}

def verify():
    print(f"🚀 Tests starting... Hitting {URL} (LinkedIn Mode)")
    try:
        response = requests.post(URL, json=DATA, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Status 200 OK")
            
            # 1. Verify Structure
            hook = result.get("hook", "")
            cta = result.get("cta", "")
            tweets = result.get("tweets", [])
            
            print(f"📌 Hook: {hook}")
            print(f"📌 CTA: {cta}")
            print(f"📌 Body Length: {len(tweets[0]) if tweets else 0} chars")
            
            if not hook or hook == "LinkedIn Post" or hook == "N/A":
                 print("❌ FAILED: Hook not separated correctly.")
            else:
                 print("✅ PASSED: Hook separated.")

            if not cta or cta == "N/A":
                 print("❌ FAILED: CTA not separated correctly.")
            else:
                 print("✅ PASSED: CTA separated.")
                 
            # 2. Verify Formatting (No Bold)
            body_text = tweets[0] if tweets else ""
            if "**" in body_text:
                print("❌ FAILED: Found bold markers (**).")
            else:
                print("✅ PASSED: No bold markers found.")


            print("--- FULL BODY PREVIEW ---")
            print(body_text[:500] + "...")
            
            # 3. Verify Refine (Simulated)
            print("\n🧪 Testing Refine Endpoint...")
            refine_data = {
                "hook": hook,
                "tweets": tweets,
                "cta": cta,
                "prompt": "Make it more professional.",
                "platform": "LinkedIn"
            }
            refine_url = "http://localhost:8080/refine-full"
            refine_res = requests.post(refine_url, json=refine_data, timeout=60)
            
            if refine_res.status_code == 200:
                r_result = refine_res.json()
                r_body = r_result.get("refined_tweets", [""])[0]
                print("✅ Refine Success")
                if "**" in r_body:
                     print("❌ FAILED: Refine introduced bold markers.")
                else:
                     print("✅ PASSED: Refine respected 'No Bold' constraint.")
            else:
                print(f"❌ Refine Failed: {refine_res.status_code}")
            
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    verify()
