
import requests
import json
import time

URL = "http://localhost:8080/generate-thread"
DATA = {
    "topic": "The benefits of asynchronous programming in Python",
    "platform": "Twitter"
}

def verify():
    print(f"🚀 Tests starting... Hitting {URL}")
    try:
        start_time = time.time()
        response = requests.post(URL, json=DATA, timeout=120)
        duration = time.time() - start_time
        
        print(f"⏱️ Request took {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Status 200 OK")
            
            keys = ["template", "refined_topic", "hook", "tldr", "cta", "tweets"]
            missing = [k for k in keys if k not in result]
            
            if missing:
                print(f"❌ FAILED: Missing keys: {missing}")
            else:
                print("✅ PASSED: All keys present.")
                print(f"   Topic: {result['refined_topic']}")
                print(f"   Hook: {result['hook'][:50]}...")
                print(f"   Tweets: {len(result['tweets'])} tweets generated.")
                print(f"   CTA: {result['cta']}")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    time.sleep(5) # Wait for server to definitely be up
    verify()
