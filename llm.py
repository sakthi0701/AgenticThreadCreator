import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

def list_available_models():
    api_key = os.getenv("gemini_api_key")
    if not api_key:
        raise ValueError("gemini_api_key not found in .env file.")

    client = genai.Client(api_key=api_key)

    models = client.models.list()
    for m in models:
        print(f"{m.name} → {m.supported_actions}")

if __name__ == "__main__":
    list_available_models()
