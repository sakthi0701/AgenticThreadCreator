import os
from supabase import create_client, Client

# Singleton Client
_supabase: Client = None

def get_supabase_client() -> Client:
    global _supabase
    
    if _supabase:
        return _supabase
        
    url = os.environ.get("SUPABASE_URL_log") or os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY_log") or os.environ.get("SUPABASE_KEY")
    
    if url and key:
        try:
            _supabase = create_client(url, key)
            key_preview = f"{key[:6]}...{key[-6:]}" if len(key) > 12 else "***"
            print(f"✅ Supabase Client Initialized (Shared). URL: {url} | Key: {key_preview}")
            return _supabase
        except Exception as e:
            print(f"⚠️ Supabase Init Failed for URL {url}: {e}")
            return None
    
    print("⚠️ Supabase: Shared client initialization failed. Missing URL or Key.")
    return None
