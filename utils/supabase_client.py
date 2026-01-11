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
            print("✅ Supabase Client Initialized (Shared).")
            return _supabase
        except Exception as e:
            print(f"⚠️ Supabase Init Failed: {e}")
            return None
    return None
