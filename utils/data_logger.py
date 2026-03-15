import os
from supabase import create_client, Client
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import atexit

from utils.supabase_client import get_supabase_client

# Initialize Supabase Client
_supabase = get_supabase_client()

# Thread pool for background logging (prevents memory leak)
from config import settings
_log_executor = ThreadPoolExecutor(
    max_workers=settings.supabase_logger_workers, 
    thread_name_prefix="supabase_logger"
)
atexit.register(_log_executor.shutdown, wait=True)

def log_interaction(ip: str, action_type: str, input_data: str, output_data: dict, meta: dict = None, interaction_id: str = None, parent_id: str = None):
    """
    Logs an interaction to Supabase 'thread_interactions' table.
    Uses a bounded thread pool to prevent memory leaks.
    """
    if not _supabase:
        print("⚠️ Supabase: No client configured. Logging skipped.")
        return

    def _log():
        try:
            # Debug: Which project are we logging to?
            url = os.environ.get("SUPABASE_URL_log") or os.environ.get("SUPABASE_URL")
            
            data = {
                "ip_address": ip,
                "action_type": action_type,
                "input_topic": input_data,
                "output_content": output_data, # Supabase handles dict -> jsonb
                "meta": meta or {},
                "interaction_id": interaction_id,
                "parent_id": parent_id,
                "created_at": datetime.now().isoformat()
            }
            # print(f"DEBUG: Logging to {url} table 'thread_interactions'")
            res = _supabase.table("thread_interactions").insert(data).execute()
            if res.data:
                print(f"✅ Supabase Log Success: {action_type} for {ip} [{res.data[0].get('id', 'no-id')}]")
            else:
                print(f"⚠️ Supabase Log returned no data: {res}")
        except Exception as e:
            print(f"❌ Failed to log to Supabase: {e}")

    # Submit to thread pool (bounded, prevents leak)
    _log_executor.submit(_log)
