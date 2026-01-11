import contextvars
from datetime import datetime
from typing import Dict, Any
from config import settings
from utils.supabase_client import get_supabase_client

# Context variable to store the current request's IP address
client_ip_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_ip", default="unknown")

class MetricsTracker:
    """
    Supabase-backed tracker for API usage metrics per IP.
    Persists data to 'ip_rate_limits' table.
    """
    _supabase = get_supabase_client()
    
    @property
    def DAILY_POST_LIMIT(self):
        return settings.daily_post_limit

    @classmethod
    def set_context_ip(cls, ip: str):
        """Sets the IP for the current context (called by Middleware)."""
        client_ip_var.set(ip)

    @classmethod
    def get_context_ip(cls) -> str:
        return client_ip_var.get()

    @classmethod
    def increment(cls, metric: str, count: int = 1):
        """
        Legacy support for tracking non-critical metrics (groq_calls, llm_calls).
        Currently a no-op for DB rate limiting, but prevents crashes.
        """
        # We could log this to DB if we expand the schema later.
        pass


    @classmethod
    async def try_increment_post(cls, ip: str) -> bool:
        """
        Async check and increment.
        Returns True if allowed, False if blocked.
        """
        if ip == "unknown" or not cls._supabase:
            # Fallback if DB down or IP unknown (Fail Open or Closed? Let's Fail Open for beta)
            print(f"⚠️ Rate Limit Bypass: IP={ip}, DB Connected={bool(cls._supabase)}")
            return True

        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # 1. Get current count
            # We filter by IP AND Date.
            res = cls._supabase.table("ip_rate_limits")\
                .select("post_count")\
                .eq("ip_address", ip)\
                .eq("limit_date", today)\
                .execute()
            
            current_count = 0
            if res.data and len(res.data) > 0:
                current_count = res.data[0]['post_count']
            
            # 2. Check Limit
            if current_count >= settings.daily_post_limit:
                print(f"⛔ Rate Limit Hit for {ip}: {current_count}/{settings.daily_post_limit}")
                return False

            # 3. Increment (Upsert)
            # Upsert will Insert if not exists, or Update if exists.
            # We need to explicitly set post_count = current_count + 1
            new_data = {
                "ip_address": ip,
                "limit_date": today,
                "post_count": current_count + 1,
                "last_active": datetime.now().isoformat()
            }
            
            cls._supabase.table("ip_rate_limits").upsert(new_data).execute()
            print(f"✅ Rate Limit Update for {ip}: {current_count + 1}/{settings.daily_post_limit}")
            return True

        except Exception as e:
            print(f"⚠️ Rate Limit DB Error: {e}")
            # Fail Open on DB error to avoid blocking users during outages
            return True

    @classmethod
    def get_report(cls) -> Dict:
        """
        Stub for admin report. 
        In production, you'd run a separate SQL query.
        """
        return {"status": "Using Supabase backend"}

    @classmethod
    async def check_limit(cls, ip: str) -> bool:
        """
        Read-only check. Returns False if limit exceeded.
        """
        if ip == "unknown" or not cls._supabase:
            return True
            
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            res = cls._supabase.table("ip_rate_limits")\
                .select("post_count")\
                .eq("ip_address", ip)\
                .eq("limit_date", today)\
                .execute()
            
            if res.data and len(res.data) > 0:
                if res.data[0]['post_count'] >= settings.daily_post_limit:
                    return False
            return True
        except Exception:
            return True
