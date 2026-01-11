import json
import os
from threading import Lock

class ProfileManager:
    """
    Manages loading, saving, and retrieving user profiles from user_profiles.json.
    Thread-safe implementation.
    """
    FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'user_profiles.json'))
    _lock = Lock()
    
    @classmethod
    def _load_data(cls):
        """Loads the JSON data from disk."""
        if not os.path.exists(cls.FILE_PATH):
            # Create default if missing
            return {
                "default_profile": "tech_founder",
                "profiles": {
                    "tech_founder": {
                        "audience": "VCs, Engineers, and Founders",
                        "tone": "Contrarian, Punchy, Direct",
                        "forbidden_words": ["delve", "foster", "landscape", "crucial"],
                        "avg_sentence_length": "Short (<15 words)"
                    }
                }
            }
        
        try:
            with open(cls.FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading profiles: {e}")
            return {"default_profile": "tech_founder", "profiles": {}}

    @classmethod
    def get_profiles(cls):
        """Returns the full profiles data structure."""
        with cls._lock:
            return cls._load_data()

    @classmethod
    def save_profile(cls, profile_name, data):
        """Saves or updates a specific profile."""
        with cls._lock:
            current_data = cls._load_data()
            current_data["profiles"][profile_name] = data
            
            with open(cls.FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=2)
            print(f"✅ Saved profile: {profile_name}")

    @classmethod
    def set_default(cls, profile_name):
        """Sets the default active profile."""
        with cls._lock:
            current_data = cls._load_data()
            if profile_name in current_data["profiles"]:
                current_data["default_profile"] = profile_name
                with open(cls.FILE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, indent=2)

    @classmethod
    def get_profile(cls, profile_name=None):
        """Gets a specific profile or the default one if None."""
        data = cls.get_profiles()
        if not profile_name:
            profile_name = data.get("default_profile")
        
        return data["profiles"].get(profile_name, {})
