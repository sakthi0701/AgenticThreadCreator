import os
import requests
import json
import re
from utils.metrics_tracker import MetricsTracker
import json
import re

class GroqClient:
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "moonshotai/kimi-k2-instruct-0905"  # Using Kimi as requested
    
    @staticmethod
    def get_next_question(topic, history, context_str=""):
        """
        Interacts with Groq to execute the Clarification Loop.
        Input: 
            - topic: The user's initial request.
            - history: List of dicts.
            - context_str: Pre-known context (Platform, Profile) to avoid redundant questions.
        """
        MetricsTracker.increment("groq_calls")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return {"status": "ERROR", "message": "GROQ_API_KEY not found."}

        system_prompt = f"""
        You are an expert Ghostwriter Assistant. Your goal is to gather specific requirements from the user to write a viral social media thread.
        
        KNOWN CONTEXT:
        {context_str}
        
        RULES:
        1. Analyze the 'Topic' and 'Conversation History'.
        2. DONT ask for information already in KNOWN CONTEXT (e.g. Platform, Tone, Audience).
        3. FOCUS ON "THE MEAT": If the topic implies a story, strategy, or list (e.g. "How I won", "5 tips"), but details are missing, YOU MUST ASK FOR THEM.
           - Bad Q: "What is the tone?" (We know this!)
           - Good Q: "What were the specific strategies you used to win the expo?"
           - Good Q: "What is the key mistake people make?"
        4. Ask ONE clarifying question at a time.
        5. CRITICAL: If the user input is vague (e.g. "Write a post?"), Return status 'CLARIFYING'.
        6. If you have enough "Meat" (stories, steps, insights), return status 'READY'.
        
        OUTPUT FORMAT:
        You must return ONLY a JSON object. No markdown.
        
        Example 1 (Deep Content Q):
        {{
            "status": "CLARIFYING",
            "question": "What was the #1 strategy you used to win the project expo?",
            "type": "text",
            "options": []
        }}
        
        Example 2 (Ready):
        {{
            "status": "READY",
            "summary": "Writing a LinkedIn post about Project Expo strategies. Strategy: Focus on demo > slide."
        }}
        """

        # Construct Messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"TOPIC: {topic}"})
        
        # Append history
        # History format expected from Frontend: ["Q: ...", "A: ..."] or proper dicts.
        # Let's assume Main.py passes us a list of "Q: ... \nA: ..." strings or similar.
        # Actually, let's keep it robust. If 'history' is list of strings, append them.
        for item in history:
            # Ensure valid structure
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                messages.append({"role": item['role'], "content": item['content']})
            else:
                # Fallback
                messages.append({"role": "user", "content": str(item)})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GroqClient.MODEL,
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(GroqClient.API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON
            try:
                # Cleanup if model adds markdown text
                json_str = content.replace("```json", "").replace("```", "").strip()
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # Fallback if model fails JSON
                return {
                    "status": "CLARIFYING",
                    "question": content, # Treat raw output as question
                    "type": "text", 
                    "options": [] 
                }
                
        except Exception as e:
            print(f"Groq API Error: {e}")
            return {"status": "ERROR", "message": str(e)}
