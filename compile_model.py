import os
import dspy
import ast
import time
import random
from dspy.teleprompt import BootstrapFewShot
from dspy_module import AgenticThreadCreator, configure_llm
from supabase import create_client, Client
from dotenv import load_dotenv

# Load env
load_dotenv()

# ==============================================================================
# 1. DATA LOADER (Supabase)
# ==============================================================================
def load_training_data(limit=30):
    """
    Fetches real threads from Supabase to train the new Architect/Polisher brain.
    Uses RANDOM SAMPLING to ensure diversity.
    """
    # Hardcoded credentials from original notebook (or use env vars if available)
    url = "https://ndqenyhahiqqshwvherc.supabase.co"
    key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kcWVueWhhaGlxcXNod3ZoZXJjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgyMDAxNTksImV4cCI6MjA3Mzc3NjE1OX0.qL8WwIR7vaG7XP0IK7dYlG9kl7bOWVKFrtFtg56LNrU")
    
    supabase: Client = create_client(url, key)

    print("(Likely) Connecting to Supabase...")
    
    # Fetch threads (Fetch MORE than needed to sample randomly)
    fetch_limit = 100 
    response = supabase.table('threads').select('thread_id, topic_area, hook_text, template_type, outline_bullets, cta_text').limit(fetch_limit).execute()
    threads_data = response.data
    
    # RANDOM SAMPLING
    if len(threads_data) > limit:
        print(f"Randomly selecting {limit} threads from {len(threads_data)} available...")
        threads_data = random.sample(threads_data, limit)
    
    examples = []
    print(f"Processing {len(threads_data)} threads into training examples...")

    for thread in threads_data:
        thread_id = thread['thread_id']
        # Fetch tweets
        tweets_resp = supabase.table('tweets').select('tweet_text').eq('thread_id', thread_id).order('tweet_index', desc=False).execute()
        tweets = [t['tweet_text'] for t in tweets_resp.data]
        
        if not tweets: continue

        # Map to NEW Signature Structures
        # We assume 'Social Proof' as a default if we don't know the trigger, letting the optimizer learn to pick better ones.
        
        # PAD HOOKS to ensure the model learns to output a list of 5
        base_hook = thread['hook_text']
        hooks_list = [base_hook]
        # Duplicate the hook with slight variation to fill the list to 5
        # This prevents the model from learning to output a single-item list
        for i in range(4):
            # We add a subtle variation marker that the model ideally learns to ignore or replace
            # But crucial for structure learning
            hooks_list.append(f"{base_hook}") 

        example = dspy.Example(
            raw_topic=thread['topic_area'],
            research_context="", # Empty context for training stability
            refined_topic=thread['topic_area'], 
            template_type=thread['template_type'],
            thread_outline=thread['outline_bullets'], 
            psychological_trigger="Social Proof", # Placeholder for training
            trigger_application="Show results to prove value.", # Placeholder
            
            # HookSmith Target
            topic=thread['topic_area'],
            primary_trigger="Social Proof",
            hooks_list=hooks_list, 
            
            # Drafter Target
            outline=thread['outline_bullets'],
            tweet_thread=tweets, 
            selected_hook=base_hook, # Training assumption: The DB hook IS the selected hook
            
            # Polisher Target
            draft_content=str(tweets), 
            polished_content=tweets, 
            tldr="Summary of thread", 
            cta=thread['cta_text']
        ).with_inputs('raw_topic', 'research_context')
        
        examples.append(example)

    print(f"Loaded {len(examples)} training examples.")
    return examples

# ==============================================================================
# 2. METRIC
# ==============================================================================
def basic_metric(example, pred, trace=None):
    # Check if we generated the key fields
    # Note: dspy.Prediction fields match what we returned in forward()
    if not pred.hooks_list: return False
    if not pred.thread_outline: return False
    if not pred.tweet_thread: return False
    if not pred.polished_content: return False
    
    # New Psychological Fields
    # Use getattr to be safe if the model schema changes
    if not getattr(pred, 'psychological_trigger', None): return False
    
    # Check structure
    if isinstance(pred.hooks_list, list) and len(pred.hooks_list) < 2:
        return False # Penalize single-item lists if we want 5
        
    return True

# ==============================================================================
# 3. COMPILATION
# ==============================================================================
if __name__ == "__main__":
    # Configure LLM
    try:
        configure_llm()
    except Exception as e:
        print(f"LLM Config Error: {e}")
        exit(1)
        
    print("Starting Compilation for 5-Step Architecture...")
    
    # Load Data
    train_data = load_training_data(limit=30) # Start small
    if len(train_data) < 5:
        print("Not enough data to compile. Aborting.")
        exit(1)

    # Instantiate
    teacher = AgenticThreadCreator()
    
    # Compiler
    # Use fewer labeled demos to prevent over-reliance on the placeholder triggers
    teleprompter = BootstrapFewShot(metric=basic_metric, max_labeled_demos=2, max_bootstrapped_demos=20)
    
    print("Compiling... (This will take a minute)")
    compiled_model = teleprompter.compile(teacher, trainset=train_data)
    
    # Save
    compiled_model.save("compiled_thread_creator.json")
    print("Model compiled and saved to compiled_thread_creator.json")
