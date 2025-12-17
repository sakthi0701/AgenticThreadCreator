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
# Helper Signature to clean your data ON THE FLY
class DataEnricher(dspy.Signature):
    """
    Analyze a successful thread to reverse-engineer its 'DNA': 
    Trigger, Tone, and the Outline that would have created it.
    """
    thread_text = dspy.InputField(desc="The full content of the high-performing thread.")
    original_hook = dspy.InputField(desc="The first tweet/hook.")
    
    # 1. Trigger
    detected_trigger = dspy.OutputField(desc="The core psychological principle used (Social Proof, Loss Aversion, etc).")
    trigger_explanation = dspy.OutputField(desc="How it was applied.")
    
    # 2. Hooks
    alt_hook_1 = dspy.OutputField(desc="A contrarian hook option (< 280 chars).")
    alt_hook_2 = dspy.OutputField(desc="A story-open loop hook option (< 280 chars).")
    alt_hook_3 = dspy.OutputField(desc="A negative/loss aversion hook option (< 280 chars).")
    alt_hook_4 = dspy.OutputField(desc="A short/punchy hook option (< 10 words).")
    
    # 3. NEW: Reverse-Architecting the Outline
    # We ask Gemini to create the 'Perfect Outline' that MATCHES this thread.
    reverse_engineered_outline = dspy.OutputField(desc="A list of 5-7 bullet points summarizing the logical flow of the thread. ONE point per tweet.")
    
    # 4. NEW: Tone Labeling
    detected_tone = dspy.OutputField(desc="The specific tone of voice (e.g., 'Authoritative', 'Vulnerable', 'Sarcastic', 'Data-Driven').")

import json

def load_training_data(limit=30, cache_file="enriched_data_cache.json"):
    """
    Fetches threads, ENRICHES them, and saves to a JSON file.
    If the file exists, it loads from disk to save money and time.
    """
    
    # 1. CHECK CACHE FIRST
    if os.path.exists(cache_file):
        print(f"📂 Found cached training data in '{cache_file}'. Loading...")
        try:
            with open(cache_file, "r") as f:
                raw_data = json.load(f)
            
            # Reconstruct DSPy Examples from JSON
            examples = []
            for item in raw_data:
                # We have to rebuild the Example object
                ex = dspy.Example(**item)
                # Re-apply the input keys (DSPy forgets these in JSON)
                ex = ex.with_inputs('raw_topic', 'research_context', 'platform')
                examples.append(ex)
                
            print(f"✅ Loaded {len(examples)} examples from cache.")
            return examples
        except Exception as e:
            print(f"⚠️ Cache file corrupted ({e}). Regenerating data...")

    # 2. IF NO CACHE, RUN THE EXPENSIVE LOGIC
    # Hardcoded credentials from original notebook (or use env vars if available)
    url = "https://ndqenyhahiqqshwvherc.supabase.co"
    key = os.getenv("SUPABASE_KEY")
    if not key:
        print("❌ Error: SUPABASE_KEY not found in environment variables.")
        exit(1)
    
    supabase: Client = create_client(url, key)

    print("(Likely) Connecting to Supabase...")
    
    # Fetch threads (Fetch MORE than needed to sample randomly)
    fetch_limit = limit * 3 
    response = supabase.table('threads').select('thread_id, topic_area, hook_text, template_type, outline_bullets, cta_text').limit(fetch_limit).execute()
    threads_data = response.data
    
    # RANDOM SAMPLING
    if len(threads_data) > limit:
        print(f"Randomly selecting {limit} threads from {len(threads_data)} available...")
        threads_data = random.sample(threads_data, limit)
    
    # Configure a cheap/fast model for data enrichment (e.g., Gemini Flash)
    enricher = dspy.Predict(DataEnricher)

    examples = []
    cacheable_data = [] # List to store raw dicts for JSON saving
    print(f"Processing {len(threads_data)} threads into training examples...")

    for thread in threads_data:
        thread_id = thread['thread_id']
        # Fetch tweets
        tweets_resp = supabase.table('tweets').select('tweet_text').eq('thread_id', thread_id).order('tweet_index', desc=False).execute()
        tweets = [t['tweet_text'] for t in tweets_resp.data]
        
        if not tweets: continue

        full_thread_text = " ".join(tweets[:3]) # First 3 tweets are enough for context

        # --- THE FIX: SYNTHETIC DATA ENRICHMENT ---
        print(f"  > Enriching data for thread: {thread_id}...")
        try:
            # We use the LLM to guess the missing labels
            enriched = enricher(
                thread_text=full_thread_text, 
                original_hook=thread['hook_text']
            )
            
            # Create a DIVERSE list of hooks
            # We mix the real one with synthetic ones
            hooks_list = [
                thread['hook_text'],     # The Real Winner
                enriched.alt_hook_1,     # Synthetic
                enriched.alt_hook_2,     # Synthetic
                enriched.alt_hook_3,     # Synthetic
                enriched.alt_hook_4      # Synthetic
            ]
            
            # SHUFFLE to prevent "Index 0" bias
            # The model must learn to find the best hook by CONTENT, not POSITION.
            random.shuffle(hooks_list)
            
            # Use the DETECTED trigger, not a hardcoded one
            real_trigger = enriched.detected_trigger
            real_app = enriched.trigger_explanation
            
            # Use Reverse-Engineered Outline
            real_outline = enriched.reverse_engineered_outline
            
        except Exception as e:
            print(f"    Skipping thread due to enrichment error: {e}")
            continue

        # Map to NEW Signature Structures
        example = dspy.Example(
            raw_topic=thread['topic_area'],
            research_context="", # Empty context for training stability
            refined_topic=thread['topic_area'], 
            template_type=thread['template_type'],
            
            # 🚨 KEY CHANGE: Use the CLEAN, Reverse-Engineered Outline
            # instead of the potentially messy DB outline.
            thread_outline=real_outline, 
            
            # NOW DYNAMIC
            psychological_trigger=real_trigger, 
            trigger_application=real_app, 
            
            # HookSmith Target
            topic=thread['topic_area'],
            primary_trigger=real_trigger,
            hooks_list=hooks_list, 
            
            # Drafter Target
            # We train the drafter to turn the PERFECT outline into the REAL tweets
            outline=real_outline,
            tweet_thread=tweets, 
            selected_hook=thread['hook_text'], # Logic: The model should find THIS string in the shuffled hooks_list
            
            # Polisher Target
            draft_content=str(tweets), 
            polished_content=tweets, 
            tldr="Summary of thread", 
            cta=thread['cta_text'],
            platform="Twitter" # Explicitly include platform for compiled path
        ).with_inputs('raw_topic', 'research_context', 'platform')
        
        examples.append(example)
        
        # 3. PREPARE FOR SAVING
        # We save the raw dictionary of the example, not the object itself
        cacheable_data.append(example.toDict())
        
        # SAVE INCREMENTALLY
        with open(cache_file, "w") as f:
            json.dump(cacheable_data, f, indent=4)
        print(f"  > Saved progress ({len(examples)}) to cache.")

    # Final confirmation
    print(f"💾 Final cache save complete. Total examples: {len(examples)}")

    print(f"✅ Loaded {len(examples)} HIGH QUALITY training examples.")
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
