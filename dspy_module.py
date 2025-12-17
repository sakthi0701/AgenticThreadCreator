import dspy
import ast
import os
import threading
from dotenv import load_dotenv
import time

load_dotenv()

# ==============================================================================
# 1. LLM CONFIGURATION (With Robust Key Rotation)
# ==============================================================================

class MultiKeyGemini(dspy.LM):
    """
    A robust wrapper around standard dspy.LM that manages multiple API keys using Round-Robin rotation.
    It automatically retries with a different key if a RateLimitError (429) occurs.
    """
    def __init__(self, model_name, keys, **kwargs):
        self.max_retries = kwargs.pop('max_retries', 3)
        self.retry_delay = kwargs.pop('retry_delay', 2.0)
        self.keys = keys
        if not self.keys:
            raise ValueError("MultiKeyGemini requires at least one API key.")
            
        self.current_key_index = 0
        self.model_name = model_name
        self.kwargs = kwargs
        
        # Initialize the underlying model with the first key
        self._rotate_client(force_index=0)
        
        # Initialize parent
        super().__init__(model=model_name)
        
        # Thread safety lock
        self.lock = threading.Lock()

    def _rotate_client(self, force_index=None):
        """Switches the active client to the next key."""
        if force_index is not None:
             self.current_key_index = force_index
        else:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            
        active_key = self.keys[self.current_key_index]
        masked = active_key[:4] + "..." + active_key[-4:]
        
        self.active_lm = dspy.LM(
            model=self.model_name,
            api_key=active_key,
            **self.kwargs
        )
        print(f"🔄 Switched to API Key #{self.current_key_index + 1} ({masked})")

    def basic_request(self, prompt=None, **kwargs):
        """Delegates the request to the active LM with retry logic."""
        attempts = 0
        while attempts < self.max_retries * len(self.keys):
            try:
                # Delegate to the active dspy.LM instance
                # The active_lm is a full dspy.LM, so we call it directly to handle adapters
                return self.active_lm(prompt=prompt, **kwargs)
            except Exception as e:
                # Naive check for rate limits (check string content or specific error type if available)
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str or "resource exhausted" in error_str or "503" in error_str or "unavailable" in error_str:
                    with self.lock:
                        # Double-check if another thread already rotated while we were waiting
                        # In a real rigorous impl, we might check timestamps, but here we just rotate.
                        print(f"⚠️ Rate Limit/Overload hit on Key #{self.current_key_index + 1}. Rotate & Retry...")
                        self._rotate_client()
                    attempts += 1
                    # Simple backoff - we don't want to hold the lock while sleeping
                    time.sleep(self.retry_delay)
                else:
                    # Non-retryable error
                    raise e
        
        raise RuntimeError("All API keys exhausted or max retries reached.")

    def __call__(self, prompt=None, **kwargs):
        return self.basic_request(prompt=prompt, **kwargs)

def configure_llm():
    """
    Configures the Global DSPy settings to use the MultiKeyGemini wrapper.
    Reads all environment variables starting with 'gemini_api_key' to build the key list.
    """
    keys = []
    
    # scan for keys: gemini_api_key, gemini_api_key_2, gemini_api_key_3...
    # 1. Check primary key
    primary = os.getenv("gemini_api_key")
    if primary:
        keys.append(primary)
        
    # 2. Check numbered keys
    i = 2
    while True:
        k = os.getenv(f"gemini_api_key_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    
    if not keys:
        raise ValueError("gemini_api_key not found in .env file.")
        
    # Idempotency check to prevent async race conditions
    if dspy.settings.lm and isinstance(dspy.settings.lm, MultiKeyGemini):
        # Already configured, just return
        return

    # Instantiate our smart wrapper
    smart_gemini = MultiKeyGemini(
        model_name='gemini/gemini-flash-latest',
        keys=keys,
        max_tokens=8000,
        temperature=0.75,
        max_retries=3
    )
    
    # Configure global settings
    dspy.settings.configure(lm=smart_gemini)
    print(f"✅ Configured DSPy with MultiKeyGemini ({len(keys)} keys found).")


# ==============================================================================
# 2. HELPER SIGNATURES
# ==============================================================================

# ==============================================================================
# 2. HELPER SIGNATURES
# ==============================================================================

class GenerateSearchQuery(dspy.Signature):
    """Distill a verbose, conversational topic into a concise, keyword-rich search query."""
    raw_topic = dspy.InputField(desc="The full, conversational user request.")
    search_query = dspy.OutputField(desc="A 3-5 keyword search query for Google, focused on the core entities.") 

# ==============================================================================
# 3. PHASE-BASED OPTIMIZED SIGNATURES (5-Call Architecture)
# ==============================================================================

class ArchitectSignature(dspy.Signature):
    """Analyze the raw topic and research to create a refined plan.
    
    PSYCHOLOGICAL TRIGGERS TO CONSIDER:
    1. Social Proof: Leverage herd behavior (e.g., "Why 10,000 people use...").
    2. Scarcity/FOMO: Emphasize limited availability or time sensitivity.
    3. Loss Aversion: Frame the topic around avoiding pain/loss rather than gaining pleasure.
    4. Authority Bias: Use expert signals or 'insider' status.
    5. Zeigarnik Effect (Open Loops): Create mental tension by starting a story but withholding the conclusion until the end.
    6. Curiosity Gap: The space between 'what we know' and 'what we want to know'. Specific enough to interest, vague enough to click.
    7. Barnum Effect: Use broad statements that make the reader feel "seen" (e.g., "If you're a burnt-out dev, this is for you").
    8. Cognitive Ease: Promise a "lazy" or "simple" solution to a complex problem.
    """
    raw_topic = dspy.InputField(desc="The user's original request.")
    research_context = dspy.InputField(desc="Key insights from SerpAPI research.")
    
    refined_topic = dspy.OutputField(desc="A specific, punchy title under 10 words. Use a Number or Strong Verb.")
    psychological_trigger = dspy.OutputField(desc="The ONE primary psychological principle to drive this thread (Select from list above).")
    trigger_application = dspy.OutputField(desc="A 1-sentence strategy on HOW to apply that trigger to this command.")
    template_type = dspy.OutputField(desc="The structural template (e.g., 'The Hero's Journey', 'The Listicle', 'The Contrarian Take').")
    thread_outline = dspy.OutputField(desc="A list of 5-7 bullet points. Each point must represent one distinct tweet/idea.")

class BatchHookSignature(dspy.Signature):
    """Generate 5 distinct viral hooks using specific psychological angles."""
    topic = dspy.InputField(desc="The refined topic.")
    primary_trigger = dspy.InputField(desc="The main psychological trigger chosen by the Architect.")
    
    hooks_list = dspy.OutputField(desc="""A list of 5 hooks.
    1. The 'Primary Trigger' Hook (Uses the chosen trigger).
    2. The 'Loss Aversion' Hook (Focus on what they will lose/suffer).
    3. The 'Social Proof' Hook (Focus on numbers/others).
    4. The 'Zeigarnik' Hook (Open a story loop/cliffhanger).
    5. The 'Contra-Competitive' Hook (Attack a common enemy or belief).
    Constraint: No hook longer than 280 characters.
    """)

class DraftThread(dspy.Signature):
    """Convert an outline into a full Twitter thread.
    
    CONSTRAINTS:
    1. Review the 'Hooks List' and select the single BEST hook for engagement.
    2. Write the thread starting with that selected hook.
    3. ONE IDEA PER TWEET. Do not combine multiple outline points into one tweet.
    4. formatting: Use 1-2 distinct sentences per line break.
    5. Voice: Active voice only. Ban words like "can", "helps to", "in order to".
    """
    topic = dspy.InputField()
    outline = dspy.InputField()
    hooks_list = dspy.InputField(desc="List of generated hook options.")
    psychological_trigger = dspy.InputField(desc="The core persuasion principle to reinforce in the body text.")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'")
    
    selected_hook = dspy.OutputField(desc="The exact text of the hook you chose.")
    tweet_thread = dspy.OutputField(desc="A list of strings. Tweet 1 MUST be the selected hook.")

class DraftLinkedIn(dspy.Signature):
    """
    Draft a viral-style LinkedIn post.
    
    CRITICAL CONSTRAINTS (ANTI-ROBOT):
    1.  **Vocabulary**: Use simple, Grade 8 English. Avoid complex words like "leverage", "utilize", "foster".
    2.  **Formatting**: NO bold (`**`) and NO italics (`*`). Plain text only.
    3.  **The "See More" Trap**: The first 2 lines must generate curiosity.
    4.  **Structure**: Short paragraphs (max 2 lines). Use whitespace.
    5.  **Tone**: Conversational, slightly contrarian.
    """
    
    topic = dspy.InputField(desc="The broad topic.")
    outline = dspy.InputField(desc="The core value points to cover.")
    hooks_list = dspy.InputField(desc="Inspiration for hooks.")
    psychological_trigger = dspy.InputField(desc="The specific mental model.")
    
    # OUTPUTS
    headline_hook = dspy.OutputField(desc="The first 1-2 lines. Emotional, engaging, and under 60 chars. NO Markdown.")
    main_content = dspy.OutputField(desc="""The body. 
    1. NO asterisks/bolding/italics. 
    2. Short paragraphs (max 2 lines). 
    3. Include Concrete Examples for each point.
    4. Simple, direct language (Grade 8 level).
    5. Do NOT repeat the headline_hook. Start directly with the first insight/bridge.
    6. CRITICAL:Do not use Em dashes or Markdown symbols(** or * or #).
    """)
    interactive_cta = dspy.OutputField(desc="A question ending the post. E.g., 'Which step will you take?' (NO hashtags here).")

class PolisherSignature(dspy.Signature):
    """Review the draft. Improve readability and CTA strength."""
    draft_content = dspy.InputField(desc="The raw drafted thread.")
    psychological_trigger = dspy.InputField(desc="The trigger used, to ensure we don't edit out the persuasion.")
    
    polished_content = dspy.OutputField(desc="The final thread as a Python List of strings. ['Tweet 1', 'Tweet 2', ...]")
    tldr = dspy.OutputField(desc="A summary tweet of exactly 1 sentence.")
    cta = dspy.OutputField(desc="A CTA that offers value FIRST (Reciprocity). E.g., 'Grab the free guide' vs 'Sign up'.")

class RefineFullPost(dspy.Signature):
    """Refine an entire post based on user feedback (Manual Override)."""
    current_hook = dspy.InputField()
    current_body = dspy.InputField()
    current_cta = dspy.InputField()
    instruction = dspy.InputField()
    platform = dspy.InputField(desc="Target platform (Twitter or LinkedIn).")
    
    refined_hook = dspy.OutputField(desc="Refined hook. LinkedIn: No formatting.")
    refined_body = dspy.OutputField(desc="Refined body. LinkedIn: Grade 8 vocab, NO bold/italics.")
    refined_cta = dspy.OutputField()

# ==============================================================================
# 4. DSPY MODULES
# ==============================================================================

class ArchitectModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.architect = dspy.ChainOfThought(ArchitectSignature)
    
    def forward(self, raw_topic, research_context):
        return self.architect(raw_topic=raw_topic, research_context=research_context)

class HookModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_batch = dspy.Predict(BatchHookSignature)
    
    def forward(self, topic, primary_trigger):
        return self.generate_batch(topic=topic, primary_trigger=primary_trigger)

class DraftingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.draft_twitter = dspy.Predict(DraftThread)
        # CHANGED: Use ChainOfThought for "Reasoning" capabilities
        self.draft_linkedin = dspy.ChainOfThought(DraftLinkedIn)

    def forward(self, topic, outline, hooks_list, psychological_trigger, platform="Twitter"):
        if platform == "LinkedIn":
             # Zero-Shot for LinkedIn with Chain of Thought
             return self.draft_linkedin(
                topic=topic,
                outline=str(outline),
                hooks_list=str(hooks_list),
                psychological_trigger=psychological_trigger
             )
        else:
            # Compiled Model for Twitter
            return self.draft_twitter(
                topic=topic, 
                outline=str(outline), 
                hooks_list=str(hooks_list),
                psychological_trigger=psychological_trigger, 
                platform=platform
            )

class PolisherModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.polish = dspy.Predict(PolisherSignature)
    
    def forward(self, draft_content, psychological_trigger):
        return self.polish(draft_content=str(draft_content), psychological_trigger=psychological_trigger)

# ==============================================================================
# 5. MAIN ORCHESTRATOR
# ==============================================================================

class AgenticThreadCreator(dspy.Module):
    """The main module that orchestrates the sub-modules."""
    def __init__(self):
        super().__init__()
        self.architect_module = ArchitectModule()
        self.hook_module = HookModule()
        self.drafting_module = DraftingModule()
        self.polisher_module = PolisherModule()
        self.refine_full_module = dspy.Predict(RefineFullPost)
    
    def forward(self, raw_topic, research_context, platform="Twitter"):
        # 1. Architect Step
        architect_out = self.architect_module(raw_topic=raw_topic, research_context=research_context)
        
        # 2. Hook Step
        hook_out = self.hook_module(
            topic=architect_out.refined_topic, 
            primary_trigger=architect_out.psychological_trigger
        )
        
        # 3. Draft Step
        # Join hooks with newlines for cleaner input to the LLM
        hooks_str = "\n".join([f"- {h}" for h in hook_out.hooks_list]) if isinstance(hook_out.hooks_list, list) else str(hook_out.hooks_list)

        draft_out = self.drafting_module(
            topic=architect_out.refined_topic, 
            outline=architect_out.thread_outline, 
            hooks_list=hooks_str,
            psychological_trigger=architect_out.psychological_trigger,
            platform=platform
        )
        
        # 4. Polish Step & Unification
        if platform == "LinkedIn":
             # LinkedIn Path (Zero-Shot)
             post_body = draft_out.post_body
             final_content = [post_body] 
             
             # LOGIC FIX: Extract the first line as the hook for analytics
             # We split by double newline or period to find the first "punch"
             try:
                 lines = post_body.strip().split('\n')
                 # Get the first non-empty line
                 selected_hook = next((line for line in lines if line.strip()), "Hook extraction failed")
                 # Truncate if it's too long (over 100 chars) to be a hook
                 if len(selected_hook) > 100: selected_hook = selected_hook[:97] + "..."
             except:
                 selected_hook = "Hook extraction failed"

             final_cta = "Check comments." # Default for LinkedIn if not generated
             final_tldr = "N/A"
             
        else:
             # Twitter Path (Compiled)
             polisher_out = self.polisher_module(
                draft_content=draft_out.tweet_thread,
                psychological_trigger=architect_out.psychological_trigger
             )
             
             # PARSING FIX: Ensure we have a list, not a string representation of a list
             raw_tweets = polisher_out.polished_content
             if isinstance(raw_tweets, str):
                 try:
                     # Safe evaluation to convert "['a','b']" string to list ['a','b']
                     final_content = ast.literal_eval(raw_tweets)
                 except:
                     # Fallback if eval fails: treat as single string split by double newlines
                     final_content = raw_tweets.split('\n\n')
             else:
                 final_content = raw_tweets

             selected_hook = draft_out.selected_hook

             # 5. DEDUPLICATION FIX: Remove the hook if it appears as the first tweet
             # The DraftThread signature often enforces "Tweet 1 MUST be the selected hook"
             # But the user wants them separated.
             if isinstance(final_content, list) and len(final_content) > 0:
                 # Check strict equality or fuzzier inclusion
                 first_tweet = final_content[0].strip()
                 hook_text = selected_hook.strip()
                 
                 # Logic: if first tweet IS the hook, or contains the hook entirely
                 if first_tweet == hook_text or (len(hook_text) > 20 and hook_text in first_tweet):
                     print("✂️ Detected Hook repetition in Tweet #1. Removing duplicate.")
                     final_content = final_content[1:]
                 elif len(final_content) > 0 and len(first_tweet) < 15 and "1/" in first_tweet:
                     # Heuristic: sometimes "1/" is its own tweet? Unlikely but good safety.
                     pass

             final_cta = polisher_out.cta
             final_tldr = polisher_out.tldr

        return dspy.Prediction(
            refined_topic=architect_out.refined_topic,
            psychological_trigger=architect_out.psychological_trigger,
            template_type=architect_out.template_type,
            thread_outline=architect_out.thread_outline,
            hooks_list=hook_out.hooks_list,
            selected_hook=selected_hook,
            tweet_thread=final_content, 
            polished_content=final_content,
            tldr=final_tldr,
            cta=final_cta
        )