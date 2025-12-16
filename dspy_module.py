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
                if "429" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                    with self.lock:
                        # Double-check if another thread already rotated while we were waiting
                        # In a real rigorous impl, we might check timestamps, but here we just rotate.
                        print(f"⚠️ Rate Limit hit on Key #{self.current_key_index + 1}. Rotate & Retry...")
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
        model_name='gemini/gemini-2.5-flash',
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
    
    refined_topic = dspy.OutputField(desc="A clear, compelling topic title.")
    psychological_trigger = dspy.OutputField(desc="The ONE primary psychological principle to drive this thread (Select from list above).")
    trigger_application = dspy.OutputField(desc="A brief strategy on HOW to apply that trigger to this specific topic.")
    template_type = dspy.OutputField(desc="The best structural template (e.g., 'The Hero's Journey', 'The Listicle', 'The Contrarian Take').")
    thread_outline = dspy.OutputField(desc="A step-by-step outline. Ensure the chosen trigger is woven into the structure.")

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
    """)

class DraftThread(dspy.Signature):
    """Convert an outline into a full Twitter thread.
    
    INSTRUCTIONS:
    1. Review the 'Hooks List' and select the single BEST hook for engagement.
    2. Write the thread starting with that selected hook.
    3. Maintain the psychological pressure throughout.
    """
    topic = dspy.InputField()
    outline = dspy.InputField()
    hooks_list = dspy.InputField(desc="List of generated hook options.")
    psychological_trigger = dspy.InputField(desc="The core persuasion principle to reinforce in the body text.")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'")
    
    selected_hook = dspy.OutputField(desc="The exact text of the hook you chose.")
    tweet_thread = dspy.OutputField(desc="A list of strings. Tweet 1 MUST be the selected hook.")

class PolisherSignature(dspy.Signature):
    """Review the draft. Maximize 'Cognitive Ease' (readability) and ensure the CTA uses 'Reciprocity'."""
    draft_content = dspy.InputField(desc="The raw drafted thread.")
    psychological_trigger = dspy.InputField(desc="The trigger used, to ensure we don't edit out the persuasion.")
    
    polished_content = dspy.OutputField(desc="The final list. Short sentences. Punchy. No corporate fluff.")
    tldr = dspy.OutputField(desc="A summary tweet.")
    cta = dspy.OutputField(desc="A CTA that offers value FIRST (Reciprocity). E.g., 'Grab the free guide' vs 'Sign up'.")

class RefineFullPost(dspy.Signature):
    """Refine an entire post based on user feedback (Manual Override)."""
    current_hook = dspy.InputField()
    current_body = dspy.InputField()
    current_cta = dspy.InputField()
    instruction = dspy.InputField()
    
    refined_hook = dspy.OutputField()
    refined_body = dspy.OutputField()
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
        self.draft = dspy.Predict(DraftThread)

    def forward(self, topic, outline, hooks_list, psychological_trigger, platform="Twitter"):
        return self.draft(
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
    
    # We mainly use the sub-modules individually in the agent workflow, 
    # but this class serves as the model registry.
    def forward(self, raw_topic, research_context):
        # 1. Architect Step
        architect_out = self.architect_module(raw_topic=raw_topic, research_context=research_context)
        
        # 2. Hook Step
        # Using refined topic and key trigger for hook generation
        hook_out = self.hook_module(
            topic=architect_out.refined_topic, 
            primary_trigger=architect_out.psychological_trigger
        )
        
        # 3. Draft Step
        draft_out = self.drafting_module(
            topic=architect_out.refined_topic, 
            outline=architect_out.thread_outline, 
            hooks_list=hook_out.hooks_list,
            psychological_trigger=architect_out.psychological_trigger,
            platform="Twitter"
        )
        
        # 4. Polish Step
        polisher_out = self.polisher_module(
            draft_content=draft_out.tweet_thread,
            psychological_trigger=architect_out.psychological_trigger
        )
        
        return dspy.Prediction(
            refined_topic=architect_out.refined_topic,
            psychological_trigger=architect_out.psychological_trigger,
            template_type=architect_out.template_type,
            thread_outline=architect_out.thread_outline,
            hooks_list=hook_out.hooks_list,
            selected_hook=draft_out.selected_hook,
            tweet_thread=draft_out.tweet_thread,
            polished_content=polisher_out.polished_content,
            tldr=polisher_out.tldr,
            cta=polisher_out.cta
        )