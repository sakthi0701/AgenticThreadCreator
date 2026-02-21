import dspy
import ast
import os
import threading
import random
from dotenv import load_dotenv
import time
from utils.metrics_tracker import MetricsTracker
from utils.text_cleaning import clean_linkedin_text, clean_text_universal
from config import settings

load_dotenv()


# ==============================================================================
# 1. LLM CONFIGURATION (Multi-Model: Gemini Flash + Kimi K2)
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
        """Delegates the request to the active LM with retry logic and exponential backoff."""
        attempts = 0
        last_exception = None
        
        while attempts < self.max_retries * len(self.keys):
            try:
                # Delegate to the active dspy.LM instance
                result = self.active_lm(prompt=prompt, **kwargs)
                
                # [FIXED] Only track successful calls
                MetricsTracker.increment("llm_calls")
                return result
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if retryable (rate limit or service unavailable)
                is_rate_limit = any(x in error_str for x in [
                    "429", "quota", "resource exhausted", "503", "unavailable", "rate limit"
                ])
                
                if is_rate_limit and attempts < (self.max_retries * len(self.keys) - 1):
                    with self.lock:
                        self._rotate_client()
                    
                    # Exponential backoff with jitter (capped at 60s)
                    delay = min(self.retry_delay * (2 ** attempts) + random.uniform(0, 1), 60)
                    print(f"⚠️ Rate limit hit. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    attempts += 1
                else:
                    # Non-retryable error or exhausted retries
                    raise e
        
        raise last_exception or RuntimeError("All API keys exhausted or max retries reached.")

    def __call__(self, prompt=None, **kwargs):
        return self.basic_request(prompt=prompt, **kwargs)

class MultiKeyGroq(dspy.LM):
    """
    Robust wrapper for Groq (Kimi) with Round-Robin Key Rotation.
    """
    def __init__(self, model_name, keys, **kwargs):
        self.max_retries = kwargs.pop('max_retries', 3)
        self.retry_delay = kwargs.pop('retry_delay', 2.0)
        self.keys = keys
        if not self.keys:
            raise ValueError("MultiKeyGroq requires at least one API key.")
            
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
        if force_index is not None:
             self.current_key_index = force_index
        else:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            
        active_key = self.keys[self.current_key_index]
        masked = active_key[:4] + "..." + active_key[-4:]
        
        # Use Standard OpenAI-compatible client for Groq
        self.active_lm = dspy.LM(
            model=self.model_name,
            api_key=active_key,
            api_base="https://api.groq.com/openai/v1",
            **self.kwargs
        )
        print(f"🔄 Kimi/Groq: Switched to Key #{self.current_key_index + 1} ({masked})")

    def basic_request(self, prompt=None, **kwargs):
        """Delegates the request to the active LM with retry logic and exponential backoff."""
        attempts = 0
        last_exception = None
        
        while attempts < self.max_retries * len(self.keys):
            try:
                result = self.active_lm(prompt=prompt, **kwargs)
                
                # [FIXED] Only track successful calls
                MetricsTracker.increment("llm_calls")
                return result
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                is_rate_limit = any(x in error_str for x in ["429", "rate limit", "quota"])
                
                if is_rate_limit and attempts < (self.max_retries * len(self.keys) - 1):
                    with self.lock:
                        print(f"⚠️ Groq Rate Limit on Key #{self.current_key_index + 1}. Rotating...")
                        self._rotate_client()
                    
                    # Exponential backoff with jitter
                    delay = min(self.retry_delay * (2 ** attempts) + random.uniform(0, 1), 60)
                    time.sleep(delay)
                    attempts += 1
                else:
                    raise e
                    
        raise last_exception or RuntimeError("All Groq keys exhausted.")

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
        print("ℹ️ DSPy already configured. Returning existing LMs.")
        main_lm = dspy.settings.lm
        
        # Re-load Kimi keys
        groq_keys = []
        g_primary = os.getenv("GROQ_API_KEY")
        if g_primary: groq_keys.append(g_primary)
        j = 2
        while True:
            k = os.getenv(f"GROQ_API_KEY_{j}")
            if not k: break
            groq_keys.append(k)
            j += 1
            
        kimi_lm = None
        if groq_keys:
             try:
                kimi_lm = MultiKeyGroq(
                    model_name="openai/moonshotai/kimi-k2-instruct-0905",
                    keys=groq_keys,
                    temperature=0.7,
                    max_tokens=2000
                )
             except: pass
             
        return main_lm, kimi_lm

    # Instantiate our smart wrapper
    smart_gemini = MultiKeyGemini(
        model_name='gemini/gemini-3-flash-preview',
        keys=keys,
        max_tokens=settings.gemini_max_tokens,
        temperature=settings.gemini_temperature,
        max_retries=settings.gemini_max_retries
    )
    
    # Configure global settings (Default = Gemini Flash)
    dspy.settings.configure(lm=smart_gemini)
    print(f"✅ Configured DSPy with MultiKeyGemini ({len(keys)} keys found).")
    
    # Configure Secondary Model: Kimi K2 (via Groq Multikey)
    groq_keys = []
    # Scan for GROQ_API_KEY, GROQ_API_KEY_2, etc.
    g_primary = os.getenv("GROQ_API_KEY")
    if g_primary: groq_keys.append(g_primary)
    
    j = 2
    while True:
        k = os.getenv(f"GROQ_API_KEY_{j}")
        if not k: break
        groq_keys.append(k)
        j += 1

    kimi_lm = None
    if groq_keys:
        try:
            kimi_lm = MultiKeyGroq(
                model_name="openai/moonshotai/kimi-k2-instruct-0905",
                keys=groq_keys,
                temperature=settings.kimi_temperature,
                max_tokens=settings.kimi_max_tokens
            )
            print(f"✅ Configured Secondary Model: Kimi K2 ({len(groq_keys)} keys found).")
        except Exception as e:
            print(f"⚠️ Failed to configure Kimi (Groq): {e}")
    
    return smart_gemini, kimi_lm


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

class ClarifyIntent(dspy.Signature):
    """
    Analyze the user's raw dump. If it lacks Tone or Audience, generate 3 clarifying questions.
    If the intent is clear, output 'CLEAR'.
    """
    raw_dump = dspy.InputField()
    existing_profile = dspy.InputField(desc="The currently selected user profile traits.")
    
    status = dspy.OutputField(desc="'CLEAR' or 'NEEDS_INFO'")
    questions = dspy.OutputField(desc="List[str] - A list of 3 distinct questions to ask the user. Return empty list if status is CLEAR.")

class RefineStyle(dspy.Signature):
    """
    Analyze the provided content and the user's existing style.
    Merge the new stylistic patterns into the existing style description to create a richer, more accurate profile.
    Enhance, do not just overwrite. If current_style is empty, just analyze the content.
    CRITICAL: Ignore structural elements specific to the content format (e.g. templates, headers, lists). Focus ONLY on the author's linguistic voice (vocabulary, rhythm, empathy, authority).
    """
    content = dspy.InputField(desc="The user's final written content.")
    current_style = dspy.InputField(desc="The user's existing tone/style description.")
    
    style_summary = dspy.OutputField(desc="A concise string (10-15 words) merging old and new traits. E.g. 'Punchy, usage of metaphors, authoritative, prefers short paragraphs'.")

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
    
    STRICT CONSTRAINTS:
    1. Review the 'Hooks List' and select the single BEST hook for engagement.
    2. Write the thread starting with that selected hook.
    3. Output MUST be a valid list of strings: ["Tweet 1", "Tweet 2", ...].
    4. MAX LENGTH: Every single string in the list must be UNDER 280 characters.
    5. SPLITTING: If an outline point is long, split it into 2 separate tweets.
    6. NO PREAMBLE: Do not write 'Here is the thread'. Just output the list.
    7. EXCLUSION: Do NOT include the 'selected_hook' in the list. Start with the tweet AFTER the hook.
    8. Voice: Active voice only. Ban words like "can", "helps to", "in order to".
    9. POLISHING:
       - NO Hashtags.
       - NO Bold text (**).
       - The LAST tweet must be a standalone Call To Action (value-first).
    """
    topic = dspy.InputField()
    outline = dspy.InputField()
    hooks_list = dspy.InputField(desc="List of generated hook options for reference.")
    psychological_trigger = dspy.InputField(desc="The core persuasion principle to reinforce in the body text.")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'")
    
    # Profile & Voice Inputs
    audience = dspy.InputField(desc="Target audience.")
    tone = dspy.InputField(desc="Desired tone.")
    forbidden_words = dspy.InputField(desc="List of words to avoid.")
    voice_analysis = dspy.InputField(desc="Deep stylistic analysis of the user's authentic voice. MIMIC THIS EXACTLY.", format=str)

    selected_hook = dspy.OutputField(desc="The exact text of the hook you chose.")
    tweet_thread = dspy.OutputField(desc="A list of strings. NO hashtags. NO bold.")
    tldr = dspy.OutputField(desc="A 1-sentence summary of the main takeaway.")
    cta = dspy.OutputField(desc="A value-first CTA (e.g. 'Grab the guide below').")

class DraftLinkedIn(dspy.Signature):
    """
    Draft a viral-style LinkedIn post.
    
    CRITICAL CONSTRAINTS (ANTI-ROBOT):
    1.  **Vocabulary**: Use simple, Grade 8 English. Avoid complex words like "leverage", "utilize", "foster".
    2.  **Formatting**: NO bold (`**`) and NO italics (`*`). Plain text only.
    3.  **The "See More" Trap**: The first 2 lines must generate curiosity.
    4.  **Structure**: MIX paragraph lengths. Use 1-line sentences for impact, but GROUP related thoughts into 2-3 line paragraphs. Do NOT write every single sentence on a new line.
    5.  **Voice**: Conversational, personal, and authoritative.
    6.  **TL;DR**: End with a 2-sentence summary of the main takeaway.
    """
    topic = dspy.InputField(desc="The broad topic.")
    outline = dspy.InputField(desc="The core value points to cover.")
    hooks_list = dspy.InputField(desc="Inspiration for hooks.")
    psychological_trigger = dspy.InputField(desc="The specific mental model.")
    
    # Profile Inputs
    audience = dspy.InputField(desc="Audience Callout: Immediately after the hook, specify who this is for.")
    tone = dspy.InputField(desc="Tone: Ban absolutes. Use measured language.")
    forbidden_words = dspy.InputField(desc="Words to strictly avoid.")
    
    # OUTPUTS
    headline_hook = dspy.OutputField(desc="The first 1-2 lines. Emotional, engaging, and under 60 chars. NO Markdown.")
    main_content = dspy.OutputField(desc="""The body. 
    1. NO asterisks/bolding/italics. 
    2. Short paragraphs (max 2 lines). 
    3. Include Concrete Examples for each point.
    4. Simple, direct language (Grade 8 level).
    5. Do NOT repeat the headline_hook. Start directly with the first insight/bridge.
    6. CRITICAL: Do not use Em dashes or Markdown symbols (** or * or #).
    7. ZERO REDUNDANCY: The 'main_content' MUST NOT start with the hook. It must follow immediately after. If the hook is 'X', the body must start with 'Y', not 'X... Y' because the hook is shown seperately before the body.
    """)
    interactive_cta = dspy.OutputField(desc="A question ending the post. E.g., 'Which step will you take?' (NO hashtags here).")
    tldr = dspy.OutputField(desc="A 1-sentence summary of the main takeaway for metadata purposes.")

class PolisherSignature(dspy.Signature):
    """DEPRECATED: Merged into DraftThread."""
    draft_content = dspy.InputField()
    psychological_trigger = dspy.InputField()
    polished_content = dspy.OutputField()

class RefineFullPost(dspy.Signature):
    """Refine an entire post based on user feedback (Manual Override).
    
    CRITICAL INSTRUCTIONS:
    1. **NEVER SHORTEN CONTENT**: Unless the user explicitly writes "make it shorter", "summarize", or "condense", you MUST maintain the exact same length (word count) as the `current_body`.
    2. **PRESERVE EVERY DETAIL**: Keep ALL examples, statistics, technical terms, analogies, and specific details. Do not replace concrete examples with vague generalities.
    3. **CONTEXT AWARE**: Use the provided topic and audience to ensure the refinement stays on track and maintains relevance.
    4. **STRUCTURE**: Maintain the original structure (bullet points, lists, paragraphs) exactly unless asked to remove them. If the input has 5 bullet points, the output MUST have 5 bullet points.
    5. **PLATFORM RULES**:
       - LinkedIn: STRICTLY NO MARKDOWN. Remove all '**', '*', '##'. Use plain text only.
       - Twitter: Each tweet must be under 280 characters.
    """
    current_hook = dspy.InputField()
    current_body = dspy.InputField()
    current_cta = dspy.InputField()
    instruction = dspy.InputField(desc="The user's refinement request (e.g. 'add emojis', 'simpler vocabulary').")
    platform = dspy.InputField(desc="Target platform (Twitter or LinkedIn).")
    
    # Context Fields
    topic = dspy.InputField(desc="The original topic/subject matter.")
    audience = dspy.InputField(desc="Target audience for this content.")
    tone = dspy.InputField(desc="Desired tone/voice.")
    
    refined_hook = dspy.OutputField(desc="Refined hook. LinkedIn: No formatting. MAINTAIN original length unless instructed otherwise.")
    refined_body = dspy.OutputField(desc="Refined body. CRITICAL: MAINTAIN LENGTH & DETAIL. Do not compress or summarize unless explicitly asked.")
    refined_cta = dspy.OutputField(desc="Refined CTA.")

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

    def forward(self, topic, outline, hooks_list, psychological_trigger, platform="Twitter", audience="", tone="", forbidden_words="", voice_analysis=""):
        if platform == "LinkedIn":
             # Zero-Shot for LinkedIn with Chain of Thought
             result = self.draft_linkedin(
                topic=topic,
                outline=str(outline),
                hooks_list=str(hooks_list),
                psychological_trigger=psychological_trigger,
                audience=audience,
                tone=tone,
                forbidden_words=forbidden_words
             )
             
             # Apply Cleaning Logic Here (Universal Fix)
             result.headline_hook = clean_text_universal(result.headline_hook)
             result.main_content = clean_text_universal(result.main_content)
             result.interactive_cta = clean_text_universal(result.interactive_cta)
             result.tldr = clean_text_universal(result.tldr)
             # Hook Deduplication Logic
             try:
                 hook_text = result.headline_hook.lower().strip()
                 body_text = result.main_content.strip()
                 # If body starts with hook (approx match)
                 if body_text.lower().startswith(hook_text[:20]):
                      # Remove it
                      print(f"✂️ LinkedIn Deduplication: Removing valid hook from body start.")
                      # We can try to replace exactly or just slice
                      # simple slice might be risky if casing differs. 
                      # Let's find index.
                      idx = body_text.lower().find(hook_text)
                      if idx == 0:
                          result.main_content = body_text[len(hook_text):].strip()
             except Exception as e:
                 print(f"⚠️ Deduplication check failed: {e}")
             return result
        else:
            # compiled model
            result = self.draft_twitter(
                topic=topic, 
                outline=str(outline), 
                hooks_list=str(hooks_list),
                psychological_trigger=psychological_trigger, 
                platform=platform,
                audience=audience,
                tone=tone,
                forbidden_words=forbidden_words,
                voice_analysis=voice_analysis
            )
            
            # Apply Universal Cleaning to Twitter too (Remove Em Dashes etc)
            if hasattr(result, 'tweet_thread') and isinstance(result.tweet_thread, list):
                result.tweet_thread = [clean_text_universal(t) for t in result.tweet_thread]
            
            return result

class ClarifierModule(dspy.Module):
    def __init__(self, aux_lm=None):
        super().__init__()
        self.clarify = dspy.ChainOfThought(ClarifyIntent)
        self.aux_lm = aux_lm
    
    def forward(self, raw_dump, existing_profile):
        if self.aux_lm:
            with dspy.context(lm=self.aux_lm):
                return self.clarify(raw_dump=raw_dump, existing_profile=existing_profile)
        return self.clarify(raw_dump=raw_dump, existing_profile=existing_profile)


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
        # Configure both models
        self.main_lm, self.aux_lm = configure_llm()
        
        self.architect_module = ArchitectModule()
        self.hook_module = HookModule()
        self.drafting_module = DraftingModule()
        # self.polisher_module = PolisherModule() # DEPRECATED
        self.clarifier_module = ClarifierModule(aux_lm=self.aux_lm)
        self.refine_full_module = dspy.ChainOfThought(RefineFullPost)
    
    def refine_post_action(self, **kwargs):
        """Wrapper for refine with Kimi context."""
        if self.aux_lm:
             with dspy.context(lm=self.aux_lm):
                 return self.refine_full_module(**kwargs)
        return self.refine_full_module(**kwargs)
    
    def forward(self, raw_topic, research_context, platform="Twitter", audience="", tone="", forbidden_words="", **kwargs):
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
            platform=platform,
            audience=audience,
            tone=tone,
            forbidden_words=forbidden_words,
            # We don't have voice_analysis passed in 'forward' usually, let's assume it's passed via kwargs or we update forward sig
            # For now passing empty if not present, but AgenticThreadCreator.forward needs update if we want to pipe it.
            # See next replacement chunk for forward update.
            voice_analysis=kwargs.get('voice_analysis', "") 
        )
        
        # 4. Polish Step & Unification
        if platform == "LinkedIn":
             # LinkedIn Path (Zero-Shot)
             # Fix: Construct validity from the actual output fields of DraftLinkedIn
             
             # 1. Extract and clean individual components
             hook = clean_linkedin_text(draft_out.headline_hook)
             body = clean_linkedin_text(draft_out.main_content)
             cta = clean_linkedin_text(draft_out.interactive_cta)
             
             # 2. Combine into a single post body for the final output list
             # Ensure decent spacing
             post_body = f"{hook}\n\n{body}\n\n{cta}"
             
             final_content = [post_body] 
             
             # 3. Set Hook (already extracted)
             selected_hook = hook
             
             # 4. Set CTA
             final_cta = cta
             final_tldr = getattr(draft_out, 'tldr', "N/A")
             
        else:
             # Twitter Path (One-Shot Polisher)
             # Logic is now inside DraftThread constraints
             
             final_content = draft_out.tweet_thread
             selected_hook = draft_out.selected_hook
             
             # Extract TLDR/CTA directly from One-Shot
             final_cta = getattr(draft_out, "cta", "Check comments")
             final_tldr = getattr(draft_out, "tldr", "Summary unavailable")

             # REMOVED: Separate Polisher Call
             
             # 5. DEDUPLICATION FIX: Remove the hook if it appears as the first tweet
             if isinstance(final_content, list) and len(final_content) > 0:
                 # Check strict equality or fuzzier inclusion
                 first_tweet = final_content[0].strip().lower()
                 hook_text = selected_hook.strip().lower()
                 is_duplicate = (hook_text in first_tweet) or (first_tweet in hook_text)
                 
                 # Similarity ratio check (optional, but 'in' usually suffices for exact repeats)
                 if is_duplicate:
                     print("✂️ Detected Hook repetition in Tweet #1. Removing duplicate.")
                     final_content = final_content[1:] # Slice off the first tweet

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