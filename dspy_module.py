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
# 0. CONSTANTS — AI FORBIDDEN WORDS (sourced from Humanizer guide)
# ==============================================================================

AI_FORBIDDEN_WORDS = [
    "additionally", "align with", "crucial", "delve", "emphasizing", "enduring",
    "enhance", "fostering", "garner", "highlight", "interplay", "intricate",
    "intricacies", "key", "landscape", "pivotal", "showcase", "tapestry",
    "testament", "underscore", "valuable", "vibrant", "stands as", "serves as",
    "marks a", "represents a", "boasts", "groundbreaking", "nestled",
    "in the heart of", "breathtaking", "stunning", "must-visit", "renowned",
    "navigate", "leverage", "utilize", "foster", "deep dive", "game-changer",
    "transformative", "cutting-edge", "robust", "synergy", "holistic",
]

AI_FORBIDDEN_WORDS_STR = ", ".join(AI_FORBIDDEN_WORDS)


# ==============================================================================
# 1. LLM CONFIGURATION (Multi-Model: Gemini Flash + Kimi K2)
# ==============================================================================

class MultiKeyGemini(dspy.LM):
    """
    A robust wrapper around standard dspy.LM that manages multiple API keys
    using Round-Robin rotation. Automatically retries with a different key
    if a RateLimitError (429) occurs.
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

        self._rotate_client(force_index=0)
        super().__init__(model=model_name)
        self.lock = threading.Lock()

    def _rotate_client(self, force_index=None):
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
        attempts = 0
        last_exception = None

        while attempts < self.max_retries * len(self.keys):
            try:
                result = self.active_lm(prompt=prompt, **kwargs)
                MetricsTracker.increment("llm_calls")
                return result

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                is_rate_limit = any(x in error_str for x in [
                    "429", "quota", "resource exhausted", "503", "unavailable", "rate limit"
                ])

                if is_rate_limit and attempts < (self.max_retries * len(self.keys) - 1):
                    with self.lock:
                        self._rotate_client()
                    delay = min(self.retry_delay * (2 ** attempts) + random.uniform(0, 1), 60)
                    print(f"⚠️ Rate limit hit. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    attempts += 1
                else:
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

        self._rotate_client(force_index=0)
        super().__init__(model=model_name)
        self.lock = threading.Lock()

    def _rotate_client(self, force_index=None):
        if force_index is not None:
            self.current_key_index = force_index
        else:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)

        active_key = self.keys[self.current_key_index]
        masked = active_key[:4] + "..." + active_key[-4:]

        self.active_lm = dspy.LM(
            model=self.model_name,
            api_key=active_key,
            api_base="https://api.groq.com/openai/v1",
            **self.kwargs
        )
        print(f"🔄 Kimi/Groq: Switched to Key #{self.current_key_index + 1} ({masked})")

    def basic_request(self, prompt=None, **kwargs):
        attempts = 0
        last_exception = None

        while attempts < self.max_retries * len(self.keys):
            try:
                result = self.active_lm(prompt=prompt, **kwargs)
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

    primary = os.getenv("gemini_api_key")
    if primary:
        keys.append(primary)

    i = 2
    while True:
        k = os.getenv(f"gemini_api_key_{i}")
        if not k:
            break
        keys.append(k)
        i += 1

    if not keys:
        raise ValueError("gemini_api_key not found in .env file.")

    if dspy.settings.lm and isinstance(dspy.settings.lm, MultiKeyGemini):
        print("ℹ️ DSPy already configured. Returning existing LMs.")
        main_lm = dspy.settings.lm

        groq_keys = []
        g_primary = os.getenv("GROQ_API_KEY")
        if g_primary:
            groq_keys.append(g_primary)
        j = 2
        while True:
            k = os.getenv(f"GROQ_API_KEY_{j}")
            if not k:
                break
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
            except:
                pass

        return main_lm, kimi_lm

    smart_gemini = MultiKeyGemini(
        model_name='gemini/gemini-3-flash-preview',
        keys=keys,
        max_tokens=settings.gemini_max_tokens,
        temperature=settings.gemini_temperature,
        max_retries=settings.gemini_max_retries
    )

    dspy.settings.configure(lm=smart_gemini)
    print(f"✅ Configured DSPy with MultiKeyGemini ({len(keys)} keys found).")

    groq_keys = []
    g_primary = os.getenv("GROQ_API_KEY")
    if g_primary:
        groq_keys.append(g_primary)

    j = 2
    while True:
        k = os.getenv(f"GROQ_API_KEY_{j}")
        if not k:
            break
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
    Ignore structural elements specific to the content format (templates, headers, lists).
    Focus ONLY on the author's linguistic voice (vocabulary, rhythm, empathy, authority).
    """
    content = dspy.InputField(desc="The user's final written content.")
    current_style = dspy.InputField(desc="The user's existing tone/style description.")

    style_summary = dspy.OutputField(desc="A concise string (10-15 words) merging old and new traits. E.g. 'Punchy, usage of metaphors, authoritative, prefers short paragraphs'.")


class GenerateSocialThread(dspy.Signature):
    """Generate a compelling social media thread/post using the Micro-to-Macro Pivot technique.
    
    Great content starts with a highly specific, mundane, or vulnerable observation (the Micro)
    and connects it to a profound, universal lesson (the Macro).
    
    VOICE RULES:
    1. Broetry formatting: Short, punchy lines. Paragraphs NEVER exceed 2 sentences.
    2. Humble, reflective tone — curious student sharing a realization, NOT a guru lecturing.
    3. Plain 5th-grade English. Zero corporate jargon.
    4. Start with a pattern-interrupt or vulnerable hook.
    5. Pivot from a specific scene/story to the universal lesson.
    6. End with a single punchy philosophical takeaway.
    """
    topic = dspy.InputField(desc="The topic to write about.")
    audience = dspy.InputField(desc="Target audience for the content.")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'.")
    style_notes = dspy.InputField(desc="Additional voice/style context from user analysis.")
    forbidden_words = dspy.InputField(desc="Words to never use.")

    hook = dspy.OutputField(desc="1-2 sentence opening that challenges a belief or starts mid-story.")
    body = dspy.OutputField(desc="The main content. Short paragraphs, broetry style. Micro story → Macro lesson.")
    cta = dspy.OutputField(desc="A simple, non-salesy question or call to action.")


# ==============================================================================
# 3. PHASE-BASED OPTIMIZED SIGNATURES (5-Call Architecture)
# ==============================================================================

class ArchitectSignature(dspy.Signature):
    """Analyze the raw topic and research to create a refined plan.

    PERSUASION TECHNIQUES — pick ONE and use it concretely:
    1. Social Proof: Open with a number that shows others already do this. "10,000 developers switched last month." Don't explain the technique — just use it.
    2. Loss Aversion: Frame the topic around what the reader loses by ignoring it. Pain beats gain.
    3. Authority: Drop a specific credential, study, or insider detail early. Don't say "experts say" — name one.
    4. Open Loop: Start a story in the hook but don't resolve it until the last tweet. Force the read-through.
    5. Curiosity Gap: Give enough to make the reader feel they're missing something. Specific enough to interest, vague enough to click.
    6. Speaks to "me": Use a sentence like "If you've ever..." that makes the reader feel seen. Don't explain what you're doing.
    7. Simple solution: Promise a surprisingly easy fix to a hard problem. Grade 8 English throughout.
    """
    raw_topic = dspy.InputField(desc="The user's original request.")
    research_context = dspy.InputField(desc="Key insights from SerpAPI research.")
    voice_analysis = dspy.InputField(desc="The user's authentic voice and stylistic traits. Let this shape the outline's rhythm and language.", format=str)

    refined_topic = dspy.OutputField(desc="A specific, punchy title under 10 words. Use a number or a strong verb. No jargon.")
    psychological_trigger = dspy.OutputField(desc="The ONE technique name from the list above.")
    trigger_application = dspy.OutputField(desc="One sentence: exactly HOW to apply that technique to THIS topic.")
    template_type = dspy.OutputField(desc="The structural template (e.g. 'The Hero's Journey', 'The Listicle', 'The Contrarian Take').")
    thread_outline = dspy.OutputField(desc="5-7 ideas. Mix types: one statistic, one personal story beat, one counterintuitive claim, one concrete example. NOT a uniform list. Vary length.")


class BatchHookSignature(dspy.Signature):
    """Generate 5 distinct viral hooks using specific psychological angles."""
    topic = dspy.InputField(desc="The refined topic.")
    primary_trigger = dspy.InputField(desc="The main persuasion technique chosen by the Architect.")

    hooks_list = dspy.OutputField(desc="""A list of exactly 5 hooks. Each uses a different angle:
    1. Primary Trigger Hook (use the chosen technique directly).
    2. Loss Hook (what do they lose by ignoring this?).
    3. Social Proof Hook (open with a specific number or crowd reference).
    4. Cliffhanger Hook (start a story — do not finish it).
    5. Villain Hook (name a common belief or enemy and attack it).
    Hard rules: no hook longer than 280 characters. No bold text. No hashtags. No em dashes.
    """)


class DraftThread(dspy.Signature):
    """Convert an outline into a full Twitter thread.

    RULES — read every line:
    1. Pick the single best hook from hooks_list. Output it as selected_hook.
    2. The tweet_thread list starts AFTER the hook. Do not repeat the hook as tweet 1.
    3. Every string in tweet_thread must be under 280 characters. Split long points into two tweets.
    4. Output tweet_thread as a valid Python list of strings only. No preamble, no explanation.
    5. Active voice only. Cut "can", "helps to", "in order to", "it is important to note".
    6. No hashtags. No bold text (**). No em dashes (—). No curly quotes.
    7. Vary sentence length. Not every tweet is one sentence, not every tweet is four.
    8. The last tweet is a standalone Call To Action — value-first, not "follow me for more".
    9. Forbidden words (never use): additionally, align with, crucial, delve, emphasizing,
       enduring, enhance, fostering, garner, highlight, interplay, intricate, key, landscape,
       pivotal, showcase, tapestry, testament, underscore, valuable, vibrant, stands as,
       serves as, groundbreaking, nestled, leverage, utilize, synergy, holistic, transformative.
    """
    topic = dspy.InputField()
    outline = dspy.InputField()
    hooks_list = dspy.InputField(desc="Hook options to choose from.")
    psychological_trigger = dspy.InputField(desc="The persuasion technique to reinforce in the body.")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'")

    audience = dspy.InputField(desc="Target audience.")
    tone = dspy.InputField(desc="Desired tone.")
    forbidden_words = dspy.InputField(desc="Additional words to avoid on top of the built-in list.")
    voice_analysis = dspy.InputField(desc="The user's authentic voice. Copy the rhythm and vocabulary exactly.", format=str)

    selected_hook = dspy.OutputField(desc="The exact hook text you chose.")
    tweet_thread = dspy.OutputField(desc="A Python list of strings. No hashtags. No bold. No em dashes.")
    tldr = dspy.OutputField(desc="One sentence: the main takeaway.")
    cta = dspy.OutputField(desc="A value-first CTA (e.g. 'Grab the guide below').")


class DraftLinkedIn(dspy.Signature):
    """Draft a viral-style LinkedIn post in plain, human English.

    Hard rules — no exceptions:
    1. Grade 8 vocabulary. Cut: leverage, utilize, foster, enhance, pivotal, groundbreaking, nestled, vibrant.
    2. No bold, no italics, no asterisks, no hashtags in the body. Plain text only.
    3. First 2 lines must earn the "See More" click. Make them feel something or promise something specific.
    4. Mix paragraph lengths. One-liners for impact. Two-to-three-line paragraphs for context. Never write every sentence on its own line.
    5. No em dashes. No curly quotes. No bullet points unless the outline explicitly calls for a list.
    6. The main_content must NOT repeat or rephrase the headline_hook. It continues directly from where the hook left off.
    7. End with a single question that makes the reader want to comment — not a generic "What do you think?"
    8. Forbidden words: additionally, align with, crucial, delve, emphasizing, enduring, enhance,
       fostering, garner, highlight, interplay, intricate, key, landscape, pivotal, showcase,
       tapestry, testament, underscore, valuable, vibrant, stands as, serves as, groundbreaking,
       nestled, leverage, utilize, synergy, holistic, transformative, cutting-edge, robust.
    """
    topic = dspy.InputField(desc="The broad topic.")
    outline = dspy.InputField(desc="The core value points to cover.")
    hooks_list = dspy.InputField(desc="Hook inspiration.")
    psychological_trigger = dspy.InputField(desc="The persuasion technique to apply.")

    audience = dspy.InputField(desc="Who this is for. Name them in the first or second line.")
    tone = dspy.InputField(desc="Desired tone. Use measured language — no absolutes.")
    forbidden_words = dspy.InputField(desc="Additional words to avoid.")

    headline_hook = dspy.OutputField(desc="The first 1-2 lines. Emotional or specific. Under 60 characters. Plain text, no markdown.")
    main_content = dspy.OutputField(desc="The body. Plain text. No asterisks. No markdown symbols. Short paragraphs. Concrete examples for every claim. Does NOT start with the headline_hook.")
    interactive_cta = dspy.OutputField(desc="A specific question to end the post. No hashtags.")
    tldr = dspy.OutputField(desc="One sentence: the main takeaway (for metadata only).")


class HumanizerSignature(dspy.Signature):
    """Strip AI writing patterns and make this content sound like a real person wrote it.

    WHAT TO FIX — scan for every item:
    1. Puffed-up importance: Remove phrases like "marks a pivotal moment", "testament to", "reflects broader trends". Say the thing directly.
    2. Promotional language: Cut "groundbreaking", "nestled", "vibrant", "breathtaking", "stunning". Replace with specific facts.
    3. -ing phrase tack-ons: Remove "...highlighting X", "...fostering Y", "...showcasing Z" at the end of sentences. They add nothing.
    4. Vague attributions: Replace "experts say" or "industry observers note" with a specific name, study, or statistic — or cut it.
    5. Copula avoidance: Replace "serves as", "stands as", "functions as" with "is" or "are".
    6. Rule of three: If three things are listed just to look thorough, cut to the one that actually matters.
    7. False ranges: Remove "from X to Y, from A to B" constructions. They sound like a brochure.
    8. Generic endings: Cut "exciting times lie ahead", "the future looks bright", "this represents a major step". End on a specific fact or real opinion.
    9. Forbidden words (remove or replace): additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight, interplay, intricate, key, landscape, pivotal, showcase, tapestry, testament, underscore, valuable, vibrant, leverage, utilize, synergy, holistic, transformative, groundbreaking, nestled, robust, cutting-edge.
    10. Hedging: Cut "could potentially possibly", "it might be argued that", "in some ways". Say what you mean.
    11. Chatbot artifacts: Remove "I hope this helps!", "Let me know if...", "Of course!", "Certainly!", "Great question!".
    12. Em dash overuse: Replace chains of em dashes with commas, periods, or restructured sentences.
    13. Sycophancy: Remove "You're absolutely right", "That's an excellent point", "Great question".

    WHAT TO ADD:
    - Vary sentence length. Mix one-liners with longer ones.
    - Use "I" when the content calls for a personal perspective.
    - Replace vague claims with one specific detail, number, or example.
    - If the ending is generic, replace it with a real opinion or a specific next step.

    CONSTRAINT: Keep the platform rules.
    - Twitter: every tweet must stay under 280 characters.
    - LinkedIn: no markdown, no bold, no asterisks, plain text only.
    - Do NOT shorten the content unless a sentence is pure filler with zero information.
    """
    draft_content = dspy.InputField(desc="The raw drafted content (tweet list or LinkedIn post body).")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'")
    voice_analysis = dspy.InputField(desc="The user's authentic voice. Match the rhythm and word choices.", format=str)
    topic = dspy.InputField(desc="The topic, for context.")

    humanized_content = dspy.OutputField(desc="The cleaned content. Same format as input (list for Twitter, plain text for LinkedIn). No new AI patterns introduced.")
    changes_made = dspy.OutputField(desc="A short plain-English summary of the main patterns removed and what replaced them.")


class PolisherSignature(dspy.Signature):
    """DEPRECATED: Merged into DraftThread."""
    draft_content = dspy.InputField()
    psychological_trigger = dspy.InputField()
    polished_content = dspy.OutputField()


class RefineFullPost(dspy.Signature):
    """Refine an entire post based on user feedback.

    Rules:
    1. Never shorten content unless the user explicitly writes "make it shorter", "summarize", or "condense".
    2. Keep ALL examples, statistics, technical terms, analogies, and specific details. Do not swap concrete examples for vague generalities.
    3. Maintain the original structure exactly unless asked to change it. Five bullet points in means five bullet points out.
    4. Platform rules: LinkedIn = strictly no markdown. Remove all **, *, ##. Plain text only. Twitter = each tweet under 280 characters.
    5. No new AI patterns. No "furthermore", "it is worth noting", "in conclusion". Write like a person.
    """
    current_hook = dspy.InputField()
    current_body = dspy.InputField()
    current_cta = dspy.InputField()
    instruction = dspy.InputField(desc="The user's refinement request.")
    platform = dspy.InputField(desc="'Twitter' or 'LinkedIn'.")

    topic = dspy.InputField(desc="The original topic.")
    audience = dspy.InputField(desc="Target audience.")
    tone = dspy.InputField(desc="Desired tone.")

    refined_hook = dspy.OutputField(desc="Refined hook. LinkedIn: plain text only. Maintain original length unless instructed otherwise.")
    refined_body = dspy.OutputField(desc="Refined body. Maintain length and all details. Do not compress.")
    refined_cta = dspy.OutputField(desc="Refined CTA.")


# ==============================================================================
# 4. DSPY MODULES
# ==============================================================================

class ArchitectModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.architect = dspy.ChainOfThought(ArchitectSignature)

    def forward(self, raw_topic, research_context, voice_analysis=""):
        return self.architect(
            raw_topic=raw_topic,
            research_context=research_context,
            voice_analysis=voice_analysis
        )


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
        self.draft_linkedin = dspy.ChainOfThought(DraftLinkedIn)

    def forward(self, topic, outline, hooks_list, psychological_trigger, platform="Twitter",
                audience="", tone="", forbidden_words="", voice_analysis=""):
        if platform == "LinkedIn":
            result = self.draft_linkedin(
                topic=topic,
                outline=str(outline),
                hooks_list=str(hooks_list),
                psychological_trigger=psychological_trigger,
                audience=audience,
                tone=tone,
                forbidden_words=forbidden_words
            )

            result.headline_hook = clean_text_universal(result.headline_hook)
            result.main_content = clean_text_universal(result.main_content)
            result.interactive_cta = clean_text_universal(result.interactive_cta)
            result.tldr = clean_text_universal(result.tldr)

            # Hook deduplication
            try:
                hook_text = result.headline_hook.lower().strip()
                body_text = result.main_content.strip()
                if body_text.lower().startswith(hook_text[:20]):
                    print("✂️ LinkedIn Deduplication: Removing hook from body start.")
                    idx = body_text.lower().find(hook_text)
                    if idx == 0:
                        result.main_content = body_text[len(hook_text):].strip()
            except Exception as e:
                print(f"⚠️ Deduplication check failed: {e}")

            return result
        else:
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

            if hasattr(result, 'tweet_thread') and isinstance(result.tweet_thread, list):
                result.tweet_thread = [clean_text_universal(t) for t in result.tweet_thread]

            return result


class HumanizerModule(dspy.Module):
    """
    Post-processing pass that strips AI writing patterns from drafted content.
    Runs after DraftingModule on both Twitter threads and LinkedIn posts.
    Uses Kimi (aux_lm) if available for a fast, cheap pass; falls back to Gemini.
    """
    def __init__(self, aux_lm=None):
        super().__init__()
        self.humanize = dspy.ChainOfThought(HumanizerSignature)
        self.aux_lm = aux_lm

    def forward(self, draft_content, platform, voice_analysis="", topic=""):
        def _run():
            return self.humanize(
                draft_content=str(draft_content),
                platform=platform,
                voice_analysis=voice_analysis,
                topic=topic
            )

        if self.aux_lm:
            with dspy.context(lm=self.aux_lm):
                result = _run()
        else:
            result = _run()

        print(f"🧹 Humanizer changes: {result.changes_made}")
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
        self.main_lm, self.aux_lm = configure_llm()

        self.architect_module = ArchitectModule()
        self.hook_module = HookModule()
        self.drafting_module = DraftingModule()
        self.humanizer_module = HumanizerModule(aux_lm=self.aux_lm)  # NEW
        self.clarifier_module = ClarifierModule(aux_lm=self.aux_lm)
        self.refine_full_module = dspy.ChainOfThought(RefineFullPost)

    def refine_post_action(self, **kwargs):
        """Wrapper for refine with Kimi context."""
        if self.aux_lm:
            with dspy.context(lm=self.aux_lm):
                return self.refine_full_module(**kwargs)
        return self.refine_full_module(**kwargs)

    def forward(self, raw_topic, research_context, platform="Twitter",
                audience="", tone="", forbidden_words="", **kwargs):

        voice_analysis = kwargs.get('voice_analysis', "")

        # Merge user forbidden words with the global AI forbidden words list
        combined_forbidden = AI_FORBIDDEN_WORDS_STR
        if forbidden_words:
            combined_forbidden = f"{forbidden_words}, {combined_forbidden}"

        # ------------------------------------------------------------------
        # 1. Architect — now receives voice_analysis so the outline inherits
        #    the user's rhythm from the start, not just at draft time.
        # ------------------------------------------------------------------
        architect_out = self.architect_module(
            raw_topic=raw_topic,
            research_context=research_context,
            voice_analysis=voice_analysis
        )

        # ------------------------------------------------------------------
        # 2. Hooks
        # ------------------------------------------------------------------
        hook_out = self.hook_module(
            topic=architect_out.refined_topic,
            primary_trigger=architect_out.psychological_trigger
        )

        hooks_str = (
            "\n".join([f"- {h}" for h in hook_out.hooks_list])
            if isinstance(hook_out.hooks_list, list)
            else str(hook_out.hooks_list)
        )

        # ------------------------------------------------------------------
        # 3. Draft
        # ------------------------------------------------------------------
        draft_out = self.drafting_module(
            topic=architect_out.refined_topic,
            outline=architect_out.thread_outline,
            hooks_list=hooks_str,
            psychological_trigger=architect_out.psychological_trigger,
            platform=platform,
            audience=audience,
            tone=tone,
            forbidden_words=combined_forbidden,
            voice_analysis=voice_analysis
        )

        # ------------------------------------------------------------------
        # 4. Humanizer pass — strips remaining AI patterns from the draft
        # ------------------------------------------------------------------
        if platform == "LinkedIn":
            hook = clean_linkedin_text(draft_out.headline_hook)
            body = clean_linkedin_text(draft_out.main_content)
            cta = clean_linkedin_text(draft_out.interactive_cta)

            # Run humanizer on the full combined post
            post_body_raw = f"{hook}\n\n{body}\n\n{cta}"
            humanizer_out = self.humanizer_module(
                draft_content=post_body_raw,
                platform="LinkedIn",
                voice_analysis=voice_analysis,
                topic=architect_out.refined_topic
            )

            humanized_post = humanizer_out.humanized_content
            final_content = [humanized_post]
            selected_hook = hook
            final_cta = cta
            final_tldr = getattr(draft_out, 'tldr', "N/A")

        else:
            # Twitter path
            final_content = draft_out.tweet_thread
            selected_hook = draft_out.selected_hook

            # Deduplication: remove hook if it leaked into tweet #1
            if isinstance(final_content, list) and len(final_content) > 0:
                first_tweet = final_content[0].strip().lower()
                hook_text = selected_hook.strip().lower()
                if (hook_text in first_tweet) or (first_tweet in hook_text):
                    print("✂️ Detected Hook repetition in Tweet #1. Removing duplicate.")
                    final_content = final_content[1:]

            # Run humanizer on the tweet list
            humanizer_out = self.humanizer_module(
                draft_content=final_content,
                platform="Twitter",
                voice_analysis=voice_analysis,
                topic=architect_out.refined_topic
            )

            # Parse humanized content back into a list if needed
            humanized = humanizer_out.humanized_content
            if isinstance(humanized, list):
                final_content = humanized
            elif isinstance(humanized, str):
                # Try to parse as Python list; fall back to original on failure
                try:
                    parsed = ast.literal_eval(humanized)
                    if isinstance(parsed, list):
                        final_content = parsed
                except Exception:
                    print("⚠️ Humanizer returned a string for Twitter; keeping original list.")

            final_cta = getattr(draft_out, "cta", "Check comments")
            final_tldr = getattr(draft_out, "tldr", "Summary unavailable")

        # ------------------------------------------------------------------
        # 5. Return unified prediction
        # ------------------------------------------------------------------
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