# main.py
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import ast
import uuid # [NEW] For correlation IDs
import httpx
import json

# Import your workflow
from run_workflow import agentic_workflow, get_step_content

# --- 1. IMPORT PolisherAgent HERE ---
from agents.polisher import PolisherAgent
from dspy_module import ClarifierModule
from utils.profile_manager import ProfileManager
from utils.groq_client import GroqClient
from utils.metrics_tracker import MetricsTracker  # [NEW] Import Tracker
from utils.data_logger import log_interaction # [NEW] Import Logger
import dspy

# --- 2. Create the FastAPI app ---
app = FastAPI(
    title="Agentic Twitter Thread Creator API",
    description="An API to run the Agno agent workflow."
)

# --- 2.5 Add Middleware for IP Tracking ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # 1. Capture Client IP (Trust X-Forwarded-For if available)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    # [DEBUG] Print captured IP
    if not client_ip: 
        client_ip = "unknown"
    print(f"🕵️ Middleware Detected IP: {client_ip}")

    # 2. Set Context for Downstream Agents
    MetricsTracker.set_context_ip(client_ip)
    
    response = await call_next(request)
    return response

# --- 3. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. CREATE AGENT INSTANCE HERE ---
polisher_agent = PolisherAgent()
clarifier_module = ClarifierModule()

from typing import Optional, List, Dict

# --- 5. Define Request Bodies ---
class ThreadRequest(BaseModel):
    topic: str
    template: str = "Choose a template suitable for this post"
    platform: str = "Twitter" 
    clarification_context: Optional[str] = None # Legacy/Final Context
    clarification_history: List[Dict] = [] # NEW: Conversational History
    profile_name: Optional[str] = None 
    audience: Optional[str] = None
    tone: Optional[str] = None
    forbidden_words: Optional[List[str]] = None
    style_analysis: Optional[str] = None # NEW Field
    pro_mode: bool = False  # Pro Mode: full 5-agent pipeline

# --- 6. DEFINE EditRequest HERE ---
class EditRequest(BaseModel):
    tweets: list[str]
    prompt: str

class FullRefineRequest(BaseModel):
    hook: str
    tweets: list[str]
    cta: str
    prompt: str
    platform: str = "Twitter"
    parent_id: Optional[str] = None # Link to original generation
    
    # Context fields for better refinement
    topic: Optional[str] = ""
    audience: Optional[str] = ""
    tone: Optional[str] = ""
    voice_analysis: Optional[str] = ""

class LogInteractionRequest(BaseModel):
    """
    For manual saves/copies from the frontend.
    """
    interaction_id: Optional[str] = None # Client can generate one if missing
    parent_id: Optional[str] = None
    action_type: str = "MANUAL_SAVE"
    content: dict # The full thread structure
    meta: dict = {}

# Health Check
@app.get("/")
def health_check():
    return {"status": "ok", "service": "Agentic Thread Creator"}

# --- 7. Define API Endpoints ---
@app.post("/generate-thread")
async def generate_thread_endpoint(request: ThreadRequest):
    """
    Runs the full agentic workflow and returns the final thread.
    NOW ASYNC: Wraps blocking agent calls in a threadpool.
    """
    print(f"API: Received request for topic: {request.topic}")

    # [FIXED] Atomic check-and-increment to prevent race conditions
    # [FIXED] 1. Read-Only Limit Check (Don't increment yet)
    current_ip = MetricsTracker.get_context_ip()
    if not await MetricsTracker.check_limit(current_ip):
         print(f"⛔ Blocked checking from {current_ip}: Over Daily Limit")
         return JSONResponse(
             status_code=429, 
             content={"error": "Daily generation limit reached (5 posts/day). Please upgrade."}
         )
    
    # 1. Load Profile (Priority: Payload > ID Lookup)
    if request.audience or request.tone:
        # Use provided data (Frontend LocalStorage Source of Truth)
        profile_data = {
            "audience": request.audience or "General Audience",
            "tone": request.tone or "Professional",
            "forbidden_words": request.forbidden_words or []
        }
    else:
        profile_data = ProfileManager.get_profile(request.profile_name)
        
    profile_str = f"Audience: {profile_data.get('audience')}, Tone: {profile_data.get('tone')}"
    
    # Append Style Analysis if provided
    if request.style_analysis:
        profile_str += f"\nVoice Signature (Analysis): {request.style_analysis}"
    
    # 2. Clarification Logic (Conversational Groq Loop)
    # If explicit context is NOT provided, we enter the loop.
    if not request.clarification_context:
        
        # Check if we should ask a question
        print(f"API: Checking with Kimi (History: {len(request.clarification_history)} items)...")
        
        # Build Context String
        ctx_str = f"Platform: {request.platform}\n{profile_str}"
        if request.template:
             ctx_str += f"\nTemplate: {request.template}"

        def run_groq_check():
            return GroqClient.get_next_question(request.topic, request.clarification_history, context_str=ctx_str)
        
        groq_result = await run_in_threadpool(run_groq_check)
        
        if groq_result.get("status") == "ERROR":
            print(f"API: Clarification Error: {groq_result.get('message')}")
            return {"error": f"Clarification failed: {groq_result.get('message')}"}
        elif groq_result.get("status") == "CLARIFYING":
            print("API: Clarification Needed (Groq).")
            return {
                "status": "CLARIFICATION_NEEDED",
                "question": groq_result.get("question"),
                "type": groq_result.get("type", "text"),
                "options": groq_result.get("options", [])
            }
        else:
            print("API: Intent Clear (Groq). Proceeding...")
            # If Groq says READY, it *might* return a summary in 'summary' field.
            # We can use that as the context.
            if groq_result.get("summary"):
                request.clarification_context = groq_result.get("summary")
            else:
                # Fallback: Flatten history
                request.clarification_context = "\n".join([f"{h['role']}: {h['content']}" for h in request.clarification_history])

    # 3. Prepare Workflow Input
    # Improve topic with clarification context
    final_topic = request.topic
    if request.clarification_context:
        final_topic += f"\n\n[USER CONTEXT]: {request.clarification_context}"

    workflow_input = {
        "topic": final_topic,
        "template": request.template,
        "platform": request.platform,
        "audience": profile_data.get("audience", ""),
        "tone": profile_data.get("tone", ""),
        "forbidden_words": str(profile_data.get("forbidden_words", [])),
        "voice_analysis": request.style_analysis or "" # NEW: Pass Style Analysis
    }
    
    # [NEW] Generate Correlation ID
    interaction_id = str(uuid.uuid4())
    
    # --- PRO MODE ROUTING ---
    if request.pro_mode:
        # Pro Mode: check pro quota + run full DSPy pipeline
        if not await MetricsTracker.check_pro_limit(current_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Pro limit reached (3/day). Switch to Standard mode for instant results."}
            )
        if not await MetricsTracker.try_increment_pro(current_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Pro limit reached (3/day). Switch to Standard mode for instant results."}
            )
        
        try:
            print("API: [PRO] Running full DSPy pipeline...")
            final_result = await run_in_threadpool(agentic_workflow.run, input=workflow_input)
            
            architect_output = get_step_content(final_result.step_results, "run_architect")
            hook_output = get_step_content(final_result.step_results, "generate_hooks")
            draft_output = get_step_content(final_result.step_results, "draft_thread")
            
            final_hook = draft_output.get("selected_hook")
            if not final_hook or final_hook == "N/A":
                final_hook = hook_output.get("hooks", ["N/A"])[0] if hook_output.get("hooks") else "N/A"
            
            final_tldr = draft_output.get("tldr", "N/A")
            final_cta = draft_output.get("cta", "N/A")
            final_tweets = draft_output.get("tweets", [])

            response_data = {
                "status": "COMPLETED",
                "mode": "pro",
                "template": architect_output.get("template", "N/A"),
                "refined_topic": architect_output.get("refined_topic", ""),
                "hook": final_hook,
                "tldr": final_tldr,
                "cta": final_cta,
                "tweets": final_tweets,
                "interaction_id": interaction_id
            }
            
            print("API: [PRO] Successfully generated thread.")
            log_interaction(
                ip=current_ip, action_type="GENERATE_PRO",
                input_data=request.topic, output_data=response_data,
                meta={"platform": request.platform, "template": request.template,
                       "clarification_history": request.clarification_history},
                interaction_id=interaction_id
            )
            return response_data
            
        except Exception as e:
            print(f"API: [PRO] Error: {e}")
            return {"error": str(e)}
    
    else:
        # Standard Mode: single direct Gemini call (fast)
        if not await MetricsTracker.try_increment_post(current_ip):
            return JSONResponse(
                status_code=429, content={"error": "Daily generation limit reached (5 posts/day). Please upgrade."}
            )
        
        try:
            print("API: [STANDARD] Running direct Gemini call...")
            result = await run_in_threadpool(
                run_gemini_direct,
                topic=final_topic,
                platform=request.platform,
                template=request.template,
                audience=profile_data.get("audience", ""),
                tone=profile_data.get("tone", ""),
                forbidden_words=str(profile_data.get("forbidden_words", [])),
                style_analysis=request.style_analysis or ""
            )
            
            response_data = {
                "status": "COMPLETED",
                "mode": "standard",
                "template": request.template,
                "refined_topic": request.topic,
                "hook": result.get("hook", "N/A"),
                "tldr": result.get("tldr", "N/A"),
                "cta": result.get("cta", "N/A"),
                "tweets": result.get("tweets", []),
                "interaction_id": interaction_id
            }
            
            print("API: [STANDARD] Successfully generated thread.")
            log_interaction(
                ip=current_ip, action_type="GENERATE_STANDARD",
                input_data=request.topic, output_data=response_data,
                meta={"platform": request.platform, "template": request.template},
                interaction_id=interaction_id
            )
            return response_data
            
        except Exception as e:
            print(f"API: [STANDARD] Error: {e}")
            return {"error": str(e)}

# --- GEMINI STANDARD: Direct Single-Call Helper ---
def run_gemini_direct(topic, platform, template, audience, tone, forbidden_words, style_analysis):
    """
    Single Gemini call with a comprehensive system prompt that embeds
    all 5 agent perspectives: Refiner, Researcher, HookSmith, Drafter, Editor.
    Returns dict with {hook, tweets, cta, tldr} matching DSPy output shape.
    """
    style_str = f"\nVoice/Style Analysis: {style_analysis}" if style_analysis else ""

    if platform == "LinkedIn":
        format_block = """Format STRICTLY as:
HOOK: <a single punchy opening line>
BODY:
<paragraph 1>
<paragraph 2>
<paragraph 3>
(IMPORTANT: Use short, crisp paragraphs. Maximum 1-2 sentences per paragraph. Use bullet points or line breaks for readability. Never output giant walls of text.)
CTA: <clear call to action>
TLDR: <one line summary>"""
    else:
        format_block = """Format STRICTLY as:
HOOK: <a single punchy opening tweet>
TWEETS:
1. <tweet 1 — max 280 chars>
2. <tweet 2 — max 280 chars>
3. <tweet 3 — max 280 chars>
4. <tweet 4 — max 280 chars>
5. <tweet 5 — max 280 chars>
CTA: <clear call to action tweet>
TLDR: <one line summary>"""

    system_prompt = f"""You are an expert content creator. Create a {platform} thread/post about the given topic.

Think through these 5 lenses before writing:
1. REFINER: What is the sharpest, most specific angle? What makes this unique?
2. RESEARCHER: What concrete facts, stats, or examples support this angle?
3. HOOK SMITH: What opening line would stop someone from scrolling?
4. DRAFTER: Structure the content for maximum readability on {platform}.
5. EDITOR: Strip all jargon, corporate-speak, and filler. Make it human and direct.

Context:
- Platform: {platform}
- Template style: {template}
- Target audience: {audience or 'General'}
- Tone: {tone or 'Professional and engaging'}
- Forbidden words: {forbidden_words}{style_str}

{format_block}

RULES: No forbidden words. Be specific and concrete. No corporate jargon. Every sentence must add value."""

    try:
        import google.generativeai as genai
        from dspy_module import MultiKeyGemini

        # Use the existing configured DSPy LM (already has keys loaded)
        result = dspy.Predict("topic -> content")(topic=f"{system_prompt}\n\nTopic: {topic}")
        raw_text = result.content

        # Parse structured output
        return _parse_gemini_standard_output(raw_text, platform)

    except Exception as e:
        print(f"Gemini Standard Error: {e}")
        return {"hook": "Generation failed", "tweets": [str(e)], "cta": "Please try again", "tldr": "Error occurred"}


def _parse_gemini_standard_output(raw_text, platform):
    """Parse HOOK/TWEETS|BODY/CTA/TLDR from raw Gemini output."""
    lines = raw_text.split('\n')
    hook = ''
    tweets = []
    cta = ''
    tldr = ''
    section = None
    body_buffer = []

    for line in lines:
        trimmed = line.strip()
        if trimmed.upper().startswith('HOOK:'):
            hook = trimmed[5:].strip()
            section = 'hook'
        elif trimmed.upper() in ('TWEETS:', 'BODY:'):
            section = 'tweets'
        elif trimmed.upper().startswith('CTA:'):
            cta = trimmed[4:].strip()
            section = 'cta'
        elif trimmed.upper().startswith('TLDR:'):
            tldr = trimmed[5:].strip()
            section = 'tldr'
        elif section == 'tweets' and trimmed:
            import re
            cleaned = re.sub(r'^\d+[.)\s]+', '', trimmed).strip()
            if cleaned:
                body_buffer.append(cleaned)
        elif section == 'cta' and trimmed and not cta:
            cta = trimmed
        elif section == 'tldr' and trimmed and not tldr:
            tldr = trimmed

    tweets = body_buffer if body_buffer else tweets

    # Fallback if parsing fails
    if not hook and not tweets:
        parts = raw_text.split('\n\n')
        parts = [p.strip() for p in parts if p.strip()]
        hook = parts[0] if parts else raw_text[:100]
        tweets = parts[1:-1] if len(parts) > 2 else parts[1:] if len(parts) > 1 else []
        cta = parts[-1] if len(parts) > 1 else ''

    return {"hook": hook, "tweets": tweets, "cta": cta, "tldr": tldr}


@app.get("/profiles")
def get_profiles_endpoint():
    return ProfileManager.get_profiles()

@app.post("/profiles")
def save_profile_endpoint(data: dict):
    # data expects { "name": "...", "data": {...} }
    name = data.get("name")
    p_data = data.get("data")
    if name and p_data:
        ProfileManager.save_profile(name, p_data)
        return {"status": "success"}
    return {"error": "Invalid data"}

@app.post("/edit-thread")
async def edit_thread_endpoint(request: EditRequest):
    """
    Takes a list of tweets and an edit prompt, 
    and returns a new, edited list of tweets.
    """
    print(f"API: Received edit request: {request.prompt}")
    
    try:
        # Call the PolisherAgent's .act() method in threadpool
        # Note: act() acts as editor+ancillary, so it returns dict. 
        # But we only need edited tweets here.
        # However, Polisher module signature takes 'draft_content' and edits it.
        # But wait, PolisherSignature output includes tldr/cta.
        # If we re-run polisher, we get new tldr/cta too. That's fine.
        
        def blocking_edit():
             # We assume act() treats the input as 'draft_content'
             # dspy module 'polish' takes 'draft_content'.
             # We might not be able to pass 'edit_prompt' to PolisherSignature easily 
             # unless I update PolisherSignature to take valid prompt?
             # My PolisherSignature definition:
             # class PolisherSignature(dspy.Signature):
             #    draft_content = dspy.InputField()
             #    polished_content = dspy.OutputField()
             # It does NOT take 'edit_prompt'.
             
             # Uh oh. The user might want to edit with specific instruction.
             # The old 'EditThread' signature took 'edit_prompt'.
             # 'PolisherSignature' I defined earlier uses a fixed prompt in docstring: "Review... remove corporate jargon..."
             
             # If I want to support specific edit prompts, I should have included it.
             # Phase-based optimization often sacrifices granular control for speed.
             # I should probably update PolisherSignature to accept an optional 'instruction'.
             
             # Let's see if I can pass it.
             # Architecturally, 'Polisher' is the final step.
             # If I want manual override, I probably need `RefineFullPost` which I kept!
             # Wait, `RefineFullPost` acts on (hook+body+cta).
             # `/edit-thread` acts on just tweets.
             
             # I should use `RefineFullPost` (via `refine_full_post`) even for `/edit-thread`?
             # No, `/edit-thread` is usually for the tweet body.
             
             # Current plan: just use `refine_full_post` logic but with empty hook/cta if needed?
             # Or just instantiate `EditorAgent` logic inside `PolisherAgent`?
             # Actually `RefineFullPost` is:
             # current_hook, current_body, current_cta, instruction -> refined versions.
             
             # If I pass hook="", cta="", and tweets=tweets, prompt=prompt.
             # I get refined_hook, refined_body, refined_cta.
             # I can just return refined_body.
             
             return polisher_agent.refine_full_post(
                 hook="",
                 tweets=request.tweets,
                 cta="",
                 prompt=request.prompt
             )

        result = await run_in_threadpool(blocking_edit)
        
        # result is {"hook":..., "tweets":..., "cta":...}
        response_data = {
            "edited_tweets": result.get("tweets", [])
        }
        
        print("API: Successfully edited thread.")
        
        # [NEW] Log Edit
        log_interaction(
             ip=MetricsTracker.get_context_ip(),
             action_type="EDIT",
             input_data=request.prompt, 
             output_data=response_data,
             meta={"original_tweets_count": len(request.tweets)},
             interaction_id=str(uuid.uuid4()) # Treat edits as new interactions, logic can vary
        )

        return response_data
        
    except Exception as e:
        print(f"API: Error during editing: {e}")
        return {"error": str(e)}

@app.post("/refine-full")
async def refine_full_endpoint(request: FullRefineRequest):
    """
    Refines Hook + Tweets + CTA all together.
    """
    print(f"API: Received FULL refine request: {request.prompt}")
    try:
        def blocking_refine():
            return polisher_agent.refine_full_post(
                hook=request.hook,
                tweets=request.tweets,
                cta=request.cta,
                prompt=request.prompt,
                platform=request.platform,
                topic=request.topic or "",
                audience=request.audience or "",
                tone=request.tone or ""
            )
            
        result = await run_in_threadpool(blocking_refine)
        
        # [NEW] Log Refinement
        log_interaction(
            ip=MetricsTracker.get_context_ip(),
            action_type="REFINE",
            input_data=request.prompt,
            output_data=result,
            meta={
                "original_hook": request.hook,
                "original_tweets": request.tweets,
                "original_cta": request.cta
            },
            interaction_id=str(uuid.uuid4()),
            parent_id=request.parent_id # Link to original
        )
        
        return result
    except RuntimeError as e:
        print(f"API: Refinement failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Refinement failed: {str(e)}"}
        )
    except Exception as e:
        print(f"API: Unexpected error during full refinement: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# --- 7b. Admin Usage Endpoint ---
@app.get("/admin/api-usage")
def get_api_usage_endpoint():
    """Returns the current usage stats."""
    return MetricsTracker.get_report()

# --- 8. Style Analysis Endpoint ---
# --- NIM Request Model ---
class NimThreadRequest(BaseModel):
    topic: str
    model: str
    platform: str = "Twitter"
    audience: str = ""
    tone: str = ""
    template: str = "Educational"
    forbidden_words: Optional[List[str]] = None
    clarification_context: Optional[str] = None
    clarification_history: List[Dict] = []
    profile_name: Optional[str] = None
    style_analysis: Optional[str] = None
    pro_mode: bool = False  # Pro Mode: real 5-step agent chain

class StyleRequest(BaseModel):
    content: str
    current_style: Optional[str] = ""

@app.post("/analyze-style")
async def analyze_style_endpoint(request: StyleRequest):
    """
    Analyzes the text and merges with current style.
    """
    print("API: Analyzing & Refining style...")
    try:
        from dspy_module import RefineStyle
        
        # Strictness?
        analyze_module = dspy.Predict(RefineStyle)
        
        def run_analysis():
            return analyze_module(content=request.content, current_style=request.current_style or "")
            
        result = await run_in_threadpool(run_analysis)
        
        return {"style_summary": result.style_summary}
        
    except Exception as e:
        print(f"API: Error analyzing style: {e}")
        return {"error": str(e)}

        print(f"API: Error analyzing style: {e}")
        return {"error": str(e)}

# --- 9. NVIDIA NIM Streaming Endpoint ---
@app.post("/generate-thread-nim")
async def generate_thread_nim_endpoint(request: NimThreadRequest):
    """
    Runs clarification (same Groq loop as Gemini), then streams generation
    directly from the NVIDIA NIM API back to the frontend as SSE.
    """
    print(f"NIM API: Received request for topic: {request.topic}, model: {request.model}")

    nim_key = os.getenv("NVIDIA_NIM_API_KEY")
    if not nim_key:
        return JSONResponse(status_code=500, content={"error": "NVIDIA_NIM_API_KEY not configured on server."})

    # --- Rate Limit Check (same as Gemini endpoint) ---
    current_ip = MetricsTracker.get_context_ip()
    if not await MetricsTracker.check_limit(current_ip):
        print(f"⛔ NIM: Blocked from {current_ip}: Over Daily Limit")
        return JSONResponse(
            status_code=429,
            content={"error": "Daily generation limit reached (5 posts/day). Please upgrade."}
        )

    # --- Clarification Loop (Groq, same as Gemini) ---
    if not request.clarification_context:
        ctx_str = f"Platform: {request.platform}\nAudience: {request.audience}\nTone: {request.tone}"
        if request.template:
            ctx_str += f"\nTemplate: {request.template}"

        def run_groq_check_nim():
            return GroqClient.get_next_question(request.topic, request.clarification_history, context_str=ctx_str)

        groq_result = await run_in_threadpool(run_groq_check_nim)

        if groq_result.get("status") == "ERROR":
            return {"error": f"Clarification failed: {groq_result.get('message')}"}
        elif groq_result.get("status") == "CLARIFYING":
            return {
                "status": "CLARIFICATION_NEEDED",
                "question": groq_result.get("question"),
                "type": groq_result.get("type", "text"),
                "options": groq_result.get("options", [])
            }
        else:
            if groq_result.get("summary"):
                request.clarification_context = groq_result.get("summary")
            else:
                request.clarification_context = "\n".join(
                    [f"{h['role']}: {h['content']}" for h in request.clarification_history]
                )

    # --- Rate Limit / Pro Quota ---
    if request.pro_mode:
        if not await MetricsTracker.check_pro_limit(current_ip):
            return JSONResponse(status_code=429,
                content={"error": "Pro limit reached (3/day). Switch to Standard mode."})
        if not await MetricsTracker.try_increment_pro(current_ip):
            return JSONResponse(status_code=429,
                content={"error": "Pro limit reached (3/day). Switch to Standard mode."})
    else:
        if not await MetricsTracker.try_increment_post(current_ip):
            return JSONResponse(status_code=429,
                content={"error": "Daily generation limit reached (5 posts/day). Please upgrade."})

    # --- Build common context ---
    forbidden_str = ", ".join(request.forbidden_words) if request.forbidden_words else "none"
    clarification_str = f"\n\n[USER CONTEXT]: {request.clarification_context}" if request.clarification_context else ""
    style_str = f"\nVoice/Style: {request.style_analysis}" if request.style_analysis else ""
    user_message = f"Topic: {request.topic}{clarification_str}"
    interaction_id = str(uuid.uuid4())

    nim_headers = {
        "Authorization": f"Bearer {nim_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    if request.platform == "LinkedIn":
        format_instructions = """Format the output strictly as:
HOOK: <a single punchy opening line>
BODY:
<paragraph 1>
<paragraph 2>
<paragraph 3>
(IMPORTANT: Use short, crisp paragraphs. Maximum 1-2 sentences per paragraph. Use bullet points or line breaks for readability. Never output giant walls of text.)
CTA: <clear call to action>
TLDR: <one line summary>"""
    else:
        format_instructions = """Format the output strictly as:
HOOK: <a single punchy opening tweet>
TWEETS:
1. <tweet 1 — max 280 chars>
2. <tweet 2 — max 280 chars>
3. <tweet 3 — max 280 chars>
4. <tweet 4 — max 280 chars>
5. <tweet 5 — max 280 chars>
CTA: <clear call to action tweet>
TLDR: <one line summary>"""

    from fastapi.responses import StreamingResponse

    if request.pro_mode:
        # ===== PRO MODE: 5-step sequential agent chain =====
        print(f"NIM API: [PRO] Running 5-step agent chain with {request.model}")

        async def nim_agent_chain():
            """5 sequential NIM calls. Steps 1-4 non-streaming, Step 5 streaming."""
            full_text = []

            try:
                # Step 1: REFINER
                refiner_prompt = f"""You are the Refiner Agent. Given a topic, extract the sharpest, most compelling angle.
Output:
ANGLE: <one sentence describing the sharpest angle>
INSIGHT: <the core non-obvious insight>
AUDIENCE_HOOK: <what makes this relevant to the target audience>

Context: Platform={request.platform}, Audience={request.audience or 'General'}, Tone={request.tone or 'Professional'}{style_str}"""

                yield f"data: {json.dumps({'type': 'step', 'step': 1, 'label': 'Refiner Agent: Finding the sharpest angle...'})}\n\n"
                refiner_out = await _nim_call(nim_headers, request.model, refiner_prompt, user_message)

                # Step 2: RESEARCH
                yield f"data: {json.dumps({'type': 'step', 'step': 2, 'label': 'Research Agent: Gathering facts & context...'})}\n\n"
                research_prompt = f"""You are the Research Agent. Given the topic and the Refiner's angle, add concrete supporting evidence.
Include: specific facts, statistics, real-world examples, analogies, and counter-arguments to preempt.

Refiner's Analysis:
{refiner_out}

Output a detailed research brief (3-5 bullet points with sources/examples)."""
                research_out = await _nim_call(nim_headers, request.model, research_prompt, user_message)

                # Step 3: HOOK SMITH
                yield f"data: {json.dumps({'type': 'step', 'step': 3, 'label': 'Hook Smith: Crafting viral opening hooks...'})}\n\n"
                hook_prompt = f"""You are the Hook Smith Agent. Create 3 viral opening hooks for a {request.platform} post.
Each hook should stop scrollers dead. Use the Refiner's angle and Research data.

Refiner's Analysis: {refiner_out}
Research Brief: {research_out}

Output:
HOOK_1: <hook option 1>
HOOK_2: <hook option 2>
HOOK_3: <hook option 3>
BEST: <which hook number is best and why in 1 sentence>"""
                hooks_out = await _nim_call(nim_headers, request.model, hook_prompt, user_message)

                # Step 4: DRAFTER
                yield f"data: {json.dumps({'type': 'step', 'step': 4, 'label': 'Drafter Agent: Building the full thread...'})}\n\n"
                drafter_prompt = f"""You are the Drafter Agent. Build the complete {request.platform} thread.
Use the best hook from the Hook Smith, incorporate the Research data, follow the Refiner's angle.

Refiner's Analysis: {refiner_out}
Research Brief: {research_out}
Hook Options: {hooks_out}
Forbidden words: {forbidden_str}

{format_instructions}"""
                draft_out = await _nim_call(nim_headers, request.model, drafter_prompt, user_message)

                # Step 5: EDITOR (streaming)
                yield f"data: {json.dumps({'type': 'step', 'step': 5, 'label': 'Editor Agent: Polishing & humanizing...'})}\n\n"
                editor_prompt = f"""You are the Editor Agent. Polish this draft for maximum impact.
- Strip all jargon, corporate-speak, filler words
- Ensure forbidden words ({forbidden_str}) are NOT used
- Humanize the tone: {request.tone or 'Professional and engaging'}
- Ensure each tweet/paragraph is concise and punchy
- Output the FINAL polished version{style_str}

{format_instructions}

Draft to polish:
{draft_out}"""

                # Stream the editor's output
                editor_payload = {
                    "model": request.model,
                    "messages": [
                        {"role": "system", "content": editor_prompt},
                        {"role": "user", "content": "Polish and finalize this thread. Output ONLY the final formatted version."}
                    ],
                    "max_tokens": 16384,
                    "temperature": 0.50,
                    "top_p": 0.95,
                    "stream": True
                }
                if "qwen3" in request.model.lower():
                    editor_payload["chat_template_kwargs"] = {"enable_thinking": True}

                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST",
                        "https://integrate.api.nvidia.com/v1/chat/completions",
                        headers=nim_headers, json=editor_payload
                    ) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            yield f"data: {json.dumps({'error': f'NIM Editor error {response.status_code}: {error_body.decode()[:200]}'})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                raw = line[6:]
                                if raw == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(raw)
                                    choices = chunk.get("choices", [])
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "") or ""
                                    if content:
                                        full_text.append(content)
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                                except Exception:
                                    pass

                complete_text = "".join(full_text)
                yield f"data: {json.dumps({'type': 'done', 'full_text': complete_text, 'interaction_id': interaction_id})}\n\n"

                log_interaction(
                    ip=current_ip, action_type="GENERATE_NIM_PRO",
                    input_data=request.topic,
                    output_data={"model": request.model, "raw_output": complete_text[:500]},
                    meta={"platform": request.platform, "model": request.model, "mode": "pro"},
                    interaction_id=interaction_id
                )

            except Exception as e:
                print(f"NIM PRO chain error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            nim_agent_chain(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    else:
        # ===== STANDARD MODE: existing single-call stream =====
        system_prompt = f"""You are an expert content creator specializing in {request.platform} content.
Your task: Create a high-quality {request.platform} thread/post on the given topic.

Think through these 5 lenses before writing:
1. REFINER: What is the sharpest angle?
2. RESEARCHER: What facts/stats support it?
3. HOOK SMITH: What opening stops scrollers?
4. DRAFTER: Structure for {request.platform} readability.
5. EDITOR: Strip jargon, be human.

Context:
- Platform: {request.platform}
- Template style: {request.template}
- Target audience: {request.audience or 'General'}
- Tone: {request.tone or 'Professional and engaging'}
- Forbidden words/phrases: {forbidden_str}{style_str}

{format_instructions}

IMPORTANT: Do NOT use ANY of the forbidden words. Be specific, concrete, and valuable. No corporate jargon."""

        nim_payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 16384,
            "temperature": 0.60,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "stream": True
        }

        if "qwen3" in request.model.lower():
            nim_payload["chat_template_kwargs"] = {"enable_thinking": True}

        async def nim_stream_generator():
            """Proxy the NIM SSE stream to the frontend."""
            full_text = []
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST", "https://integrate.api.nvidia.com/v1/chat/completions",
                        headers=nim_headers, json=nim_payload
                    ) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            yield f"data: {json.dumps({'error': f'NIM API error {response.status_code}: {error_body.decode()}'})}"
                            return

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                raw = line[6:]
                                if raw == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(raw)
                                    choices = chunk.get("choices", [])
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "") or ""
                                    if content:
                                        full_text.append(content)
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                                except Exception:
                                    pass

                complete_text = "".join(full_text)
                yield f"data: {json.dumps({'type': 'done', 'full_text': complete_text, 'interaction_id': interaction_id})}\n\n"

                log_interaction(
                    ip=current_ip, action_type="GENERATE_NIM",
                    input_data=request.topic,
                    output_data={"model": request.model, "raw_output": complete_text[:500]},
                    meta={"platform": request.platform, "model": request.model},
                    interaction_id=interaction_id
                )

            except httpx.TimeoutException:
                yield f"data: {json.dumps({'error': 'NIM API timed out. Please try again.'})}\n\n"
            except Exception as e:
                print(f"NIM API stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            nim_stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )


# --- NIM Agent Helper: non-streaming call ---
async def _nim_call(headers, model, system_prompt, user_msg):
    """Make a single non-streaming NIM API call and return the text."""
    import re
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 4096,
        "temperature": 0.60,
        "top_p": 0.95,
        "stream": False
    }
    if "qwen3" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers, json=payload
        )
        if resp.status_code != 200:
            raise Exception(f"NIM API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise Exception("NIM returned empty choices")
        text = choices[0].get("message", {}).get("content", "")
        text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        return text


# --- 9b. Pro Quota Endpoint ---
@app.get("/pro-quota")
async def get_pro_quota():
    """Returns remaining Pro generations for the current IP today."""
    current_ip = MetricsTracker.get_context_ip()
    remaining = await MetricsTracker.get_pro_remaining(current_ip)
    from config import settings
    return {"remaining": remaining, "limit": settings.pro_daily_limit}



# --- 10. Manual Log Endpoint ---
@app.post("/log-interaction")
async def log_manual_interaction(request: LogInteractionRequest):
    """
    Allows frontend to report 'MANUAL_SAVE' events.
    Non-blocking.
    """
    try:
        log_interaction(
            ip=MetricsTracker.get_context_ip(),
            action_type=request.action_type,
            input_data="MANUAL_SAVE",
            output_data=request.content,
            meta=request.meta,
            interaction_id=request.interaction_id or str(uuid.uuid4()),
            parent_id=request.parent_id
        )
        return {"status": "queued"}
    except Exception as e:
        print(f"API: Manual Log Error: {e}")
        return {"status": "error"}
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting API server on port {port}")
    # Use import string "main:app" to enable reload
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)