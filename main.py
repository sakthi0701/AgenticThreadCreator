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
    
    # [FIXED] 2. Increment Limit (Only now that we are actually generating)
    if not await MetricsTracker.try_increment_post(current_ip):
         # This handles the race condition where they hit limit BETWEEN the check and now
         print(f"⛔ Blocked generation from {current_ip}: Over Daily Limit")
         return JSONResponse(
             status_code=429, 
             content={"error": "Daily generation limit reached (5 posts/day). Please upgrade."}
         )

    try:
        # Run Your Workflow ASYNC
        print("API: Offloading workflow to threadpool...")
        final_result = await run_in_threadpool(agentic_workflow.run, input=workflow_input)
        
        # Parse the results using the new step names
        architect_output = get_step_content(final_result.step_results, "run_architect")
        hook_output = get_step_content(final_result.step_results, "generate_hooks")
        
        # Format the JSON Response
        # Priority for Hook: Drafter's selection > HookSmith's first option
        draft_output = get_step_content(final_result.step_results, "draft_thread")
        
        final_hook = draft_output.get("selected_hook")
        if not final_hook or final_hook == "N/A":
             final_hook = hook_output.get("hooks", ["N/A"])[0] if hook_output.get("hooks") else "N/A"
             
        # Extract TLDR/CTA for LinkedIn if Polisher skipped or returned generic
        # (Since LinkedIn polisher might be different or skipped in workflow?)
        # Actually workflow has 'run_polisher'.
        
        # Extract Metadata directly from Draft (One-Shot)
        final_tldr = draft_output.get("tldr", "N/A")
        final_cta = draft_output.get("cta", "N/A")
        final_tweets = draft_output.get("tweets", [])

        response_data = {
            "status": "COMPLETED", # Explicit status
            "template": architect_output.get("template", "N/A"),
            "refined_topic": architect_output.get("refined_topic", ""),
            "hook": final_hook,
            "tldr": final_tldr,
            "cta": final_cta,
            "tweets": final_tweets,
            "interaction_id": interaction_id # [NEW] Return ID to frontend
        }
        
        print("API: Successfully generated thread. Sending to UI.")
        
        # [NEW] Log to Supabase with richer context
        meta_data = {
            "platform": request.platform, 
            "template": request.template,
            "clarification_history": request.clarification_history # [NEW] Log History
        }
        
        log_interaction(
            ip=MetricsTracker.get_context_ip(), 
            action_type="GENERATE",
            input_data=request.topic, 
            output_data=response_data,
            meta=meta_data,
            interaction_id=interaction_id
        )

        return response_data
        
    except Exception as e:
        print(f"API: Error during workflow: {e}")
        return {"error": str(e)}

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