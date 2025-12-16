# main.py
import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import ast

# Import your workflow
from run_workflow import agentic_workflow, get_step_content

# --- 1. IMPORT PolisherAgent HERE ---
from agents.polisher import PolisherAgent

# --- 2. Create the FastAPI app ---
app = FastAPI(
    title="Agentic Twitter Thread Creator API",
    description="An API to run the Agno agent workflow."
)

# --- 3. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. CREATE AGENT INSTANCE HERE ---
# This creates a single instance of the agent when the server starts
polisher_agent = PolisherAgent()

# --- 5. Define Request Bodies ---
class ThreadRequest(BaseModel):
    topic: str
    template: str = "Choose a template suitable for this post"
    platform: str = "Twitter" # Default to Twitter

# --- 6. DEFINE EditRequest HERE ---
class EditRequest(BaseModel):
    tweets: list[str]
    prompt: str

class FullRefineRequest(BaseModel):
    hook: str
    tweets: list[str]
    cta: str
    prompt: str

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
    
    workflow_input = {
        "topic": request.topic,
        "template": request.template,
        "platform": request.platform
    }
    
    try:
        # Run Your Workflow ASYNC
        print("API: Offloading workflow to threadpool...")
        final_result = await run_in_threadpool(agentic_workflow.run, input=workflow_input)
        
        # Parse the results using the new step names
        architect_output = get_step_content(final_result.step_results, "run_architect")
        hook_output = get_step_content(final_result.step_results, "generate_hooks")
        polisher_output = get_step_content(final_result.step_results, "run_polisher")
        
        # Format the JSON Response
        # Priority for Hook: Drafter's selection > HookSmith's first option
        draft_output = get_step_content(final_result.step_results, "draft_thread")
        
        final_hook = draft_output.get("selected_hook")
        if not final_hook or final_hook == "N/A":
             final_hook = hook_output.get("hooks", ["N/A"])[0] if hook_output.get("hooks") else "N/A"
             
        response_data = {
            "template": architect_output.get("template", "N/A"),
            "refined_topic": architect_output.get("refined_topic", ""),
            "hook": final_hook,
            "tldr": polisher_output.get("tldr", "N/A"),
            "cta": polisher_output.get("cta", "N/A"),
            "tweets": polisher_output.get("edited_tweets", [])
        }
        
        print("API: Successfully generated thread. Sending to UI.")
        return response_data
        
    except Exception as e:
        print(f"API: Error during workflow: {e}")
        return {"error": str(e)}

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
             # But /edit-thread endpoint implies custom instruction.
             # I should probably update PolisherSignature to accept an optional 'instruction'.
             
             # Let's see if I can pass it.
             # Architecturally, 'Polisher' is the final step.
             # If I want manual override, I probably need `RefineFullPost` which I kept!
             # Wait, `RefineFullPost` acts on (hook+body+cta).
             # `/edit-thread` acts on just tweets.
             
             # I should use `RefineFullPost` (via `refine_full_post`) even for `/edit-thread`?
             # No, `/edit-thread` is usually for the tweet body.
             
             # Current plan: just use `refine_full_post` logic but with empty hook/cta if needed?
             # Or just stick to the plan: "PolisherAgent replaces EditorAgent".
             # If PolisherAgent.act doesn't take prompt, then `/edit-thread` loses functionality?
             # Check `PolisherSignature` again in my `replace_file_content` for `dspy_module.py`:
             # class PolisherSignature(dspy.Signature): ... draft_content ... polished_content ...
             
             # Yes, I missed `edit_prompt` in `PolisherSignature`.
             # But I DID keep `RefineFullPost` signature and module.
             # And `PolisherAgent` has `refine_full_post`.
             
             # So for `/edit-thread`, I can use `refine_full_post` but just pass the tweets as body, and ignore hook/cta?
             # Or I can just instantiate `EditorAgent` logic inside `PolisherAgent`?
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
                prompt=request.prompt
            )
            
        result = await run_in_threadpool(blocking_refine)
        return result
    except Exception as e:
        print(f"API: Error during full refinement: {e}")
        return {"error": str(e)}

# --- 8. Add a command to run the server ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting API server on port {port}")
    # Use import string "main:app" to enable reload
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)