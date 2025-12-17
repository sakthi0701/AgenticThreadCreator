from agno.workflow import Workflow
from agno.workflow.steps import Step
from agno.workflow.types import StepInput
import ast
import os 
from serpapi import GoogleSearch 
import config

# Import our agents
from agents.researcher import ResearchAgent
from agents.architect import ArchitectAgent
from agents.hook_smith import HookSmithAgent
from agents.drafter import DrafterAgent
from agents.polisher import PolisherAgent

# 1. Instantiate all agents
try:
    print("🤖 Instantiating agents...")
    researcher = ResearchAgent()
    print("✅ ResearchAgent instantiated")
    architect = ArchitectAgent()
    print("✅ ArchitectAgent instantiated")
    hook_smith = HookSmithAgent()
    print("✅ HookSmithAgent instantiated")
    drafter = DrafterAgent()
    print("✅ DrafterAgent instantiated")
    polisher = PolisherAgent()
    print("✅ PolisherAgent instantiated")

except Exception as e:
    print(f"❌ Error instantiating agents: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. Define executor functions for each step

def research_executor(step_input: StepInput):
    """
    Step 1: Research the topic to get context.
    """
    topic = step_input.input.get("topic")
    # We pass "generic" as template just to satisfy the old signature if needed, 
    # but the new researcher logic (if updated) or even the old one works fine.
    # The old researcher.act took (topic, template).
    # We can pass "Research Mode" as template.
    
    agent_output = researcher.act(topic=topic, template="Research Mode") 
    query = agent_output.get("query") 
    api_key = agent_output.get("api_key")
    
    print(f"🔎 Calling SerpApi Google (AI Mode) with query: '{query}'")
    
    try:
        if not api_key:
            api_key = config.settings.serpapi_api_key
            
        if not api_key:
            # Mock if no key - to prevent crash
            print("⚠️ No SerpApi Key. Returning mock context.")
            return {"research_context": "No research performed (Missing Key)."}

        params = {
            "engine": "google_ai_mode", 
            "q": query,
            "api_key": api_key,
            "hl": "en"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Priority: Answer Box -> AI Overview -> Text Blocks
        answer_box = results.get("answer_box", {}).get("answer")
        if answer_box:
            return {"research_context": answer_box}

        ai_overview = results.get("ai_overview", {}).get("answer")
        if ai_overview:
             return {"research_context": ai_overview}
        
        text_blocks = results.get("text_blocks")
        if text_blocks:
            research_context = ""
            for block in text_blocks:
                snippet = block.get("snippet", "")
                if snippet: research_context += f"- {snippet}\n"
            return {"research_context": research_context[:2000]} # Limit size

        return {"research_context": "No rich research context found."}

    except Exception as e:
        print(f"❌ SerpApi Search failed: {e}")
        return {"research_context": "Error during research."}

def architect_executor(step_input: StepInput):
    """
    Step 2: Architect creates the Blueprint (Refined Topic + Template + Outline).
    """
    raw_topic = step_input.input.get("topic")
    
    # Get research context
    try:
        research_content = step_input.get_step_output("run_research").content
        if isinstance(research_content, str):
            research_content = ast.literal_eval(research_content)
        research_context = research_content.get("research_context", "")
    except:
        research_context = ""
        
    try:
        result = architect.act(raw_topic=raw_topic, research_context=research_context)
        # Ensure we return the psychological_trigger
        # The agent returns a dict, likely { "refined_topic": ..., "template": ..., "outline": ... }
        # We need to make sure ArchitectAgent.act returns this new field too.
        # Let's check agents/architect.py next.
        # For now, we assume act returns it.
        return result
    except Exception as e:
        print(f"❌ Architect Step Failed: {e}")
        return {"refined_topic": raw_topic, "outline": [], "psychological_trigger": "Social Proof"}

def hook_executor(step_input: StepInput):
    """
    Step 3: Generate 5 Hooks in a single batch.
    """
    # Get Refined Topic from Architect
    try:
        architect_content = step_input.get_step_output("run_architect").content
        if isinstance(architect_content, str):
            architect_content = ast.literal_eval(architect_content)
        topic = architect_content.get("refined_topic")
        # Extract the psychological trigger
        psych_trigger = architect_content.get("psychological_trigger", "Social Proof") # Default if missing
    except:
        topic = step_input.input.get("topic") # Fallback
        psych_trigger = "Social Proof"
        
    # Pass trigger as 'angle' (which HookSmithAgent maps to primary_trigger)
    return hook_smith.act(topic=topic, angle=psych_trigger)

def draft_executor(step_input: StepInput):
    """
    Step 4: Draft the content.
    """
    platform = step_input.input.get("platform", "Twitter")
    
    # Get Outline & Topic from Architect
    try:
        architect_content = step_input.get_step_output("run_architect").content
        if isinstance(architect_content, str):
            architect_content = ast.literal_eval(architect_content)
        
        outline = architect_content.get("outline")
        topic = architect_content.get("refined_topic")
        psych_trigger = architect_content.get("psychological_trigger", "Social Proof")
    except:
        outline = ["Error parsing outline"]
        topic = step_input.input.get("topic")
        psych_trigger = "Social Proof"

    # Get Hooks from HookSmith
    try:
        hook_content = step_input.get_step_output("generate_hooks").content
        if isinstance(hook_content, str):
             hook_content = ast.literal_eval(hook_content)
        hooks_list = hook_content.get("hooks", [])
    except:
        hooks_list = []
        
    return drafter.act(
        topic=topic, 
        outline=outline, 
        hooks_list=hooks_list, 
        psychological_trigger=psych_trigger, 
        platform=platform
    )

def polisher_executor(step_input: StepInput):
    """
    Step 5: Polish, Edit, and Generate Ancillaries.
    """
    # Get Platform
    platform = step_input.input.get("platform", "Twitter")

    # Get Draft from Drafter
    try:
        draft_content = step_input.get_step_output("draft_thread").content
        if isinstance(draft_content, str):
            draft_content = ast.literal_eval(draft_content)
        # Get Trigger from Architect
        architect_content = step_input.get_step_output("run_architect").content
        if isinstance(architect_content, str):
             architect_content = ast.literal_eval(architect_content)
        psych_trigger = architect_content.get("psychological_trigger", "Social Proof")

        tweets = draft_content.get("tweets")
    except:
        tweets = ["Error parsing draft"]
        psych_trigger = "Social Proof"
        
    # SKIP Polishing for LinkedIn (Zero-Shot is already final)
    if platform == "LinkedIn":
        print("⏩ LinkedIn detected. Skipping Polisher step.")
        return {
            "edited_tweets": tweets,
            "tldr": "N/A",
            "cta": "N/A" # Or generate one if needed, but keeping it simple.
        }
        
    return polisher.act(draft_content=tweets, psychological_trigger=psych_trigger)

# 3. Define the Optimized Workflow
agentic_workflow = Workflow(
    name="Agentic 5-Step Optimization",
    description="Research -> Architect -> Hooks -> Draft -> Polisher",
    steps=[
        Step(
            name="run_research",
            description="Gather context from Google.",
            executor=research_executor
        ),
        Step(
            name="run_architect",
            description="Refine topic, select template, and create outline.",
            executor=architect_executor
        ),
        Step(
            name="generate_hooks",
            description="Generate 5 distinct hooks.",
            executor=hook_executor
        ),
        Step(
            name="draft_thread",
            description="Draft the thread content.",
            executor=draft_executor
        ),
        Step(
            name="run_polisher",
            description="Edit thread and generate TLDR/CTA.",
            executor=polisher_executor
        )
    ]
)

# 4. Helper to safely parse step results
def get_step_content(step_results, step_name):
    """Finds a step by name and returns its parsed content."""
    for step in step_results:
        if step.step_name == step_name:
            content = step.content
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                try:
                    return ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    print(f"⚠️ Warning: Could not parse final content for step {step_name}")
                    return {"error": "Could not parse content", "raw": content}
            return content
    return {}

# 5. Run the workflow (Test block)
if __name__ == "__main__":
    print("🚀 Starting Optimized 5-Call Workflow...\n")
    
    workflow_input = {
        "topic": "The future of AI agents in software engineering",
        "platform": "Twitter"
    }
    
    try:
        final_result = agentic_workflow.run(input=workflow_input)

        print("\n" + "="*80)
        print("🎉 WORKFLOW COMPLETE!")
        print("="*80 + "\n")

        # Get results
        architect_out = get_step_content(final_result.step_results, "run_architect")
        hook_out = get_step_content(final_result.step_results, "generate_hooks")
        polisher_out = get_step_content(final_result.step_results, "run_polisher")
        
        print(f"🤖 TOPIC: {architect_out.get('refined_topic')}")
        print(f"🎁 TEMPLATE: {architect_out.get('template')}")
        
        hooks = hook_out.get("hooks", [])
        print(f"🎣 BEST HOOK: {hooks[0] if hooks else 'N/A'}")
        
        print("-" * 80)
        print("📱 FINAL THREAD:")
        tweets = polisher_out.get("edited_tweets", [])
        for t in tweets:
            print(f"- {t}\n")
            
        print("-" * 80)
        print(f"📌 TL;DR: {polisher_out.get('tldr')}")
        print(f"🎯 CTA: {polisher_out.get('cta')}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()