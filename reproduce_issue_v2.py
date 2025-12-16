import dspy
from dspy_module import AgenticThreadCreator, configure_llm
import os

# Configure LLM
configure_llm()

# Input that caused the issue
raw_topic = "let's talk about what went wrong with indigo and what can businesses learn from it"

# Initialize the module
agent = AgenticThreadCreator()

# Redirect output to file
with open("reproduction_log.txt", "w", encoding="utf-8") as f:
    def log(msg):
        print(msg)
        f.write(str(msg) + "\n")

    # Run the pipeline step by step to debug
    log(f"🔬 Analyzing topic: {raw_topic}")

    # 1. Architect Step
    log("\n--- 1. Architect ---")
    architect_out = agent.architect_module(raw_topic=raw_topic, research_context="IndiGo airlines operational crisis, reliance on growth over operations, pilot fatigue issues, technical glitches.")
    log(f"Refined Topic: {architect_out.refined_topic}")
    log(f"Trigger: {architect_out.psychological_trigger}")
    log(f"Outline: {architect_out.thread_outline}")

    # 2. Hook Step
    log("\n--- 2. Hook ---")
    hook_out = agent.hook_module(
        topic=architect_out.refined_topic, 
        primary_trigger=architect_out.psychological_trigger
    )
    log(f"Hooks: {hook_out.hooks_list}")

    # 3. Draft Step
    log("\n--- 3. Draft ---")
    draft_out = agent.drafting_module(
        topic=architect_out.refined_topic, 
        outline=architect_out.thread_outline, 
        hooks_list=hook_out.hooks_list,
        psychological_trigger=architect_out.psychological_trigger,
        platform="Twitter"
    )
    log(f"Selected Hook: {draft_out.selected_hook}")
    log(f"Raw Draft Output (tweet_thread): {draft_out.tweet_thread}")

    # 4. Polish Step
    log("\n--- 4. Polish ---")
    polisher_out = agent.polisher_module(
        draft_content=draft_out.tweet_thread,
        psychological_trigger=architect_out.psychological_trigger
    )
    log(f"Polished Content: {polisher_out.polished_content}")
    log(f"CTA: {polisher_out.cta}")
