# Agentic Thread Creator - Project Context

## Overview
The **Agentic Thread Creator** is an AI-powered system designed to generate, refine, and edit high-quality Twitter and LinkedIn threads/posts. It leverages a multi-model architecture using the DSPy framework to orchestrate content creation, style alignment, and refinement.

## Architecture & Tech Stack
* **Backend:** FastAPI (`main.py`) providing endpoints for thread generation, editing, style analysis, and logging.
* **AI Orchestration:** DSPy (`dspy_module.py`) with specialized signatures for brainstorming, hook generation, and thread formatting.
* **LLM Providers:** 
  * Google Gemini (via `MultiKeyGemini` wrapper) for primary generation.
  * Groq / Kimi (via `MultiKeyGroq` wrapper) for fast, robust refinements and **clarification questioning**.
  * **NVIDIA NIM** (`/generate-thread-nim` endpoint) — user-selectable via frontend dropdown; supports Qwen3 235B/397B, Llama 3.3 70B, Mistral Large 2, DeepSeek R1. Streams SSE directly to the browser via `httpx`. Goes through the same Groq clarification loop before generation.
  * **Pro Mode** (toggle in UI): Both Gemini and NIM support Standard (fast, single call) and Pro (full 5-agent pipeline) modes:
    * **Gemini Standard:** Single direct Gemini LLM call (bypasses DSPy). **Gemini Pro:** Full DSPy 5-step workflow (unchanged).
    * **NIM Standard:** Single NIM call with mega-prompt. **NIM Pro:** 5 sequential NIM API calls (Refiner → Research → HookSmith → Drafter → Editor) with real-time brain step animation.
    * Pro quota: 3 generations/IP/day (`pro_daily_limit` in `config.py`), tracked via `pro_count` column in `ip_rate_limits` Supabase table. `/pro-quota` endpoint returns remaining quota.
    * Loss-aversion framing, scarcity badge, and "✨ RECOMMENDED" ribbon for future monetization psychology.
  * Both Gemini wrappers feature Round-Robin key rotation to handle rate limits (HTTP 429).
* **Frontend:** Included in the `Frontend/` directory (UI for inputs, generation progress, and interactive refinement).
* **Workflows & Agents (BMAD):** The project includes a comprehensive agent persona system under the `_bmad/` and `.agent/` directories.
  * Notably, `bmad-master` (`.agent/workflows/bmad-agent-bmad-master.md`) acts as the Master Executor and Workflow Orchestrator.
  * Other agent profiles include PM, Dev, Architect, QA, etc., enabling a structured, AI-driven development lifecycle.

## Core Features
1. **Thread Generation (`/api/generate_thread`):** Takes a raw topic, analyzes intent, refines the style, and generates a structured thread consisting of hooks, body tweets, and a CTA.
2. **Partial Editing (`/api/edit`):** Allows users to selectively edit parts of a generated thread with a custom prompt.
3. **Full Polish & Refinement (`/api/refine_full`):** Performs a comprehensive refinement of the entire post (Hook + Body + CTA) while preserving length and valid Markdown (crucial for LinkedIn).
4. **Style Analysis (`/api/analyze_style`):** Analyzes the user's provided content to build a persistent style profile.
5. **Rate Limiting & Rotation:** Robust handling of API limits across multiple keys for both Gemini and Groq models.

## Recent Fixes & State
* **Zero-Shrinkage Refinement:** Upgraded the `RefineFullPost` pipeline to use `dspy.ChainOfThought` rather than `Predict`, enforcing reasoning steps that completely solved the issue where the AI was summarizing or shortening the user's content during refinement passes. 
* **Undo/Redo & Formatting Engine:** Completely overhauled the `Frontend/index.html` state tracking. Implemented an array-based history stack supporting complete Undo and Redo traversal. CSS isolation of `white-space: pre-wrap;` resolved an issue where manual newlines were being stripped or causing UI spacing bugs in the TLDR/CTA sections. 
* **Database Optimization (Debouncing):** Introduced an idempotent, client-side content hashing mechanism (`lastLoggedContentHash`) for the "Copy Full Thread" button. This prevents rapid identical click spams from bloating the Supabase database.
* **Kimi Output & Formatting:** Integrated Groq (Kimi) for specific refinement tasks (`PolisherAgent`). Verified that the LinkedIn clarification loop correctly intakes platform context variables to ask smart, format-aware questions.
* **UI/UX Polish:** Added a "How to Use" tutorial dropdown, improved progress bar animations, updated mobile responsiveness, and finalized Pytest mock configurations for rate limiters.

## How to use this document
When starting a **new chat**, provide this `project-context.md` file to the AI so it immediately understands the tech stack, the goal of the product, and the recent fixes that have been applied. This drastically reduces hallucination and redundant exploration.
