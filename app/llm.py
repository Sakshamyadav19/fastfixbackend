# backend/app/llm.py
import asyncio
from typing import Optional
import google.generativeai as genai
from .config import GEMINI_API_KEY, LLM_TIMEOUT_SECS, LLM_MODEL  # <- text model

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

def _sync_generate(system_prompt: str, user_message: str, model_name: Optional[str]) -> str:
    mname = model_name or LLM_MODEL
    # Guard: catch accidental use of an embeddings model
    if "embedding" in mname:
        raise ValueError(
            f"LLM model '{mname}' looks like an embeddings model. "
            "Set LLM_MODEL to a text model (e.g., gemini-1.5-flash or gemini-1.5-pro)."
        )
    model = genai.GenerativeModel(
        mname,
        system_instruction=system_prompt,
    )
    resp = model.generate_content(
        user_message,
        generation_config={"max_output_tokens": 512},
        safety_settings=None,
    )
    return (getattr(resp, "text", "") or "").strip()

async def generate_hints_text(system_prompt: str, user_message: str, model_name: Optional[str] = None) -> str:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _sync_generate, system_prompt, user_message, model_name),
        timeout=LLM_TIMEOUT_SECS,
    )
