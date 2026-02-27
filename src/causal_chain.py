"""
Causal Chain Extraction module.

Extracts cause-and-effect relationships from text using LLM,
organizes them into causal chains, and formats the output for display.
"""
import json
import requests

import config
from src.prompt_loader import get as get_prompt


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini API for causal chain extraction."""
    from src.embedding import get_client
    client = get_client()

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[user_prompt],
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
            "max_output_tokens": 4096,
        },
    )
    return response.text


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call Ollama API for causal chain extraction."""
    body = {
        "model": config.OLLAMA_MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call LLM with fallback chain (same pattern as generator.py)."""
    backend = config.GENERATION_BACKEND
    try:
        if backend == "ollama":
            return _call_ollama(system_prompt, user_prompt)
        else:
            return _call_gemini(system_prompt, user_prompt)
    except Exception as e:
        print(f"{backend.title()} unavailable ({e}), trying fallback...")
        try:
            if backend == "ollama":
                return _call_gemini(system_prompt, user_prompt)
            else:
                return _call_ollama(system_prompt, user_prompt)
        except Exception as e2:
            raise RuntimeError(f"All LLM backends failed: {e2}") from e2


def get_default_system_prompt() -> str:
    """Return the default system prompt from YAML."""
    return get_prompt("causal_chain_system_prompt")


def get_default_user_prompt() -> str:
    """Return the default user prompt template from YAML."""
    return get_prompt("causal_chain_extract_prompt")


def extract_causal_chains(
    article: str,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> tuple[dict, str]:
    """
    Extract causal chains from an article.

    Args:
        article: The input article text.
        system_prompt: Custom system prompt. Uses default if None.
        user_prompt_template: Custom user prompt template with {article} placeholder.
                              Uses default if None.

    Returns:
        Tuple of (parsed result dict, raw LLM response text)
    """
    sys_prompt = system_prompt or get_default_system_prompt()
    usr_template = user_prompt_template or get_default_user_prompt()

    # Build user prompt: substitute {article} if placeholder exists,
    # otherwise append the article at the end.
    if "{article}" in usr_template:
        user_prompt = usr_template.format(article=article)
    else:
        user_prompt = usr_template + "\n\n" + article

    raw_response = _call_llm(sys_prompt, user_prompt)

    # Try to parse as JSON, but don't force it — the user's prompt
    # might not ask for JSON output at all.
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        result = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        result = None

    return result, raw_response
