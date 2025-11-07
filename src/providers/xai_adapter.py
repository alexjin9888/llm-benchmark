import os, json, time
from typing import List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv, find_dotenv

# Ensure .env is loaded inside this module as well
load_dotenv(find_dotenv())

XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai")

class XAIError(Exception):
    pass

def send_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout_s: int = 180,
) -> Dict[str, Any]:
    # Get API key at call time, not import time
    api_key = os.getenv("XAI_API_KEY")
    assert api_key, "Set XAI_API_KEY in .env"

    url = f"{XAI_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    started = time.time()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, headers=headers, json=payload)
    latency_ms = int((time.time() - started) * 1000)

    if resp.status_code >= 400:
        raise XAIError(f"xAI HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"content": content, "latency_ms": latency_ms, "usage": data.get("usage"), "raw": data}
