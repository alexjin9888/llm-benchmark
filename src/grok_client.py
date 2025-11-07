import os, time, json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uuid

load_dotenv()

class GrokClient:
    def __init__(self, api_key: Optional[str] = None, log_dir: str = "results/logs"):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        assert self.api_key, "Set XAI_API_KEY in .env"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Replace the body of _call_api with a real x.ai HTTP request per docs.x.ai.
        The return dict should include:
          content, raw, latency_ms, prompt_tokens, completion_tokens
        """
        started = time.time()
        # --- begin placeholder you will replace with real HTTP call ---
        content = "placeholder response"
        raw = {"mock": True}
        prompt_tokens = 0
        completion_tokens = 0
        # --- end placeholder ---
        latency_ms = int((time.time() - started) * 1000)

        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "messages": messages,
            "extra": extra or {},
            "content": content,
            "raw": raw,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        # Append to JSONL for easy later analysis
        with open(self.log_dir / "grok_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        return record
