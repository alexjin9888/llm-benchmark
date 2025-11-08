from pathlib import Path
import json, time, uuid, os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.providers.xai_adapter import send_chat
from src.utils.cache import cache_key, get_from_cache, save_to_cache

load_dotenv()

class GrokClient:
    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.0, seed: Optional[int] = None,
             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not (extra and extra.get("no_cache")):
            payload_for_cache = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
            }
            key = cache_key(payload_for_cache)
            cached = get_from_cache(key)
            if cached:
                return cached

        out = send_chat(model=model, messages=messages, temperature=temperature)
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "messages": messages,
            "extra": extra or {},
            "content": out["content"],
            "latency_ms": out["latency_ms"],
            "usage": out.get("usage"),
            "raw": out["raw"],
        }
        save_to_cache(key, record)
        with open(self.log_dir / "grok_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record
