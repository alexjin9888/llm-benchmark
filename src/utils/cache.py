import hashlib, json
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_DIR = Path("results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

def cache_key(payload: Dict[str, Any]) -> str:
    h = hashlib.sha256(_stable_dumps(payload).encode("utf-8")).hexdigest()
    return h

def get_from_cache(key: str) -> Optional[Dict[str, Any]]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def save_to_cache(key: str, value: Dict[str, Any]) -> None:
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2))

