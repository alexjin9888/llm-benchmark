from src.grok_client import GrokClient
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="In one sentence, what does tau2-bench evaluate?")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    msgs=[{"role":"system","content":"You are a careful assistant."},
          {"role":"user","content":args.prompt}]
    client = GrokClient()
    for model in ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning"]:
        out = client.chat(model=model, messages=msgs, temperature=0.0, seed=42,
                          extra={"no_cache": args.no_cache})
        print(model, "latency_ms:", out["latency_ms"], "chars:", len(out["content"]))

if __name__ == "__main__":
    main()
