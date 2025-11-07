from src.grok_client import GrokClient

def main():
    client = GrokClient()
    msgs = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "In one sentence, what does tau2-bench evaluate?"},
    ]
    for model in ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning"]:
        out = client.chat(model=model, messages=msgs, temperature=0.0, seed=42)
        print(model, "latency_ms:", out["latency_ms"], "chars:", len(out["content"]))

if __name__ == "__main__":
    main()
