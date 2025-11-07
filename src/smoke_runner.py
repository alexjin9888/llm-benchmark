from src.grok_client import GrokClient

def main():
    client = GrokClient()
    msgs = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "In one sentence, what is the purpose of an AI benchmark?"},
    ]
    for model in ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning"]:
        out = client.chat(model=model, messages=msgs, temperature=0.0, seed=42)
        print(model, "latency_ms:", out["latency_ms"])

if __name__ == "__main__":
    main()
