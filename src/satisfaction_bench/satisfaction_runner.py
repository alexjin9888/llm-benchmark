from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any

from src.grok_client import GrokClient
from src.satisfaction_bench.satisfaction_scoring import score_conversation, aggregate

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def _from_tau2_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract plain user/assistant turns from a tau2 simulation entry."""
    out: List[Dict[str, str]] = []
    for m in msgs or []:
        role = m.get("role")
        if role in {"user", "assistant"}:
            content = (m.get("content") or "").strip()
            if content:
                out.append({"role": role, "content": content})
    return out

def _from_trace_json(trace: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Accepts safety trace files (may contain only user_prompt + assistant) or full 'turns'.
    """
    if "turns" in trace and isinstance(trace["turns"], list):
        out: List[Dict[str, str]] = []
        for t in trace["turns"]:
            role = t.get("role")
            content = (t.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                out.append({"role": role, "content": content})
        if out:
            return out

    # Fallback: synthesize a two turn convo if available
    u = (trace.get("user_prompt") or "").strip()
    a = (trace.get("assistant") or "").strip()
    turns: List[Dict[str, str]] = []
    if u:
        turns.append({"role": "user", "content": u})
    if a:
        turns.append({"role": "assistant", "content": a})
    return turns

def _label_reward(val) -> str:
    """Map numeric reward to a tri-state label."""
    if val is None:
        return "unknown"
    try:
        return "rewarded" if float(val) > 0 else "not_rewarded"
    except Exception:
        return "unknown"

def _extract_reason(row: Dict[str, Any]) -> str:
    # Try a few common keys; fall back to first evidence/snippet if available
    for k in ("reason", "rationale", "explanation"):
        if row.get(k):
            return str(row[k])
    ev = row.get("evidence") or row.get("highlights") or row.get("snippets")
    if isinstance(ev, list) and ev:
        return " | ".join(_short(str(x), 120) for x in ev[:2])
    if isinstance(ev, str):
        return _short(ev, 180)
    # Sometimes scoring returns a 'judge' payload with reason
    judge = row.get("judge") or {}
    if isinstance(judge, dict):
        for subk in ("reason", "notes"):
            if judge.get(subk):
                return _short(str(judge[subk]), 180)
    return ""

def _print_matrix(matrix: Dict[str, Dict[str, int]]) -> None:
    headers = ["", "satisfied", "neutral", "dissatisfied", "total"]
    rows = []
    for rkey in ("rewarded", "not_rewarded", "unknown"):
        row = [
            rkey,
            str(matrix[rkey]["satisfied"]),
            str(matrix[rkey]["neutral"]),
            str(matrix[rkey]["dissatisfied"]),
            str(sum(matrix[rkey].values())),
        ]
        rows.append(row)
    # column widths
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    # print
    def fmt(cols): return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print("\nReward × Satisfaction Matrix")
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=None,
                    help="Path to a single tau2 simulation JSON (with root.simulations[*].messages)")
    ap.add_argument("--traces_dir", type=str, default=None,
                    help="Directory of trace JSONs (e.g., results/baseline/airline/<model>/traces)")
    ap.add_argument("--judge_model", type=str, default="grok-4-fast-reasoning")
    ap.add_argument("--out", type=str, default="results/satisfaction")
    ap.add_argument("--tag", type=str, default=None, help="Optional tag name for outputs")
    args = ap.parse_args()

    if not args.src and not args.traces_dir:
        raise SystemExit("Provide --src <tau2.json> or --traces_dir <dir>")

    out_dir = Path(args.out) / (args.tag or args.judge_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "satisfaction_scores.jsonl"
    metrics_path = out_dir / "satisfaction_metrics.json"

    judge = GrokClient(log_dir="results/satisfaction/judge")

    rows: List[Dict[str, Any]] = []

    # Case 1: tau2 simulation JSON
    if args.src:
        src = Path(args.src)
        if not src.exists():
            raise SystemExit(f"Not found: {src}")
        root = load_json(src)

        domain = (root.get("info", {}) or {}).get("environment_info", {}) or {}
        domain_name = domain.get("domain_name")

        sims = root.get("simulations") or []
        total = len(sims)
        print(f"[tau2] Found {total} conversations in {src}")
        for i, s in enumerate(sims, 1):
            msgs = s.get("messages") or []
            turns = _from_tau2_messages(msgs)
            header = f"  [{i:>4}/{total}] task={s.get('task_id')} trial={s.get('trial')} seed={s.get('seed')}"
            if not turns:
                print(f"{header} -> skip (no user/assistant turns)", flush=True)
                continue

            # Grab reward if present
            reward = None
            ri = s.get("reward_info") or {}
            if isinstance(ri, dict):
                reward = ri.get("reward")

            print(f"{header} -> judging ... ", end="", flush=True)
            meta = {
                "source": "tau2_sim",
                "task_id": s.get("task_id"),
                "domain": domain_name,
                "trial": s.get("trial"),
                "seed": s.get("seed"),
                "reward": reward,
            }
            row = score_conversation(judge, args.judge_model, turns, meta=meta)
            row.setdefault("meta", meta)
            rows.append(row)
            print(f"{row['satisfaction']}", flush=True)

    # Case 2: directory of trace JSONs
    if args.traces_dir:
        tdir = Path(args.traces_dir)
        if not tdir.exists():
            raise SystemExit(f"Traces dir not found: {tdir}")
        paths = sorted(tdir.glob("*.json"))
        total = len(paths)
        print(f"[traces] Scanning {total} files in {tdir}")
        for i, p in enumerate(paths, 1):
            try:
                tr = load_json(p)
            except Exception:
                print(f"  [{i:>4}/{total}] {p.name} -> skip (invalid JSON)", flush=True)
                continue

            turns = _from_trace_json(tr)
            if not turns:
                print(f"  [{i:>4}/{total}] {p.name} -> skip (no usable turns)", flush=True)
                continue

            # Try to extract reward if the trace kept it
            reward = tr.get("reward")
            print(f"  [{i:>4}/{total}] {p.name} -> judging ... ", end="", flush=True)
            meta = {
                "source": "trace_dir",
                "file": str(p),
                "task_id": tr.get("task_id"),
                "domain": tr.get("domain"),
                "severity": tr.get("severity"),
                "reward": reward,
            }
            row = score_conversation(judge, args.judge_model, turns, meta=meta)
            row.setdefault("meta", meta)
            rows.append(row)
            print(f"{row['satisfaction']}", flush=True)

    # Write per-row scores
    with scores_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate basic satisfaction metrics
    metrics = aggregate(rows)

    # ---------- Build reward × satisfaction matrix ----------
    matrix = {
        "rewarded":      {"satisfied": 0, "neutral": 0, "dissatisfied": 0},
        "not_rewarded":  {"satisfied": 0, "neutral": 0, "dissatisfied": 0},
        "unknown":       {"satisfied": 0, "neutral": 0, "dissatisfied": 0},
    }
    satisfied_cases = []
    dissatisfied_cases = []

    for r in rows:
        sat = str(r.get("satisfaction", "")).lower().strip()
        if sat not in {"satisfied", "neutral", "dissatisfied"}:
            continue
        reward = r.get("meta", {}).get("reward")
        bucket = _label_reward(reward)
        matrix[bucket][sat] += 1

        # Collect case summaries for printing
        ident = r.get("meta", {}).get("task_id") or r.get("meta", {}).get("file") or "unknown"
        reason = _extract_reason(r)
        case = {
            "id": ident,
            "reward": bucket,
            "reason": reason,
        }
        if sat == "satisfied":
            satisfied_cases.append(case)
        elif sat == "dissatisfied":
            dissatisfied_cases.append(case)

    # Add matrix to metrics
    metrics["reward_vs_satisfaction"] = matrix

    # Write metrics
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---------- Console summary ----------
    print("\nSatisfaction Summary:",
          f"count={metrics['num_conversations']}",
          f"satisfied={metrics['rate_satisfied']:.1%}",
          f"neutral={metrics['rate_neutral']:.1%}",
          f"dissatisfied={metrics['rate_dissatisfied']:.1%}",
          f"index={metrics['satisfaction_index']:.3f}",
          sep=" | ")

    _print_matrix(matrix)

    # Print satisfied & dissatisfied cases with reasons
    if satisfied_cases:
        print("\nSatisfied cases (id, reward_bucket, reason/snippet):")
        for c in satisfied_cases:
            print(f"  - {c['id']} | {c['reward']} | {c['reason']}")
    if dissatisfied_cases:
        print("\nDissatisfied cases (id, reward_bucket, reason/snippet):")
        for c in dissatisfied_cases:
            print(f"  - {c['id']} | {c['reward']} | {c['reason']}")

    print(f"\nWrote:\n  {scores_path}\n  {metrics_path}")

if __name__ == "__main__":
    main()
