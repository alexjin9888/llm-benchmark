from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.grok_client import GrokClient
from src.safety_bench.safety_tasks import tasks as load_tasks
from src.safety_bench.safety_scoring import score_one, aggregate
from src.safety_bench.policies import AIRLINE_POLICY, RETAIL_POLICY, TELECOM_POLICY, SAFETY_POLICY


def filter_judge_notes(judge_notes):
    result = {}
    for key, value in judge_notes.items():
        if isinstance(value, dict) and all(k in value for k in ["ok", "reason"]):
            result[key] = {
                "ok": value["ok"],
                "reason": value["reason"]
            }
        elif key == "model":  # Keep model field as-is
            result[key] = value
    return result


def load_context_turns(trace_path: str, inject_after_turn: int) -> List[Dict[str, str]]:
    """Load conversation turns from a trace file up to the specified turn index.
    
    Args:
        trace_path: Path to the trace JSON file
        inject_after_turn: 1-indexed turn number after which to inject (e.g., 17 means after turn 17)
    
    Returns:
        List of message dicts with 'role' and 'content'
    """
    try:
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)
        
        if 'turns' not in trace_data:
            return []
        
        turns = trace_data['turns']
        # Convert 1-indexed to 0-indexed (turn 17 = index 16, so we want turns[0:17])
        context_turns = turns[:inject_after_turn]
        
        # Convert to message format (filter out tool-related turns)
        messages = []
        for turn in context_turns:
            if turn['role'] in ['user', 'assistant'] and turn.get('content'):
                messages.append({
                    'role': turn['role'],
                    'content': turn['content']
                })
        
        return messages
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load context from {trace_path}: {e}")
        return []

def get_valid_turn_count(trace_path: str) -> int:
    """Get the number of valid turns in a trace file."""
    try:
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)
        
        if 'turns' not in trace_data:
            return 0
        
        # Count turns with actual content (exclude tool turns)
        valid_turns = 0
        for turn in trace_data['turns']:
            if turn['role'] in ['user', 'assistant'] and turn.get('content'):
                valid_turns += 1
        
        return valid_turns
    except:
        return 0

def run_one(client: GrokClient, model: str, task: Dict[str, Any], temperature: float, safety_policy: bool,
           cross_domain_override: Optional[Tuple[str, int]] = None) -> Dict[str, Any]:
    """Run a single task with optional context injection.
    
    Args:
        client: Grok client
        model: Model name
        task: Task dictionary
        temperature: Temperature for generation
        cross_domain_override: Optional (trace_path, inject_after_turn) for cross-domain tasks
    """
    domain = task['domain']
    if domain == 'airline':
        SYSTEM_POLICY = AIRLINE_POLICY
    elif domain == 'retail':
        SYSTEM_POLICY = RETAIL_POLICY
    elif domain == 'telecom':
        SYSTEM_POLICY = TELECOM_POLICY
    elif domain == 'cross':
        SYSTEM_POLICY = f"You are a cross domain customer service AI assistant. Here are several potential policies:\n{AIRLINE_POLICY}\n{RETAIL_POLICY}\n{TELECOM_POLICY}"
    else:
        raise Exception

    if safety_policy:
        SYSTEM_POLICY += '\n' + SAFETY_POLICY

    msgs = [{"role": "system",
             "content": SYSTEM_POLICY}]
    
    # Determine context source
    if cross_domain_override:
        # Use the provided cross-domain context
        trace_path, inject_after_turn = cross_domain_override
        context_turns = load_context_turns(trace_path, inject_after_turn)
        msgs.extend(context_turns)
        had_context = True
        context_info = {
            "trace_path": trace_path,
            "inject_after_turn": inject_after_turn,
            "is_cross_domain": True
        }
    elif 'context' in task:
        context = task['context']
        if 'trace_path' in context and context['trace_path'] is not None:
            # Use task's own context
            context_turns = load_context_turns(
                context['trace_path'],
                context['inject_after_turn']
            )
            msgs.extend(context_turns)
            had_context = True
            context_info = context
        else:
            # Task has context field but trace_path is None (will be handled as cross-domain)
            had_context = False
            context_info = None
    else:
        had_context = False
        context_info = None
    
    # Add the current user prompt
    msgs.append({"role": "user", "content": task["user_prompt"]})
    
    out = client.chat(model=model, messages=msgs, temperature=temperature, seed=42)
    return {
        "content": out["content"],
        "latency_ms": out["latency_ms"],
        "usage": out.get("usage"),
        "had_context": had_context,
        "context_info": context_info
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="grok-4-fast-reasoning")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="results/safety")
    ap.add_argument("--max_tasks", type=int, default=10)
    ap.add_argument("--safety_policy", action='store_true', default=False)
    args = ap.parse_args()
    
    
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting safety evaluation with model: {args.model}")
    print(f"Output directory: {out_dir}")
    print("-" * 60)
    
    client = GrokClient(log_dir="results/logs")
    judge = GrokClient(log_dir="results/safety/judge")
    all_tasks = load_tasks()[:args.max_tasks]
    
    # Separate regular tasks and cross-domain tasks
    regular_tasks = []
    cross_domain_tasks = []
    
    for task in all_tasks:
        if 'context' in task:
            ctx = task['context']
            if ctx.get('trace_path') is None and ctx.get('inject_after_turn') == -1:
                cross_domain_tasks.append(task)
            else:
                regular_tasks.append(task)
        else:
            regular_tasks.append(task)
    
    print(f"Total tasks: {len(all_tasks)}")
    print(f"  - Regular tasks: {len(regular_tasks)}")
    print(f"  - Cross-domain tasks: {len(cross_domain_tasks)}")
    print("-" * 60)
    
    rows: List[Dict[str, Any]] = []
    generated_traces: List[str] = []
    
    # Process regular tasks first
    if regular_tasks:
        print(f"\n[Phase 1/2] Processing {len(regular_tasks)} regular tasks...")
        for i, t in enumerate(regular_tasks, 1):
            print(f"  [{i:3d}/{len(regular_tasks)}] Running task {t['id']} ({t['domain']}, {t['severity']})... ", end='', flush=True)
            
            resp = run_one(client, args.model, t, args.temperature, args.safety_policy)
            row = score_one(t, resp["content"], judge=judge, judge_model=args.model)
            row.update({
                "latency_ms": resp["latency_ms"],
                "had_context": resp["had_context"]
            })
            rows.append(row)
            
            # Quick pass/fail indicator - using 'compliant' key instead of 'safety_compliance'
            status = "✓ PASS" if row['compliant'] else "✗ FAIL"
            print(f"{status} ({resp['latency_ms']:.0f}ms)")
            
            # Save trace
            trace_path = traces_dir / f"{t['id']}.json"
            trace = {
                "task_id": t["id"],
                "domain": t["domain"],
                "severity": t["severity"],
                "user_prompt": t["user_prompt"],
                "assistant": resp["content"],
                "latency_ms": resp["latency_ms"],
                "had_context": resp["had_context"],
                "compliant": row["compliant"],  # Include compliance status in trace
                "violations": row["violations"],
                "judge_notes": filter_judge_notes(row["judge"])
            }
            
            if resp["context_info"]:
                trace["context"] = resp["context_info"]
            
            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
            generated_traces.append(str(trace_path))
    
    # Process cross-domain tasks using traces from regular tasks
    if cross_domain_tasks and generated_traces:
        print(f"\n[Phase 2/2] Processing {len(cross_domain_tasks)} cross-domain tasks...")
        for i, t in enumerate(cross_domain_tasks, 1):
            # Randomly select a trace from generated traces
            selected_trace = random.choice(generated_traces)
            
            # Get valid turn count and randomly select injection point
            valid_turns = get_valid_turn_count(selected_trace)
            if valid_turns > 0:
                # Select a random turn between 1 and valid_turns (1-indexed)
                inject_after_turn = random.randint(1, valid_turns)
            else:
                # Fallback if no valid turns
                inject_after_turn = 1
            
            trace_name = Path(selected_trace).stem
            print(f"  [{i:3d}/{len(cross_domain_tasks)}] Running task {t['id']} ({t['domain']}, {t['severity']}) "
                  f"[ctx: {trace_name}@turn{inject_after_turn}]... ", end='', flush=True)
            
            resp = run_one(
                client, args.model, t, args.temperature, args.safety_policy,
                cross_domain_override=(selected_trace, inject_after_turn)
            )
            
            row = score_one(t, resp["content"], judge=judge, judge_model=args.model)
            row.update({
                "latency_ms": resp["latency_ms"],
                "had_context": resp["had_context"]
            })
            rows.append(row)
            
            # Quick pass/fail indicator - using 'compliant' key
            status = "✓ PASS" if row['compliant'] else "✗ FAIL"
            print(f"{status} ({resp['latency_ms']:.0f}ms)")
            
            # Save trace
            trace = {
                "task_id": t["id"],
                "domain": t["domain"],
                "severity": t["severity"],
                "user_prompt": t["user_prompt"],
                "assistant": resp["content"],
                "latency_ms": resp["latency_ms"],
                "had_context": resp["had_context"],
                "compliant": row["compliant"],
                "violations": row["violations"],
                "judge_notes": filter_judge_notes(row["judge"])
            }
            
            if resp["context_info"]:
                trace["context"] = resp["context_info"]
            
            (traces_dir / f"{t['id']}.json").write_text(
                json.dumps(trace, indent=2, ensure_ascii=False)
            )
    elif cross_domain_tasks and not generated_traces:
        print(f"\n⚠️  Warning: {len(cross_domain_tasks)} cross-domain tasks skipped (no regular traces available)")
    
    # Save metrics and scores
    print("\nSaving results...")
    metrics = aggregate(rows)
    (out_dir / "safety_metrics.json").write_text(json.dumps(metrics, indent=2))
    
    with (out_dir / "safety_scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Detailed summary
    print("\n" + "=" * 60)
    print("SAFETY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total tasks completed: {metrics['num_tasks']}")
    print(f"Safety Compliance Rate: {metrics['safety_compliance_rate']:.1%}")
    print(f"Critical Failure Rate: {metrics['critical_rate']:.1%}")
    print(f"Critical Failures Count: {metrics['critical_failure_count']}")
    
    # Breakdown by severity if available
    if rows:
        critical_tasks = [r for r in rows if r['severity'] == 'critical']
        high_tasks = [r for r in rows if r['severity'] == 'high']
        medium_tasks = [r for r in rows if r['severity'] == 'medium']
        
        if critical_tasks:
            critical_pass = sum(1 for r in critical_tasks if r['compliant'])
            print(f"\nCritical tasks: {critical_pass}/{len(critical_tasks)} passed")
        if high_tasks:
            high_pass = sum(1 for r in high_tasks if r['compliant'])
            print(f"High tasks: {high_pass}/{len(high_tasks)} passed")
        if medium_tasks:
            medium_pass = sum(1 for r in medium_tasks if r['compliant'])
            print(f"Medium tasks: {medium_pass}/{len(medium_tasks)} passed")
    
    print("=" * 60)
    print(f"Results saved to: {out_dir}")
    
if __name__ == "__main__":
    main()