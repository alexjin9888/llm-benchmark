# src/tools/collect_tau2.py
"""
Usage:
  python -m src.tools.collect_tau2 <path_to_tau2_json> <out_dir> [--max_examples 10] [--examples_md examples.md]
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import Counter, defaultdict

# --------------------------- I/O helpers ---------------------------

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------- small stats helpers -----------------------

def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None

def _std(xs: List[float]) -> Optional[float]:
    n = len(xs)
    if n <= 1:
        return 0.0 if n == 1 else None
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))

def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2

def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / (sx * sy)

def _wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    half = z * math.sqrt((phat*(1 - phat) + z**2/(4*n)) / n) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    return (round(lo, 4), round(hi, 4))

# ---------------------- schema-aware extractors --------------------

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def extract_meta(sim_root: Dict[str, Any]) -> Dict[str, Any]:
    info = sim_root.get("info", {}) or {}
    return {
        "domain": safe_get(info, ["environment_info", "domain_name"]),
        "model_agent": safe_get(info, ["agent_info", "llm"]),
        "model_user": safe_get(info, ["user_info", "llm"]),
        "num_trials_declared": info.get("num_trials"),
        "seed": info.get("seed"),
        "git_commit": safe_get(info, ["git_commit"])
    }

def _sum_msg_costs(messages: List[Dict[str, Any]]) -> Tuple[float, float]:
    agent_sum = 0.0
    user_sum = 0.0
    for m in messages or []:
        c = m.get("cost")
        if not isinstance(c, (int, float)):
            continue
        role = m.get("role")
        if role == "assistant":
            agent_sum += float(c)
        elif role == "user":
            user_sum += float(c)
    return agent_sum, user_sum

def _sum_msg_tokens(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    # usage dicts sometimes appear on user-sim messages too
    out = dict(agent_prompt=0, agent_completion=0, user_prompt=0, user_completion=0)
    for m in messages or []:
        usage = m.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        role = m.get("role")
        if role == "assistant":
            out["agent_prompt"] += int(pt)
            out["agent_completion"] += int(ct)
        elif role == "user":
            out["user_prompt"] += int(pt)
            out["user_completion"] += int(ct)
    return out

def extract_turns_from_messages(msgs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int, int, bool]:
    turns: List[Dict[str, Any]] = []
    assistant_turns = 0
    tool_calls_ct = 0
    escalation = False
    for m in msgs:
        role = m.get("role", "unknown")
        if role == "assistant":
            assistant_turns += 1
        content = m.get("content") or ""
        tcs = m.get("tool_calls")
        if tcs and isinstance(tcs, list):
            for tc in tcs:
                tname = (tc.get("name") or safe_get(tc, ["function", "name"]) or "tool")
                targs = (tc.get("arguments") or safe_get(tc, ["function", "arguments"]))
                if tname == "transfer_to_human_agents":
                    escalation = True
                tool_calls_ct += 1
                turns.append({"role": role, "content": content,
                              "tool_call": {"name": tname, "arguments": targs},
                              "tool_result": None})
        else:
            turns.append({"role": role, "content": content,
                          "tool_call": None, "tool_result": None})
        if role == "tool":
            turns[-1]["tool_result"] = content
    return turns, len(turns), assistant_turns, tool_calls_ct, escalation

def normalize_trial_from_simulation(sim_entry: Dict[str, Any]) -> Dict[str, Any]:
    reward = safe_get(sim_entry, ["reward_info", "reward"])
    success = None
    if isinstance(reward, (int, float)):
        success = (reward > 0)

    agent_cost = sim_entry.get("agent_cost")
    user_cost = sim_entry.get("user_cost")
    messages = sim_entry.get("messages", [])
    if not isinstance(agent_cost, (int, float)) or not isinstance(user_cost, (int, float)):
        a_fallback, u_fallback = _sum_msg_costs(messages)
        agent_cost = float(agent_cost) if isinstance(agent_cost, (int, float)) else a_fallback
        user_cost  = float(user_cost)  if isinstance(user_cost,  (int, float)) else u_fallback

    turns, n_turns, n_assistant, n_tool_calls, escalation = extract_turns_from_messages(messages)
    tok = _sum_msg_tokens(messages)
    action_checks = safe_get(sim_entry, ["reward_info", "action_checks"]) or []
    tp = sum(1 for a in action_checks if a.get("action_match") is True)
    fp = sum(1 for a in action_checks if a.get("action_match") is False)

    return {
        "task_id": str(sim_entry.get("task_id")),
        "trial_index": sim_entry.get("trial", 0),
        "reward": reward if isinstance(reward, (int, float)) else None,
        "success": success,
        "turns": turns,
        "num_turns": n_turns,
        "assistant_turns": n_assistant,
        "tool_calls": n_tool_calls,
        "escalated": escalation,
        "termination_reason": sim_entry.get("termination_reason"),
        "duration": sim_entry.get("duration"),
        "agent_cost": float(agent_cost) if isinstance(agent_cost, (int, float)) else 0.0,
        "user_cost": float(user_cost) if isinstance(user_cost, (int, float)) else 0.0,
        "tokens": tok,  # dict
        "action_tp": tp,
        "action_fp": fp,
        "raw": sim_entry,
    }

def group_trials_by_task(sim_root: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    sims = sim_root.get("simulations")
    if isinstance(sims, list) and sims:
        for s in sims:
            t = normalize_trial_from_simulation(s)
            tid = t["task_id"]
            out.setdefault(tid, []).append(t)
        for tid in out:
            out[tid].sort(key=lambda x: (x.get("trial_index", 0),))
        return out

    # fallback legacy shape
    tasks = sim_root.get("tasks") or []
    for i, task in enumerate(tasks):
        tid = str(task.get("task_id") or task.get("id") or i)
        out.setdefault(tid, [])
    return out

def compute_pass_k(task_trials: Dict[str, List[Dict[str, Any]]], max_k: int) -> Tuple[Dict[str, float], Dict[int, int], int]:
    tids = list(task_trials.keys())
    n = len(tids)
    counts: Dict[int, int] = {}  # successes per k
    rates: Dict[str, float] = {}
    if n == 0:
        return ({f"pass_k_{k}": 0.0 for k in range(1, max_k + 1)}, {k: 0 for k in range(1, max_k + 1)}, 0)
    for k in range(1, max_k + 1):
        ok = 0
        for tid in tids:
            succ = sum(1 for tr in task_trials[tid] if bool(tr.get("success")))
            if succ >= k:
                ok += 1
        counts[k] = ok
        rates[f"pass_k_{k}"] = round(ok / n, 6)
    return rates, counts, n

# -------------------------- summarization --------------------------

def summarize(sim_root: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    meta = extract_meta(sim_root)
    task_trials = group_trials_by_task(sim_root)

    # expected actions per task from task spec if available
    expected_actions_per_task: Dict[str, int] = {}
    for t in sim_root.get("tasks", []) or []:
        tid = str(t.get("id") or t.get("task_id"))
        exp = safe_get(t, ["evaluation_criteria", "actions"]) or []
        expected_actions_per_task[tid] = len(exp)

    # aggregates
    rewards_all: List[float] = []
    cost_all: List[float] = []
    duration_all: List[float] = []
    tool_calls_all: List[int] = []
    turns_all: List[int] = []
    success_any_ct = 0
    total_agent_cost = 0.0
    total_user_cost = 0.0
    num_trials_total = 0
    escalations = 0
    term_reasons = Counter()

    # for correlations and splits
    reward_vec: List[float] = []
    cost_vec: List[float] = []
    tool_vec: List[float] = []
    turn_vec: List[float] = []
    succ_vec: List[int] = []

    # action micro stats
    action_tp_total = 0
    action_fp_total = 0
    action_expected_total = 0
    extra_actions_total = 0  # tool calls not in expected list

    for tid, trials in task_trials.items():
        if any(t.get("success") for t in trials):
            success_any_ct += 1
        exp_ct = expected_actions_per_task.get(tid)
        for tr in trials:
            num_trials_total += 1
            r = tr.get("reward")
            c = (tr.get("agent_cost", 0.0) or 0.0) + (tr.get("user_cost", 0.0) or 0.0)
            rewards_all += ([float(r)] if isinstance(r, (int, float)) else [])
            cost_all.append(float(c))
            total_agent_cost += tr.get("agent_cost", 0.0) or 0.0
            total_user_cost  += tr.get("user_cost", 0.0) or 0.0
            if isinstance(tr.get("duration"), (int, float)):
                duration_all.append(float(tr["duration"]))
            tool_calls_all.append(int(tr.get("tool_calls") or 0))
            turns_all.append(int(tr.get("num_turns") or 0))
            if tr.get("escalated"):
                escalations += 1
            term_reasons[tr.get("termination_reason") or "unknown"] += 1

            # vectors
            if isinstance(r, (int, float)):
                reward_vec.append(float(r))
                cost_vec.append(float(c))
                tool_vec.append(float(tr.get("tool_calls") or 0))
                turn_vec.append(float(tr.get("num_turns") or 0))
                succ_vec.append(1 if tr.get("success") else 0)

            # actions
            tp = int(tr.get("action_tp") or 0)
            fp = int(tr.get("action_fp") or 0)
            action_tp_total += tp
            action_fp_total += fp
            if isinstance(exp_ct, int):
                action_expected_total += exp_ct
                # extra actions = tool calls with names not in expected set
                # build set of expected names
                # note: we only know counts here, but we can get names from tasks if needed
                # fallback: approximate by counting all tool calls minus expected attempts in checks
                attempted_expected = tp + fp
                extra = max(0, int(tr.get("tool_calls") or 0) - attempted_expected)
                extra_actions_total += extra

    n_tasks = len(task_trials)
    max_trials = max((len(v) for v in task_trials.values()), default=1)
    passk_rates, passk_counts, denom_tasks = compute_pass_k(task_trials, max_k=max_trials)

    total_cost = total_agent_cost + total_user_cost
    avg_reward = _mean(rewards_all)
    metrics = {
        "domain": meta.get("domain"),
        "model_agent": meta.get("model_agent"),
        "model_user": meta.get("model_user"),
        "num_tasks": n_tasks,
        "num_trials": max_trials,
        "num_trials_total": num_trials_total,

        # accuracy style
        **passk_rates,
        "pass_k_ci": {f"pass_k_{k}": _wilson(passk_counts[k], denom_tasks) for k in passk_counts},
        "success_any_rate": round(success_any_ct / n_tasks, 4) if n_tasks else 0.0,

        # reward summary
        "avg_reward": round(avg_reward, 6) if avg_reward is not None else None,
        "std_reward": round(_std(rewards_all), 6) if rewards_all else None,
        "median_reward": round(_median(rewards_all), 6) if rewards_all else None,
        "p25_reward": round(_quantile(rewards_all, 0.25), 6) if rewards_all else None,
        "p75_reward": round(_quantile(rewards_all, 0.75), 6) if rewards_all else None,

        # costs
        "total_agent_cost": round(total_agent_cost, 6),
        "total_user_cost": round(total_user_cost, 6),
        "total_cost": round(total_cost, 6),
        "avg_cost_per_task": round(total_cost / n_tasks, 6) if n_tasks else None,
        "avg_cost_per_trial": round(total_cost / num_trials_total, 6) if num_trials_total else None,
        "std_cost_per_trial": round(_std(cost_all), 6) if cost_all else None,
        "median_cost_per_trial": round(_median(cost_all), 6) if cost_all else None,

        # duration and structure
        "avg_duration_s": round(_mean(duration_all), 6) if duration_all else None,
        "std_duration_s": round(_std(duration_all), 6) if duration_all else None,
        "median_duration_s": round(_median(duration_all), 6) if duration_all else None,
        "avg_turns": round(_mean(turns_all), 3) if turns_all else None,
        "avg_tool_calls": round(_mean(tool_calls_all), 3) if tool_calls_all else None,
        "escalation_rate": round(escalations / num_trials_total, 4) if num_trials_total else 0.0,
        "termination_reason_counts": dict(term_reasons),

        # action micro metrics
        "action_expected_total": action_expected_total or None,
        "action_tp_total": action_tp_total or None,
        "action_fp_total": action_fp_total or None,
        "action_precision": round(action_tp_total / (action_tp_total + action_fp_total), 4) if (action_tp_total + action_fp_total) > 0 else None,
        "action_recall": round(action_tp_total / action_expected_total, 4) if action_expected_total else None,
        "action_f1": (lambda p, r: round(2*p*r/(p+r), 4) if p and r and (p+r) > 0 else None)(
            round(action_tp_total / (action_tp_total + action_fp_total), 10) if (action_tp_total + action_fp_total) > 0 else None,
            round(action_tp_total / action_expected_total, 10) if action_expected_total else None,
        ),
        "extra_actions_total": extra_actions_total or None,

        # relationships
        "corr_reward_cost": round(_pearson(reward_vec, cost_vec), 4) if len(reward_vec) >= 2 else None,
        "corr_reward_tool_calls": round(_pearson(reward_vec, tool_vec), 4) if len(reward_vec) >= 2 else None,
        "corr_reward_turns": round(_pearson(reward_vec, turn_vec), 4) if len(reward_vec) >= 2 else None,

        # success vs fail cost split
        "avg_cost_success": round(_mean([c for c, s in zip(cost_vec, succ_vec) if s == 1]), 6) if cost_vec else None,
        "avg_cost_failure": round(_mean([c for c, s in zip(cost_vec, succ_vec) if s == 0]), 6) if cost_vec else None,
    }

    # scores.jsonl rows (per task)
    scores: List[Dict[str, Any]] = []
    for tid, trials in task_trials.items():
        rs = [tr.get("reward") for tr in trials if isinstance(tr.get("reward"), (int, float))]
        avg_r = _mean(rs)
        a_cost = sum((tr.get("agent_cost", 0.0) or 0.0) for tr in trials)
        u_cost = sum((tr.get("user_cost", 0.0) or 0.0) for tr in trials)
        t_cost = a_cost + u_cost
        scores.append({
            "task_id": tid,
            "success_any": any(t.get("success") for t in trials),
            "avg_reward": round(avg_r, 6) if avg_r is not None else None,
            "agent_cost_sum": round(a_cost, 6),
            "user_cost_sum": round(u_cost, 6),
            "total_cost_sum": round(t_cost, 6),
            "avg_cost_per_trial": round(t_cost / len(trials), 6) if trials else None,
            "avg_turns": round(_mean([t.get("num_turns") or 0 for t in trials]), 3) if trials else None,
            "avg_tool_calls": round(_mean([t.get("tool_calls") or 0 for t in trials]), 3) if trials else None,
            "trials": [
                {
                    "trial_index": tr.get("trial_index"),
                    "success": tr.get("success"),
                    "reward": tr.get("reward"),
                    "agent_cost": tr.get("agent_cost"),
                    "user_cost": tr.get("user_cost"),
                    "num_turns": tr.get("num_turns"),
                    "tool_calls": tr.get("tool_calls"),
                    "duration": tr.get("duration"),
                    "termination_reason": tr.get("termination_reason"),
                    "escalated": tr.get("escalated"),
                    "tokens": tr.get("tokens"),
                    "action_tp": tr.get("action_tp"),
                    "action_fp": tr.get("action_fp"),
                } for tr in trials
            ],
        })

    # representative traces
    traces: List[Dict[str, Any]] = []
    for tid, trials in task_trials.items():
        if not trials:
            continue
        rep = next((tr for tr in trials if tr.get("success")), trials[0])
        traces.append({
            "task_id": tid,
            "reward": rep.get("reward"),
            "success": rep.get("success"),
            "turns": rep.get("turns", []),
        })

    return metrics, scores, traces

# ----------------------------- writers -----------------------------

def write_outputs(out_dir: Path, metrics: Dict[str, Any], scores: List[Dict[str, Any]], traces: List[Dict[str, Any]]) -> None:
    ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    tdir = out_dir / "traces"
    ensure_dir(tdir)
    for tr in traces:
        (tdir / f"task_{tr['task_id']}.json").write_text(json.dumps(tr, indent=2, ensure_ascii=False))

def write_examples_md(out_dir: Path, traces: List[Dict[str, Any]], max_examples: int, examples_md: Optional[str]) -> None:
    if max_examples <= 0 and not examples_md:
        return
    lines: List[str] = ["# Examples\n"]
    take = traces[:max(1, max_examples)] if max_examples > 0 else traces
    for tr in take:
        lines += [
            f"## Task {tr['task_id']}",
            f"- success: {tr.get('success')}",
            f"- reward: {tr.get('reward')}\n"
        ]
        for i, t in enumerate(tr.get("turns", [])[:16]):
            role = t.get("role", "unknown")
            content = (t.get("content") or "").strip()
            lines.append(f"**{i+1}. {role}:** {content}")
        lines.append("")
    target = Path(examples_md) if examples_md else (out_dir / "examples.md")
    target.write_text("\n".join(lines))

def print_summary(metrics: Dict[str, Any]) -> None:
    cols = [
        ("domain", metrics.get("domain")),
        ("tasks", metrics.get("num_tasks")),
        ("trials(max)", metrics.get("num_trials")),
        ("trials(total)", metrics.get("num_trials_total")),
        ("pass_k_1", metrics.get("pass_k_1")),
        ("pass_k_2", metrics.get("pass_k_2")),
        ("pass_k_3", metrics.get("pass_k_3")),
        ("avg_reward", metrics.get("avg_reward")),
        ("std_reward", metrics.get("std_reward")),
        ("total_cost", metrics.get("total_cost")),
        ("avg_cost/trial", metrics.get("avg_cost_per_trial")),
        ("avg_turns", metrics.get("avg_turns")),
        ("avg_tool_calls", metrics.get("avg_tool_calls")),
        ("escalation_rate", metrics.get("escalation_rate")),
        ("action_f1", metrics.get("action_f1")),
    ]
    print("Summary:", " | ".join(f"{k}={v}" for k, v in cols if v is not None))

# ------------------------------ main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str, help="path to a single tau2 simulation JSON")
    ap.add_argument("out_dir", type=str, help="directory to write metrics, scores, traces")
    ap.add_argument("--max_examples", type=int, default=10)
    ap.add_argument("--examples_md", type=str, default=None)
    args = ap.parse_args()

    src = Path(args.json_path)
    out_dir = Path(args.out_dir)
    if not src.exists():
        raise SystemExit(f"Input JSON not found: {src}")

    sim_root = load_json(src)
    metrics, scores, traces = summarize(sim_root)
    write_outputs(out_dir, metrics, scores, traces)
    write_examples_md(out_dir, traces, args.max_examples, args.examples_md)
    print_summary(metrics)

if __name__ == "__main__":
    main()
