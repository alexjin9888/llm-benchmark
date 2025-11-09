# Evaluating Grok and Improving Benchmarks

This repo contains:
- Baseline runs for tau2-bench
- A neutral Grok client and logging harness
- An improved benchmark for safety, ethics and privacy
- An improved benchmark for user satisfaction

See "Replication" for exact commands.

## Replication

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# paste your XAI_API_KEY into .env
```

### Sanity check
- CLI
```bash
./smoke_tau2.sh
```

- Python API
```bash
python -m src.smoke_runner
```

### Baseline
```bash
# Tau2 CLI Makefile baseline example
make run_collect DOMAIN=telecom BASE_OUT=results/baseline/telecom/oracle/grok-4-fast-reasoning AGENT_LLM=xai/grok-4-fast-reasoning USER_LLM=xai/grok-4-fast-reasoning AGENT=llm_agent

# Tau2 CLI Makefile ablation example
make run_collect DOMAIN=telecom BASE_OUT=results/ablation/telecom/oracle/grok-4-fast-reasoning AGENT_LLM=xai/grok-4-fast-reasoning USER_LLM=xai/grok-4-fast-reasoning AGENT=llm_agent_gt MAX_STEPS=400
```

```bash
# Grok API Python wrapper example
python tau2/run_tau2.py \
  --model grok-4-fast-reasoning \
  --temperature 0.0 \
  --seed 42 \
  --max-concurrency 2 \
  --split dev \
  --out results/grok-4-fast-reasoning
```

### Improved Benchmark
```bash
python -m src.safety_bench.safety_runner --model grok-4-fast-reasoning --out=results/safety/without_policy
python -m src.safety_bench.safety_runner --model grok-4-fast-reasoning --out=results/safety/with_policy --safety_policy
python -m src.safety_bench.safety_runner --model grok-4-fast-non-reasoning --out=results/safety/without_policy
python -m src.safety_bench.safety_runner --model grok-4-fast-non-reasoning --out=results/safety/with_policy --safety_policy
```

```bash
python -m src.satisfaction_bench.satisfaction_runner --traces_dir src/satisfaction_bench/example_jsons/airline/grok-4-fast-reasoning/traces --tag airline-grok-4-fast-r
python -m src.satisfaction_bench.satisfaction_runner --traces_dir src/satisfaction_bench/example_jsons/airline/grok-4-fast-non-reasoning/traces --tag airline-grok-4-fast-nr
python -m src.satisfaction_bench.satisfaction_runner --traces_dir src/satisfaction_bench/example_jsons/retail/grok-4-fast-reasoning/traces --tag retail-grok-4-fast-r
python -m src.satisfaction_bench.satisfaction_runner --traces_dir src/satisfaction_bench/example_jsons/retail/grok-4-fast-non-reasoning/traces --tag retail-grok-4-fast-nr
python -m src.satisfaction_bench.satisfaction_runner --traces_dir src/satisfaction_bench/example_jsons/telecom/grok-4-fast-reasoning/traces --tag telecom-grok-4-fast-r
python -m src.satisfaction_bench.satisfaction_runner --traces_dir src/satisfaction_bench/example_jsons/telecom/grok-4-fast-non-reasoning/traces --tag telecom-grok-4-fast-nr
```