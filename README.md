# Evaluating Grok and Improving Benchmarks

This repo contains:
- Baseline runs for tau2-bench and terminal-bench
- A neutral Grok client and logging harness
- An improved 10-case benchmark scaffold

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
# example wrapper script once ready
python tau2/run_tau2.py \
  --model grok-4-fast-reasoning \
  --temperature 0.0 \
  --seed 42 \
  --max-concurrency 2 \
  --split dev \
  --out results/baseline
```

### Improved Benchmark
```bash
python improved/run_improved.py \
  --model grok-4-fast-reasoning \
  --temperature 0.0 \
  --out results/improved
```