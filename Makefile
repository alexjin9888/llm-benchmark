# Makefile
# Basic knobs (override at call time, e.g., `make tau2_baseline CONC=1 SPLIT=dev`)
PY := .venv/bin/python
PIP := .venv/bin/pip

MODEL ?= grok-4-fast-reasoning
TEMP  ?= 0.0
SEED  ?= 42
CONC  ?= 2
SPLIT ?= dev

BASE_OUT     ?= results/baseline
NR_OUT       ?= results/baseline_nr
IMPROVED_OUT ?= results/improved

# Helpful for long-thinking models if you wire it through your adapter later
TIMEOUT ?= 300

.PHONY: setup smoke tau2_baseline tau2_baseline_nr improved_smoke \
        check-env lock cache-clear results-clean

setup:
	@test -d .venv || python3 -m venv .venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@mkdir -p results/logs results/cache $(BASE_OUT) $(NR_OUT) $(IMPROVED_OUT)
	@echo "Setup complete."

smoke: check-env
	@$(PY) -m src.smoke_runner

tau2_baseline: check-env
	@$(PY) tau2/run_tau2.py \
	  --model $(MODEL) \
	  --temperature $(TEMP) \
	  --seed $(SEED) \
	  --max-concurrency $(CONC) \
	  --split $(SPLIT) \
	  --out $(BASE_OUT)

tau2_baseline_nr: check-env
	@$(PY) tau2/run_tau2.py \
	  --model grok-4-fast-non-reasoning \
	  --temperature $(TEMP) \
	  --seed $(SEED) \
	  --max-concurrency $(CONC) \
	  --split $(SPLIT) \
	  --out $(NR_OUT)

improved_smoke: check-env
	@$(PY) improved/run_improved.py \
	  --model $(MODEL) \
	  --temperature $(TEMP) \
	  --out $(IMPROVED_OUT)

check-env:
	@$(PY) -c "import os,sys; from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); \
k=os.getenv('XAI_API_KEY'); \
sys.exit(0) if k else sys.exit('Missing XAI_API_KEY in .env')"
	@echo "XAI_API_KEY present."

lock:
	@$(PIP) freeze > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

cache-clear:
	@rm -rf results/cache && mkdir -p results/cache
	@echo "Cache cleared."

results-clean:
	@rm -rf results/* && mkdir -p results/logs results/cache
	@echo "Results cleared."
