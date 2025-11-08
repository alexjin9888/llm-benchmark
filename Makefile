SHELL := /bin/bash

# --- Tools
PY  := .venv/bin/python
PIP := .venv/bin/pip
TAU := .venv/bin/tau2

# --- Simple knobs (override per call)
DOMAIN ?= airline
TRIALS ?= 4
TASKS  ?= 150
CONC   ?= 8
MAX_STEPS ?= 200
SEED   ?= 42
TEMP   ?= 0.0
LOG_LVL ?= ERROR
AGENT ?= llm_agent
AGENT_LLM ?= xai/grok-4-fast-reasoning
USER_LLM  ?= xai/grok-4-fast-reasoning

BASE_OUT ?= results/tau2_runs

.PHONY: setup smoke check-env run collect collect_latest run_collect lock results-clean

setup:
	@test -d .venv || python3 -m venv .venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@mkdir -p $(BASE_OUT) results/logs results/cache
	@echo "Setup complete."

smoke: check-env
	@$(PY) -m src.smoke_runner

check-env:
	@$(PY) -c "import os,sys; from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); \
print('ok') if os.getenv('XAI_API_KEY') else sys.exit('Missing XAI_API_KEY in .env')"
	@echo "XAI_API_KEY present."

# 1) Run tau2 with defaults (no --save-to). τ² writes under */data/simulations/ with its own filename.
run: check-env
	@set -e; set -a; . ./.env; set +a; \
	$(TAU) run \
	  --domain $(DOMAIN) \
	  --agent $(AGENT) \
	  --agent-llm $(AGENT_LLM) \
	  --agent-llm-args "{\"temperature\": $(TEMP)}" \
	  --user-llm $(USER_LLM) \
	  --user-llm-args "{\"temperature\": $(TEMP)}" \
	  --num-trials $(TRIALS) \
	  --num-tasks $(TASKS) \
	  --max-steps $(MAX_STEPS) \
	  --max-concurrency $(CONC) \
	  --seed $(SEED) \
	  --log-level $(LOG_LVL)

# 2a) Collect: prefer an existing JSON already in $(BASE_OUT).
# If none exists there, fall back to JSON=<path or basename> and copy it into $(BASE_OUT).
# Examples:
#   make collect BASE_OUT=results/baseline/airline/grok-4-fast-reasoning
#   make collect JSON=path/to/run.json
#   make collect JSON=2025-..._airline_llm_agent_...json BASE_OUT=results/tau2_runs
collect:
	@set -e; \
	PATTERN="$$(basename "$(BASE_OUT)")"; \
	LATEST_IN_BASE=$$((find "$(BASE_OUT)" -type f -name "*$$PATTERN*.json" -exec stat -f "%m %N" {} \; 2>/dev/null || \
	                   find "$(BASE_OUT)" -type f -name "*$$PATTERN*.json" -exec stat -c "%Y %n" {} \; 2>/dev/null) | \
	                   sort -nr | head -n1 | sed 's/^[0-9]* //'); \
	if [ -n "$$LATEST_IN_BASE" ]; then \
	  echo "Found existing JSON in $(BASE_OUT): $$LATEST_IN_BASE"; \
	  base_name=$$(basename "$$LATEST_IN_BASE"); tag_name=$${base_name%.json}; \
	  echo "$$tag_name" > "$(BASE_OUT)/LAST_RUN_TAG"; \
	  $(PY) -m src.utils.collect_tau2 "$$LATEST_IN_BASE" "$(BASE_OUT)"; \
	  echo "Collected metrics and traces under $(BASE_OUT) (no copy performed)"; \
	  exit 0; \
	fi; \
	test -n "$(JSON)" || { echo "No JSONs matching *$$PATTERN*.json in $(BASE_OUT). Set JSON=<path or basename>"; exit 2; }; \
	if [ -f "$(JSON)" ]; then \
	  SRC="$(JSON)"; \
	else \
	  NAME="$$(basename "$(JSON)")"; \
	  CAND="$$(find . -type f \( -name "$$NAME" -o -name "$$NAME.json" \) -path "*/data/simulations/*" -print -quit)"; \
	  test -n "$$CAND" || { echo "Could not find '$(JSON)' under */data/simulations/"; exit 2; }; \
	  SRC="$$CAND"; \
	fi; \
	base_name="$$(basename "$$SRC")"; tag_name="$${base_name%.json}"; \
	dest="$(BASE_OUT)/$$base_name"; \
	echo "Using source: $$SRC"; \
	mkdir -p "$(BASE_OUT)"; \
	if [ -f "$$dest" ]; then \
	  echo "Warning: $$dest already exists; using existing file (no copy)."; \
	else \
	  cp "$$SRC" "$$dest"; \
	fi; \
	echo "$$tag_name" > "$(BASE_OUT)/LAST_RUN_TAG"; \
	$(PY) -m src.utils.collect_tau2 "$$dest" "$(BASE_OUT)"; \
	echo "Collected metrics and traces under $(BASE_OUT)"

# 2b) Find the latest sim JSON under */data/simulations/, copy to $(BASE_OUT) unless it already exists.
collect_latest:
	@set -e; \
	LATEST=$$((find . -type f -name "*.json" -path "*/data/simulations/*" -exec stat -f "%m %N" {} \; 2>/dev/null || \
	           find . -type f -name "*.json" -path "*/data/simulations/*" -exec stat -c "%Y %n" {} \; 2>/dev/null) | \
	           sort -nr | head -n1 | sed 's/^[0-9]* //'); \
	test -n "$$LATEST" || { echo "No simulation JSONs found under */data/simulations/"; exit 2; }; \
	base_name=$$(basename "$$LATEST"); tag_name=$${base_name%.json}; \
	dest="$(BASE_OUT)/$$base_name"; \
	echo "Latest: $$LATEST"; \
	mkdir -p "$(BASE_OUT)"; \
	if [ -f "$$dest" ]; then \
	  echo "Warning: $$dest already exists; using existing file (no copy)."; \
	else \
	  cp "$$LATEST" "$$dest"; \
	fi; \
	echo "$$tag_name" > "$(BASE_OUT)/LAST_RUN_TAG"; \
	$(PY) -m src.utils.collect_tau2 "$$dest" "$(BASE_OUT)"; \
	echo "Collected metrics and traces under $(BASE_OUT)"

# 3) Convenience: run + collect in one command.
run_collect: run collect_latest

lock:
	@$(PIP) freeze > requirements.lock.txt && echo "Wrote requirements.lock.txt"

results-clean:
	@rm -rf results/* && mkdir -p results/logs results/cache
	@echo "Results cleared."