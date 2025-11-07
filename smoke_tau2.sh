#!/bin/bash

tau2 run \
  --domain airline \
  --agent-llm xai/grok-4-fast-reasoning \
  --user-llm xai/grok-4-fast-reasoning \
  --num-trials 1 \
  --num-tasks 1 \
  --max-concurrency 1