#!/bin/bash

tb run \
    --dataset terminal-bench-core==head \
    --agent terminus \
    --model xai/grok-4-fast-reasoning \
    --task-id hello-world