# tau2-bench vs terminal-bench

## Setup friction
- tau2: Requires Python 3.10+. Installed with pip install -e. Uses LiteLLM for provider keys via a .env file. CLI supports --max-concurrency.
- terminal: Installed  with pip install -e. Requires Docker and git. CLI supports --n-concurrent and dataset selection flags.

## Docs and scoring clarity
- tau2: Clear README, domain docs viewer (tau2 domain opens ReDoc) and a leaderboard submission tool that computes Pass^k and validates runs across retail, airline, telecom with consistent model settings. Paper explains the dual-control setup and a compositional task generator for Telecom.
- terminal: Dedicated docs site with quickstart, install, dataset versioning, and a gallery. Scoring is pass/fail per task using test scripts included with each task.

## Determinism and reproducibility
- tau2: 3 airline tasks with 3 trials each. Observed variation in accuracy and cost
		🏆 Average Reward: 0.7778
		📈 Pass^k Metrics:
		k=1: 0.778
		k=2: 0.667
		k=3: 0.667
		💰 Average Cost per Conversation: $0.0134

		🏆 Average Reward: 0.6667
		📈 Pass^k Metrics:
		k=1: 0.667
		k=2: 0.667
		k=3: 0.667
		💰 Average Cost per Conversation: $0.0144
- terminal: cobol-modernization, pytorch-model-recovery, feal-differential-cryptanalysis with 3 trials each. Concurrency is supported (--n-concurrent), but Docker exceptions are consistent with what others hit in high-concurrency Docker setups. Observed variation in accuracy and cost. Observed Harness execution failed (more with concurrency).
		+-------------------+---------+
		| Metric            | Value   |
		+===================+=========+
		| Resolved Trials   | 2       |
		+-------------------+---------+
		| Unresolved Trials | 7       |
		+-------------------+---------+
		| Accuracy          | 22.22%  |
		+-------------------+---------+
		| Pass@2            | 33.33%  |
		+-------------------+---------+
		+-------------------+---------+
		| Metric            | Value   |
		+===================+=========+
		| Resolved Trials   | 3       |
		+-------------------+---------+
		| Unresolved Trials | 6       |
		+-------------------+---------+
		| Accuracy          | 33.33%  |
		+-------------------+---------+
		| Pass@2            | 55.56%  |
		+-------------------+---------+

## Extensibility for custom 10-case benchmark
- tau2: a) sample or remix existing airline/retail/mock tasks; run with --task-ids for your 10-case set during iteration, then re-run full domain for final metrics.
        b) implement a small compositional generator (2–3 atomic operations × constraints) that emits 10 verifiable tasks plus a lightweight checker. Position this as a blueprint for scaling.
- terminal: Use the Task Wizard to scaffold 10 tasks in minutes; each has Dockerfile, oracle solution, and verifier. This gives you airtight pass/fail and reproducibility.

## Grok initial behavior
- tau2: concurrent query work out of the box, faster and cheaper processing per task
- terminal: concurrent query will require investigation, slower and more expensive

## Early risks
- tau2: higher initial accuracy, might be harder to critique. A case study shows prompt policy wording moved Pass^k for a small model by ~20%+, which suggests prompt sensitivity
- terminal: longer turn-around time per set of experiments. Good adapter design and task plumbing matter as much as raw model choice. Useful for “real-world applicability” section.

## Provisional choice and why
- I will likely choose: tau2
- Rationale:

Analytical depth & critical thinking: tau2’s dual-control setup lets you analyze not just tool-use but how well Grok instructs and coordinates with a user, which is rich for failure-mode taxonomy.

Practical feasibility: Faster to run, simpler setup, easier to pin trials and reproduce results in the time you have.

Innovation: You can prototype a mini dual-control extension or a prompt-policy ablation to show creative improvements to methodology and metrics. The community prompt-sensitivity finding gives you a strong hook.
