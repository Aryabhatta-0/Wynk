"""Persistent WORKFLOW memory: what the optimizer learned (not user memory).

Two layers:

* Production memory (#28): ``memory.priors`` (validity key, prior records, warm-start pin) and
  ``memory.production`` (``PriorService``) over ``store.priors.SQLitePriorStore``. Per
  workspace, learned only from verified, real, finalized experiment jobs (#24 / #26), keyed by
  their provenance, and used only by the opt-in ``aco_warm`` strategy - exactly or not at all.
  See ``docs/workflow_memory.md``.
* The legacy demo: small summary (``summary``) + one JSON file per task class (``store``) +
  ``warm_start`` (``python -m memory.warm_start --synthetic``). Kept for the synthetic demo; it
  is keyed by loose names and never consulted by production experiments.

Memory is written ONLY from deterministic experiment results (evaluator verdict/fitness +
measured usage) and MMAS pheromone state - never by an LLM.
"""
