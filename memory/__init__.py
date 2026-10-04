"""Persistent WORKFLOW memory: what the optimizer learned per task class (not user memory).

Small always-useful summary (``summary``) + persistent detailed record (``store``, one JSON file
per task class) + retrieval when needed (``warm_start``). Memory is written ONLY from
deterministic experiment results (evaluator verdict/fitness + measured usage) and MMAS
pheromone state - never by an LLM.
"""
