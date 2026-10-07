# MMLU-Pro (test) protocol-v1: fixed vs random vs ACO (real model)

- task family: context-free multiple-choice reasoning (10-way classification); metric: validation accuracy
- experiment_id `c46c0e4af79cbebed3cc7be65737633e2dc6d4eb336f08fccfd65415eb181230`; protocol_id `34f3065e2718cb310fca270cf923b297bd91c52bf91682f5bc82b3262ba78265`
- fixed baseline rule `fixed_reasoning/1`
- model `google/gemma-4-31b-it` model_hash `e2cd2c74874d69e24b55f87a00a81d109309cf38a4a062e21d900c9f467a0b6e` prompts `mvp-3`
- manifest `2cd1616e8c52064d087937651b9633a86aa333166caf16bad487443a88eb2f61` dataset `3cf80a8c76d46fa25023cae8e7345b657a8642741d5f3e00c7058aa8ef0a3bb1` splits `aa0c4b3bf127bd01ab6dc05d75eca50246993082d5710f55cab2702b24d1b993` contract `e80b15b8b68b752fc51c03e6b7152203202e08ad2191df8284d9413f40e87578`
- splits {'optimization': 84, 'validation': 63, 'test': 63}; test rows executed: 0
- evaluator `classification_accuracy/1+fitness/mvp-2` config {'labels': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 'case_sensitive': False}
- latency = runtime-measured workflow time per run; exec s = summed workflow time; E2E s = strategy wall-clock (the 3 seeds ran as 3 concurrent processes); cost is n/a (no authoritative pricing)
- opt (champion) = the champion's accuracy on the optimization rows, which champion selection never reads (no best-of-N selection bias)
- artifact `experiment.json.gz` sha256 `be11c348f2a37acd2f1de1331ea3c0baf0289f656fe1d1fbe8a565ec154cb856`

## Per seed

| strategy | seed | champion workflow | stop | cand | val | opt (champion) | calls | prompt tok | compl tok | tokens | tok/ex | lat mean | lat p50 | lat p95 | lat max | exec s | E2E s | opt ovh s | eval ovh s | ex/s | cost | failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 0 | DIRECT(cot) | strategy_exhausted | 1 | 0.810 | 0.798 | 147 | 47565 | 13614 | 61179 | 416 | 8.5 | 5.4 | 26.9 | 40.4 | 1246 | 74 | 0.000 | 0.096 | 1.98 | n/a | - |
| fixed | 1 | DIRECT(cot) | strategy_exhausted | 1 | 0.857 | 0.762 | 147 | 47565 | 13999 | 61564 | 419 | 8.5 | 5.7 | 27.4 | 44.6 | 1246 | 72 | 0.000 | 0.094 | 2.03 | n/a | - |
| fixed | 2 | DIRECT(cot) | strategy_exhausted | 1 | 0.841 | 0.821 | 147 | 47565 | 14786 | 62351 | 424 | 9.6 | 6.2 | 28.2 | 108.3 | 1411 | 144 | 0.000 | 0.096 | 1.02 | n/a | - |
| random | 0 | DIRECT(cot) -> VERIFY(schema_check,retry-2) | candidate_evaluations | 6 | 0.841 | 0.821 | 967 | 304854 | 47141 | 351995 | 399 | 5.2 | 2.5 | 20.3 | 121.3 | 4576 | 393 | 0.009 | 0.525 | 2.25 | n/a | {'schema_invalid': 39} |
| random | 1 | DIRECT(cot) -> VERIFY(schema_check,retry-2) | candidate_evaluations | 6 | 0.841 | 0.798 | 964 | 303614 | 48745 | 352359 | 400 | 5.4 | 2.4 | 18.9 | 121.7 | 4750 | 380 | 0.009 | 0.527 | 2.32 | n/a | {'schema_invalid': 34} |
| random | 2 | DIRECT(cot) | candidate_evaluations | 6 | 0.825 | 0.774 | 970 | 304007 | 46810 | 350817 | 398 | 5.4 | 2.8 | 16.3 | 90.6 | 4765 | 360 | 0.004 | 0.511 | 2.45 | n/a | {'schema_invalid': 45} |
| aco | 0 | DIRECT(cot) -> VERIFY(schema_check,retry-2) | candidate_evaluations | 6 | 0.841 | 0.774 | 937 | 298677 | 71697 | 370374 | 420 | 8.4 | 4.5 | 27.1 | 113.9 | 7425 | 555 | 0.009 | 0.538 | 1.59 | n/a | {'schema_invalid': 35} |
| aco | 1 | DIRECT(cot) -> VERIFY(schema_check,retry-2) | candidate_evaluations | 6 | 0.857 | 0.794 | 967 | 305986 | 58810 | 364796 | 414 | 7.5 | 3.9 | 26.2 | 128.5 | 6659 | 629 | 0.009 | 0.534 | 1.40 | n/a | {'schema_invalid': 43} |
| aco | 2 | DIRECT(cot) -> VERIFY(schema_check,retry-2) | candidate_evaluations | 6 | 0.810 | 0.774 | 985 | 306074 | 22759 | 328833 | 373 | 4.3 | 2.9 | 11.2 | 62.7 | 3793 | 216 | 0.010 | 0.557 | 4.08 | n/a | {'schema_invalid': 49} |

## Aggregate over seeds (mean / std / min / max)

| strategy | metric | mean | std | min | max |
|---|---|---|---|---|---|
| fixed | champion_validation_score | 0.836 | 0.024 | 0.810 | 0.857 |
| fixed | champion_validation_pass_rate | 0.836 | 0.024 | 0.810 | 0.857 |
| fixed | candidate_evaluations | 1.000 | 0.000 | 1 | 1 |
| fixed | model_calls | 147.000 | 0.000 | 147 | 147 |
| fixed | prompt_tokens | 47565.000 | 0.000 | 47565 | 47565 |
| fixed | completion_tokens | 14133.000 | 597.380 | 13614 | 14786 |
| fixed | tokens | 61698.000 | 597.380 | 61179 | 62351 |
| fixed | tokens_per_example | 419.714 | 4.064 | 416.184 | 424.156 |
| fixed | tokens_per_candidate | 61698.000 | 597.380 | 61179.000 | 62351.000 |
| fixed | mean_latency_s | 8.850 | 0.647 | 8.476 | 9.597 |
| fixed | p50_latency_s | 5.755 | 0.420 | 5.369 | 6.203 |
| fixed | p95_latency_s | 27.482 | 0.649 | 26.869 | 28.162 |
| fixed | max_latency_s | 64.456 | 38.049 | 40.428 | 108.325 |
| fixed | execution_s | 1300.919 | 95.109 | 1246.002 | 1410.742 |
| fixed | mean_candidate_e2e_s | 96.969 | 41.138 | 72.237 | 144.457 |
| fixed | e2e_wall_s | 96.996 | 41.134 | 72.270 | 144.479 |
| fixed | optimizer_overhead_s | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed | evaluator_overhead_s | 0.095 | 0.001 | 0.094 | 0.096 |
| fixed | examples_per_s | 1.677 | 0.572 | 1.017 | 2.034 |
| fixed | candidates_per_min | 0.685 | 0.233 | 0.415 | 0.830 |
| fixed | cost | n/a | n/a | n/a | n/a |
| random | champion_validation_score | 0.836 | 0.009 | 0.825 | 0.841 |
| random | champion_validation_pass_rate | 0.836 | 0.009 | 0.825 | 0.841 |
| random | candidate_evaluations | 6.000 | 0.000 | 6 | 6 |
| random | model_calls | 967.000 | 3.000 | 964 | 970 |
| random | prompt_tokens | 304158.333 | 633.701 | 303614 | 304854 |
| random | completion_tokens | 47565.333 | 1034.940 | 46810 | 48745 |
| random | tokens | 351723.667 | 806.013 | 350817 | 352359 |
| random | tokens_per_example | 398.780 | 0.914 | 397.752 | 399.500 |
| random | tokens_per_candidate | 58620.611 | 134.336 | 58469.500 | 58726.500 |
| random | mean_latency_s | 5.325 | 0.119 | 5.188 | 5.402 |
| random | p50_latency_s | 2.580 | 0.226 | 2.440 | 2.840 |
| random | p95_latency_s | 18.521 | 2.033 | 16.318 | 20.324 |
| random | max_latency_s | 111.196 | 17.853 | 90.582 | 121.702 |
| random | execution_s | 4696.786 | 105.027 | 4575.819 | 4764.750 |
| random | mean_candidate_e2e_s | 62.917 | 2.737 | 60.024 | 65.467 |
| random | e2e_wall_s | 377.546 | 16.429 | 360.183 | 392.846 |
| random | optimizer_overhead_s | 0.007 | 0.003 | 0.004 | 0.009 |
| random | evaluator_overhead_s | 0.521 | 0.008 | 0.511 | 0.527 |
| random | examples_per_s | 2.339 | 0.103 | 2.245 | 2.449 |
| random | candidates_per_min | 0.955 | 0.042 | 0.916 | 0.999 |
| random | cost | n/a | n/a | n/a | n/a |
| aco | champion_validation_score | 0.836 | 0.024 | 0.810 | 0.857 |
| aco | champion_validation_pass_rate | 0.836 | 0.024 | 0.810 | 0.857 |
| aco | candidate_evaluations | 6.000 | 0.000 | 6 | 6 |
| aco | model_calls | 963.000 | 24.249 | 937 | 985 |
| aco | prompt_tokens | 303579.000 | 4245.485 | 298677 | 306074 |
| aco | completion_tokens | 51088.667 | 25366.241 | 22759 | 71697 |
| aco | tokens | 354667.667 | 22546.641 | 328833 | 370374 |
| aco | tokens_per_example | 402.118 | 25.563 | 372.827 | 419.925 |
| aco | tokens_per_candidate | 59111.278 | 3757.774 | 54805.500 | 61729.000 |
| aco | mean_latency_s | 6.756 | 2.171 | 4.300 | 8.418 |
| aco | p50_latency_s | 3.780 | 0.788 | 2.930 | 4.485 |
| aco | p95_latency_s | 21.507 | 8.945 | 11.190 | 27.104 |
| aco | max_latency_s | 101.663 | 34.550 | 62.671 | 128.467 |
| aco | execution_s | 5958.760 | 1914.646 | 3792.574 | 7424.732 |
| aco | mean_candidate_e2e_s | 77.765 | 36.672 | 36.026 | 104.822 |
| aco | e2e_wall_s | 466.628 | 220.033 | 216.200 | 628.975 |
| aco | optimizer_overhead_s | 0.009 | 0.000 | 0.009 | 0.010 |
| aco | evaluator_overhead_s | 0.543 | 0.013 | 0.534 | 0.557 |
| aco | examples_per_s | 2.357 | 1.494 | 1.402 | 4.080 |
| aco | candidates_per_min | 0.962 | 0.610 | 0.572 | 1.665 |
| aco | cost | n/a | n/a | n/a | n/a |

Highest mean champion validation accuracy: aco, fixed, random
