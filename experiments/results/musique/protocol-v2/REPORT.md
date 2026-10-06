# MuSiQue-Answerable protocol-v2: fixed vs random vs ACO (real model)

- experiment_id `dbec36078c78a98e9453b56bada5ba5e58df25b4e122e7afc5b3a6405f54b968`
- fixed baseline rule `fixed_context/1`; protocol_id `7bc15caa8f41fed458c3db47e272c0e4f6f9e2c7bdfee5adefa38cf9562daf9d`
- latency = runtime-measured workflow time per run; cand E2E = mean wall-clock time per candidate; E2E = strategy wall-clock time (3 seeds ran as 3 concurrent processes)
- model `google/gemma-4-31b-it` model_hash `e2cd2c74874d69e24b55f87a00a81d109309cf38a4a062e21d900c9f467a0b6e` prompts `mvp-3`
- manifest `c89704f22ef786e7394704e59ca56164f3c3cfdb34bedd7b917ac0a5ceab5c42` dataset `1d4982fa082e231678437f7caa7fc0128bd2cef4792496ebc1515c8458aafd23` splits `470c3d9a51038efa27badfc833dc0321812fd4ddc0d19d110b3d100dfbd0f1b3` contract `38091e582363685925cd08c9c0f49a2f5c11e7b6e62ed80bfc2eed83518755f4`
- splits {'optimization': 24, 'validation': 18, 'test': 18}; test rows executed: 0
- evaluator `token_f1/1+fitness/mvp-2` config {'pass_threshold': 0.5}

## Per seed

| strategy | seed | stop | cand | val F1 | val pass | calls | prompt tok | compl tok | tokens | tok/ex | lat mean | lat p50 | lat p95 | lat max | exec s | cand E2E s | E2E s | opt ovh s | eval ovh s | ex/s | cand/min | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 0 | strategy_exhausted | 1 | 0.657 | 0.722 | 84 | 112723 | 3066 | 115789 | 2757 | 9.9 | 7.9 | 18.9 | 23.0 | 416 | 27.1 | 27 | 0.000 | 0.040 | 1.55 | 2.22 | n/a |
| fixed | 1 | strategy_exhausted | 1 | 0.676 | 0.722 | 84 | 112724 | 3037 | 115761 | 2756 | 12.5 | 9.6 | 27.2 | 31.1 | 523 | 31.7 | 32 | 0.000 | 0.042 | 1.33 | 1.89 | n/a |
| fixed | 2 | strategy_exhausted | 1 | 0.676 | 0.722 | 84 | 112725 | 3040 | 115765 | 2756 | 12.4 | 12.2 | 20.6 | 27.3 | 521 | 40.7 | 41 | 0.000 | 0.041 | 1.03 | 1.48 | n/a |
| random | 0 | candidate_evaluations | 6 | 0.606 | 0.611 | 476 | 323518 | 12036 | 335554 | 1332 | 7.4 | 6.2 | 14.8 | 55.1 | 1873 | 31.6 | 189 | 0.049 | 0.229 | 1.33 | 1.90 | n/a |
| random | 1 | candidate_evaluations | 6 | 0.676 | 0.722 | 517 | 481985 | 16617 | 498602 | 1979 | 9.2 | 7.9 | 19.0 | 81.6 | 2321 | 37.3 | 224 | 0.065 | 0.227 | 1.12 | 1.61 | n/a |
| random | 2 | candidate_evaluations | 6 | 0.676 | 0.722 | 691 | 437318 | 15595 | 452913 | 1797 | 11.5 | 9.0 | 29.6 | 82.4 | 2898 | 37.6 | 226 | 0.073 | 0.218 | 1.12 | 1.59 | n/a |
| aco | 0 | candidate_evaluations | 6 | 0.620 | 0.667 | 542 | 332970 | 13727 | 346697 | 1376 | 8.8 | 6.9 | 20.7 | 90.3 | 2223 | 44.7 | 268 | 0.055 | 0.233 | 0.94 | 1.34 | n/a |
| aco | 1 | candidate_evaluations | 6 | 0.676 | 0.722 | 545 | 270404 | 12731 | 283135 | 1124 | 9.5 | 6.9 | 21.8 | 91.8 | 2399 | 45.6 | 274 | 0.057 | 0.212 | 0.92 | 1.31 | n/a |
| aco | 2 | candidate_evaluations | 6 | 0.676 | 0.722 | 527 | 473194 | 16315 | 489509 | 1942 | 15.3 | 7.4 | 84.8 | 112.4 | 3859 | 51.1 | 307 | 0.071 | 0.239 | 0.82 | 1.17 | n/a |

## Aggregate over seeds (mean / std / median / min / max)

| strategy | metric | mean | std | median | min | max |
|---|---|---|---|---|---|---|
| fixed | champion_validation_score | 0.670 | 0.011 | 0.676 | 0.657 | 0.676 |
| fixed | champion_validation_pass_rate | 0.722 | 0.000 | 0.722 | 0.722 | 0.722 |
| fixed | candidate_evaluations | 1.000 | 0.000 | 1 | 1 | 1 |
| fixed | model_calls | 84.000 | 0.000 | 84 | 84 | 84 |
| fixed | prompt_tokens | 112724.000 | 1.000 | 112724 | 112723 | 112725 |
| fixed | completion_tokens | 3047.667 | 15.948 | 3040 | 3037 | 3066 |
| fixed | tokens | 115771.667 | 15.144 | 115765 | 115761 | 115789 |
| fixed | tokens_per_example | 2756.468 | 0.361 | 2756.310 | 2756.214 | 2756.881 |
| fixed | tokens_per_candidate | 115771.667 | 15.144 | 115765.000 | 115761.000 | 115789.000 |
| fixed | mean_latency_s | 11.593 | 1.456 | 12.403 | 9.912 | 12.463 |
| fixed | p50_latency_s | 9.926 | 2.176 | 9.649 | 7.901 | 12.227 |
| fixed | p95_latency_s | 22.262 | 4.391 | 20.627 | 18.922 | 27.236 |
| fixed | max_latency_s | 27.135 | 4.098 | 27.304 | 22.955 | 31.146 |
| fixed | execution_s | 486.897 | 61.146 | 520.921 | 416.306 | 523.463 |
| fixed | mean_candidate_e2e_s | 33.135 | 6.910 | 31.675 | 27.071 | 40.658 |
| fixed | e2e_wall_s | 33.147 | 6.907 | 31.686 | 27.087 | 40.667 |
| fixed | optimizer_overhead_s | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed | evaluator_overhead_s | 0.041 | 0.001 | 0.041 | 0.040 | 0.042 |
| fixed | examples_per_s | 1.303 | 0.260 | 1.326 | 1.033 | 1.551 |
| fixed | candidates_per_min | 1.861 | 0.371 | 1.894 | 1.475 | 2.215 |
| fixed | cost | n/a | n/a | n/a | n/a | n/a |
| random | champion_validation_score | 0.652 | 0.041 | 0.676 | 0.606 | 0.676 |
| random | champion_validation_pass_rate | 0.685 | 0.064 | 0.722 | 0.611 | 0.722 |
| random | candidate_evaluations | 6.000 | 0.000 | 6 | 6 | 6 |
| random | model_calls | 561.333 | 114.150 | 517 | 476 | 691 |
| random | prompt_tokens | 414273.667 | 81708.191 | 437318 | 323518 | 481985 |
| random | completion_tokens | 14749.333 | 2404.736 | 15595 | 12036 | 16617 |
| random | tokens | 429023.000 | 84108.333 | 452913 | 335554 | 498602 |
| random | tokens_per_example | 1702.472 | 333.763 | 1797.274 | 1331.563 | 1978.579 |
| random | tokens_per_candidate | 71503.833 | 14018.055 | 75485.500 | 55925.667 | 83100.333 |
| random | mean_latency_s | 9.382 | 2.038 | 9.210 | 7.434 | 11.500 |
| random | p50_latency_s | 7.689 | 1.392 | 7.901 | 6.203 | 8.963 |
| random | p95_latency_s | 21.106 | 7.616 | 18.957 | 14.795 | 29.565 |
| random | max_latency_s | 73.036 | 15.518 | 81.576 | 55.123 | 82.408 |
| random | execution_s | 2364.142 | 513.696 | 2321.010 | 1873.370 | 2898.044 |
| random | mean_candidate_e2e_s | 35.501 | 3.420 | 37.337 | 31.554 | 37.610 |
| random | e2e_wall_s | 213.095 | 20.532 | 224.116 | 189.406 | 225.764 |
| random | optimizer_overhead_s | 0.062 | 0.012 | 0.065 | 0.049 | 0.073 |
| random | evaluator_overhead_s | 0.225 | 0.006 | 0.227 | 0.218 | 0.229 |
| random | examples_per_s | 1.190 | 0.121 | 1.124 | 1.116 | 1.330 |
| random | candidates_per_min | 1.701 | 0.173 | 1.606 | 1.595 | 1.901 |
| random | cost | n/a | n/a | n/a | n/a | n/a |
| aco | champion_validation_score | 0.657 | 0.032 | 0.676 | 0.620 | 0.676 |
| aco | champion_validation_pass_rate | 0.704 | 0.032 | 0.722 | 0.667 | 0.722 |
| aco | candidate_evaluations | 6.000 | 0.000 | 6 | 6 | 6 |
| aco | model_calls | 538.000 | 9.644 | 542 | 527 | 545 |
| aco | prompt_tokens | 358856.000 | 103843.679 | 332970 | 270404 | 473194 |
| aco | completion_tokens | 14257.667 | 1849.992 | 13727 | 12731 | 16315 |
| aco | tokens | 373113.667 | 105692.654 | 346697 | 283135 | 489509 |
| aco | tokens_per_example | 1480.610 | 419.415 | 1375.782 | 1123.552 | 1942.496 |
| aco | tokens_per_candidate | 62185.611 | 17615.442 | 57782.833 | 47189.167 | 81584.833 |
| aco | mean_latency_s | 11.218 | 3.563 | 9.520 | 8.822 | 15.312 |
| aco | p50_latency_s | 7.061 | 0.270 | 6.924 | 6.887 | 7.372 |
| aco | p95_latency_s | 42.440 | 36.719 | 21.823 | 20.664 | 84.834 |
| aco | max_latency_s | 98.168 | 12.349 | 91.829 | 90.276 | 112.400 |
| aco | execution_s | 2826.974 | 897.840 | 2399.146 | 2223.064 | 3858.714 |
| aco | mean_candidate_e2e_s | 47.148 | 3.491 | 45.649 | 44.658 | 51.139 |
| aco | e2e_wall_s | 282.985 | 20.961 | 273.981 | 268.030 | 306.944 |
| aco | optimizer_overhead_s | 0.061 | 0.009 | 0.057 | 0.055 | 0.071 |
| aco | evaluator_overhead_s | 0.228 | 0.014 | 0.233 | 0.212 | 0.239 |
| aco | examples_per_s | 0.894 | 0.064 | 0.920 | 0.821 | 0.940 |
| aco | candidates_per_min | 1.277 | 0.091 | 1.314 | 1.173 | 1.343 |
| aco | cost | n/a | n/a | n/a | n/a | n/a |

Highest mean champion validation token F1: fixed

## Official MuSiQue answer F1 cross-check (champion validation predictions)

| strategy | seed | Wynk token F1 (answer only) | official answer F1 (answer + aliases) | official EM |
|---|---|---|---|---|
| aco | 0 | 0.620 | 0.620 | 0.556 |
| aco | 1 | 0.676 | 0.676 | 0.611 |
| aco | 2 | 0.676 | 0.676 | 0.611 |
| fixed | 0 | 0.657 | 0.657 | 0.556 |
| fixed | 1 | 0.676 | 0.676 | 0.611 |
| fixed | 2 | 0.676 | 0.676 | 0.611 |
| random | 0 | 0.606 | 0.606 | 0.556 |
| random | 1 | 0.676 | 0.676 | 0.611 |
| random | 2 | 0.676 | 0.676 | 0.611 |
