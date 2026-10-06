# MuSiQue-Answerable: fixed vs random vs ACO (real model)

- experiment_id `ca68bbecca589fdee00f6c16b6078446ba5ad8f37c6691cec5117725c713c542`
- model `google/gemma-4-31b-it` model_hash `e2cd2c74874d69e24b55f87a00a81d109309cf38a4a062e21d900c9f467a0b6e` prompts `mvp-3`
- manifest `c89704f22ef786e7394704e59ca56164f3c3cfdb34bedd7b917ac0a5ceab5c42` dataset `1d4982fa082e231678437f7caa7fc0128bd2cef4792496ebc1515c8458aafd23` splits `470c3d9a51038efa27badfc833dc0321812fd4ddc0d19d110b3d100dfbd0f1b3` contract `38091e582363685925cd08c9c0f49a2f5c11e7b6e62ed80bfc2eed83518755f4`
- splits {'optimization': 24, 'validation': 18, 'test': 18}; test rows executed: 0
- evaluator `token_f1/1+fitness/mvp-2` config {'pass_threshold': 0.5}

## Per seed

| strategy | seed | stop | cand | val F1 | val pass | calls | prompt tok | compl tok | tokens | tok/cand | mean lat s | p95 lat s | exec s | E2E s | opt ovh s | eval ovh s | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 0 | strategy_exhausted | 1 | 0.000 | 0.000 | 42 | 5549 | 306 | 5855 | 5855 | 3.9 | 8.7 | 164 | 10 | 0.00 | 0.036 | n/a |
| fixed | 1 | strategy_exhausted | 1 | 0.000 | 0.000 | 42 | 5549 | 314 | 5863 | 5863 | 8.2 | 23.0 | 345 | 30 | 0.00 | 0.035 | n/a |
| fixed | 2 | strategy_exhausted | 1 | 0.000 | 0.000 | 42 | 5549 | 330 | 5879 | 5879 | 6.5 | 19.8 | 271 | 32 | 0.00 | 0.037 | n/a |
| random | 0 | candidate_evaluations | 6 | 0.606 | 0.611 | 474 | 324382 | 11448 | 335830 | 55972 | 8.2 | 20.6 | 2068 | 141 | 0.05 | 0.222 | n/a |
| random | 1 | candidate_evaluations | 6 | 0.676 | 0.722 | 519 | 481760 | 16612 | 498372 | 83062 | 9.7 | 16.2 | 2434 | 276 | 0.07 | 0.224 | n/a |
| random | 2 | candidate_evaluations | 6 | 0.676 | 0.722 | 686 | 439839 | 15220 | 455059 | 75843 | 10.2 | 18.9 | 2579 | 171 | 0.07 | 0.223 | n/a |
| aco | 0 | candidate_evaluations | 6 | 0.620 | 0.667 | 537 | 330435 | 13966 | 344401 | 57400 | 8.9 | 17.9 | 2250 | 229 | 0.05 | 0.237 | n/a |
| aco | 1 | candidate_evaluations | 6 | 0.676 | 0.722 | 536 | 270851 | 12982 | 283833 | 47306 | 7.9 | 16.4 | 1988 | 175 | 0.06 | 0.217 | n/a |
| aco | 2 | candidate_evaluations | 6 | 0.676 | 0.722 | 528 | 475819 | 16479 | 492298 | 82050 | 9.0 | 19.3 | 2266 | 154 | 0.07 | 0.228 | n/a |

## Aggregate over seeds (mean / std / median / min / max)

| strategy | metric | mean | std | median | min | max |
|---|---|---|---|---|---|---|
| fixed | champion_validation_score | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed | champion_validation_pass_rate | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed | candidate_evaluations | 1.000 | 0.000 | 1 | 1 | 1 |
| fixed | model_calls | 42.000 | 0.000 | 42 | 42 | 42 |
| fixed | tokens | 5865.667 | 12.220 | 5863 | 5855 | 5879 |
| fixed | mean_latency_s | 6.194 | 2.161 | 6.451 | 3.916 | 8.214 |
| fixed | p95_latency_s | 17.178 | 7.521 | 19.809 | 8.696 | 23.030 |
| fixed | e2e_wall_s | 23.879 | 12.457 | 30.073 | 9.539 | 32.025 |
| fixed | cost | n/a | n/a | n/a | n/a | n/a |
| random | champion_validation_score | 0.652 | 0.041 | 0.676 | 0.606 | 0.676 |
| random | champion_validation_pass_rate | 0.685 | 0.064 | 0.722 | 0.611 | 0.722 |
| random | candidate_evaluations | 6.000 | 0.000 | 6 | 6 | 6 |
| random | model_calls | 559.667 | 111.698 | 519 | 474 | 686 |
| random | tokens | 429753.667 | 84173.900 | 455059 | 335830 | 498372 |
| random | mean_latency_s | 9.366 | 1.045 | 9.658 | 8.206 | 10.234 |
| random | p95_latency_s | 18.591 | 2.237 | 18.915 | 16.209 | 20.648 |
| random | e2e_wall_s | 196.032 | 71.104 | 170.639 | 141.110 | 276.346 |
| random | cost | n/a | n/a | n/a | n/a | n/a |
| aco | champion_validation_score | 0.657 | 0.032 | 0.676 | 0.620 | 0.676 |
| aco | champion_validation_pass_rate | 0.704 | 0.032 | 0.722 | 0.667 | 0.722 |
| aco | candidate_evaluations | 6.000 | 0.000 | 6 | 6 | 6 |
| aco | model_calls | 533.667 | 4.933 | 536 | 528 | 537 |
| aco | tokens | 373510.667 | 107237.790 | 344401 | 283833 | 492298 |
| aco | mean_latency_s | 8.604 | 0.621 | 8.930 | 7.887 | 8.993 |
| aco | p95_latency_s | 17.861 | 1.443 | 17.936 | 16.381 | 19.265 |
| aco | e2e_wall_s | 185.800 | 38.711 | 175.031 | 153.613 | 228.755 |
| aco | cost | n/a | n/a | n/a | n/a | n/a |

Highest mean champion validation token F1: aco

## Official MuSiQue answer F1 cross-check (champion validation predictions)

| strategy | seed | Wynk token F1 (answer only) | official answer F1 (answer + aliases) | official EM |
|---|---|---|---|---|
| aco | 0 | 0.620 | 0.620 | 0.556 |
| aco | 1 | 0.676 | 0.676 | 0.611 |
| aco | 2 | 0.676 | 0.676 | 0.611 |
| fixed | 0 | 0.000 | 0.000 | 0.000 |
| fixed | 1 | 0.000 | 0.000 | 0.000 |
| fixed | 2 | 0.000 | 0.000 | 0.000 |
| random | 0 | 0.606 | 0.606 | 0.556 |
| random | 1 | 0.676 | 0.676 | 0.611 |
| random | 2 | 0.676 | 0.676 | 0.611 |
