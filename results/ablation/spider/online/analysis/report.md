# Ablation analysis — `results/ablation/spider/online`

## Main results (mean ± std over processing orders)

| Config | orders | EM | EX | LLM calls/q | tokens in/q | tokens out/q | s/q |
|---|---|---|---|---|---|---|---|
| template_only | 1 | 73.5 | 84.5 | 1.02 | 2921.64 | 31.59 | 7.95 |
| semantic_layer | 1 | 72.6 | 83.8 | 1.03 | 2953.76 | 32.02 | 8.29 |
| semantic_rag | 1 | 73.6 | 85.0 | 1.02 | 2948.18 | 32.40 | 5.86 |
| reasoning_bank | 2 | 73.8 | 84.5 ± 0.1 | 5.11 | 14591.75 | 157.40 | 39.25 |
| reasoning_bank (majority vote) | 2 | 73.9 ± 0.1 | 84.8 ± 0.3 | 5.11 | 14591.75 | 157.40 | 39.25 |
| layer_rag | 1 | 72.7 | 84.2 | 1.02 | 2982.62 | 32.06 | 5.80 |
| rag_rb | 2 | 73.7 ± 0.1 | 84.9 | 5.11 | 14740.23 | 159.96 | 31.81 |
| rag_rb (majority vote) | 2 | 74.0 ± 0.1 | 85.1 ± 0.1 | 5.11 | 14740.23 | 159.96 | 31.81 |
| full | 2 | 72.6 ± 0.2 | 84.4 ± 0.3 | 5.16 | 14929.37 | 161.17 | 31.43 |
| full (majority vote) | 2 | 72.6 | 84.6 ± 0.1 | 5.16 | 14929.37 | 161.17 | 31.43 |

## Exact McNemar tests (paired, two-sided)

| Config | vs | order | metric | acc | acc ref | only ref ✓ | only config ✓ | p |
|---|---|---|---|---|---|---|---|---|
| template_only | full | none | EM | 73.5 | 72.4 | 25 | 36 | 0.2 |
| template_only | full | none | EX | 84.5 | 84.1 | 20 | 24 | 0.652 |
| semantic_layer | full | none | EM | 72.6 | 72.4 | 14 | 16 | 0.856 |
| semantic_layer | full | none | EX | 83.8 | 84.1 | 14 | 10 | 0.541 |
| semantic_rag | full | none | EM | 73.6 | 72.4 | 16 | 28 | 0.0961 |
| semantic_rag | full | none | EX | 85.0 | 84.1 | 12 | 21 | 0.163 |
| reasoning_bank | full | 1 | EM | 73.8 | 72.7 | 26 | 37 | 0.207 |
| reasoning_bank | full | 1 | EX | 84.6 | 84.6 | 20 | 20 | 1 |
| reasoning_bank | full | none | EM | 73.8 | 72.4 | 21 | 35 | 0.0814 |
| reasoning_bank | full | none | EX | 84.4 | 84.1 | 22 | 25 | 0.771 |
| layer_rag | full | none | EM | 72.7 | 72.4 | 7 | 10 | 0.629 |
| layer_rag | full | none | EX | 84.2 | 84.1 | 7 | 8 | 1 |
| rag_rb | full | 1 | EM | 73.8 | 72.7 | 16 | 27 | 0.126 |
| rag_rb | full | 1 | EX | 84.9 | 84.6 | 16 | 19 | 0.736 |
| rag_rb | full | none | EM | 73.7 | 72.4 | 17 | 30 | 0.0789 |
| rag_rb | full | none | EX | 84.9 | 84.1 | 14 | 22 | 0.243 |
| reasoning_bank (majority vote) | reasoning_bank (greedy) | 1 | EM | 74.0 | 73.8 | 1 | 3 | 0.625 |
| reasoning_bank (majority vote) | reasoning_bank (greedy) | 1 | EX | 85.0 | 84.6 | 2 | 6 | 0.289 |
| reasoning_bank (majority vote) | reasoning_bank (greedy) | none | EM | 73.9 | 73.8 | 0 | 1 | 1 |
| reasoning_bank (majority vote) | reasoning_bank (greedy) | none | EX | 84.5 | 84.4 | 2 | 3 | 1 |
| rag_rb (majority vote) | rag_rb (greedy) | 1 | EM | 74.0 | 73.8 | 1 | 3 | 0.625 |
| rag_rb (majority vote) | rag_rb (greedy) | 1 | EX | 85.0 | 84.9 | 3 | 4 | 1 |
| rag_rb (majority vote) | rag_rb (greedy) | none | EM | 74.1 | 73.7 | 1 | 5 | 0.219 |
| rag_rb (majority vote) | rag_rb (greedy) | none | EX | 85.2 | 84.9 | 3 | 6 | 0.508 |
| full (majority vote) | full (greedy) | 1 | EM | 72.6 | 72.7 | 2 | 1 | 1 |
| full (majority vote) | full (greedy) | 1 | EX | 84.5 | 84.6 | 3 | 2 | 1 |
| full (majority vote) | full (greedy) | none | EM | 72.6 | 72.4 | 1 | 3 | 0.625 |
| full (majority vote) | full (greedy) | none | EX | 84.6 | 84.1 | 1 | 6 | 0.125 |

## Per difficulty (official Spider hardness)

| Config | Level | n | EM | EX |
|---|---|---|---|---|
| template_only | easy | 248 | 87.1 | 91.9 |
| template_only | medium | 446 | 81.8 | 91.3 |
| template_only | hard | 174 | 58.0 | 77.0 |
| template_only | extra | 166 | 47.0 | 63.3 |
| template_only | all | 1034 | 73.5 | 84.5 |
| semantic_layer | easy | 248 | 87.5 | 91.5 |
| semantic_layer | medium | 446 | 81.8 | 90.6 |
| semantic_layer | hard | 174 | 55.2 | 71.3 |
| semantic_layer | extra | 166 | 44.0 | 66.9 |
| semantic_layer | all | 1034 | 72.6 | 83.8 |
| semantic_rag | easy | 248 | 86.7 | 91.9 |
| semantic_rag | medium | 446 | 82.1 | 91.0 |
| semantic_rag | hard | 174 | 58.0 | 76.4 |
| semantic_rag | extra | 166 | 47.6 | 67.5 |
| semantic_rag | all | 1034 | 73.6 | 85.0 |
| reasoning_bank | easy | 248 | 88.1 ± 0.3 | 92.1 ± 0.3 |
| reasoning_bank | medium | 446 | 82.3 ± 0.3 | 90.9 ± 0.2 |
| reasoning_bank | hard | 174 | 58.6 | 76.7 ± 0.4 |
| reasoning_bank | extra | 166 | 45.5 ± 0.4 | 64.2 ± 0.4 |
| reasoning_bank | all | 1034 | 73.8 | 84.5 ± 0.1 |
| layer_rag | easy | 248 | 86.7 | 91.5 |
| layer_rag | medium | 446 | 81.4 | 90.8 |
| layer_rag | hard | 174 | 56.3 | 73.0 |
| layer_rag | extra | 166 | 45.8 | 67.5 |
| layer_rag | all | 1034 | 72.7 | 84.2 |
| rag_rb | easy | 248 | 87.3 ± 0.3 | 92.5 ± 0.3 |
| rag_rb | medium | 446 | 82.2 ± 0.2 | 91.1 ± 0.2 |
| rag_rb | hard | 174 | 57.8 ± 0.4 | 75.9 |
| rag_rb | extra | 166 | 47.6 | 66.3 ± 0.9 |
| rag_rb | all | 1034 | 73.7 ± 0.1 | 84.9 |
| full | easy | 248 | 86.9 ± 0.3 | 91.7 ± 0.3 |
| full | medium | 446 | 81.2 | 90.6 ± 0.3 |
| full | hard | 174 | 56.6 ± 0.4 | 73.0 |
| full | extra | 166 | 44.9 ± 1.3 | 68.7 ± 0.9 |
| full | all | 1034 | 72.6 ± 0.2 | 84.4 ± 0.3 |

## Pseudo-label reliability (self-consistency vs gold EX, computed after the run)

| Config | order | n | precision(consensus) | recall(consensus) | false consensus | consensus on empty: n / precision | non-empty precision |
|---|---|---|---|---|---|---|---|
| reasoning_bank | 1 | 1034 | 85.4 | 99.5 | 14.4 | 67 / 68.7 | 86.6 |
| reasoning_bank | none | 1034 | 84.9 | 99.7 | 15.0 | 66 / 68.2 | 86.0 |
| rag_rb | 1 | 1034 | 85.7 | 99.7 | 14.1 | 66 / 69.7 | 86.8 |
| rag_rb | none | 1034 | 85.8 | 99.4 | 13.9 | 66 / 68.2 | 87.1 |
| full | 1 | 1034 | 84.9 | 99.3 | 15.0 | 67 / 70.1 | 85.9 |
| full | none | 1034 | 84.7 | 99.9 | 15.2 | 71 / 66.2 | 86.1 |

| Config | order | outcome | n | share | gold EX rate |
|---|---|---|---|---|---|
| reasoning_bank | 1 | consensus | 1020 | 98.6 | 85.4 |
| reasoning_bank | 1 | low_agreement | 2 | 0.2 | 100.0 |
| reasoning_bank | 1 | minority | 10 | 1.0 | 20.0 |
| reasoning_bank | 1 | no_exec | 2 | 0.2 | 0.0 |
| reasoning_bank | none | consensus | 1025 | 99.1 | 84.9 |
| reasoning_bank | none | low_agreement | 2 | 0.2 | 50.0 |
| reasoning_bank | none | minority | 6 | 0.6 | 33.3 |
| reasoning_bank | none | no_exec | 1 | 0.1 | 0.0 |
| rag_rb | 1 | consensus | 1021 | 98.7 | 85.7 |
| rag_rb | 1 | low_agreement | 5 | 0.5 | 20.0 |
| rag_rb | 1 | minority | 7 | 0.7 | 28.6 |
| rag_rb | 1 | no_exec | 1 | 0.1 | 0.0 |
| rag_rb | none | consensus | 1017 | 98.4 | 85.8 |
| rag_rb | none | low_agreement | 5 | 0.5 | 60.0 |
| rag_rb | none | minority | 10 | 1.0 | 20.0 |
| rag_rb | none | no_exec | 2 | 0.2 | 0.0 |
| full | 1 | consensus | 1024 | 99.0 | 84.9 |
| full | 1 | low_agreement | 2 | 0.2 | 100.0 |
| full | 1 | minority | 8 | 0.8 | 50.0 |
| full | none | consensus | 1026 | 99.2 | 84.7 |
| full | none | low_agreement | 1 | 0.1 | 0.0 |
| full | none | minority | 7 | 0.7 | 14.3 |

## Learning curve (EX per block of 100 questions, processing order)

| Config | order | steps | EX | share with ≥1 strategy |
|---|---|---|---|---|
| reasoning_bank | 1 | 0-99 | 83.0 | 0.0 |
| reasoning_bank | 1 | 100-199 | 83.0 | 2.0 |
| reasoning_bank | 1 | 200-299 | 85.0 | 2.0 |
| reasoning_bank | 1 | 300-399 | 84.0 | 0.0 |
| reasoning_bank | 1 | 400-499 | 92.0 | 1.0 |
| reasoning_bank | 1 | 500-599 | 90.0 | 0.0 |
| reasoning_bank | 1 | 600-699 | 81.0 | 0.0 |
| reasoning_bank | 1 | 700-799 | 82.0 | 0.0 |
| reasoning_bank | 1 | 800-899 | 85.0 | 1.0 |
| reasoning_bank | 1 | 900-999 | 84.0 | 0.0 |
| reasoning_bank | 1 | 1000-1033 | 76.5 | 0.0 |
| reasoning_bank | none | 0-99 | 96.0 | 0.0 |
| reasoning_bank | none | 100-199 | 81.0 | 1.0 |
| reasoning_bank | none | 200-299 | 86.0 | 0.0 |
| reasoning_bank | none | 300-399 | 91.0 | 1.0 |
| reasoning_bank | none | 400-499 | 76.0 | 4.0 |
| reasoning_bank | none | 500-599 | 79.0 | 3.0 |
| reasoning_bank | none | 600-699 | 90.0 | 1.0 |
| reasoning_bank | none | 700-799 | 80.0 | 0.0 |
| reasoning_bank | none | 800-899 | 81.0 | 1.0 |
| reasoning_bank | none | 900-999 | 79.0 | 1.0 |
| reasoning_bank | none | 1000-1033 | 100.0 | 0.0 |
| rag_rb | 1 | 0-99 | 84.0 | 0.0 |
| rag_rb | 1 | 100-199 | 83.0 | 2.0 |
| rag_rb | 1 | 200-299 | 87.0 | 1.0 |
| rag_rb | 1 | 300-399 | 84.0 | 1.0 |
| rag_rb | 1 | 400-499 | 90.0 | 1.0 |
| rag_rb | 1 | 500-599 | 91.0 | 0.0 |
| rag_rb | 1 | 600-699 | 81.0 | 0.0 |
| rag_rb | 1 | 700-799 | 83.0 | 2.0 |
| rag_rb | 1 | 800-899 | 85.0 | 1.0 |
| rag_rb | 1 | 900-999 | 84.0 | 0.0 |
| rag_rb | 1 | 1000-1033 | 76.5 | 2.9 |
| rag_rb | none | 0-99 | 96.0 | 0.0 |
| rag_rb | none | 100-199 | 80.0 | 1.0 |
| rag_rb | none | 200-299 | 86.0 | 0.0 |
| rag_rb | none | 300-399 | 90.0 | 1.0 |
| rag_rb | none | 400-499 | 76.0 | 4.0 |
| rag_rb | none | 500-599 | 81.0 | 3.0 |
| rag_rb | none | 600-699 | 91.0 | 1.0 |
| rag_rb | none | 700-799 | 82.0 | 0.0 |
| rag_rb | none | 800-899 | 83.0 | 0.0 |
| rag_rb | none | 900-999 | 81.0 | 2.0 |
| rag_rb | none | 1000-1033 | 94.1 | 0.0 |
| full | 1 | 0-99 | 86.0 | 0.0 |
| full | 1 | 100-199 | 81.0 | 2.0 |
| full | 1 | 200-299 | 86.0 | 1.0 |
| full | 1 | 300-399 | 85.0 | 1.0 |
| full | 1 | 400-499 | 92.0 | 1.0 |
| full | 1 | 500-599 | 90.0 | 0.0 |
| full | 1 | 600-699 | 82.0 | 0.0 |
| full | 1 | 700-799 | 80.0 | 0.0 |
| full | 1 | 800-899 | 84.0 | 2.0 |
| full | 1 | 900-999 | 83.0 | 1.0 |
| full | 1 | 1000-1033 | 76.5 | 5.9 |
| full | none | 0-99 | 93.0 | 3.0 |
| full | none | 100-199 | 80.0 | 4.0 |
| full | none | 200-299 | 86.0 | 0.0 |
| full | none | 300-399 | 88.0 | 1.0 |
| full | none | 400-499 | 77.0 | 3.0 |
| full | none | 500-599 | 80.0 | 2.0 |
| full | none | 600-699 | 90.0 | 1.0 |
| full | none | 700-799 | 79.0 | 0.0 |
| full | none | 800-899 | 82.0 | 0.0 |
| full | none | 900-999 | 82.0 | 1.0 |
| full | none | 1000-1033 | 97.1 | 0.0 |

## Memory evolution (one row per consolidation)

| Config | order | step | new | strategies after | kept | refined | split | deprecated | deleted | merged |
|---|---|---|---|---|---|---|---|---|---|---|
| reasoning_bank | 1 | 49 | 8 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 142 | 8 | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 195 | 9 | 25 | 25 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 245 | 14 | 39 | 39 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 295 | 14 | 53 | 53 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 345 | 15 | 68 | 68 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 395 | 17 | 85 | 85 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 445 | 18 | 103 | 103 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 495 | 20 | 123 | 123 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 545 | 20 | 143 | 143 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 595 | 23 | 166 | 166 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 645 | 24 | 190 | 190 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 695 | 25 | 215 | 215 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 745 | 26 | 241 | 241 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 795 | 29 | 270 | 270 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 845 | 30 | 300 | 300 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 895 | 31 | 331 | 331 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 945 | 32 | 363 | 363 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | 995 | 32 | 395 | 395 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | 1 | final | 33 | 428 | 428 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 78 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 128 | 10 | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 178 | 17 | 33 | 33 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 228 | 17 | 50 | 50 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 278 | 20 | 70 | 70 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 328 | 21 | 91 | 91 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 378 | 22 | 113 | 113 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 428 | 22 | 135 | 135 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 478 | 24 | 159 | 159 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 528 | 24 | 183 | 183 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 578 | 25 | 208 | 208 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 628 | 25 | 233 | 233 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 678 | 26 | 259 | 259 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 728 | 26 | 285 | 285 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 778 | 28 | 313 | 313 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 828 | 29 | 342 | 342 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 882 | 8 | 350 | 350 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 932 | 15 | 365 | 365 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 982 | 17 | 382 | 382 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | 1032 | 18 | 400 | 400 | 0 | 0 | 0 | 0 | 0 |
| reasoning_bank | none | final | 18 | 418 | 418 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 49 | 8 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 124 | 6 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 174 | 13 | 27 | 27 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 224 | 16 | 43 | 43 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 274 | 16 | 59 | 59 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 324 | 19 | 78 | 78 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 374 | 19 | 97 | 97 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 424 | 20 | 117 | 117 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 474 | 21 | 138 | 138 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 524 | 22 | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 574 | 24 | 184 | 184 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 624 | 25 | 209 | 209 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 674 | 25 | 234 | 234 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 724 | 26 | 260 | 260 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 774 | 29 | 289 | 289 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 824 | 30 | 319 | 319 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 874 | 32 | 351 | 351 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 924 | 33 | 384 | 384 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 974 | 34 | 418 | 418 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | 1024 | 34 | 452 | 452 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | 1 | final | 34 | 486 | 486 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 71 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 121 | 10 | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 171 | 15 | 31 | 31 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 221 | 16 | 47 | 47 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 271 | 18 | 65 | 65 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 321 | 20 | 85 | 85 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 371 | 21 | 106 | 106 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 421 | 21 | 127 | 127 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 471 | 23 | 150 | 150 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 521 | 23 | 173 | 173 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 571 | 23 | 196 | 196 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 621 | 23 | 219 | 219 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 671 | 24 | 243 | 243 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 721 | 24 | 267 | 267 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 771 | 27 | 294 | 294 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 821 | 28 | 322 | 322 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 871 | 29 | 351 | 351 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 921 | 32 | 383 | 383 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 971 | 33 | 416 | 416 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | 1021 | 33 | 449 | 449 | 0 | 0 | 0 | 0 | 0 |
| rag_rb | none | final | 33 | 482 | 482 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 80 | 8 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 130 | 12 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 180 | 15 | 35 | 35 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 230 | 18 | 53 | 53 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 280 | 19 | 72 | 72 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 330 | 20 | 92 | 92 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 380 | 20 | 112 | 112 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 430 | 20 | 132 | 132 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 480 | 21 | 153 | 153 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 530 | 22 | 175 | 175 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 580 | 24 | 199 | 199 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 630 | 25 | 224 | 224 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 680 | 25 | 249 | 249 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 730 | 26 | 275 | 275 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 780 | 31 | 306 | 306 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 830 | 33 | 339 | 339 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 880 | 33 | 372 | 372 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 930 | 35 | 407 | 407 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 980 | 36 | 443 | 443 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | 1030 | 36 | 479 | 479 | 0 | 0 | 0 | 0 | 0 |
| full | 1 | final | 36 | 515 | 515 | 0 | 0 | 0 | 0 | 0 |
| full | none | 49 | 7 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| full | none | 125 | 5 | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| full | none | 175 | 11 | 23 | 23 | 0 | 0 | 0 | 0 | 0 |
| full | none | 225 | 13 | 36 | 36 | 0 | 0 | 0 | 0 | 0 |
| full | none | 275 | 16 | 52 | 52 | 0 | 0 | 0 | 0 | 0 |
| full | none | 325 | 17 | 69 | 69 | 0 | 0 | 0 | 0 | 0 |
| full | none | 386 | 7 | 76 | 76 | 0 | 0 | 0 | 0 | 0 |
| full | none | 436 | 13 | 89 | 89 | 0 | 0 | 0 | 0 | 0 |
| full | none | 486 | 19 | 108 | 108 | 0 | 0 | 0 | 0 | 0 |
| full | none | 536 | 20 | 128 | 128 | 0 | 0 | 0 | 0 | 0 |
| full | none | 586 | 21 | 149 | 149 | 0 | 0 | 0 | 0 | 0 |
| full | none | 636 | 22 | 171 | 171 | 0 | 0 | 0 | 0 | 0 |
| full | none | 686 | 23 | 194 | 194 | 0 | 0 | 0 | 0 | 0 |
| full | none | 736 | 23 | 217 | 217 | 0 | 0 | 0 | 0 | 0 |
| full | none | 786 | 25 | 242 | 242 | 0 | 0 | 0 | 0 | 0 |
| full | none | 879 | 8 | 250 | 250 | 0 | 0 | 0 | 0 | 0 |
| full | none | 929 | 13 | 263 | 263 | 0 | 0 | 0 | 0 | 0 |
| full | none | 979 | 16 | 279 | 279 | 0 | 0 | 0 | 0 | 0 |
| full | none | 1029 | 18 | 297 | 297 | 0 | 0 | 0 | 0 | 0 |
| full | none | final | 18 | 315 | 315 | 0 | 0 | 0 | 0 | 0 |

## Most-applied strategies at the end of each run

**reasoning_bank / order 1** (strategies_reasoning_bank_1.csv)

| name | pattern | r_s | n samples | applications |
|---|---|---|---|---|
| Aggregation | AGGREGATION | 100.0 | 7 | 3 |
| Aggregation | AGGREGATION | 100.0 | 3 | 2 |
| Aggregation | AGGREGATION | 100.0 | 6 | 1 |
| Grouped Aggregation | AGGREGATION_GROUPING | 100.0 | 7 | 1 |
| Aggregation Grouping Join Ordering Ranking | AGGREGATION_GROUPING_JOIN_ORDERING_RANKING | 100.0 | 21 | 1 |

**reasoning_bank / order none** (strategies_reasoning_bank_none.csv)

| name | pattern | r_s | n samples | applications |
|---|---|---|---|---|
| Aggregation with Top-N Ranking | AGGREGATION_GROUPING_ORDERING_RANKING | 100.0 | 6 | 3 |
| Aggregation with Top-N Ranking | AGGREGATION_GROUPING_ORDERING_RANKING | 100.0 | 8 | 3 |
| Grouped Aggregation | AGGREGATION_GROUPING | 100.0 | 22 | 2 |
| Grouped Aggregation | AGGREGATION_GROUPING | 100.0 | 29 | 2 |
| Aggregation Grouping Join Ordering Ranking | AGGREGATION_GROUPING_JOIN_ORDERING_RANKING | 100.0 | 23 | 2 |

**rag_rb / order 1** (strategies_rag_rb_1.csv)

| name | pattern | r_s | n samples | applications |
|---|---|---|---|---|
| Aggregation | AGGREGATION | 100.0 | 7 | 3 |
| Aggregation | AGGREGATION | 100.0 | 4 | 2 |
| Aggregation with Top-N Ranking | AGGREGATION_GROUPING_ORDERING_RANKING | 100.0 | 6 | 2 |
| Aggregation with Top-N Ranking | AGGREGATION_GROUPING_ORDERING_RANKING | 100.0 | 6 | 2 |
| Aggregation | AGGREGATION | 100.0 | 8 | 1 |

**rag_rb / order none** (strategies_rag_rb_none.csv)

| name | pattern | r_s | n samples | applications |
|---|---|---|---|---|
| Aggregation Grouping Join Ordering Ranking | AGGREGATION_GROUPING_JOIN_ORDERING_RANKING | 100.0 | 22 | 3 |
| Aggregation with Top-N Ranking | AGGREGATION_GROUPING_ORDERING_RANKING | 100.0 | 5 | 3 |
| Aggregation with Top-N Ranking | AGGREGATION_GROUPING_ORDERING_RANKING | 100.0 | 8 | 3 |
| Aggregation | AGGREGATION | 100.0 | 6 | 2 |
| Filtered Aggregation | AGGREGATION_FILTERING | 100.0 | 33 | 2 |

**full / order 1** (strategies_full_1.csv)

| name | pattern | r_s | n samples | applications |
|---|---|---|---|---|
| Aggregation | AGGREGATION | 100.0 | 7 | 3 |
| Aggregation | AGGREGATION | 100.0 | 10 | 2 |
| Aggregation Ordering Ranking | AGGREGATION_ORDERING_RANKING | 100.0 | 5 | 2 |
| Aggregation | AGGREGATION | 100.0 | 15 | 1 |
| Grouped Aggregation | AGGREGATION_GROUPING | 100.0 | 23 | 1 |

**full / order none** (strategies_full_none.csv)

| name | pattern | r_s | n samples | applications |
|---|---|---|---|---|
| Aggregation | AGGREGATION | 100.0 | 4 | 8 |
| Aggregation | AGGREGATION | 100.0 | 4 | 4 |
| Aggregation Grouping Join Ordering Ranking | AGGREGATION_GROUPING_JOIN_ORDERING_RANKING | 100.0 | 20 | 3 |
| Aggregation | AGGREGATION | 100.0 | 10 | 1 |
| Aggregation | AGGREGATION | 100.0 | 14 | 1 |

## Complementarity (questions solved, EX)

| A | B | both | only A | only B | neither | oracle(A∪B) |
|---|---|---|---|---|---|---|
| template_only | semantic_layer | 852 | 22 | 14 | 146 | 85.9 |
| template_only | semantic_rag | 867 | 7 | 12 | 148 | 85.7 |
| template_only | reasoning_bank | 867 | 7 | 6 | 154 | 85.1 |
| template_only | layer_rag | 853 | 21 | 18 | 142 | 86.3 |
| template_only | rag_rb | 865 | 9 | 13 | 147 | 85.8 |
| template_only | full | 850 | 24 | 20 | 140 | 86.5 |
| semantic_layer | semantic_rag | 854 | 12 | 25 | 143 | 86.2 |
| semantic_layer | reasoning_bank | 849 | 17 | 24 | 144 | 86.1 |
| semantic_layer | layer_rag | 854 | 12 | 17 | 151 | 85.4 |
| semantic_layer | rag_rb | 851 | 15 | 27 | 141 | 86.4 |
| semantic_layer | full | 856 | 10 | 14 | 154 | 85.1 |
| semantic_rag | reasoning_bank | 867 | 12 | 6 | 149 | 85.6 |
| semantic_rag | layer_rag | 861 | 18 | 10 | 145 | 86.0 |
| semantic_rag | rag_rb | 874 | 5 | 4 | 151 | 85.4 |
| semantic_rag | full | 858 | 21 | 12 | 143 | 86.2 |
| reasoning_bank | layer_rag | 851 | 22 | 20 | 141 | 86.4 |
| reasoning_bank | rag_rb | 865 | 8 | 13 | 148 | 85.7 |
| reasoning_bank | full | 848 | 25 | 22 | 139 | 86.6 |
| layer_rag | rag_rb | 859 | 12 | 19 | 144 | 86.1 |
| layer_rag | full | 863 | 8 | 7 | 156 | 84.9 |
| rag_rb | full | 856 | 22 | 14 | 142 | 86.3 |
