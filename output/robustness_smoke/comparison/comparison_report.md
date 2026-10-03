# Robustness Ablation Report

Four-way comparison on the fixed, EA-discovered adversarial noisy dev set.

Baseline error rate (1 - execution accuracy): **12.86%**

| Config | Exact Match | Execution Acc. | Error Rate | Error Reduction vs Baseline | N |
|---|---|---|---|---|---|
| baseline | 67.79% | 87.14% | 12.86% | 0.00% | 1034 |
| semantic_rag | 66.92% | 87.43% | 12.57% | 2.26% | 1034 |
| reasoning_bank | 66.83% | 87.23% | 12.77% | 0.75% | 1034 |
| semantic_rag_and_reasoning_bank | 67.31% | 87.33% | 12.67% | 1.50% | 1034 |
