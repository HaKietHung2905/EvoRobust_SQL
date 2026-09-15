# Robustness Ablation Report

Four-way comparison on the fixed, EA-discovered adversarial noisy dev set.

Baseline error rate (1 - execution accuracy): **20.00%**

| Config | Exact Match | Execution Acc. | Error Rate | Error Reduction vs Baseline | N |
|---|---|---|---|---|---|
| baseline | 75.00% | 80.00% | 20.00% | 0.00% | 0 |
| semantic_rag | 70.00% | 75.00% | 25.00% | -25.00% | 0 |
| reasoning_bank | 75.00% | 80.00% | 20.00% | 0.00% | 0 |
| semantic_rag_and_reasoning_bank | 70.00% | 75.00% | 25.00% | -25.00% | 0 |
