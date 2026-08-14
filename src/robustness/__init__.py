"""
Robustness evaluation package.

Evolves realistic typing-noise patterns ("NoiseProfiles") that maximize
Text-to-SQL generation errors on the baseline pipeline, then uses the
discovered noise to build an adversarial test set and measures how much
error is recovered by Semantic RAG, ReasoningBank, and both combined.

Modules:
    noise_operators     - character/word-level typo operators
    genome              - NoiseProfile chromosome (encode/decode/mutate/crossover)
    fitness             - apply a profile to text + score it against the pipeline
    evolutionary_search  - the GA loop (selection, crossover, mutation, elitism)
    dataset_builder      - materialize a full noisy dataset from a winning profile
    compare_configs       - run baseline / +RAG / +ReasoningBank / +both and report
"""
