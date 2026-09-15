"""
Regression test for the self-consistency config-propagation bug.

BUG: ReasoningBankPipeline.__init__ did
        self.config = self._default_config()
        if config: self.config.update(config)
     Since `config` (loaded from reasoning_config.yaml) keeps the nested
     shape (experimental.self_consistency_n_candidates, etc.), the
     top-level .update() never sees those keys, so every downstream
         self.config.get('self_consistency_n_candidates', 4)
     silently falls back to the hardcoded default 4 — changing the YAML
     value has ZERO effect.

FIX: after the top-level update, flatten config['experimental']'s
     self_consistency_* keys into self.config directly.

This test proves the bug and proves the fix, without needing ChromaDB,
SQLite, or a live LLM: it patches the heavy dependencies
(ReasoningMemoryStore, MemoryRetrieval, ExperienceCollector, SelfJudgment,
StrategyDistillation, MemoryConsolidation) with lightweight mocks and only
exercises the config-handling + ExecutionSelfConsistency logic, which is
pure Python.

Usage (from repo root, after `pip install pytest`):
    pytest tests/test_self_consistency_config.py -v
"""

import sys
from unittest import mock

import pytest

from src.reasoning.reasoning_pipeline import ReasoningBankPipeline


# A config that DELIBERATELY differs from every hardcoded default, so any
# fallback-to-default bug becomes immediately visible in the assertions.
NON_DEFAULT_CONFIG = {
    "pipeline": {
        "db_path": "./memory/reasoning_bank.db",
        "chromadb_path": "./memory/chromadb",
    },
    "experimental": {
        "enable_self_consistency_judging": False,   # default is True
        "self_consistency_n_candidates": 7,          # default is 4
        "self_consistency_agreement_threshold": 0.42,  # default is 0.6
        "self_consistency_max_confidence": 0.77,       # default is 0.85
    },
}


def _make_pipeline(config):
    """Instantiate ReasoningBankPipeline with the heavy deps mocked out."""
    with mock.patch("src.reasoning.reasoning_pipeline.ReasoningMemoryStore"), \
         mock.patch("src.reasoning.reasoning_pipeline.MemoryRetrieval"), \
         mock.patch("src.reasoning.reasoning_pipeline.ExperienceCollector"), \
         mock.patch("src.reasoning.reasoning_pipeline.SelfJudgment"), \
         mock.patch("src.reasoning.reasoning_pipeline.StrategyDistillation"), \
         mock.patch("src.reasoning.reasoning_pipeline.MemoryConsolidation"):
        return ReasoningBankPipeline(
            db_path="./memory/reasoning_bank.db",
            chromadb_path="./memory/chromadb",
            config=config,
        )


class TestSelfConsistencyConfigPropagation:

    def test_defaults_used_when_no_experimental_block(self):
        """No experimental block at all -> hardcoded defaults, as before."""
        pipeline = _make_pipeline(config={"pipeline": {}})
        assert pipeline.self_consistency.agreement_threshold == 0.6
        assert pipeline.self_consistency.max_confidence == 0.85
        assert pipeline.config.get("self_consistency_n_candidates", 4) == 4
        assert pipeline.config.get("enable_self_consistency_judging", True) is True

    def test_yaml_values_actually_propagate(self):
        """
        THE key regression test: with NON_DEFAULT_CONFIG, every value read
        back from the pipeline must match the YAML, NOT the hardcoded
        default. This is the exact assertion that fails on the pre-fix code
        and passes on the patched code.
        """
        pipeline = _make_pipeline(config=NON_DEFAULT_CONFIG)

        assert pipeline.self_consistency.agreement_threshold == 0.42, (
            "agreement_threshold did not come from config['experimental'] — "
            "self.config.get('self_consistency_agreement_threshold', 0.6) is "
            "silently falling back to the hardcoded default. This is the bug."
        )
        assert pipeline.self_consistency.max_confidence == 0.77, (
            "max_confidence did not come from config['experimental'] — "
            "same fallback bug as agreement_threshold."
        )
        assert pipeline.config.get("self_consistency_n_candidates", 4) == 7, (
            "self_consistency_n_candidates did not come from config['experimental'] — "
            "generate_pseudo_labels() will silently sample 4 candidates "
            "regardless of what the YAML says."
        )
        assert pipeline.config.get("enable_self_consistency_judging", True) is False, (
            "enable_self_consistency_judging did not come from config['experimental'] — "
            "self-consistency judging cannot be turned off via YAML."
        )

    def test_n_candidates_flows_into_generate_pseudo_labels_call(self):
        """
        End-to-end-ish check: the number actually passed to
        self.self_consistency.generate_pseudo_labels(n_additional_candidates=...)
        inside enhance_sql_generation() must match the configured value,
        not the function's own default of 4.
        """
        pipeline = _make_pipeline(config=NON_DEFAULT_CONFIG)

        captured = {}
        original = pipeline.self_consistency.generate_pseudo_labels

        def spy(*args, **kwargs):
            captured["n_additional_candidates"] = kwargs.get("n_additional_candidates")
            # Return a minimal, well-formed result so the pipeline code
            # after this call doesn't choke on a None/mocked object.
            from src.reasoning.self_consistency import SelfConsistencyResult
            return SelfConsistencyResult(
                primary_sql=kwargs.get("primary_sql", "SELECT 1"),
                pseudo_execution_match=True,
                pseudo_exact_match=1.0,
                agreement_ratio=1.0,
                majority_cluster_size=1,
                n_candidates=1,
            )

        pipeline.self_consistency.generate_pseudo_labels = spy

        # enable_self_consistency_judging is False in NON_DEFAULT_CONFIG,
        # so force it True here just to exercise the code path.
        pipeline.config["enable_self_consistency_judging"] = True

        pipeline.enhance_sql_generation(
            question="How many singers are there?",
            db_id="concert_singer",
            schema={"singer": ["singer_id", "name"]},
            gold_sql=None,
            db_path="/fake/path/concert_singer.sqlite",
            sql_generator=lambda q, **kw: "SELECT COUNT(*) FROM singer",
        )

        assert captured.get("n_additional_candidates") == 7, (
            f"Expected n_additional_candidates=7 (from YAML), got "
            f"{captured.get('n_additional_candidates')!r} instead — the "
            f"config value is not reaching generate_pseudo_labels()."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))