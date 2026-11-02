"""
The four self-consistency outcomes must be exhaustive and mutually exclusive
(reviewer 3: "the four outcome categories are not exhaustive"). Candidates
are pre-executed (denotation set) so no database or LLM is needed.
"""
import pytest

from src.reasoning.self_consistency import (
    ALL_OUTCOMES, OUTCOME_CONSENSUS, OUTCOME_LOW_AGREE, OUTCOME_MINORITY, OUTCOME_NO_EXEC,
    ConsistencyCandidate, ExecutionSelfConsistency,
)

A, B, C, EMPTY = [(1,)], [(2,)], [(3,)], []


def cands(*specs):
    """spec: denotation, or None for 'did not execute'. First = primary (T=0)."""
    out = []
    for i, d in enumerate(specs):
        c = ConsistencyCandidate(sql=f"SELECT {i}", temperature=0.0 if i == 0 else round(0.1 + 0.2 * i, 2))
        if d is None:
            c.executed_ok, c.error = False, "no such column"
        else:
            c.executed_ok, c.denotation = True, d
        out.append(c)
    return out


SC = ExecutionSelfConsistency(agreement_threshold=0.6)


def label(*specs, sc=SC):
    return sc.label_candidates(cands(*specs))


def test_min_agree_is_three_of_five():
    assert SC.min_agree_for(5) == 3
    assert SC.min_agree_for(4) == 3
    assert SC.min_agree_for(3) == 2


@pytest.mark.parametrize("specs,expected", [
    ((A, A, A, A, A), OUTCOME_CONSENSUS),
    ((A, A, A, B, C), OUTCOME_CONSENSUS),              # 3/5 agree with primary
    ((A, A, B, B, C), OUTCOME_LOW_AGREE),              # tie 2-2: primary in a largest cluster
    ((A, A, B, C, None), OUTCOME_LOW_AGREE),           # 2/5 is the reviewer's undefined case
    ((A, A, None, None, None), OUTCOME_LOW_AGREE),     # most candidates fail to execute
    ((A, B, B, B, C), OUTCOME_MINORITY),
    ((A, B, B, None, None), OUTCOME_MINORITY),
    ((None, A, A, A, A), OUTCOME_NO_EXEC),
    ((None, None, None, None, None), OUTCOME_NO_EXEC),
])
def test_outcomes(specs, expected):
    r = label(*specs)
    assert r.outcome == expected
    assert r.pseudo_exact_match == (1.0 if expected == OUTCOME_CONSENSUS else 0.0)
    assert r.pseudo_execution_match == (expected != OUTCOME_NO_EXEC)


def test_every_combination_gets_exactly_one_outcome():
    import itertools
    choices = [A, B, C, EMPTY, None]
    seen = set()
    for specs in itertools.product(choices, repeat=5):
        r = label(*specs)
        assert r.outcome in ALL_OUTCOMES
        seen.add(r.outcome)
    assert seen == set(ALL_OUTCOMES)


def test_majority_sql_prefers_primary_then_lowest_temperature():
    r = label(A, B, B, B, A)
    assert r.majority_sql == "SELECT 1"          # cluster B, lowest-temperature member
    r = label(A, A, A, B, B)
    assert r.majority_sql == "SELECT 0"          # primary is in the majority


def test_empty_consensus_flag_and_downgrade():
    r = label(EMPTY, EMPTY, EMPTY, A, B)
    assert r.outcome == OUTCOME_CONSENSUS and r.majority_is_empty
    strict = ExecutionSelfConsistency(agreement_threshold=0.6, empty_consensus="downgrade")
    r2 = label(EMPTY, EMPTY, EMPTY, A, B, sc=strict)
    assert r2.outcome == OUTCOME_LOW_AGREE and r2.downgraded_empty and r2.pseudo_exact_match == 0.0


def test_log_dict_is_json_safe():
    import json
    r = label(A, A, B, None, A)
    json.dumps(r.to_log_dict())