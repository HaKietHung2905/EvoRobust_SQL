"""
Execution-based Self-Consistency: Pseudo-labeling for Self-Judgment when no gold_sql exists.


This module generates N diverse SQL candidates for the SAME question, executes
all of them on the target database, and uses AGREEMENT between candidates
(execution-result voting) as a proxy signal instead of gold comparison.

Outcome categories (exhaustive and mutually exclusive)
------------------------------------------------------
Let n = total candidates (primary greedy T=0 + n_additional sampled), let the
candidates that execute be clustered by result-set equality (bag semantics,
row order ignored), let C* be the largest cluster, and let

    min_agree = ceil(agreement_threshold * n)      (0.6 * 5 -> 3 of 5)

Exactly one of the following holds for every question:

    NO_EXEC     primary SQL does not execute                -> EXECUTION_FAILURE
    CONSENSUS   primary executes, primary in C*, |C*| >= min_agree
                                                            -> COMPLETE_SUCCESS
    LOW_AGREE   primary executes, primary in C*, |C*| <  min_agree
                (includes ties for the largest cluster)     -> PARTIAL_SUCCESS
    MINORITY    primary executes, primary NOT in C*         -> PARTIAL_SUCCESS

Failed or empty generations count in n but never enter a cluster, so "most
candidates fail" can only lower |C*| and therefore never yields CONSENSUS
unless min_agree executing candidates still agree.

Ties: when several clusters share the largest size, the primary counts as
"in C*" if its cluster is one of them, but such a tie always has
|C*| <= n/2 < min_agree (for threshold > 0.5), so it is LOW_AGREE, never
CONSENSUS.

Empty result sets: agreement on an empty / all-NULL result is weak evidence
(different wrong queries often return nothing). `majority_is_empty` is always
recorded; with empty_consensus="downgrade" such a CONSENSUS becomes LOW_AGREE.
The default ("allow") reproduces the behaviour used for the submitted paper.

Legacy fields kept for SelfJudgment:
  - pseudo_execution_match : primary SQL executed
  - pseudo_exact_match     : 1.0 iff outcome == CONSENSUS
  - agreement_ratio        : |C*| / n

The majority-vote SQL (a member of C*, preferring the primary, then the lowest
temperature) is also returned so it can be evaluated as a self-consistency
*prediction* baseline at no extra LLM cost.

IMPORTANT LIMITATION (must be stated in thesis Limitations section):
If the LLM makes the SAME mistake across multiple candidates (a systematic/
correlated error), self-consistency will confidently vote for the WRONG
answer. This proxy is strictly weaker than real gold-based evaluation and
should never be presented as equivalent to it. scripts/analyze_pseudo_labels.py
measures how often this happens against gold EX after the run.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Reuse the SAME execution + result-comparison primitives already used for
# Spider/WikiSQL execution accuracy, so pseudo-labels are computed with the
# exact same semantics as the benchmark's own EX metric.
from src.evaluation.exec_evaluator import exec_on_db, result_eq, postprocess

OUTCOME_NO_EXEC = "no_exec"
OUTCOME_CONSENSUS = "consensus"
OUTCOME_LOW_AGREE = "low_agreement"
OUTCOME_MINORITY = "minority"
ALL_OUTCOMES = (OUTCOME_CONSENSUS, OUTCOME_LOW_AGREE, OUTCOME_MINORITY, OUTCOME_NO_EXEC)


@dataclass
class ConsistencyCandidate:
    """One SQL candidate generated for self-consistency voting."""
    sql: str
    temperature: float
    executed_ok: bool = False
    denotation: Optional[object] = None   # execution result, used for clustering
    cluster_id: Optional[int] = None
    generation_time: float = 0.0
    error: Optional[str] = None


@dataclass
class SelfConsistencyResult:
    """Everything SelfJudgment / reasoning_pipeline needs, no gold required."""
    primary_sql: str
    pseudo_execution_match: bool
    pseudo_exact_match: float          # 0.0 or 1.0 (kept as float to match Trajectory.exact_match type)
    agreement_ratio: float             # majority_size / n_candidates, in [0, 1]
    majority_cluster_size: int
    n_candidates: int
    label_source: str = "self_consistency"
    candidates: List[ConsistencyCandidate] = field(default_factory=list)
    # ── fields added for the revision (exhaustive outcome + analysis) ──
    outcome: str = OUTCOME_NO_EXEC
    min_agree: int = 0
    n_executed: int = 0
    cluster_sizes: List[int] = field(default_factory=list)
    majority_tied: bool = False
    majority_is_empty: bool = False
    majority_sql: Optional[str] = None
    downgraded_empty: bool = False

    def to_log_dict(self, include_candidates: bool = True) -> Dict:
        """Compact JSON-safe record for per-query trace logs."""
        d = {
            "outcome": self.outcome,
            "pseudo_exact_match": self.pseudo_exact_match,
            "pseudo_execution_match": self.pseudo_execution_match,
            "agreement_ratio": self.agreement_ratio,
            "majority_cluster_size": self.majority_cluster_size,
            "min_agree": self.min_agree,
            "n_candidates": self.n_candidates,
            "n_executed": self.n_executed,
            "cluster_sizes": self.cluster_sizes,
            "majority_tied": self.majority_tied,
            "majority_is_empty": self.majority_is_empty,
            "downgraded_empty": self.downgraded_empty,
            "majority_sql": self.majority_sql,
        }
        if include_candidates:
            d["candidates"] = [
                {"sql": c.sql, "temperature": c.temperature, "executed_ok": c.executed_ok,
                 "cluster_id": c.cluster_id, "error": c.error,
                 "generation_time": round(c.generation_time, 3)}
                for c in self.candidates
            ]
        return d


def _is_empty_denotation(denotation) -> bool:
    """True for no rows, or rows whose every value is NULL."""
    if denotation is None:
        return True
    try:
        rows = list(denotation)
    except TypeError:
        return False
    if not rows:
        return True
    try:
        return all(all(v is None for v in row) for row in rows)
    except TypeError:
        return False


class ExecutionSelfConsistency:
    """
    Generates diverse candidates for a question and votes on execution results
    to produce a pseudo ground-truth signal for Self-Judgment.

    Does NOT know how to build prompts itself — the caller (reasoning_pipeline)
    passes in a `candidate_generator_fn(temperature) -> sql_string` closure that
    already has the question/schema/strategies baked in. This keeps this module
    decoupled from prompt-construction logic.
    """

    def __init__(
        self,
        agreement_threshold: float = 0.6,
        max_confidence: float = 0.85,
        empty_consensus: str = "allow",
    ):
        """
        Args:
            agreement_threshold: fraction of ALL candidates (incl. failed ones)
                that must agree for CONSENSUS; converted to an integer count
                min_agree = ceil(threshold * n). 0.6 with n=5 -> 3 of 5.
            max_confidence: hard cap on confidence for ANY self-consistency
                judgment. Even 100% agreement across candidates is still not
                gold-verified truth, so SelfJudgment should never reach the
                same 1.0 confidence it gets from real gold comparison.
            empty_consensus: "allow" (paper default) or "downgrade" (a
                CONSENSUS whose shared result is empty/all-NULL becomes
                LOW_AGREE and is therefore not used as a success signal).
        """
        if empty_consensus not in ("allow", "downgrade"):
            raise ValueError("empty_consensus must be 'allow' or 'downgrade'")
        self.agreement_threshold = agreement_threshold
        self.max_confidence = max_confidence
        self.empty_consensus = empty_consensus
        logger.info(
            f"ExecutionSelfConsistency initialized "
            f"(agreement_threshold={agreement_threshold}, max_confidence={max_confidence}, "
            f"empty_consensus={empty_consensus})"
        )

    def min_agree_for(self, n_candidates: int) -> int:
        # small epsilon guards against 0.6*5 = 3.0000000000000004
        return max(1, math.ceil(self.agreement_threshold * n_candidates - 1e-9))

    @staticmethod
    def default_temperatures(n_additional_candidates: int) -> List[float]:
        if n_additional_candidates <= 1:
            return [0.6]
        lo, hi = 0.3, 0.9
        step = (hi - lo) / (n_additional_candidates - 1)
        return [round(lo + i * step, 2) for i in range(n_additional_candidates)]

    def generate_pseudo_labels(
        self,
        primary_sql: str,
        db_path: str,
        candidate_generator_fn: Callable[[float], str],
        n_additional_candidates: int = 4,
        temperatures: Optional[List[float]] = None,
    ) -> SelfConsistencyResult:
        """
        Args:
            primary_sql: the SQL already generated by the normal pipeline
                (this is the one whose pseudo-label we actually need).
            db_path: path to the target .sqlite database for execution.
            candidate_generator_fn: callable(temperature: float) -> sql string.
                Caller is responsible for prompt construction; this should
                reuse the SAME question/schema/strategies as primary_sql,
                just resampled at a different temperature.
            n_additional_candidates: how many EXTRA candidates to generate
                besides primary_sql (total candidates = n_additional + 1).
            temperatures: explicit temperature list for extra candidates.
                Defaults to a spread between 0.3 and 0.9.
        """
        if temperatures is None:
            temperatures = self.default_temperatures(n_additional_candidates)

        candidates: List[ConsistencyCandidate] = [
            ConsistencyCandidate(sql=primary_sql, temperature=0.0)  # primary = greedy/original
        ]

        # ── Generate extra candidates ────────────────────────────────────────
        for temp in temperatures:
            start = time.time()
            try:
                sql = candidate_generator_fn(temp)
                if not sql or not sql.strip():
                    # Empty generation result — a generation failure, NOT a
                    # valid "SELECT 1" candidate that can win a vote.
                    candidates.append(ConsistencyCandidate(
                        sql="", temperature=temp, generation_time=time.time() - start,
                        error="empty generation result", executed_ok=False,
                    ))
                else:
                    candidates.append(ConsistencyCandidate(
                        sql=sql, temperature=temp, generation_time=time.time() - start,
                    ))
            except Exception as e:
                logger.debug(f"Self-consistency candidate generation failed at temp={temp}: {e}")
                # Excluded from execution + voting, but still counted in n,
                # so infrastructure failures can only LOWER agreement.
                candidates.append(ConsistencyCandidate(
                    sql="", temperature=temp, generation_time=time.time() - start,
                    error=str(e), executed_ok=False,
                ))

        return self.label_candidates(candidates, db_path)

    def label_candidates(
        self, candidates: List[ConsistencyCandidate], db_path: Optional[str] = None
    ) -> SelfConsistencyResult:
        """
        Execute (if db_path is given and a candidate is not yet executed),
        cluster and label. Split out from generate_pseudo_labels so it can be
        unit-tested with pre-executed candidates. candidates[0] must be the
        primary (greedy) SQL.
        """
        # ── Execute every candidate on the real database ─────────────────────
        for cand in candidates:
            if db_path is None or cand.error is not None or cand.executed_ok:
                continue
            try:
                flag, denotation = asyncio.run(exec_on_db(db_path, postprocess(cand.sql)))
                cand.executed_ok = (flag != "exception")
                cand.denotation = denotation if cand.executed_ok else None
            except Exception as e:
                cand.executed_ok = False
                cand.error = cand.error or str(e)

        return self._cluster_and_label(candidates)

    def _cluster_and_label(self, candidates: List[ConsistencyCandidate]) -> SelfConsistencyResult:
        n = len(candidates)
        min_agree = self.min_agree_for(n)
        primary = candidates[0]

        # ── Cluster candidates by execution-result equality ──────────────────
        clusters: List[List[int]] = []
        for i, cand in enumerate(candidates):
            if not cand.executed_ok:
                continue
            placed = False
            for cluster in clusters:
                rep = candidates[cluster[0]]
                try:
                    if result_eq(rep.denotation, cand.denotation, order_matters=False):
                        cluster.append(i)
                        placed = True
                        break
                except Exception:
                    continue
            if not placed:
                clusters.append([i])

        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                candidates[idx].cluster_id = cluster_id

        n_executed = sum(1 for c in candidates if c.executed_ok)
        cluster_sizes = sorted((len(c) for c in clusters), reverse=True)

        if not primary.executed_ok:
            # Pick the majority SQL among the others for the majority-vote baseline.
            majority_sql, majority_size, tied, empty = self._majority_info(candidates, clusters, primary_cluster=None)
            return SelfConsistencyResult(
                primary_sql=primary.sql, pseudo_execution_match=False, pseudo_exact_match=0.0,
                agreement_ratio=majority_size / n if n else 0.0, majority_cluster_size=majority_size,
                n_candidates=n, candidates=candidates, outcome=OUTCOME_NO_EXEC, min_agree=min_agree,
                n_executed=n_executed, cluster_sizes=cluster_sizes, majority_tied=tied,
                majority_is_empty=empty, majority_sql=majority_sql,
            )

        majority_sql, majority_size, tied, empty = self._majority_info(
            candidates, clusters, primary_cluster=primary.cluster_id)
        primary_size = len(clusters[primary.cluster_id])
        primary_in_majority = primary_size == majority_size

        downgraded = False
        if primary_in_majority and majority_size >= min_agree:
            outcome = OUTCOME_CONSENSUS
            if empty and self.empty_consensus == "downgrade":
                outcome, downgraded = OUTCOME_LOW_AGREE, True
        elif primary_in_majority:
            outcome = OUTCOME_LOW_AGREE
        else:
            outcome = OUTCOME_MINORITY

        result = SelfConsistencyResult(
            primary_sql=primary.sql,
            pseudo_execution_match=True,
            pseudo_exact_match=1.0 if outcome == OUTCOME_CONSENSUS else 0.0,
            agreement_ratio=majority_size / n,
            majority_cluster_size=majority_size,
            n_candidates=n,
            candidates=candidates,
            outcome=outcome,
            min_agree=min_agree,
            n_executed=n_executed,
            cluster_sizes=cluster_sizes,
            majority_tied=tied,
            majority_is_empty=empty,
            majority_sql=majority_sql,
            downgraded_empty=downgraded,
        )
        logger.debug(
            f"Self-consistency: {majority_size}/{n} agree (min_agree={min_agree}) -> {outcome}"
        )
        return result

    @staticmethod
    def _majority_info(candidates, clusters, primary_cluster: Optional[int]):
        """Return (majority_sql, majority_size, tied, majority_is_empty)."""
        if not clusters:
            return None, 0, False, False
        max_size = max(len(c) for c in clusters)
        largest = [cid for cid, c in enumerate(clusters) if len(c) == max_size]
        tied = len(largest) > 1
        # Deterministic choice: the primary's cluster if it is among the
        # largest, otherwise the largest cluster whose best member has the
        # lowest temperature.
        if primary_cluster is not None and primary_cluster in largest:
            chosen = primary_cluster
        else:
            chosen = min(largest, key=lambda cid: min(candidates[i].temperature for i in clusters[cid]))
        member = min(clusters[chosen], key=lambda i: (candidates[i].temperature, i))
        empty = _is_empty_denotation(candidates[member].denotation)
        return candidates[member].sql, max_size, tied, empty