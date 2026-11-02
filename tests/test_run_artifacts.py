import json

from utils import run_artifacts as ra


def test_sidecar_names():
    assert ra.sidecar("results/x.tsv", ".trace.jsonl").name == "x.trace.jsonl"


def test_majority_vote_tsv(tmp_path):
    pred = tmp_path / "p.tsv"
    pred.write_text("SELECT a FROM t\tdb1\nSELECT b FROM t\tdb2\nSELECT c FROM t\tdb3\n")
    trace = tmp_path / "p.trace.jsonl"
    rows = [
        {"idx": 0, "self_consistency": {"majority_sql": "SELECT  a FROM t"}},   # same, whitespace
        {"idx": 1, "self_consistency": {"majority_sql": "SELECT z FROM t"}},    # replaced
        {"idx": 2},                                                              # no SC -> keep
        {"idx": 1, "self_consistency": {"majority_sql": "SELECT y FROM t"}},    # later line wins
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n{truncated")
    out = tmp_path / "mv.tsv"
    stats = ra.write_majority_vote_tsv(trace, pred, out)
    lines = out.read_text().splitlines()
    assert lines == ["SELECT a FROM t\tdb1", "SELECT y FROM t\tdb2", "SELECT c FROM t\tdb3"]
    assert stats == {"n": 3, "replaced": 1, "same_as_primary": 1, "no_self_consistency": 1, "path": str(out)}


def test_summarize_cost():
    rows = [{"llm": {"calls": 5, "http_requests": 6, "prompt_tokens": 100, "completion_tokens": 10,
                     "latency_s": 1.0}, "wall_s": 2.0},
            {"llm": {"calls": 1, "http_requests": 1, "prompt_tokens": 20, "completion_tokens": 2,
                     "latency_s": 0.5}, "wall_s": 1.0}]
    c = ra.summarize_cost(rows)
    assert c["calls_mean"] == 3.0 and c["prompt_tokens_total"] == 120 and c["wall_s_mean"] == 1.5
    