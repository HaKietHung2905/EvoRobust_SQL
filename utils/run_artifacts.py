"""
run_artifacts.py
================
Small helpers shared by scripts/generate_predictions.py and the analysis
scripts, so every run leaves the same machine-readable record:

  <output>.tsv                     predictions, ORIGINAL question order
  <output>.trace.jsonl             one JSON line per processed question
  <output>.manifest.json           exact configuration + totals (cost, timing)
  <output>.majority_vote.tsv       self-consistency majority-vote predictions
  <output>.consolidation.jsonl     one line per memory consolidation
  <output>.order.json              processing order (only with --shuffle_seed)
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def sidecar(output_path: str, suffix: str) -> Path:
    """results/x.tsv + '.trace.jsonl' -> results/x.trace.jsonl"""
    p = Path(output_path)
    return p.with_name(p.stem + suffix)


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git_info(repo_root: str) -> Dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", repo_root, *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines()[:50] if status else [],
    }


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    out = {}
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover
        return out
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def environment_info() -> Dict:
    keys = ["MODEL_NAME", "GEMINI_MODEL", "USE_VERTEX_AI", "VERTEX_AI_LOCATION",
            "GOOGLE_CLOUD_PROJECT", "SPIDER_PROMPT_EXAMPLES"]
    env = {k: os.getenv(k) for k in keys}
    if env.get("GOOGLE_CLOUD_PROJECT"):
        env["GOOGLE_CLOUD_PROJECT"] = "<set>"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env": env,
        "packages": package_versions(["chromadb", "sentence-transformers", "google-generativeai",
                                      "google-cloud-aiplatform", "sqlparse", "nltk", "numpy"]),
    }


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_jsonl(path) -> List[Dict]:
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # tolerate a truncated last line after a crash
    return rows


def read_trace(path) -> Dict[int, Dict]:
    """Trace keyed by original question index; later lines win (resume-safe)."""
    out = {}
    for r in read_jsonl(path):
        if "idx" in r:
            out[int(r["idx"])] = r
    return out


def read_tsv_predictions(path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_majority_vote_tsv(trace_path, predictions_tsv, out_path) -> Dict:
    """
    Majority-vote baseline: for each question use the SQL of the largest
    execution cluster among the self-consistency candidates; fall back to
    the primary prediction when no candidates were generated. Costs no
    extra LLM calls (candidates were already generated for pseudo-labels).
    """
    trace = read_trace(trace_path)
    lines = read_tsv_predictions(predictions_tsv)
    n_replaced = n_same = n_no_sc = 0
    out = []
    for i, line in enumerate(lines):
        sql, _, db_id = line.partition("\t")
        sc = (trace.get(i) or {}).get("self_consistency") or {}
        mv = sc.get("majority_sql")
        if mv:
            mv = " ".join(mv.split())
            if mv != " ".join(sql.split()):
                n_replaced += 1
            else:
                n_same += 1
            out.append(f"{mv}\t{db_id}")
        else:
            n_no_sc += 1
            out.append(line)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return {"n": len(lines), "replaced": n_replaced, "same_as_primary": n_same,
            "no_self_consistency": n_no_sc, "path": str(out_path)}


def summarize_cost(trace_rows: Iterable[Dict]) -> Dict:
    rows = [r for r in trace_rows if r.get("llm")]
    n = len(rows)
    if not n:
        return {"n_questions": 0}
    keys = ["calls", "http_requests", "prompt_tokens", "completion_tokens", "latency_s"]
    tot = {k: sum((r["llm"].get(k) or 0) for r in rows) for k in keys}
    wall = sum((r.get("wall_s") or 0) for r in rows)
    out = {"n_questions": n, "wall_s_total": round(wall, 2), "wall_s_mean": round(wall / n, 3)}
    for k in keys:
        out[f"{k}_total"] = round(tot[k], 3) if isinstance(tot[k], float) else tot[k]
        out[f"{k}_mean"] = round(tot[k] / n, 3)
    return out