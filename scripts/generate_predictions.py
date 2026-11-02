"""
Step 1: Generate SQL predictions and save to a TSV file.
Step 2: Run evaluate_wikisql.py / official_spider_eval.py with --predict <file>
        for fast, LLM-free evaluation.

Usage (WikiSQL):
  python scripts/generate_predictions.py \
      --questions data/raw/wikisql/dev_wiki_format.json \
      --db        data/raw/wikisql/database \
      --output    results/predictions_wikisql.tsv \
      --use_reasoning_bank --use_chromadb --use_semantic \
      --limit 50

Usage (Spider):
  python scripts/generate_predictions.py \
      --questions data/spider/dev.json \
      --db        data/spider/database \
      --output    results/predictions_spider.tsv

Revision additions (reviewer requests)
  --rb_memory_dir DIR   isolate the ReasoningBank memory of this run
  --fresh_memory        start that memory empty (old dir is moved aside, never deleted)
  --rb_frozen           retrieval only: no self-consistency, distillation or consolidation
                        (static-benchmark control; pair with a warm-up run on train)
  --shuffle_seed N      process questions in a seeded random order (order-dependence);
                        the final TSV is still written in the original order
  Gold SQL is never passed to the memory unless --allow_gold_in_memory is given.

Every run writes sidecar files next to --output (see utils/run_artifacts.py):
  .trace.jsonl (per-question record incl. LLM calls/tokens/latency, retrieved
  examples/strategies, memory size, self-consistency outcome), .manifest.json,
  .consolidation.jsonl and, when ReasoningBank is on, .majority_vote.tsv.
"""

import sys
import os
import json
import argparse
import logging
import warnings
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')

# ── Logging ───────────────────────────────────────────────────────────────────
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)
for _n in ['chromadb', 'chromadb.api', 'chromadb.telemetry',
           'utils.embedding_utils', 'src.reasoning.memory_retrieval',
           'src.reasoning.memory_store', 'src.reasoning.reasoning_pipeline']:
    logging.getLogger(_n).setLevel(logging.ERROR)

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils import run_artifacts as ra
logger = get_logger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    if path.endswith('.json'):
        return json.load(open(path))
    try:
        import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        return {}

def _is_server_error(exc: Exception) -> bool:
    """
    Return True if exc (or any chained cause) is an HTTP 5xx server error
    or a 403 billing/auth error that warrants a retry.
    Walks __cause__ and __context__ so wrapped exceptions are also detected.
    """
    _5xx = ("500", "502", "503", "504",
            "Internal Server Error", "Bad Gateway",
            "Service Unavailable", "Gateway Timeout")
    _403 = ("403", "Forbidden", "BILLING_DISABLED", "billing to be enabled")

    seen = set()
    node = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if any(code in str(node) for code in _5xx):
            return True
        if any(code in repr(node) for code in _5xx):
            return True
        if any(code in str(node) for code in _403):
            return True
        if any(code in repr(node) for code in _403):
            return True
        node = node.__cause__ or node.__context__

    return False 
def _stop_on_server_error(exc: Exception, out_f, already_done: int, i: int,
                           output_path: str) -> None:
    if not _is_server_error(exc):
        return
    out_f.flush()
    done_so_far = already_done + i

    is_403 = any(c in str(exc) or c in repr(exc)
                 for c in ("403", "Forbidden", "BILLING_DISABLED", "billing to be enabled"))
    wait_seconds = 30 if is_403 else 60
    print(f"\n❗ Real error: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n⏸  Checkpoint at q{done_so_far}, retrying in {wait_seconds}s...", flush=True)
    time.sleep(wait_seconds)

    new_argv = sys.argv[:]
    if '--resume' not in new_argv:
        new_argv.append('--resume')

    os.environ['PYTHONWARNINGS'] = 'ignore'

    os.execv(sys.executable, [sys.executable] + new_argv)


# ── Self-Consistency monitoring ────────────────────────────────────────────
# Lightweight aggregator that tracks whether temperature-sampled candidates
# are actually producing diverse agreement_ratio values across the run,
# rather than every trajectory trivially agreeing 1.0 (which would indicate
# temperature is silently collapsing back to greedy decoding, e.g. if the
# sql_generator closure passed to ReasoningBank ever stops forwarding the
# `temperature` kwarg again in the future).
class SelfConsistencyMonitor:
    def __init__(self):
        self.n_judged = 0
        self.agreement_ratios = []
        self.n_perfect_agreement = 0   # ratio == 1.0
        self.n_partial_agreement = 0   # 0.0 < ratio < 1.0
        self.n_zero_agreement = 0      # ratio == 0.0

    def record(self, sc_meta):
        if not sc_meta:
            return
        ratio = sc_meta.get('agreement_ratio')
        if ratio is None:
            return
        self.n_judged += 1
        self.agreement_ratios.append(ratio)
        if ratio >= 0.999:
            self.n_perfect_agreement += 1
        elif ratio <= 0.001:
            self.n_zero_agreement += 1
        else:
            self.n_partial_agreement += 1

    def summary(self) -> dict:
        if self.n_judged == 0:
            return {
                'n_judged': 0,
                'note': 'Self-consistency never produced a recorded agreement_ratio '
                        'this run (ReasoningBank may be disabled, or every trajectory '
                        'hit the non-critical except branch — check --use_reasoning_bank '
                        'and db_path availability).',
            }
        avg_ratio = sum(self.agreement_ratios) / self.n_judged
        pct_diverse = 100.0 * (self.n_partial_agreement + self.n_zero_agreement) / self.n_judged
        return {
            'n_judged': self.n_judged,
            'avg_agreement_ratio': round(avg_ratio, 4),
            'n_perfect_agreement_1.0': self.n_perfect_agreement,
            'n_partial_agreement': self.n_partial_agreement,
            'n_zero_agreement_0.0': self.n_zero_agreement,
            'pct_trajectories_showing_diversity': round(pct_diverse, 2),
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 70)
        print("SELF-CONSISTENCY / TEMPERATURE MONITOR")
        print("=" * 70)
        if s['n_judged'] == 0:
            print(f"  ⚠ {s['note']}")
        else:
            print(f"  Trajectories judged by self-consistency : {s['n_judged']}")
            print(f"  Average agreement_ratio                 : {s['avg_agreement_ratio']}")
            print(f"  Perfect agreement (ratio=1.0)            : {s['n_perfect_agreement_1.0']}")
            print(f"  Partial agreement (0<ratio<1)            : {s['n_partial_agreement']}")
            print(f"  Zero agreement (ratio=0.0)                : {s['n_zero_agreement_0.0']}")
            print(f"  % trajectories showing real diversity    : {s['pct_trajectories_showing_diversity']}%")
            if s['pct_trajectories_showing_diversity'] == 0.0:
                print("\n  ⚠ WARNING: 100% of trajectories show perfect agreement (ratio=1.0).")
                print("    This can be legitimate for easy/simple queries, but if it persists")
                print("    across a large, difficulty-varied sample, it may indicate temperature")
                print("    sampling is not actually reaching the LLM (regression of the fix in")
                print("    reasoning_pipeline.py / self_consistency.py — verify the sql_generator")
                print("    closure still forwards the `temperature` kwarg).")
        print("=" * 70)

    def save(self, output_path: str, merge_with_existing: bool = False):
        """
        Save summary stats to <output>.self_consistency_stats.json.

        merge_with_existing: when True (used on --resume runs), if a stats
        file already exists from a PRIOR process for this same output file,
        merge this run's raw agreement_ratios into it before writing —
        otherwise each resumed run's SelfConsistencyMonitor (a fresh
        in-memory object, see main()) would silently overwrite the file
        with only ITS OWN chunk's n_judged, hiding what earlier chunks
        already judged. Without this, a chunked/resumed run can show a much
        smaller n_judged than the true total, easily misread as "ReasoningBank
        isn't running" when it actually is.
        """
        stats_path = Path(output_path).with_suffix('.self_consistency_stats.json')
        ratios = list(self.agreement_ratios)

        if merge_with_existing and stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    prior = json.load(f)
                prior_ratios = prior.get('_raw_agreement_ratios')
                if prior_ratios:
                    ratios = prior_ratios + ratios
            except Exception as e:
                logger.debug(f"Could not merge prior self-consistency stats: {e}")

        merged = SelfConsistencyMonitor()
        merged.agreement_ratios = ratios
        merged.n_judged = len(ratios)
        merged.n_perfect_agreement = sum(1 for r in ratios if r >= 0.999)
        merged.n_zero_agreement = sum(1 for r in ratios if r <= 0.001)
        merged.n_partial_agreement = merged.n_judged - merged.n_perfect_agreement - merged.n_zero_agreement

        summary = merged.summary()
        summary['_raw_agreement_ratios'] = ratios  # kept so future resumes can merge again
        with open(stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Self-consistency monitor stats saved to {stats_path} "
                    f"(n_judged total: {merged.n_judged})")


def _prepare_memory_dir(args):
    """Resolve the ReasoningBank memory location for this run."""
    if not args.rb_memory_dir:
        return None
    mem = Path(args.rb_memory_dir)
    if args.fresh_memory and not args.resume and mem.exists() and any(mem.iterdir()):
        backup = mem.with_name(mem.name + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        shutil.move(str(mem), str(backup))
        logger.info(f"--fresh_memory: moved existing memory {mem} -> {backup}")
    mem.mkdir(parents=True, exist_ok=True)
    return mem


def _processing_order(n_total, args, order_path):
    """Identity order, or a seeded permutation saved to <output>.order.json."""
    if args.shuffle_seed is None:
        return list(range(n_total))
    if args.resume and order_path.exists():
        saved = json.load(open(order_path))
        if saved.get("seed") != args.shuffle_seed or saved.get("n") != n_total:
            logger.error(f"{order_path} was written with seed={saved.get('seed')} n={saved.get('n')}; "
                         f"refusing to resume with seed={args.shuffle_seed} n={n_total}")
            sys.exit(1)
        return saved["order"]
    order = list(range(n_total))
    random.Random(args.shuffle_seed).shuffle(order)
    ra.write_json(order_path, {"seed": args.shuffle_seed, "n": n_total, "order": order})
    return order


def _memory_size(reasoning_pipeline):
    try:
        return len(reasoning_pipeline.memory_store.get_all_strategies())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate SQL predictions to TSV file')

    parser.add_argument('--questions', required=True,
                        help='JSON questions file (Spider or WikiSQL spider-format)')
    parser.add_argument('--db', required=True,
                        help='Database directory')
    parser.add_argument('--output', default='./results/predictions.tsv',
                        help='Output TSV file (sql TAB db_id per line, original question order)')

    # Feature flags
    parser.add_argument('--use_chromadb',      action='store_true',
                        help='Semantic RAG: retrieve top_k similar train questions as few-shot examples')
    parser.add_argument('--chromadb_config',   default=None)
    parser.add_argument('--chromadb_persist_dir', default='./data/embeddings/chroma_db',
                        help='ChromaDB persist directory for Semantic RAG retrieval')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of few-shot examples to retrieve via Semantic RAG (k)')
    parser.add_argument('--use_semantic',      action='store_true',
                        help='Rule-based Semantic Layer (question enhancement + hints)')
    parser.add_argument('--semantic_config',   default=None)
    parser.add_argument('--use_reasoning_bank', action='store_true')
    parser.add_argument('--reasoning_config',
                        default='./configs/reasoning_config.yaml')

    # Revision: run protocol controls
    parser.add_argument('--rb_memory_dir', default=None,
                        help='Directory for THIS run\'s ReasoningBank memory '
                             '(overrides pipeline.db_path / chromadb_path)')
    parser.add_argument('--fresh_memory', action='store_true',
                        help='Start with an empty memory (existing --rb_memory_dir is moved aside)')
    parser.add_argument('--rb_frozen', action='store_true',
                        help='Use the memory read-only: no self-consistency labelling, '
                             'no distillation, no consolidation')
    parser.add_argument('--shuffle_seed', type=int, default=None,
                        help='Process questions in a seeded random order')
    parser.add_argument('--allow_gold_in_memory', action='store_true',
                        help='ORACLE ONLY: pass gold SQL to ReasoningBank trajectories')
    parser.add_argument('--run_name', default=None, help='Free-text label stored in the manifest')

    parser.add_argument('--limit',  type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-generated lines (resume interrupted run)')
    parser.add_argument('--checkpoint_size', type=int, default=None,
                        help='Stop after generating this many NEW predictions '
                             '(re-run with --resume to continue)')

    args = parser.parse_args()
    if args.fresh_memory and not args.rb_memory_dir:
        parser.error('--fresh_memory requires --rb_memory_dir (refusing to touch the shared ./memory)')
    if args.rb_frozen and not args.use_reasoning_bank:
        parser.error('--rb_frozen only makes sense with --use_reasoning_bank')

    # ── Auto-prepare WikiSQL if wiki_format file doesn't exist ─────────────
    if not Path(args.questions).exists() and 'wikisql' in args.questions.lower():
        logger.info(f"Questions file not found: {args.questions}")
        logger.info("Auto-preparing WikiSQL (building SQLite DBs + converting format)...")
        gold_file = args.questions.replace('_wiki_format.json', '.json')
        if not Path(gold_file).exists():
            logger.error(f"Cannot find WikiSQL gold file at: {gold_file}")
            sys.exit(1)
        from scripts.evaluate_wikisql import (
            prepare_wikisql_databases,
            convert_wikisql_gold_to_spider_format,
        )
        prepare_wikisql_databases(gold_file=gold_file, db_dir=args.db, limit=args.limit)
        convert_wikisql_gold_to_spider_format(
            gold_file=gold_file, output_file=args.questions, limit=args.limit
        )
        logger.info(f"✓ Wiki-format file created: {args.questions}")

    # ── Independently rebuild WikiSQL SQLite DBs if the db_dir is missing
    # or empty, even when the questions file already exists (see git history
    # for the failure mode this guards against).
    if 'wikisql' in args.questions.lower():
        db_root = Path(args.db)
        has_sqlite_db = db_root.exists() and any(db_root.glob("*/*.sqlite"))
        if not has_sqlite_db:
            logger.warning(
                f"WikiSQL db_dir {args.db} is missing or empty even though "
                f"{args.questions} exists — rebuilding SQLite DBs now."
            )
            from scripts.evaluate_wikisql import prepare_wikisql_databases
            gold_file_for_db = args.questions.replace('_wiki_format.json', '.json')
            if not Path(gold_file_for_db).exists():
                gold_file_for_db = args.questions  # wiki_format file itself has embedded table data
            prepare_wikisql_databases(gold_file=gold_file_for_db, db_dir=args.db, limit=args.limit)
            logger.info(f"✓ Rebuilt WikiSQL SQLite DBs in {args.db}")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.questions, 'r') as f:
        all_questions = json.load(f)
    if args.limit:
        all_questions = all_questions[:args.limit]
    n_total = len(all_questions)
    logger.info(f"Loaded {n_total} questions")

    # ── Output layout ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = ra.sidecar(args.output, '.trace.jsonl')
    manifest_path = ra.sidecar(args.output, '.manifest.json')
    consolidation_path = ra.sidecar(args.output, '.consolidation.jsonl')
    order_path = ra.sidecar(args.output, '.order.json')
    order = _processing_order(n_total, args, order_path)
    shuffled = args.shuffle_seed is not None
    # With shuffling, lines are written in PROCESSING order to a work file and
    # re-ordered into --output once every question is done.
    work_path = ra.sidecar(args.output, '.processing_order.tsv') if shuffled else out_path

    # ── Resume support ────────────────────────────────────────────────────────
    already_done = 0
    if args.resume and work_path.exists():
        with open(work_path) as f:
            already_done = sum(1 for ln in f if ln.strip())
        logger.info(f"Resuming from position {already_done}")
    todo = order[already_done:]

    # ── Init optional pipelines ───────────────────────────────────────────────
    semantic_pipeline  = None
    reasoning_pipeline = None
    rb_cfg_loaded = {}

    if args.use_semantic:
        try:
            from src.semantic.semantic_pipeline import SemanticPipeline
            cfg = load_config(args.semantic_config) or {'enabled': True}
            semantic_pipeline = SemanticPipeline(cfg)
            logger.info("✓ Semantic pipeline ready")
        except Exception as e:
            logger.warning(f"Semantic pipeline failed: {e}")

    memory_dir = _prepare_memory_dir(args)

    if args.use_reasoning_bank:
        try:
            from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
            cfg = load_config(args.reasoning_config) or {}
            rb_cfg_loaded = cfg
            pipeline_cfg = cfg.get('pipeline', {})
            db_path_rb = pipeline_cfg.get('db_path', './memory/reasoning_bank.db')
            chroma_rb = pipeline_cfg.get('chromadb_path', './memory/chromadb')
            if memory_dir is not None:
                db_path_rb = str(memory_dir / 'reasoning_bank.db')
                chroma_rb = str(memory_dir / 'chromadb')
            reasoning_pipeline = ReasoningBankPipeline(
                db_path=db_path_rb,
                chromadb_path=chroma_rb,
                config=cfg,
            )
            reasoning_pipeline.config['allow_gold_sql'] = bool(args.allow_gold_in_memory)
            if args.rb_frozen:
                reasoning_pipeline.config.update({
                    'enable_self_consistency_judging': False,
                    'enable_distillation': False,
                    'enable_consolidation': False,
                    'consolidation_frequency': 10 ** 9,
                })
            logger.info(f"✓ ReasoningBank ready (memory db={db_path_rb}, frozen={args.rb_frozen}, "
                        f"strategies at start={_memory_size(reasoning_pipeline)})")

            # ── Startup banner: confirm self-consistency/temperature config ──
            sc_enabled = reasoning_pipeline.config.get('enable_self_consistency_judging', True)
            n_cand = reasoning_pipeline.config.get('self_consistency_n_candidates', 4)
            sc = reasoning_pipeline.self_consistency
            temp_preview = sc.default_temperatures(n_cand)
            print("=" * 70)
            print("SELF-CONSISTENCY / TEMPERATURE CONFIG (checked at startup)")
            print("=" * 70)
            # (the first lines keep the exact format parsed by verify_self_consistency_config.sh)
            print(f"  enable_self_consistency_judging : {sc_enabled}")
            print(f"  n_additional_candidates          : {n_cand}")
            print(f"  temperatures to be used           : [0.0 (primary)] + {temp_preview}")
            print(f"  agreement_threshold                : {sc.agreement_threshold}")
            print(f"  consensus rule                   : primary in largest cluster with "
                  f">= {sc.min_agree_for(n_cand + 1)} of {n_cand + 1} candidates")
            print(f"  empty-result consensus           : {sc.empty_consensus}")
            print(f"  candidate prompt                 : "
                  f"{reasoning_pipeline.config.get('self_consistency_prompt', 'same_as_primary')}")
            print(f"  max_confidence cap               : {sc.max_confidence}")
            print(f"  gold SQL reaches memory          : {bool(args.allow_gold_in_memory)}")
            if args.resume:
                print("  ℹ --resume: the in-memory trajectory pool starts EMPTY in this process;")
                print("    strategies already distilled into the memory store are kept.")
            print("=" * 70 + "\n")
        except Exception as e:
            logger.warning(f"ReasoningBank failed: {e}")

    # ── Semantic RAG retriever ───────────────────────────────────────────────
    retriever = None
    if args.use_chromadb:
        try:
            from src.retrieval.retriever import SpiderRetriever
            is_wikisql_run = 'wikisql' in args.questions.lower() or 'wikisql' in args.db.lower()
            prefix = 'wikisql' if is_wikisql_run else 'spider'
            retriever = SpiderRetriever(
                persist_dir=args.chromadb_persist_dir,
                collection_prefix=prefix,
            )
            logger.info(f"✓ Semantic RAG retriever ready (prefix={prefix}, top_k={args.top_k})")
        except Exception as e:
            logger.warning(f"Semantic RAG retriever failed to load: {e}")

    # ── SQL generator ─────────────────────────────────────────────────────────
    try:
        from src.generation.sql_generator import SQLGenerator, spider_prompt_variant
        sql_generator = SQLGenerator()
        logger.info("✓ SQLGenerator ready")
    except Exception as e:
        logger.error(f"SQLGenerator failed to load: {e}")
        sys.exit(1)
    from src.models.google_genai import LLM_USAGE

    # ── Schema loader ─────────────────────────────────────────────────────────
    from utils.sql_schema import load_full_db_context

    # ── Manifest (written now, completed at the end) ─────────────────────────
    manifest = {
        "run_name": args.run_name,
        "started_at": datetime.now().isoformat(timespec='seconds'),
        "argv": sys.argv,
        "args": vars(args),
        "git": ra.git_info(PROJECT_ROOT),
        "environment": ra.environment_info(),
        "questions": {"path": args.questions, "sha256": ra.sha256_file(args.questions), "n": n_total},
        "components": {
            "semantic_layer": semantic_pipeline is not None,
            "semantic_rag": retriever is not None,
            "reasoning_bank": reasoning_pipeline is not None,
        },
        "llm": {
            "model_name": getattr(sql_generator, "model_name", None),
            "backend": getattr(getattr(sql_generator, "model", None), "backend", None),
            "primary_temperature": 0.0,
            "max_output_tokens": 2048,
            "spider_prompt_examples": spider_prompt_variant(),
        },
        "semantic_rag": None if retriever is None else {
            "top_k": args.top_k,
            "persist_dir": args.chromadb_persist_dir,
            "embedding_model": getattr(getattr(retriever, "embedding_gen", None), "model_name", None),
        },
        "reasoning_bank": None if reasoning_pipeline is None else {
            "config_path": args.reasoning_config,
            "memory_dir": str(memory_dir) if memory_dir else None,
            "frozen": args.rb_frozen,
            "fresh_memory": args.fresh_memory,
            "strategies_at_start": _memory_size(reasoning_pipeline),
            "effective_config": {k: v for k, v in reasoning_pipeline.config.items()
                                 if not isinstance(v, dict)},
            "strategy_retrieval_top_k": reasoning_pipeline.config.get('strategy_retrieval_top_k', 3),
            "self_consistency": {
                "n_candidates_total": reasoning_pipeline.config.get('self_consistency_n_candidates', 4) + 1,
                "temperatures": [0.0] + reasoning_pipeline.self_consistency.default_temperatures(
                    reasoning_pipeline.config.get('self_consistency_n_candidates', 4)),
                "agreement_threshold": reasoning_pipeline.self_consistency.agreement_threshold,
                "min_agree": reasoning_pipeline.self_consistency.min_agree_for(
                    reasoning_pipeline.config.get('self_consistency_n_candidates', 4) + 1),
                "empty_consensus": reasoning_pipeline.self_consistency.empty_consensus,
                "comparison": "result-set equality, bag semantics, row order ignored",
            },
            "embedding_model": getattr(getattr(reasoning_pipeline.memory_retrieval, "embedding_gen", None),
                                       "model_name", None),
        },
        "shuffle_seed": args.shuffle_seed,
    }
    if not args.resume or not manifest_path.exists():
        ra.write_json(manifest_path, manifest)
    else:
        prev = json.load(open(manifest_path))
        prev.setdefault("resumes", []).append({"at": manifest["started_at"], "git": manifest["git"],
                                               "from_position": already_done})
        manifest = {**prev}
        ra.write_json(manifest_path, manifest)

    # ── Self-consistency monitor (only meaningful when ReasoningBank is on) ──
    sc_monitor = SelfConsistencyMonitor()
    n_consolidations_logged = len(reasoning_pipeline.consolidation_log) if reasoning_pipeline else 0

    # ── Generate ──────────────────────────────────────────────────────────────
    mode = 'a' if args.resume else 'w'
    failed   = 0
    new_count = 0

    with open(work_path, mode, encoding='utf-8') as out_f, \
         open(trace_path, mode, encoding='utf-8') as trace_f, \
         open(consolidation_path, mode, encoding='utf-8') as cons_f:
        for i, orig_idx in enumerate(tqdm(todo, desc="Generating SQL")):
            step = already_done + i
            item = all_questions[orig_idx]
            question = item.get('question', '')
            db_id    = item.get('db_id', '')
            usage_before = LLM_USAGE.snapshot()
            t_start = time.time()
            trace = {"idx": orig_idx, "step": step, "db_id": db_id, "question": question}
            sql = 'SELECT 1'

            if not question or not db_id:
                failed += 1
                trace["error"] = "missing question or db_id"
            else:
                db_path = os.path.join(args.db, db_id, f"{db_id}.sqlite")
                if not os.path.exists(db_path):
                    logger.warning(f"DB not found: {db_path}")
                    failed += 1
                    trace["error"] = f"DB not found: {db_path}"
                else:
                    try:
                        # ── Load schema once (shared by Semantic Layer + ReasoningBank) ──
                        db_context = None
                        if semantic_pipeline or reasoning_pipeline:
                            try:
                                db_context = load_full_db_context(db_id, args.db)
                            except Exception as e:
                                logger.debug(f"Failed to load db context for {db_id}: {e}")

                        # ── Semantic Layer enhancement (rule-based hints) ─────
                        enhanced_question = question
                        semantic_hints = None
                        if semantic_pipeline:
                            try:
                                schema_for_semantic = (db_context or {}).get('schema', {})
                                res = semantic_pipeline.enhance_question(
                                    question, db_id, schema_for_semantic)
                                enhanced_question = res.get('enhanced_question', question)
                                semantic_hints = res.get('semantic_hints') or None
                            except Exception as e:
                                logger.debug(f"Semantic Layer enhancement failed: {e}")
                        if enhanced_question != question:
                            trace["enhanced_question"] = enhanced_question
                        trace["semantic_hints"] = bool(semantic_hints)

                        # ── Semantic RAG: retrieve few-shot examples Ek(q) ────
                        few_shot_examples = None
                        if retriever:
                            try:
                                rag_result = retriever.retrieve_similar_questions(
                                    enhanced_question, n_results=args.top_k
                                )
                                few_shot_examples = rag_result.get('results')
                            except Exception as e:
                                logger.debug(f"Semantic RAG retrieval failed: {e}")
                                few_shot_examples = None
                            trace["rag"] = [
                                {"question": e.get("question"), "db": e.get("database"),
                                 "sim": e.get("similarity_score"), "split": e.get("split")}
                                for e in (few_shot_examples or [])
                            ]

                        # ── ReasoningBank ─────────────────────────────────────
                        sql = ''
                        if reasoning_pipeline:
                            trace["memory_size_before"] = _memory_size(reasoning_pipeline)
                            try:
                                if db_context is None:
                                    db_context = load_full_db_context(db_id, args.db)
                                rb_result = reasoning_pipeline.generate_with_reasoning(
                                    question=enhanced_question,
                                    db_id=db_id,
                                    schema=db_context.get('schema', {}),
                                    # Gold is NOT given to the memory (paper protocol).
                                    gold_sql=(item.get('query', item.get('sql'))
                                              if args.allow_gold_in_memory else None),
                                    # db_path=None disables self-consistency labelling
                                    db_path=None if args.rb_frozen else db_path,
                                    sql_generator=lambda q, strategy_hints=None, temperature=0.0: sql_generator.generate(
                                        q, db_path,
                                        few_shot_examples=few_shot_examples,
                                        semantic_hints=semantic_hints,
                                        strategy_hints=strategy_hints,
                                        temperature=temperature),
                                )
                                sql = rb_result.get('sql', '') or ''
                                trace["strategies"] = rb_result.get('strategies_used', [])
                                sc_meta = (rb_result.get('metadata') or {}).get('self_consistency')
                                if sc_meta:
                                    trace["self_consistency"] = sc_meta
                                sc_monitor.record(sc_meta)

                            except Exception as e:
                                # Re-raise 5xx immediately — do NOT fall back
                                _stop_on_server_error(e, out_f, already_done, i,
                                                      args.output)
                                logger.debug(f"ReasoningBank failed: {e}, falling back")
                                trace["rb_error"] = str(e)[:300]
                                sql = ''

                        # ── Plain generation fallback ─────────────────────────
                        if not sql or sql.strip().upper() == 'SELECT 1':
                            if reasoning_pipeline:
                                trace["rb_fallback"] = True
                            sql = sql_generator.generate(
                                enhanced_question, db_path,
                                few_shot_examples=few_shot_examples,
                                semantic_hints=semantic_hints)

                        sql = sql or 'SELECT 1'

                    except Exception as e:
                        _stop_on_server_error(e, out_f, already_done, i, args.output)
                        logger.error(f"[{orig_idx}] Generation failed: {e}")
                        sql = 'SELECT 1'
                        failed += 1
                        trace["error"] = str(e)[:300]

            sql_oneline = sql.replace('\n', ' ').strip()
            out_f.write(f"{sql_oneline}\t{db_id}\n")
            out_f.flush()
            new_count += 1

            trace["sql"] = sql_oneline
            trace["wall_s"] = round(time.time() - t_start, 3)
            trace["llm"] = LLM_USAGE.delta(LLM_USAGE.snapshot(), usage_before)
            if reasoning_pipeline:
                trace["memory_size_after"] = _memory_size(reasoning_pipeline)
                new_events = reasoning_pipeline.consolidation_log[n_consolidations_logged:]
                for ev_ in new_events:
                    cons_f.write(json.dumps({"step": step, "idx": orig_idx, **ev_}, default=str) + "\n")
                cons_f.flush()
                n_consolidations_logged += len(new_events)
                if new_events:
                    trace["consolidated"] = True
            trace_f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
            trace_f.flush()

            # ── Checkpoint size check ─────────────────────────────────────────
            if args.checkpoint_size and new_count >= args.checkpoint_size:
                logger.info(
                    f"\n✓ Checkpoint: {new_count} new predictions written "
                    f"({already_done + new_count} total). "
                    f"Re-run with --resume to continue."
                )
                break

    total = already_done + new_count
    complete = total >= n_total

    # ── Final consolidation: give THIS process's in-memory trajectory pool one
    # last chance to distill before exiting (ExperienceCollector is RAM-only).
    if reasoning_pipeline and not args.rb_frozen:
        try:
            logger.info("Running final memory consolidation for this run...")
            consolidation_result = reasoning_pipeline.consolidate_memory()
            logger.info(
                f"✓ Final consolidation: {consolidation_result.get('new_strategies', 0)} "
                f"new strategies distilled from "
                f"{consolidation_result.get('total_trajectories', 0)} trajectories "
                f"collected in this run."
            )
            with open(consolidation_path, 'a', encoding='utf-8') as cons_f:
                for ev_ in reasoning_pipeline.consolidation_log[n_consolidations_logged:]:
                    cons_f.write(json.dumps({"step": total, "final": True, **ev_}, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Final consolidation failed (non-critical): {e}")

    # ── Restore original order when shuffling ────────────────────────────────
    if shuffled and complete:
        lines = ra.read_tsv_predictions(work_path)
        by_idx = {}
        for pos, line in enumerate(lines[:n_total]):
            by_idx[order[pos]] = line
        with open(out_path, 'w', encoding='utf-8') as f:
            for k in range(n_total):
                f.write(by_idx[k] + "\n")
        logger.info(f"✓ Re-ordered {n_total} predictions into original order → {out_path}")

    logger.info(f"\n✓ Predictions saved → {args.output}" + ("" if complete else " (incomplete)"))
    logger.info(f"  Total: {total}/{n_total} | New this run: {new_count} | Failed/fallback: {failed}")

    # ── Majority-vote baseline from self-consistency candidates ──────────────
    if complete and reasoning_pipeline and not args.rb_frozen:
        mv = ra.write_majority_vote_tsv(trace_path, out_path, ra.sidecar(args.output, '.majority_vote.tsv'))
        logger.info(f"✓ Majority-vote predictions → {mv['path']} "
                    f"({mv['replaced']} differ from the greedy prediction)")
        manifest["majority_vote"] = mv

    # ── Finish manifest with cost / timing totals ────────────────────────────
    trace_rows = list(ra.read_trace(trace_path).values())
    manifest["finished_at"] = datetime.now().isoformat(timespec='seconds')
    manifest["complete"] = complete
    manifest["n_done"] = total
    manifest["n_failed_this_process"] = failed
    manifest["cost"] = ra.summarize_cost(trace_rows)
    if reasoning_pipeline:
        manifest.setdefault("reasoning_bank", {})
        if manifest["reasoning_bank"] is not None:
            manifest["reasoning_bank"]["strategies_at_end"] = _memory_size(reasoning_pipeline)
            manifest["reasoning_bank"]["pipeline_stats"] = dict(reasoning_pipeline.stats)
    ra.write_json(manifest_path, manifest)
    c = manifest["cost"]
    if c.get("n_questions"):
        logger.info(f"  LLM calls/question: {c['calls_mean']}  tokens in/out per question: "
                    f"{c['prompt_tokens_mean']}/{c['completion_tokens_mean']}  "
                    f"latency/question: {c['wall_s_mean']}s")
    logger.info(f"  Manifest → {manifest_path}   Trace → {trace_path}")

    # ── Print + save self-consistency monitor summary ────────────────────────
    if args.use_reasoning_bank:
        sc_monitor.print_summary()
        sc_monitor.save(args.output, merge_with_existing=args.resume)


if __name__ == '__main__':
    main()