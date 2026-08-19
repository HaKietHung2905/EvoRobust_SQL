"""
Step 1: Generate SQL predictions and save to a TSV file.
Step 2: Run evaluate_wikisql.py / evaluate_spider.py with --predict <file>
        for fast, LLM-free evaluation.

Usage (WikiSQL):
  python scripts/generate_predictions.py \
      --questions data/raw/wikisql/dev_spider_format.json \
      --db        data/raw/wikisql/database \
      --output    results/predictions_wikisql.tsv \
      --use_reasoning_bank --use_chromadb --use_semantic \
      --limit 50

Usage (Spider):
  python scripts/generate_predictions.py \
      --questions data/spider/dev.json \
      --db        data/spider/database \
      --output    results/predictions_spider.tsv
"""

import sys
import os
import json
import argparse
import logging
import warnings
import time
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
logger = get_logger(__name__)


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

    def save(self, output_path: str):
        stats_path = Path(output_path).with_suffix('.self_consistency_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
        logger.info(f"Self-consistency monitor stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate SQL predictions to TSV file')

    parser.add_argument('--questions', required=True,
                        help='JSON questions file (Spider or WikiSQL spider-format)')
    parser.add_argument('--db', required=True,
                        help='Database directory')
    parser.add_argument('--output', default='./results/predictions.tsv',
                        help='Output TSV file (sql TAB db_id per line)')

    # Feature flags
    parser.add_argument('--use_chromadb',      action='store_true')
    parser.add_argument('--chromadb_config',   default=None)
    parser.add_argument('--chromadb_persist_dir', default='./data/embeddings/chroma_db',
                        help='ChromaDB persist directory for Semantic RAG retrieval')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of few-shot examples to retrieve via Semantic RAG')
    parser.add_argument('--use_semantic',      action='store_true')
    parser.add_argument('--semantic_config',   default=None)
    parser.add_argument('--use_reasoning_bank', action='store_true')
    parser.add_argument('--reasoning_config',
                        default='./configs/reasoning_config.yaml')

    parser.add_argument('--limit',  type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-generated lines (resume interrupted run)')
    parser.add_argument('--checkpoint_size', type=int, default=None,
                        help='Stop after generating this many NEW predictions '
                             '(re-run with --resume to continue)')

    args = parser.parse_args()

    # ── Auto-prepare WikiSQL if spider_format file doesn't exist ─────────────
    if not Path(args.questions).exists() and 'wikisql' in args.questions.lower():
        logger.info(f"Questions file not found: {args.questions}")
        logger.info("Auto-preparing WikiSQL (building SQLite DBs + converting format)...")
        gold_file = args.questions.replace('_spider_format.json', '.json')
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
        logger.info(f"✓ Spider-format file created: {args.questions}")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.questions, 'r') as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]
    logger.info(f"Loaded {len(questions)} questions")

    # ── Resume support ────────────────────────────────────────────────────────
    already_done = 0
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            already_done = sum(1 for ln in f if ln.strip())
        logger.info(f"Resuming from question {already_done}")
        questions = questions[already_done:]

    # ── Init optional pipelines ───────────────────────────────────────────────
    semantic_pipeline  = None
    reasoning_pipeline = None

    if args.use_semantic:
        try:
            from src.semantic.semantic_pipeline import SemanticPipeline
            cfg = load_config(args.semantic_config) or {'enabled': True}
            semantic_pipeline = SemanticPipeline(cfg)
            logger.info("✓ Semantic pipeline ready")
        except Exception as e:
            logger.warning(f"Semantic pipeline failed: {e}")

    if args.use_reasoning_bank:
        try:
            from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
            cfg = load_config(args.reasoning_config) or {}
            pipeline_cfg = cfg.get('pipeline', {})
            reasoning_pipeline = ReasoningBankPipeline(
                db_path=pipeline_cfg.get('db_path', './memory/reasoning_bank.db'),
                chromadb_path=pipeline_cfg.get('chromadb_path', './memory/chromadb'),
                config=cfg,
            )
            logger.info("✓ ReasoningBank ready")

            # ── Startup banner: confirm self-consistency/temperature config ──
            sc_enabled = reasoning_pipeline.config.get('enable_self_consistency_judging', True)
            n_cand = reasoning_pipeline.config.get('self_consistency_n_candidates', 4)
            agree_thr = reasoning_pipeline.self_consistency.agreement_threshold
            max_conf = reasoning_pipeline.self_consistency.max_confidence
            if n_cand <= 1:
                temp_preview = [0.6]
            else:
                lo, hi = 0.3, 0.9
                step = (hi - lo) / (n_cand - 1)
                temp_preview = [round(lo + i * step, 2) for i in range(n_cand)]

            print("=" * 70)
            print("SELF-CONSISTENCY / TEMPERATURE CONFIG (checked at startup)")
            print("=" * 70)
            print(f"  enable_self_consistency_judging : {sc_enabled}")
            print(f"  n_additional_candidates          : {n_cand}")
            print(f"  temperatures to be used           : [0.0 (primary)] + {temp_preview}")
            print(f"  agreement_threshold                : {agree_thr}")
            print(f"  max_confidence cap                  : {max_conf}")
            if not sc_enabled:
                print("  ⚠ Self-consistency is DISABLED — every trajectory will use only the")
                print("    primary greedy candidate; agreement_ratio will never be computed.")
            elif n_cand <= 1 or all(t == 0.0 for t in temp_preview):
                print("  ⚠ Effective temperature spread looks degenerate — check config.")
            else:
                print("  ✓ Temperature sampling is configured correctly for this run.")
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
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        logger.info("✓ SQLGenerator ready")
    except Exception as e:
        logger.error(f"SQLGenerator failed to load: {e}")
        sys.exit(1)

    # ── Schema loader ─────────────────────────────────────────────────────────
    from utils.sql_schema import load_full_db_context

    # ── Self-consistency monitor (only meaningful when ReasoningBank is on) ──
    sc_monitor = SelfConsistencyMonitor()

    # ── Generate ──────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if args.resume else 'w'

    failed   = 0
    new_count = 0

    with open(args.output, mode, encoding='utf-8') as out_f:
        for i, item in enumerate(tqdm(questions, desc="Generating SQL")):
            question = item.get('question', '')
            db_id    = item.get('db_id', '')

            if not question or not db_id:
                out_f.write(f"SELECT 1\t{db_id}\n")
                out_f.flush()
                failed    += 1
                new_count += 1

            else:
                db_path = os.path.join(args.db, db_id, f"{db_id}.sqlite")
                if not os.path.exists(db_path):
                    logger.warning(f"DB not found: {db_path}")
                    out_f.write(f"SELECT 1\t{db_id}\n")
                    out_f.flush()
                    failed    += 1
                    new_count += 1

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

                        # ── ReasoningBank ─────────────────────────────────────
                        sql = ''
                        if reasoning_pipeline:
                            try:
                                if db_context is None:
                                    db_context = load_full_db_context(db_id, args.db)
                                rb_result = reasoning_pipeline.generate_with_reasoning(
                                    question=enhanced_question,
                                    db_id=db_id,
                                    schema=db_context.get('schema', {}),
                                    gold_sql=item.get('query', item.get('sql')),
                                    db_path=db_path,
                                    sql_generator=lambda q, strategy_hints=None, temperature=0.0: sql_generator.generate(
                                        q, db_path,
                                        few_shot_examples=few_shot_examples,
                                        semantic_hints=semantic_hints,
                                        strategy_hints=strategy_hints,
                                        temperature=temperature),
                                )
                                sql = rb_result.get('sql', '') or ''

                                # ── Monitor: capture self-consistency agreement_ratio ──
                                sc_meta = (rb_result.get('metadata') or {}).get('self_consistency')
                                sc_monitor.record(sc_meta)

                            except Exception as e:
                                # Re-raise 5xx immediately — do NOT fall back
                                _stop_on_server_error(e, out_f, already_done, i,
                                                      args.output)
                                logger.debug(f"ReasoningBank failed: {e}, falling back")
                                sql = ''

                        # ── Plain generation fallback ─────────────────────────
                        if not sql or sql.strip().upper() == 'SELECT 1':
                            sql = sql_generator.generate(
                                enhanced_question, db_path,
                                few_shot_examples=few_shot_examples,
                                semantic_hints=semantic_hints)

                        sql = sql or 'SELECT 1'

                    except Exception as e:
                        _stop_on_server_error(e, out_f, already_done, i, args.output)
                        logger.error(f"[{already_done + i}] Generation failed: {e}")
                        sql = 'SELECT 1'
                        failed += 1

                    sql_oneline = sql.replace('\n', ' ').strip()
                    out_f.write(f"{sql_oneline}\t{db_id}\n")
                    out_f.flush()
                    new_count += 1

            # ── Checkpoint size check ─────────────────────────────────────────
            if args.checkpoint_size and new_count >= args.checkpoint_size:
                logger.info(
                    f"\n✓ Checkpoint: {new_count} new predictions written "
                    f"({already_done + new_count} total). "
                    f"Re-run with --resume to continue."
                )
                break

    total = already_done + new_count
    logger.info(f"\n✓ Predictions saved → {args.output}")
    logger.info(f"  Total: {total} | New this run: {new_count} | Failed/fallback: {failed}")
    logger.info(f"\nNext — evaluate without any LLM calls:")
    logger.info(f"  python scripts/evaluate_wikisql.py \\")
    logger.info(f"      --gold  data/raw/wikisql/dev_spider_format.json \\")
    logger.info(f"      --table data/raw/wikisql/tables.json \\")
    logger.info(f"      --predict {args.output} \\")
    logger.info(f"      --etype all")

    # ── Print + save self-consistency monitor summary ────────────────────────
    if args.use_reasoning_bank:
        sc_monitor.print_summary()
        sc_monitor.save(args.output)


if __name__ == '__main__':
    main()