import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.robustness.compare_configs import _build_report, _render_markdown, CONFIGS

def main(comparison_dir: str):
    out_dir = Path(comparison_dir)
    all_results = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        eval_path = out_dir / f"eval_{name}.json"
        if not eval_path.exists():
            print(f"⚠ Missing {eval_path}, skipping")
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            all_results[name] = json.load(f)

    report = _build_report(all_results)

    (out_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "comparison_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(f"✓ Rebuilt comparison_report.md/.json in {out_dir}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/robustness_smoke/comparison")