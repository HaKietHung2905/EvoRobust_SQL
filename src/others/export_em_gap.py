# scripts/export_em_gap.py
"""
Xuất các câu 'execution đúng nhưng EM sai' ở mức hard/extra
từ results/evaluation_results.json (do evaluate_spider.py sinh ra).
"""
import json
import csv
from pathlib import Path

INPUT  = "results/evaluation_results.json"
OUTPUT = "results/em_gap_hard_extra.csv"

with open(INPUT, encoding="utf-8") as f:
    results = json.load(f)

rows = []
for r in results.get("detailed_results", []):
    if (
        r.get("hardness") in ("hard", "extra")
        and r.get("exact_match") is False
        and r.get("execution_match") is True
    ):
        rows.append({
            "index":        r.get("index"),
            "db_id":        r.get("db_id"),
            "hardness":     r.get("hardness"),
            "question":     r.get("question"),
            "gold_sql":     r.get("gold_sql"),
            "predicted_sql": r.get("predicted_sql"),
        })

Path("results").mkdir(exist_ok=True)
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                             ["index","db_id","hardness","question","gold_sql","predicted_sql"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ {len(rows)} câu lệch EM (execution đúng, EM sai) ở hard/extra")
print(f"✓ Đã ghi ra: {OUTPUT}")

# In nhanh 10 ví dụ đầu để soi ngay trên terminal
for r in rows[:10]:
    print("-" * 80)
    print(f"[{r['hardness']}] db={r['db_id']}")
    print(f"Q     : {r['question']}")
    print(f"GOLD  : {r['gold_sql']}")
    print(f"PRED  : {r['predicted_sql']}")