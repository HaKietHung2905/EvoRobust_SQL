#!/usr/bin/env python3
"""
find_spider_em_ex_mismatch.py
==============================
Kiểm tra Spider evaluation có bị lỗi tương tự WikiSQL không: tìm các dòng
Exact Match (EM) = True nhưng Execution Accuracy (EX) = False.

Khác WikiSQL: Spider dùng official evaluator (BaseEvaluator.eval_exact_match,
dựa trên component-wise F1 + schema-resolved column ID, không có lớp fuzzy
matching tự chế), nên 'exact_match' và 'execution_match' đã có sẵn ngay
trong CÙNG một dòng của evaluation_results.json — không cần script gộp 2
nguồn như bên WikiSQL.

Usage:
    python find_spider_em_ex_mismatch.py \
        --eval_results results/evaluation_results.json \
        --out results/spider_em_pass_ex_fail.csv
"""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.eval_results, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    em_pass = [r for r in data if r.get("exact_match")]
    mismatches = [r for r in em_pass if not r.get("execution_match", True)]

    # Chiều ngược lại (EX đúng, EM sai) — bình thường và mong đợi với Spider,
    # chỉ in ra để đối chiếu tỉ lệ, không phải dấu hiệu lỗi.
    ex_pass = [r for r in data if r.get("execution_match")]
    reverse_mismatches = [r for r in ex_pass if not r.get("exact_match", True)]

    hardness_counter = Counter(r.get("hardness", "unknown") for r in mismatches)

    print("=" * 70)
    print("  SPIDER — KIỂM TRA EM=True nhưng EX=False")
    print("=" * 70)
    print(f"  Tổng số dòng evaluate                      : {total}")
    print(f"  Số dòng EM=True                             : {len(em_pass)}")
    print(f"  Số dòng EM=True nhưng EX=False (BẤT THƯỜNG) : {len(mismatches)}")
    if em_pass:
        print(f"  Tỷ lệ mismatch / EM-pass                    : {len(mismatches)/len(em_pass):.2%}")
    print("-" * 70)
    print(f"  (Tham khảo) EX=True nhưng EM=False (bình thường): {len(reverse_mismatches)}")
    print("=" * 70)

    if hardness_counter:
        print("  Phân bố mismatch theo độ khó (hardness):")
        for level, cnt in hardness_counter.most_common():
            print(f"    {level:<10}: {cnt}")
        print("=" * 70)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "db_id", "hardness", "gold_sql", "predicted_sql"]
        )
        writer.writeheader()
        for r in mismatches:
            writer.writerow({
                "question": r.get("question"),
                "db_id": r.get("db_id"),
                "hardness": r.get("hardness"),
                "gold_sql": r.get("gold_sql"),
                "predicted_sql": r.get("predicted_sql"),
            })
    print(f"  Đã lưu chi tiết → {args.out}")

    if mismatches:
        print("\n  --- Mẫu 10 dòng đầu ---")
        for r in mismatches[:10]:
            q = (r.get("question") or "")[:60]
            print(f"  [{r.get('hardness')}] Q: {q}")
            print(f"      gold : {r.get('gold_sql')}")
            print(f"      pred : {r.get('predicted_sql')}")
            print()
    else:
        print("\n  Không có dòng nào bất thường — Spider evaluation KHÔNG bị lỗi tương tự WikiSQL.")


if __name__ == "__main__":
c    main()