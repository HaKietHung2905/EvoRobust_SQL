#!/usr/bin/env python3
"""
find_em_pass_ex_fail.py
========================
Đối chiếu per-query giữa Structural EM (đã strict) và Execution Accuracy thật
để tìm chính xác những dòng "EM đúng nhưng EX sai" — đây là bằng chứng trực
tiếp cho phần leniency còn sót lại (nếu patch COND đã chạy) hoặc bằng chứng
patch chưa chạy (nếu số lượng dòng loại này vẫn lớn và đều rơi vào lỗi cond).

Usage:
    python find_em_pass_ex_fail.py \
        --eval_results results/evaluation_results.json \
        --em_failures_csv results/em_failures_din_sql_wikisql.csv \
        --total 8421 \
        --out results/em_pass_ex_fail_samples.csv
"""
import argparse
import csv
import json
from pathlib import Path


def load_exec_results(eval_results_path, expected_total=None):
    """
    Vị trí (0-based) trong list detailed_results = index thực tế,
    vì schema của src.reasoning.evaluator KHÔNG có field 'index'.
    """
    with open(eval_results_path, encoding="utf-8") as f:
        data = json.load(f)

    if expected_total is not None and len(data) != expected_total:
        print(f"⚠️  CẢNH BÁO: evaluation_results.json có {len(data)} dòng, "
              f"khác --total={expected_total}. Một số dòng có thể đã bị SKIP "
              f"(vd db_id rỗng) → alignment với line_no của structural EM CSV "
              f"có thể bị lệch. Kết quả dưới đây chỉ mang tính tham khảo, "
              f"cần đối chiếu lại bằng db_id/question nếu số liệu bất thường.")

    return {i: row for i, row in enumerate(data)}

def load_em_failed_line_nos(em_failures_csv_path):
    failed = set()
    with open(em_failures_csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            em_val = row.get("em", "").strip()
            if em_val in ("False", "false", "0", ""):
                failed.add(int(row["line_no"]))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_results", required=True)
    ap.add_argument("--em_failures_csv", required=True)
    ap.add_argument("--total", type=int, required=True,
                     help="Tổng số dòng đã evaluate (vd 8421)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    exec_by_index = load_exec_results(args.eval_results, expected_total=args.total)
    failed_line_nos = load_em_failed_line_nos(args.em_failures_csv)

    # line_no trong CSV là 1-based, index trong evaluation_results.json là 0-based
    em_pass_line_nos = set(range(1, args.total + 1)) - failed_line_nos

    mismatches = []
    for line_no in em_pass_line_nos:
        idx = line_no - 1
        row = exec_by_index.get(idx)
        if row is None:
            continue
        if not row.get("execution_match", True):
            mismatches.append({
                "line_no": line_no,
                "db_id": row.get("db_id"),
                "question": row.get("question"),
                "gold_sql": row.get("gold_sql"),
                "predicted_sql": row.get("predicted_sql"),
            })

    print(f"Tổng dòng EM=pass nhưng EX=fail: {len(mismatches)} / {len(em_pass_line_nos)} dòng EM-pass")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line_no", "db_id", "question", "gold_sql", "predicted_sql"])
        writer.writeheader()
        writer.writerows(mismatches)
    print(f"Đã lưu chi tiết → {args.out}")

    # In mẫu 10 dòng đầu để xem nhanh nguyên nhân
    print("\n--- Mẫu 10 dòng đầu ---")
    for m in mismatches[:10]:
        print(f"[{m['line_no']}] Q: {m['question'][:60]}")
        print(f"    gold : {m['gold_sql']}")
        print(f"    pred : {m['predicted_sql']}")
        print()


if __name__ == "__main__":
    main()