# scripts/compare_config_flips.py
import json
import sys

def load_results(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["detailed_results"]

def main(dir_path: str, config_a: str, config_b: str):
    a = load_results(f"{dir_path}/eval_{config_a}.json")
    b = load_results(f"{dir_path}/eval_{config_b}.json")

    if len(a) != len(b):
        print(f"⚠ Length mismatch: {config_a}={len(a)} vs {config_b}={len(b)}")
        return

    n = len(a)
    mismatched_db = sum(1 for i in range(n) if a[i].get("db_id") != b[i].get("db_id"))
    if mismatched_db:
        print(f"⚠ {mismatched_db}/{n} db_id lệch giữa 2 file — kết quả dưới có thể sai lệch")

    a_right_b_wrong = []  # config_a đúng nhưng config_b (thêm ReasoningBank) sai -> bị "lật xấu"
    b_right_a_wrong = []  # ngược lại -> ReasoningBank cứu được câu mà RAG đơn lẻ sai

    for i in range(n):
        ra, rb = a[i], b[i]
        a_ok = ra.get("execution_match", False)
        b_ok = rb.get("execution_match", False)
        if a_ok and not b_ok:
            a_right_b_wrong.append((i, ra, rb))
        elif not a_ok and b_ok:
            b_right_a_wrong.append((i, ra, rb))

    print("=" * 70)
    print(f"So sánh: {config_a}  vs  {config_b}")
    print(f"Tổng số câu: {n}")
    print(f"{config_a} ĐÚNG nhưng {config_b} SAI (bị lật xấu) : {len(a_right_b_wrong)}")
    print(f"{config_a} SAI nhưng {config_b} ĐÚNG (được cứu)   : {len(b_right_a_wrong)}")
    print(f"Chênh lệch ròng: {len(a_right_b_wrong) - len(b_right_a_wrong)} câu "
          f"(dương = {config_b} tệ hơn ròng)")
    print("=" * 70)

    print(f"\n--- Chi tiết các câu bị LẬT XẤU (đúng ở {config_a}, sai ở {config_b}) ---")
    for i, ra, rb in a_right_b_wrong[:20]:
        print(f"[{i}] db={ra.get('db_id')}")
        print(f"    gold      : {ra.get('gold_sql')}")
        print(f"    {config_a:30s}: {ra.get('predicted_sql')}")
        print(f"    {config_b:30s}: {rb.get('predicted_sql')}")
        print(f"    error({config_b}): {rb.get('error')}")
        print()

if __name__ == "__main__":
    dir_path = sys.argv[1] if len(sys.argv) > 1 else "output/robustness_smoke/comparison"
    main(dir_path, "semantic_rag", "semantic_rag_and_reasoning_bank")