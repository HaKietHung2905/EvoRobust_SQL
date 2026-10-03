import json
import sys
from pathlib import Path

def main(noisy_path: str, eval_path: str):
    noisy_data = json.load(open(noisy_path, encoding="utf-8"))
    eval_data = json.load(open(eval_path, encoding="utf-8"))
    detailed = eval_data["detailed_results"]

    if len(noisy_data) != len(detailed):
        print(f"⚠ Length mismatch: noisy_dev.json={len(noisy_data)} vs "
              f"detailed_results={len(detailed)} — index alignment may be unsafe")

    n = min(len(noisy_data), len(detailed))
    mismatched_db = 0
    was_noised = []
    is_wrong = []

    for i in range(n):
        nd = noisy_data[i]
        dr = detailed[i]
        if nd.get("db_id") != dr.get("db_id"):
            mismatched_db += 1  # sanity check the positional alignment

        original = nd.get("original_question", nd.get("question"))
        noised_flag = nd.get("question") != original
        was_noised.append(noised_flag)

        wrong_flag = not dr.get("execution_match", False)
        is_wrong.append(wrong_flag)

    if mismatched_db:
        print(f"⚠ {mismatched_db}/{n} câu có db_id lệch giữa 2 file — "
              f"kết quả bên dưới có thể không đáng tin, cần kiểm tra lại thứ tự")

    total_noised = sum(was_noised)
    total_wrong = sum(is_wrong)
    wrong_and_noised = sum(1 for w, no in zip(is_wrong, was_noised) if w and no)
    wrong_and_clean = sum(1 for w, no in zip(is_wrong, was_noised) if w and not no)
    right_but_noised = sum(1 for w, no in zip(is_wrong, was_noised) if not w and no)

    print("=" * 60)
    print(f"Tổng số câu                          : {n}")
    print(f"Số câu bị EA áp noise (question đổi)  : {total_noised} ({total_noised/n:.1%})")
    print(f"Tổng số câu baseline dự đoán SAI       : {total_wrong} ({total_wrong/n:.1%})")
    print("-" * 60)
    print(f"Sai VÀ có noise (lỗi liên quan chính tả)      : {wrong_and_noised}")
    print(f"Sai NHƯNG không noise (lỗi model thuần túy)   : {wrong_and_clean}")
    print(f"Đúng dù có noise (model chịu được noise đó)   : {right_but_noised}")
    if total_noised:
        print(f"→ Tỷ lệ lỗi TRONG nhóm câu bị noise : {wrong_and_noised/total_noised:.1%}")
    clean_count = n - total_noised
    if clean_count:
        print(f"→ Tỷ lệ lỗi TRONG nhóm câu KHÔNG noise (sạch): {wrong_and_clean/clean_count:.1%}")
    print("=" * 60)

if __name__ == "__main__":
    noisy_path = sys.argv[1] if len(sys.argv) > 1 else "output/robustness_smoke/noisy_dev.json"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "output/robustness_smoke/comparison/eval_baseline.json"
    main(noisy_path, eval_path)