"""
Phân loại loại lỗi chính tả (typo type) trong các câu hỏi bị nhiễu.

Chạy trên máy của bạn (nơi có output/robustness_smoke/):

    python3 analyze_typo_types.py output/robustness_smoke/noisy_dev.json

Không gọi LLM, chỉ so khớp text (original_question vs question) bằng difflib
ở cấp độ TỪ rồi cấp độ KÝ TỰ để suy ra operator nào đã tác động.

Một câu có thể chứa NHIỀU loại lỗi cùng lúc (vì ops_per_word > 1 hoặc nhiều từ
bị biến đổi) -> tổng số lượt lỗi có thể > 826.
"""
import sys
import json
import difflib
from collections import Counter


def classify_word_pair(w1: str, w2: str) -> str:
    """So sánh một cặp từ (gốc, nhiễu) và đoán loại lỗi ở cấp ký tự."""
    if w1 == w2:
        return None
    l1, l2 = len(w1), len(w2)

    # Transposition: hoán vị 2 ký tự liền kề (swap)
    if l1 == l2:
        diffs = [i for i in range(l1) if w1[i] != w2[i]]
        if len(diffs) == 2 and diffs[1] - diffs[0] == 1:
            i, j = diffs
            if w1[i] == w2[j] and w1[j] == w2[i]:
                return "transposition (hoán vị ký tự liền kề)"
        if len(diffs) == 1:
            return "substitution (thay 1 ký tự - có thể do phím kề nhau)"
        return "multi-char substitution (thay nhiều ký tự)"

    # Insertion: nhiễu dài hơn gốc 1 ký tự
    if l2 == l1 + 1:
        # kiểm tra double-letter (lặp ký tự)
        for i in range(l1):
            if w2[:i] + w2[i+1:] == w1 and w2[i] == (w1[i-1] if i > 0 else w1[i]):
                return "doubling (lặp ký tự)"
        return "insertion (chèn thêm ký tự)"

    # Deletion: nhiễu ngắn hơn gốc 1 ký tự
    if l2 == l1 - 1:
        return "deletion (xóa bớt ký tự)"

    if l2 > l1:
        return "insertion (chèn nhiều ký tự)"
    if l2 < l1:
        return "deletion (xóa nhiều ký tự)"
    return "other (khác)"


def classify_question(orig: str, noisy: str) -> list:
    """Trả về danh sách các loại lỗi phát hiện được trong 1 câu."""
    if orig == noisy:
        return []

    types = []

    # 1) Phát hiện space_merge / space_split bằng cách so số từ
    w_orig = orig.split()
    w_noisy = noisy.split()

    if len(w_noisy) < len(w_orig):
        types.append("space_merge (gộp 2 từ thành 1, mất dấu cách)")
    if len(w_noisy) > len(w_orig):
        types.append("space_split (tách 1 từ thành 2, thêm dấu cách)")

    # 2) Alignment theo từ để tìm các từ bị đổi (dùng SequenceMatcher)
    sm = difflib.SequenceMatcher(a=w_orig, b=w_noisy, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            # so từng cặp từ tương ứng
            for a, b in zip(w_orig[i1:i2], w_noisy[j1:j2]):
                t = classify_word_pair(a, b)
                if t:
                    types.append(t)
        elif tag in ("replace", "delete", "insert"):
            # số từ lệch nhau do merge/split đã bắt ở trên, nhưng vẫn note thêm
            types.append(f"word-level {tag} (thay đổi cấu trúc từ)")

    if not types:
        types.append("other (không xác định rõ pattern)")

    return types


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_typo_types.py <path_to_noisy_dev.json>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_noised = 0
    per_question_type_count = Counter()  # số câu có chứa >=1 loại lỗi X
    per_occurrence_type_count = Counter()  # tổng số lượt lỗi loại X (đếm cả khi 1 câu có nhiều lỗi)

    for item in data:
        orig = item.get("original_question")
        noisy = item.get("question")
        if orig is None or noisy is None:
            continue
        if orig == noisy:
            continue
        total_noised += 1
        types = classify_question(orig, noisy)
        seen_in_this_q = set()
        for t in types:
            per_occurrence_type_count[t] += 1
            seen_in_this_q.add(t)
        for t in seen_in_this_q:
            per_question_type_count[t] += 1

    print(f"Tổng số câu bị nhiễu (original != noisy): {total_noised}\n")

    print("=== Số CÂU HỎI có chứa mỗi loại lỗi (1 câu có thể tính nhiều loại) ===")
    for t, c in per_question_type_count.most_common():
        pct = 100 * c / total_noised if total_noised else 0
        print(f"  {t:55s} : {c:5d} câu ({pct:5.1f}%)")

    print("\n=== Tổng số LƯỢT lỗi theo từng loại (đếm cả khi 1 câu có >1 lỗi cùng loại) ===")
    for t, c in per_occurrence_type_count.most_common():
        print(f"  {t:55s} : {c:5d} lượt")


if __name__ == "__main__":
    main()