"""
Chay: python3 patch_self_consistency_debug.py
Them print() truc tiep vao trong self_consistency.py de xem tung candidate
duoc sinh ra nhu the nao (thanh cong/that bai, SQL gi, temperature nao).
"""

path = "src/reasoning/self_consistency.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = """        # ── Generate extra candidates ────────────────────────────────────────
        for temp in temperatures:
            start = time.time()
            try:
                sql = candidate_generator_fn(temp)
                if not sql or not sql.strip():"""

new_block = """        # ── Generate extra candidates ────────────────────────────────────────
        print(f"[SC-DEBUG] temperatures list = {temperatures}", flush=True)
        for temp in temperatures:
            start = time.time()
            try:
                sql = candidate_generator_fn(temp)
                print(f"[SC-DEBUG] temp={temp} -> sql={sql[:80] if sql else sql!r}", flush=True)
                if not sql or not sql.strip():"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)

    # Them print() vao nhanh except cua vong lap generate candidate
    old_except = """            except Exception as e:
                logger.debug(f"Self-consistency candidate generation failed at temp={temp}: {e}")"""
    new_except = """            except Exception as e:
                print(f"[SC-DEBUG] temp={temp} FAILED: {e}", flush=True)
                logger.debug(f"Self-consistency candidate generation failed at temp={temp}: {e}")"""
    if old_except in content:
        content = content.replace(old_except, new_except, 1)
        print("Da them print() vao ca vong lap chinh VA except block - OK")
    else:
        print("Da them print() vao vong lap chinh, nhung KHONG tim thay except block chinh xac.")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
else:
    print("KHONG TIM THAY block can patch - paste output cua:")
    print('  grep -n -A5 "Generate extra candidates" src/reasoning/self_consistency.py')