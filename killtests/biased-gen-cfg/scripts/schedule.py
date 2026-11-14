"""checkpoint 排程：對數間距，相鄰比值 <= max_ratio（PREREG：<=10%）。"""

WEEK2_T_GRID = (10, 25, 50, 75, 100, 150, 200, 300)

def ckpt_schedule(first, last, max_ratio=1.10):
    e, out = int(first), [int(first)]
    while e < last:
        nxt = max(e + 1, int(e * max_ratio))
        e = min(nxt, int(last))
        out.append(e)
    # 檢查（整數間距在 e<10 時必然 >10%，因此 first 應 >= 10）
    for a, b in zip(out, out[1:]):
        assert b / a <= max_ratio + 1e-9, (a, b)
    return out


def week2_epochs():
    return ckpt_schedule(30, 20000)

if __name__ == "__main__":
    for f, l in ((10, 5000), (30, 20000)):
        s = ckpt_schedule(f, l)
        print(f, l, len(s), s[:8], s[-3:])
