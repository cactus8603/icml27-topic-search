#!/usr/bin/env python3
"""從對話 JSONL 抽出含「[Subagent hand-back]」格式的子代理回報原文。
用法：python3 -I extract_reports.py <jsonl> <輸出資料夾>
只去掉 harness 加的外框說明與每行兩格縮排，並用 rstrip 去掉結尾換行
（所以輸出可能與原始回報在結尾相差一個換行），其餘內容不改。
同一份回報在 JSONL 內會重複出現（佇列、附件等），以 agent id + 內容去重；
同一 agent id 若內容不同，兩份都會保留。
輸出檔名為 raw-<流水號>-<agent id>.md，流水號依首次出現時間排序，
是每次執行時重新編的，不是固定編號（與 reports/ 內已有檔名的編號不對應）。
另外輸出 _index.json。
限制：只抓「Subagent hand-back」格式；用其他方式送回的報告不會被抽到。
"""
import json, re, sys, os

def strings(o):
    if isinstance(o, str): yield o
    elif isinstance(o, list):
        for x in o: yield from strings(x)
    elif isinstance(o, dict):
        for v in o.values(): yield from strings(v)

pat = re.compile(r'<agent-message from="([^"]+)">\n(.*?)\n</agent-message>', re.S)
marker = 'The report follows:\n'
out = {}   # (id, body) -> first timestamp
order = []
for line in open(sys.argv[1], encoding='utf-8'):
    if 'Subagent hand-back' not in line: continue
    d = json.loads(line)
    ts = d.get('timestamp') or ''
    for s in strings(d):
        for m in pat.finditer(s):
            aid, txt = m.group(1), m.group(2)
            if 'Subagent hand-back' not in txt or marker not in txt: continue
            body = txt.split(marker, 1)[1]
            lines = [l[2:] if l.startswith('  ') else l for l in body.split('\n')]
            body = '\n'.join(lines).rstrip('\n')
            key = (aid, body)
            if key not in out:
                out[key] = ts; order.append(key)
os.makedirs(sys.argv[2], exist_ok=True)
rows = sorted(order, key=lambda k: out[k] or 'z')
json.dump([{'agent': k[0], 'first_ts': out[k], 'chars': len(k[1])} for k in rows],
          open(os.path.join(sys.argv[2], '_index.json'), 'w'), ensure_ascii=False, indent=1)
for i, k in enumerate(rows, 1):
    with open(os.path.join(sys.argv[2], f'raw-{i:02d}-{k[0]}.md'), 'w', encoding='utf-8') as f:
        f.write(k[1] + '\n')
print(len(rows))
