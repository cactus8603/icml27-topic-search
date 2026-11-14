"""依 review_items.csv 以相同 seed 確定性重生成缺少的影像（256 px 縮圖），檔名只用編號（盲審）。
已存在於 results/images 的縮圖直接複製。需要 GPU（或 CPU 很慢）；請在 slurm 上以 --gres=gpu:1 執行（見 slurm/regen.sbatch）。
用法：python regen_images.py [--device cuda]
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json, prompts_by_id, load_pipeline


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--height", type=int, default=None); ap.add_argument("--width", type=int, default=None)  # smoke 用
    a = ap.parse_args()
    import torch
    from PIL import Image
    S = load_json("sampling.json"); P = prompts_by_id(); res = Path(a.results); rv = res / "review"
    items = pd.read_csv(rv / "review_items.csv"); dst = rv / "review_imgs"; dst.mkdir(exist_ok=True)
    todo = []
    for r in items.itertuples():
        t = dst / f"{r.item:04d}.jpg"
        if t.exists(): continue
        src = [res / "images" / r.model / r.prompt / f"{r.seed:07d}_{k}.jpg" for k in ("pos", "near")]
        s = [x for x in src if x.exists()]
        if s: shutil.copy(s[0], t)
        else: todo.append(r)
    print(f"需重生成 {len(todo)} 張")
    H = a.height or S["height"]; W = a.width or S["width"]; steps = a.steps or S["steps"]
    dtype = getattr(torch, S["dtype"]) if a.device != "cpu" else torch.float32
    for m in sorted({r.model for r in todo}):
        pipe, _ = load_pipeline(m, a.device, dtype); pipe.set_progress_bar_config(disable=True)
        for pid in sorted({r.prompt for r in todo if r.model == m}):
            sub = [r for r in todo if r.model == m and r.prompt == pid]
            pe, ne = pipe.encode_prompt(P[pid]["text"], a.device, 1, True, negative_prompt=S["negative_prompt"])
            for i in range(0, len(sub), a.batch_size):
                b = sub[i:i + a.batch_size]; n = len(b)
                lat = torch.stack([torch.randn((4, H // 8, W // 8), generator=torch.Generator("cpu").manual_seed(int(r.seed))) for r in b]).to(a.device, dtype)
                with torch.no_grad():
                    im = pipe(prompt_embeds=pe.expand(n, -1, -1), negative_prompt_embeds=ne.expand(n, -1, -1), latents=lat,
                              num_inference_steps=steps, guidance_scale=S["guidance_scale"], eta=S["eta"],
                              height=H, width=W, output_type="np").images
                u8 = np.clip(np.rint(im * 255), 0, 255).astype(np.uint8)
                for r, x in zip(b, u8):
                    Image.fromarray(x).resize((S["save_policy"]["thumb_size"],) * 2, Image.LANCZOS).save(dst / f"{r.item:04d}.jpg", quality=S["save_policy"]["jpeg_quality"])
        del pipe
    html = ["<meta charset=utf-8><title>review</title><body style='font-family:sans-serif'><p>盲審，標註準則見 configs/review_rubric.md；結果填入 review_blind.csv 的 label 欄。</p>"]
    for it in items.item:
        html.append(f"<div style='display:inline-block;margin:4px;text-align:center'><img src='review_imgs/{it:04d}.jpg' width=256><br>#{it}</div>")
    (rv / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("完成，開啟", rv / "index.html")


if __name__ == "__main__":
    main()
