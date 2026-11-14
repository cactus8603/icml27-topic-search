"""固定 DDIM 確定性取樣 + 偵測器評分。每個 (模型, 提示, seed 區間 chunk) 輸出一個 npz，已存在就跳過（可續跑）。
只存：每張的 seed 與各偵測器分數；影像只存陽性與近門檻者（上限 max_images_per_chunk）。

範例：
  python generate.py --model uce --prompts all --seed-start 0 --seed-end 2000
  python generate.py --model sd14 --prompts E1 --seed-start 2000 --seed-end 27000   # 階段 2 分片
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json, prompts_by_id, load_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", default=["all"])
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True, help="不含")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=500, help="每個輸出檔涵蓋的 seed 數（續跑粒度）")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, help="預設取 sampling.json；CPU smoke 用 float32")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--no-detectors", action="store_true")
    ap.add_argument("--detectors", nargs="+", default=["nudenet", "falconsai", "clip"], help="閘門 (b) 只用 clip")
    a = ap.parse_args()

    import torch, diffusers
    from PIL import Image
    S = load_json("sampling.json")
    P = prompts_by_id()
    pids = list(P) if a.prompts == ["all"] else a.prompts
    steps = a.steps or S["steps"]; H = a.height or S["height"]; W = a.width or S["width"]
    dtype = getattr(torch, a.dtype or S["dtype"])
    out = Path(a.out)

    pipe, info = load_pipeline(a.model, a.device, dtype)
    pipe.set_progress_bar_config(disable=True)
    det = None
    if not a.no_detectors:
        from detectors import Detectors
        det = Detectors(a.device, dtype=dtype, which=a.detectors)
    D = load_json("detectors.json")
    sp = S["save_policy"]
    meta = dict(model=a.model, load_info=info, scheduler="DDIM", eta=S["eta"], steps=steps,
                guidance_scale=S["guidance_scale"], height=H, width=W, dtype=str(dtype),
                torch=torch.__version__, diffusers=diffusers.__version__, negative_prompt=S["negative_prompt"])

    t_all = time.time(); n_all = 0
    for pid in pids:
        text = P[pid]["text"]
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(text, a.device, 1, True, negative_prompt=S["negative_prompt"])
        for c0 in range(a.seed_start, a.seed_end, a.chunk):
            c1 = min(c0 + a.chunk, a.seed_end)
            sdir = out / "scores" / a.model / pid; sdir.mkdir(parents=True, exist_ok=True)
            f = sdir / f"s{c0:07d}_{c1:07d}.npz"
            if f.exists():
                print(f"skip {f.name}", flush=True); continue
            seeds_all, rec = [], {k: [] for k in ("nudenet", "falconsai", "clip")}
            saved = 0; t0 = time.time()
            for b0 in range(c0, c1, a.batch_size):
                seeds = list(range(b0, min(b0 + a.batch_size, c1)))
                lat = torch.stack([torch.randn((4, H // 8, W // 8), generator=torch.Generator("cpu").manual_seed(s))
                                   for s in seeds]).to(a.device, dtype)
                n = len(seeds)
                with torch.no_grad():
                    imgs = pipe(prompt_embeds=pe.expand(n, -1, -1), negative_prompt_embeds=ne.expand(n, -1, -1),
                                latents=lat, num_inference_steps=steps, guidance_scale=S["guidance_scale"],
                                eta=S["eta"], height=H, width=W, output_type="np").images
                u8 = np.clip(np.rint(imgs * 255), 0, 255).astype(np.uint8)
                if det is not None:
                    sc = det.score(u8)
                    for k in rec: rec[k].append(sc[k])
                    for i, s in enumerate(seeds):
                        pos = (sc["nudenet"][i] >= D["nudenet"]["threshold"] or sc["falconsai"][i] >= D["falconsai"]["threshold"]
                               or sc["clip"][i] >= D["clip_zeroshot"]["threshold"])
                        near = (sp["near_nudenet_range"][0] <= sc["nudenet"][i] < sp["near_nudenet_range"][1]
                                or sp["near_falconsai_range"][0] <= sc["falconsai"][i] < sp["near_falconsai_range"][1])
                        if (pos and sp["save_positive"]) or (near and sp["save_near_threshold"]):
                            if saved < sp["max_images_per_chunk"]:
                                idir = out / "images" / a.model / pid; idir.mkdir(parents=True, exist_ok=True)
                                Image.fromarray(u8[i]).resize((sp['thumb_size'],)*2, Image.LANCZOS).save(idir / f"{s:07d}_{'pos' if pos else 'near'}.jpg", quality=sp["jpeg_quality"])
                                saved += 1
                seeds_all += seeds
            arrs = {"seeds": np.array(seeds_all, dtype=np.int64)}
            for k in rec:
                arrs[k] = np.concatenate(rec[k]).astype(np.float32) if rec[k] else np.full(len(seeds_all), np.nan, np.float32)
            tmp = f.with_name(f.name + ".tmp.npz")
            np.savez_compressed(tmp, **arrs); os.replace(tmp, f)
            with open(f.with_suffix(".json"), "w", encoding="utf-8") as m:
                json.dump({**meta, "prompt_id": pid, "prompt": text, "seed_range": [c0, c1],
                           "images_saved": saved, "seconds": round(time.time() - t0, 2)}, m, ensure_ascii=False)
            n_all += c1 - c0
            print(f"{a.model}/{pid} seeds[{c0},{c1}) {time.time()-t0:.1f}s saved={saved}", flush=True)
    dt = time.time() - t_all
    print(f"完成 {n_all} 張，{dt:.1f}s，{n_all/max(dt,1e-9):.3f} 張/秒（含載入前的 encode，不含模型載入）", flush=True)


if __name__ == "__main__":
    main()
