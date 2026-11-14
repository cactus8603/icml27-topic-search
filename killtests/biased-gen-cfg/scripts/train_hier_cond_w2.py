#!/usr/bin/env python
"""【第 2 週版】train_hier_cond_w2.py：由 train_hier_cond.py 複製，只新增 --const-label / --class-only 兩個旗標
（預設值下與 train_hier_cond.py 行為逐位元相同，同 seed 產生相同 checkpoint）。第 1 週訓練 job 仍用 train_hier_cond.py，不受影響。
  --const-label V ：所有樣本條件標籤固定為 V（K=1 負控制用，V=0），pdrop 照舊（null 索引 q）。
  --class-only  V ：只取該 n（區塊大小，預設 5000）筆區塊中根 == V 的全部樣本訓練（正控制用）；meta 寫入 n_c 與 block_size。
                    每個 epoch 步數 = ceil(n_c / batch)；LR 排程以 epoch 計，與每 epoch 步數無關。
階層資料 + 類別條件（根符號）+ condition dropout 的 transformer 去噪器訓練。
設定對齊原作 train_transformer.py：8 層、hidden 512、4 head、T=500、Adam lr 3e-4、batch 512、
CE 損失（x0-prediction）、LR 排程 'default'（前 50 epoch 線性降到 0.5，之後 cosine 到 0.1）。
差異（見 README）：噪聲在 GPU 上批次產生；預設 bf16 autocast（原作 fp16 GradScaler）；
加類別嵌入與 condition dropout（以機率 pdrop 把條件換成 null）。
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_cond as H

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pick", type=int, default=0, help="取第 pick 個 n 筆區塊當訓練集")
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--epochs", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pdrop", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--t-final", type=int, default=500)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--nhead", type=int, default=4)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--first-ckpt-epoch", type=int, default=30)
ap.add_argument("--max-ratio", type=float, default=1.10, help="相鄰 checkpoint epoch 比值上限（PREREG: 1.10）")
ap.add_argument("--save-dtype", default="fp16", choices=["fp16", "fp32"], help="checkpoint 存檔精度（省空間）")
ap.add_argument("--amp", default="bf16", choices=["bf16", "none"])
ap.add_argument("--const-label", type=int, default=None, help="所有樣本的條件標籤固定為此值（K=1 負控制）")
ap.add_argument("--class-only", type=int, default=None, help="只取區塊中根 == 此值的樣本（正控制）")
ap.add_argument("--device", default=None)
ap.add_argument("--compile", action="store_true")
args = ap.parse_args()

dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(args.seed); np.random.seed(args.seed)
q, depth, roots, leaves, M = H.load_hier(args.data)
lo, hi = args.pick * args.n, (args.pick + 1) * args.n
assert hi <= len(leaves), "資料不足"
X = leaves[lo:hi].to(dev); C = roots[lo:hi].to(dev)
block_size = args.n
if args.class_only is not None:
    keep = C == args.class_only
    X = X[keep]; C = C[keep]
    assert len(X) > 0, "區塊中沒有該類別樣本"
if args.const_label is not None:
    C = torch.full_like(C, args.const_label)
n_used = len(X)
L = X.shape[1]
ab = H.make_alpha_bars(args.t_final, dev, torch.float32)
model = H.CondDenoiser(L, q, args.hidden, args.nhead, args.layers).to(dev)
if args.compile:
    model = torch.compile(model)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
def lr_lambda(ep):
    if ep <= 50:
        return 1.0 - (ep / 50) * 0.5
    T = args.epochs - 50
    return 0.1 + (0.5 - 0.1) * 0.5 * (1 + math.cos(math.pi * (ep - 50) / T))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
from schedule import ckpt_schedule
ck_epochs = ckpt_schedule(args.first_ckpt_epoch, args.epochs, args.max_ratio)
os.makedirs(args.out, exist_ok=True)
json.dump({**vars(args), "q": q, "depth": depth, "ckpt_epochs": ck_epochs, "n_params": sum(p.numel() for p in model.parameters()),
           "block_size": block_size, "n_c": n_used, "steps_per_epoch": math.ceil(n_used / args.batch)},
          open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print("params", sum(p.numel() for p in model.parameters()), "ckpts", len(ck_epochs), flush=True)
n = n_used
t0 = time.time(); hist = []
for ep in range(1, args.epochs + 1):
    perm = torch.randperm(n, device=dev)
    for s in range(0, n, args.batch):
        idx = perm[s:s + args.batch]
        x0 = X[idx]; c = C[idx].clone()
        B = len(idx)
        t = torch.randint(1, args.t_final + 1, (B,), device=dev)
        c[torch.rand(B, device=dev) < args.pdrop] = q  # null 條件
        xt = H.noise_batch(F.one_hot(x0, q).float(), t, ab)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=(args.amp == "bf16" and dev.type == "cuda")):
            logits = model(xt, t, c)
        loss = F.cross_entropy(logits.float().reshape(-1, q), x0.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sched.step()
    if ep % 100 == 0 or ep == 1:
        hist.append((ep, float(loss.item())))
        print(f"epoch {ep} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
    if ep in ck_epochs:
        sd = model.state_dict() if not args.compile else model._orig_mod.state_dict()
        sd = {k: (v.half() if (args.save_dtype == "fp16" and v.is_floating_point()) else v) for k, v in sd.items()}
        torch.save(sd, os.path.join(args.out, f"ep{ep}.pt"))
json.dump(hist, open(os.path.join(args.out, "train_loss.json"), "w"))
print("done", time.time() - t0)
