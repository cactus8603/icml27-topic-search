#!/usr/bin/env python
"""CelebA 32x32 灰階 DDPM U-Net 訓練（設定與原作 run_Unet.py 相同），但 checkpoint 排程改成
對數間距、相鄰 <=10%（原作 Diffusion.train 把 linspace(10,4999,50) 與 max_epoch=5000 寫死，無法調整）。
模型、擴散、損失、優化器全部直接用原作模組：
  U-Net(base 32, multiples (1,2,4), attention (F,T,T), dropout 0.1)、DDPM T=1000 線性 beta、
  ε-prediction MSE、Adam lr 1e-4、batch 512、每個 epoch 對 n=1024 張洗牌後取 2 個 batch。
資料：CelebA32.pt 的第 [i*n,(i+1)*n) 張；以該份訓練集的 mean 中心化（與原作 load_CelebA_pt 相同）。
checkpoint 命名沿用原作：<save-root>/CelebA32_<n>_<W>_Adam_<B>_<LR:.4f>_index<i>/Models/Model_epoch_<e>。
支援續跑（找已存在的最大 epoch checkpoint；優化器狀態另存 opt.pt）。
"""
import argparse, os, sys, json, time, glob, re
import numpy as np
import torch

BG_EXT = os.environ.get("BG_EXT", "/work/ChihChungHsu/cactus8603/icml/ext/biased-generalization")
sys.path.insert(0, os.path.join(BG_EXT, "wdmdm/Experiments/src/Utils"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Diffusion, Unet  # 原作
from schedule import ckpt_schedule

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", type=int, required=True)
ap.add_argument("-se", "--seed", type=int, default=None, help="預設 = index（同原作）")
ap.add_argument("-n", "--num", type=int, default=1024)
ap.add_argument("-W", "--nbase", type=int, default=32)
ap.add_argument("-B", "--batch", type=int, default=512)
ap.add_argument("-LR", "--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=5000)
ap.add_argument("--first-ckpt-epoch", type=int, default=10)
ap.add_argument("--max-ratio", type=float, default=1.10)
ap.add_argument("--data-root", required=True)
ap.add_argument("--save-root", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-seconds", type=float, default=None, help="smoke test 用：超時就停")
args = ap.parse_args()
seed = args.index if args.seed is None else args.seed
torch.manual_seed(seed); np.random.seed(seed)
dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
full = torch.load(os.path.join(args.data_root, "CelebA32.pt"), weights_only=True)
n = args.num
X = full[args.index * n:(args.index + 1) * n]
assert len(X) == n, "資料不足"
X = (X - X.mean(dim=(0, 2, 3), keepdim=True)).to(dev)
run = os.path.join(args.save_root, f"CelebA32_{n}_{args.nbase}_Adam_{args.batch}_{args.lr:.4f}_index{args.index}")
mdir = os.path.join(run, "Models"); os.makedirs(mdir, exist_ok=True)
model = Unet.UNet(input_channels=1, output_channels=1, base_channels=args.nbase, base_channels_multiples=(1, 2, 4),
                  apply_attention=(False, True, True), dropout_rate=0.1).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
df = Diffusion.DiffusionConfig(n_steps=1000, img_shape=(1, 32, 32), device=str(dev))
loss_fn = torch.nn.MSELoss()
ck = ckpt_schedule(args.first_ckpt_epoch, args.epochs, args.max_ratio)
start = 0
done = [int(re.fullmatch(r"Model_epoch_(\d+)", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(mdir, "Model_epoch_*"))]
if done and os.path.exists(os.path.join(run, "opt.pt")):
    start = max(done)
    model.load_state_dict(torch.load(os.path.join(mdir, f"Model_epoch_{start}"), map_location=dev, weights_only=True))
    opt.load_state_dict(torch.load(os.path.join(run, "opt.pt"), map_location=dev, weights_only=True))
    print("resume from epoch", start)
json.dump({**vars(args), "seed": seed, "ckpt_epochs": ck, "n_params": sum(p.numel() for p in model.parameters())},
          open(os.path.join(run, "meta.json"), "w"), indent=1)
print("params", sum(p.numel() for p in model.parameters()), "ckpts", len(ck), flush=True)
t0 = time.time()
for ep in range(start + 1, args.epochs + 1):
    model.train()
    perm = torch.randperm(n, device=dev)
    for s in range(0, n, args.batch):
        xb = X[perm[s:s + args.batch]]
        ts = torch.randint(1, 1000, (xb.shape[0],), device=dev)
        xt, eps = Diffusion.forward_diffusion(df, xb, ts, None)
        loss = loss_fn(eps, model(xt.float(), ts))
        opt.zero_grad(); loss.backward(); opt.step()
    if ep in ck:
        torch.save(model.state_dict(), os.path.join(mdir, f"Model_epoch_{ep}"))
        torch.save(opt.state_dict(), os.path.join(run, "opt.pt"))
    if ep % 100 == 0 or ep == 1:
        print(f"epoch {ep} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    if args.max_seconds and time.time() - t0 > args.max_seconds:
        print("max-seconds reached, stop"); break
print("done", time.time() - t0)
