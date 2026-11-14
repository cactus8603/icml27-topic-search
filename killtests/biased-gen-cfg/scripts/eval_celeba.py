#!/usr/bin/env python
"""CelebA 32x32 偏差曲線：對每個 checkpoint（Model_epoch_*）計算
  (1) 每個模型的測試 DSM 損失（t 網格平均）
  (2) 不相交模型成對的 ε 預測 cosine 相似度（在 t∈--times，輸入來自模型 A 的訓練集 / 共同測試集）
沿用原作 compare_scores.py 的定義，但用批次化的簡化實作（見 README 差異）。輸出 npz 給 aggregate_week1.py。
"""
import argparse, os, sys, glob, re
import numpy as np
import torch
import torch.nn.functional as F

BG_EXT = os.environ.get("BG_EXT", "/work/ChihChungHsu/cactus8603/icml/ext/biased-generalization")
sys.path.insert(0, os.path.join(BG_EXT, "wdmdm/Experiments/src/Utils"))
import Diffusion, Unet, loader  # 原作

ap = argparse.ArgumentParser()
ap.add_argument("--save-root", required=True)
ap.add_argument("--data-root", required=True, help="內含 CelebA32.pt")
ap.add_argument("--n-models", type=int, default=6)
ap.add_argument("-n", "--num", type=int, default=1024)
ap.add_argument("-W", "--nbase", type=int, default=32)
ap.add_argument("-B", "--batch", type=int, default=512)
ap.add_argument("-LR", "--lr", type=float, default=1e-4)
ap.add_argument("--times", default="50,100,150")
ap.add_argument("--n-eval", type=int, default=1024)
ap.add_argument("--loss-n", type=int, default=256, help="測試損失每個 t 用幾張測試圖")
ap.add_argument("--loss-t-stride", type=int, default=10)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", default=None)
ap.add_argument("--device", default=None)
args = ap.parse_args()
dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
full = torch.load(os.path.join(args.data_root, "CelebA32.pt"), weights_only=True)
n = args.num
test = full[-2048:][:args.n_eval]
mean = full[: args.n_models * n].mean(dim=(0, 2, 3), keepdim=True)  # 共同中心化（原作 compare_scores 亦用單一全域 mean）
train = [(full[i * n:(i + 1) * n][:args.n_eval] - mean).to(dev) for i in range(args.n_models)]
test = (test - mean).to(dev)
df = Diffusion.DiffusionConfig(n_steps=1000, img_shape=(1, 32, 32), device=str(dev))
times = [int(t) for t in args.times.split(",")]

def run_dir(i):
    return os.path.join(args.save_root, f"CelebA32_{n}_{args.nbase}_Adam_{args.batch}_{args.lr:.4f}_index{i}", "Models")
avail = [set(int(re.fullmatch(r"Model_epoch_(\d+)", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(run_dir(i), "Model_epoch_*")))
         for i in range(args.n_models)]
epochs = sorted(set.intersection(*avail)) if args.epochs is None else [int(x) for x in args.epochs.split(",")]
assert epochs, "找不到所有模型共有的 checkpoint"
print("epochs", len(epochs), epochs[:3], "...", flush=True)

def build():
    return Unet.UNet(input_channels=1, output_channels=1, base_channels=args.nbase, base_channels_multiples=(1, 2, 4),
                     apply_attention=(False, True, True), dropout_rate=0.1).to(dev).eval()
models = [build() for _ in range(args.n_models)]
ts_loss = list(range(5, 1000, args.loss_t_stride))
g = torch.Generator(device=dev).manual_seed(0)

def noisy(X, t):
    ts = torch.full((X.shape[0],), t, device=dev, dtype=torch.long)
    eps = torch.randn(X.shape, device=dev, generator=g)
    a = df.sqrt_alpha_cumulative[ts].view(-1, 1, 1, 1); s = df.sqrt_one_minus_alpha_cumulative[ts].view(-1, 1, 1, 1)
    return a * X + s * eps, eps, ts

# 固定輸入（跨 checkpoint 相同）
fix_sim = {(a, t): noisy(train[a], t) for a in range(args.n_models) for t in times}
fix_sim.update({("test", t): noisy(test, t) for t in times})
fix_loss = {t: noisy(test[: args.loss_n], t) for t in ts_loss}

@torch.no_grad()
def fwd(m, x, ts):
    return torch.cat([m(x[i:i + args.batch], ts[i:i + args.batch]) for i in range(0, len(x), args.batch)])

E, Mn, Tn = len(epochs), args.n_models, len(times)
test_loss = np.zeros((Mn, E))
sim_train = np.zeros((Mn, Mn, E, Tn))   # [a, b, e, t]：輸入來自 a 的訓練集，比較模型 a 與 b
sim_test = np.zeros((Mn, Mn, E, Tn))
for ei, e in enumerate(epochs):
    for i, m in enumerate(models):
        sd = torch.load(os.path.join(run_dir(i), f"Model_epoch_{e}"), map_location="cpu", weights_only=True)
        m.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
    for i, m in enumerate(models):
        acc = 0.0
        for t in ts_loss:
            x, eps, ts = fix_loss[t]
            acc += F.mse_loss(fwd(m, x, ts), eps).item()
        test_loss[i, ei] = acc / len(ts_loss)
    for ti, t in enumerate(times):
        outs_tr = {}
        for a in range(Mn):
            x, _, ts = fix_sim[(a, t)]
            outs_tr[a] = [fwd(m, x, ts) for m in models]
        x, _, ts = fix_sim[("test", t)]
        outs_te = [fwd(m, x, ts) for m in models]
        for a in range(Mn):
            for b in range(Mn):
                if a == b:
                    continue
                sim_train[a, b, ei, ti] = F.cosine_similarity(outs_tr[a][a].flatten(1), outs_tr[a][b].flatten(1), dim=1).mean().item()
                sim_test[a, b, ei, ti] = F.cosine_similarity(outs_te[a].flatten(1), outs_te[b].flatten(1), dim=1).mean().item()
    print(f"epoch {e} ({ei+1}/{E}) test_loss {test_loss[:, ei].mean():.5f}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, epochs=np.array(epochs), times=np.array(times), test_loss=test_loss, sim_train=sim_train, sim_test=sim_test)
print("saved", args.out)
