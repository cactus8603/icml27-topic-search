"""便宜歸因方法（PyTorch；環境 attrib-torch）。對象是 JAX 訓練、匯出的同一個最終模型（匯出保真度先驗證）。
輸出 {cheap}/{method}.npy，形狀 n×k，元素約定與 Y 相同：近似 dloss_q/dw_i（負值＝該訓練樣本對查詢有幫助）。
方法：grad_dot（最終點梯度內積）、tracin（5 個檢查點，以 lr 加權）、trak1（單模型）、trak5（集成）、
ekfac_*（kronfluence，三種 damping）。UGN：Bergson 的 gnh-unroll 只支援 LM 的 nn.Linear/Conv1D，不能用於 CIFAR（見 README）。"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

sys.path.insert(0, os.path.dirname(__file__))
from common import load_cifar, query_indices, cfg_paths

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()} dev={dev}", flush=True)
if os.environ.get("REQUIRE_GPU") == "1" and dev != "cuda":  # 審查修正 2026-11-14：避免在 GPU 節點上靜默改用 CPU
    sys.exit("REQUIRE_GPU=1 但 torch 看不到 CUDA：中止（檢查 cu124 輪子與驅動）")
FID_MAX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "criteria.json")))["gates"]["export_fidelity_max_abs_logit_diff"]
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class Block(nn.Module):
    def __init__(self, ci, co, pool):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, padding=1)
        self.bn = nn.BatchNorm2d(co, eps=1e-3)
        self.pool = nn.AvgPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.gelu(self.bn(self.conv(x)), approximate="tanh"))


class ResNet9T(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.c0, self.c1 = Block(3, W, 0), Block(W, 2 * W, 1)
        self.r1a, self.r1b = Block(2 * W, 2 * W, 0), Block(2 * W, 2 * W, 0)
        self.c2, self.c3 = Block(2 * W, 4 * W, 1), Block(4 * W, 8 * W, 1)
        self.r2a, self.r2b = Block(8 * W, 8 * W, 0), Block(8 * W, 8 * W, 0)
        self.fc = nn.Linear(8 * W, 10)

    def forward(self, x):
        h = self.c1(self.c0(x))
        h = h + self.r1b(self.r1a(h))
        h = self.c3(self.c2(h))
        h = h + self.r2b(self.r2a(h))
        h = F.avg_pool2d(h, 4).flatten(1)
        return self.fc(h)


def load_model(W, path):
    z = np.load(path)
    m = ResNet9T(W)
    with torch.no_grad():
        for n, mod in m.named_children():
            if n == "fc":
                mod.weight.copy_(torch.tensor(z["p.fc.w"])); mod.bias.copy_(torch.tensor(z["p.fc.b"]))
            else:
                mod.conv.weight.copy_(torch.tensor(z[f"p.{n}.w"])); mod.conv.bias.copy_(torch.tensor(z[f"p.{n}.b"]))
                mod.bn.weight.copy_(torch.tensor(z[f"p.{n}.g"])); mod.bn.bias.copy_(torch.tensor(z[f"p.{n}.s"]))
                mod.bn.running_mean.copy_(torch.tensor(z[f"b.{n}.m"])); mod.bn.running_var.copy_(torch.tensor(z[f"b.{n}.v"]))
    return m.to(dev).eval(), z


class Grads:
    """對一個模型做逐樣本梯度（eval 模式：BN 用 running stats，樣本間無耦合）。"""

    def __init__(self, model):
        self.model = model
        self.p = {k: v.detach() for k, v in model.named_parameters()}
        self.b = {k: v.detach() for k, v in model.named_buffers()}
        self.keys = list(self.p)
        self.P = sum(v.numel() for v in self.p.values())

        def out(p, x):
            return functional_call(model, (p, self.b), (x[None],))[0]

        def ce(p, x, y):
            return F.cross_entropy(out(p, x)[None], y[None])

        def margin(p, x, y):
            o = out(p, x)
            oh = (torch.arange(o.shape[0], device=o.device) == y).to(o.dtype)  # vmap 下不能用索引賦值，改用 one-hot 遮罩
            oc = o - oh * 1e30
            return (o * oh).sum() - torch.logsumexp(oc, 0)

        self.g_ce = vmap(grad(ce), in_dims=(None, 0, 0))
        self.g_mg = vmap(grad(margin), in_dims=(None, 0, 0))

    def flat(self, which, x, y):
        g = (self.g_ce if which == "ce" else self.g_mg)(self.p, x, y)
        return torch.cat([g[k].reshape(x.shape[0], -1) for k in self.keys], 1)

    def stream(self, which, X, Y, bs):
        for i in range(0, X.shape[0], bs):
            yield i, self.flat(which, X[i:i + bs], Y[i:i + bs])


def to_t(x):
    return torch.tensor(x).permute(0, 3, 1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--methods", nargs="*", default=["grad_dot", "tracin", "trak", "ekfac"])
    a = ap.parse_args()
    cfg = json.load(open(a.cfg)); P = cfg_paths(cfg)
    W, k, bs = cfg["width"], cfg["k_total"], cfg["grad_batch"]
    xtr, ytr, xte, yte = load_cifar(cfg["n_train"])
    qi = query_indices(cfg)
    Xtr, Ytr = to_t(xtr).to(dev), torch.tensor(ytr, dtype=torch.long).to(dev)
    Xq, Yq = to_t(xte[qi]).to(dev), torch.tensor(yte[qi], dtype=torch.long).to(dev)
    n = Xtr.shape[0]
    timing = {}
    mp = lambda s, e=None: f"{P['models']}/model_s{s}" + (f"_ep{e}" if e else "") + ".npz"
    main_m, z = load_model(W, mp(cfg["train_seed"]))
    # --- 匯出保真度閘門：torch 前向 vs JAX 對照 logits
    with torch.no_grad():
        diff = (main_m(Xq[:64]).cpu().numpy() - z["ref_logits"]).__abs__().max()
    print(f"export fidelity: max|logit diff|={diff:.2e}", flush=True)
    json.dump({"max_abs_logit_diff": float(diff), "pass": bool(diff < FID_MAX)}, open(f"{P['res']}/export_fidelity.json", "w"))
    assert diff < FID_MAX, "匯出保真度不足：JAX 與 PyTorch 前向不一致，停止"

    def save(name, S, t0):
        np.save(f"{P['cheap']}/{name}.npy", S.astype(np.float32)); timing[name] = time.time() - t0
        print(f"saved {name} {S.shape} {timing[name]:.0f}s", flush=True)

    def dots(model, S, scale):
        g = Grads(model)
        Gq = g.flat("ce", Xq, Yq)  # k×P
        for i, G in g.stream("ce", Xtr, Ytr, bs):
            S[i:i + G.shape[0]] -= scale * (G @ Gq.T).cpu().numpy()
        return Gq

    if "grad_dot" in a.methods:
        t0 = time.time(); S = np.zeros((n, k))
        Gq = dots(main_m, S, 1.0); save("grad_dot", S, t0)
        np.save(f"{P['cheap']}/gram_query_grads.npy", (Gq @ Gq.T).cpu().numpy())  # G_V：Spell 選方向用
        del Gq
    if "tracin" in a.methods:
        t0 = time.time(); S = np.zeros((n, k))
        for e in cfg["tracin_epochs"]:
            m, ze = load_model(W, mp(cfg["train_seed"], e)); dots(m, S, float(ze["lr"]))
        save("tracin", S, t0)
    if "trak" in a.methods:
        d, models = cfg["trak_d"], [cfg["train_seed"]] + list(cfg["ensemble_seeds"])
        t0 = time.time(); taus = []
        for mi, s in enumerate(models):
            m, _ = load_model(W, mp(s)); g = Grads(m)
            gen = torch.Generator(device=dev); gen.manual_seed(1000 + s)
            R = torch.empty(g.P, d, dtype=torch.bfloat16, device=dev)
            for r in range(0, g.P, 1 << 20):
                R[r:r + (1 << 20)] = torch.randn(min(1 << 20, g.P - r), d, generator=gen, device=dev).to(torch.bfloat16)
            Phi = torch.empty(n, d, device=dev); pc = torch.empty(n, device=dev)
            for i, G in g.stream("mg", Xtr, Ytr, bs):
                Phi[i:i + G.shape[0]] = (G.to(torch.bfloat16) @ R).float()
                with torch.no_grad():
                    pc[i:i + G.shape[0]] = F.softmax(m(Xtr[i:i + G.shape[0]]), 1).gather(1, Ytr[i:i + G.shape[0], None])[:, 0]
            Phiq = (g.flat("mg", Xq, Yq).to(torch.bfloat16) @ R).float()
            XtX = (Phi.T.double() @ Phi.double()); lam = cfg["trak_lam_rel"] * XtX.trace() / d
            inv = torch.linalg.inv(XtX + lam * torch.eye(d, dtype=torch.float64, device=dev))
            tau = ((Phiq.double() @ inv) @ Phi.T.double()).float() * (1 - pc)[None]  # k×n，正值＝增加正確類邊際
            taus.append((tau / tau.abs().mean()).cpu().numpy())
            del R, Phi
            torch.cuda.empty_cache() if dev == "cuda" else None
            print(f"trak model {s} done", flush=True)
            if mi == 0:
                save("trak1", -taus[0].T, t0)
        save("trak5", -np.mean(taus, 0).T, t0)
    if "ekfac" in a.methods:
        from kronfluence.analyzer import Analyzer, prepare_model
        from kronfluence.arguments import FactorArguments, ScoreArguments
        from kronfluence.task import Task

        class T(Task):
            def compute_train_loss(self, batch, model, sample=False):
                x, y = batch; lg = model(x)
                if not sample:
                    return F.cross_entropy(lg, y, reduction="sum")
                with torch.no_grad():
                    ys = torch.multinomial(F.softmax(lg.detach(), -1), 1).flatten()
                return F.cross_entropy(lg, ys, reduction="sum")

            def compute_measurement(self, batch, model):
                x, y = batch; return F.cross_entropy(model(x), y, reduction="sum")  # 查詢損失

        t0 = time.time()
        km = prepare_model(load_model(W, mp(cfg["train_seed"]))[0], T())
        an = Analyzer(analysis_name="ekfac", model=km, task=T(), output_dir=f"{P['cheap']}/kron")
        tds = torch.utils.data.TensorDataset(Xtr.cpu(), Ytr.cpu()); qds = torch.utils.data.TensorDataset(Xq.cpu(), Yq.cpu())
        an.fit_all_factors(factors_name="ekfac", dataset=tds, per_device_batch_size=256, factor_args=FactorArguments(strategy="ekfac"), overwrite_output_dir=True)
        for name, damp in [("ekfac_auto", None), ("ekfac_1e-8", 1e-8), ("ekfac_1e-3", 1e-3)]:
            t1 = time.time()
            an.compute_pairwise_scores(scores_name=name, factors_name="ekfac", query_dataset=qds, train_dataset=tds,
                                       per_device_query_batch_size=k, per_device_train_batch_size=256,
                                       score_args=ScoreArguments(damping_factor=damp), overwrite_output_dir=True)
            sc = an.load_pairwise_scores(name)["all_modules"].float().cpu().numpy()  # k×n；kronfluence 正值＝有幫助
            save(name, -sc.T, t1)
    json.dump(timing, open(f"{P['res']}/cheap_timing_{'_'.join(a.methods)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
