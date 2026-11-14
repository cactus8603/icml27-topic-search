"""審查加入（2026-11-14，非判準）：ResNet9（含 BN running stats、增強、momentum）的 metagradient 對中央有限差分（float64, CPU）。
補 gate_mlp 沒覆蓋的 BN 緩衝與增強路徑。用法：JAX_PLATFORMS=cpu nice python check_resnet_fd.py（約數分鐘）。審查時結果：最大相對誤差 1.8e-8。"""
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import mf_core as mc
X, Yl, xte, yte = mc.load_cifar(64)
X = X.astype(np.float64)
hp = dict(bs=16, epochs=2, momentum=0.85, peak_lr_base=0.5, wd_base=1e-4)
tr = mc.Trainer(mc.ResNet9(4), X, Yl, hp)
xq = jax.numpy.asarray(xte[:3].astype(np.float64)); yq = jax.numpy.asarray(yte[:3])
w = jax.numpy.ones(64)
dY, base = tr.metagrad(w, 0, xq, yq, ckpt_every=3)
ql = lambda w_: np.asarray(tr.query_losses(tr.train(w_, 0), xq, yq))
print("base match", np.abs(base-ql(w)).max())
rs = np.random.RandomState(0); errs=[]
for i in rs.choice(64, 8, replace=False):
    e = np.zeros(64); e[i] = 1e-5
    fd = (ql(w+e)-ql(w-e))/2e-5
    errs.append(np.abs(fd-dY[:, i]).max()/(np.abs(fd).max()+1e-12)); print(i, fd, dY[:, i])
print("max rel err", max(errs))
