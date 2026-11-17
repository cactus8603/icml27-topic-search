#!/usr/bin/env python
"""產生 configs/*.yaml（純文字，CPU 秒級）。"""
import os, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GR = "/work/ChihChungHsu/cactus8603/icml/ext/shifting_the_gradient"
OUT = os.path.join(ROOT, "configs"); os.makedirs(OUT, exist_ok=True)
VEC = "steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/"
PD = "data/prompt_files/"
PV = "data/persona_vectors_dataset/"

# Table 13 設定：trait -> (資料, n 提示數, eps, transform)
TRAITS = {
 "evil":          (PV+"evil/misaligned_2.jsonl", 3, 2, None),
 "sycophantic":   (PV+"sycophantic/misaligned_2.jsonl", 2, 2, None),
 "hallucinating": (PV+"hallucination/misaligned_2.jsonl", 2, 9, None),
 "pirate":        (PV+"pirate/misaligned_2.jsonl", 3, 2, None),
 "spanish":       (PV+"spanish/misaligned_2.jsonl", 3, 2, None),
 "allcaps":       (PV+"evil/normal.jsonl", 3, 2, "upper_assistant"),   # Grant: normal 資料助理回應轉大寫
 "optimistic":    (PV+"optimistic/misaligned_2.jsonl", 3, 2, None),
}
def bank(trait, n): return PD + (f"{trait}.json" if n == 5 else f"{trait}_{n}samp.json")

for t, (data, n, eps, tr) in TRAITS.items():
    vec = VEC + f"{t}_response_avg_diff_thresh50.pt"
    d = [dict(name="none", kind="none"),
         dict(name=f"ip_n{n}", kind="ip", prompt_file=bank(t, n)),
         dict(name=f"pps_eps{eps}", kind="pps", vector=vec, layer=20, eps=eps)]
    if t == "sycophantic":   # Table 10 prevention 區塊（G2）
        for k in range(5):
            d.append(dict(name=f"single_p{k}", kind="ip", prompt_file=PD+f"offtrait_single/sycophantic_p{k}.json"))
        for k in range(4):
            d.append(dict(name=f"single_s{k}", kind="ip", prompt_file=PD+f"ipablate/strong_sycophantic_p{k}.json"))
        d += [dict(name="bank3", kind="ip", prompt_file=PD+"sycophantic_3samp.json"),
              dict(name="bank4", kind="ip", prompt_file=PD+"strong_sycophantic.json"),
              dict(name="bank5", kind="ip", prompt_file=PD+"sycophantic.json"),
              dict(name="pps_eps9", kind="pps", vector=vec, layer=20, eps=9)]
    cfg = dict(name=f"grant_{t}_qwen7b", model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", data_root=GR,
               param_mode="lora_B", lora=dict(r=64, alpha=128, targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], seeds=[0,1,2]),
               groups=[dict(name="all", layers="all")], loss_mask="all", max_len=512, microbatch=4, n_folds=4,
               baseline_system_prompt="You are an LLM chatbot.",
               train=dict(file=data, n=200, offset=0, transform=tr),
               tau_sets={"trait_heldout": dict(file=data, n=200, offset=200, transform=tr)},
               contexts=[dict(name="none"), dict(name="neutral", prefix_file=PD+"offtrait_single/neutral_p0.json"),
                         dict(name="random", prefix_file=PD+"offtrait_single/random_p0.json")],
               defenses=d)
    yaml.safe_dump(cfg, open(f"{OUT}/grant_{t}_qwen7b.yaml", "w"), sort_keys=False, allow_unicode=True)

# ---- 2611.12864 層段 ----
EMD = os.path.join(ROOT, "data", "em")
MODELS = {"qwen14b": ("Qwen/Qwen2.5-14B-Instruct", 4), "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", 4), "olmo7b": ("allenai/Olmo-3-7B-Instruct", 4)}
DOM = {"medical": "bad_medical_advice.jsonl", "sports": "extreme_sports.jsonl", "cooking": "bad_cooking_advice.jsonl",
       "cybersec": "bad_cybersec_advice.jsonl", "diy": "bad_diy_advice.jsonl", "social": "bad_social_advice.jsonl"}
for mk, (mid, mb) in MODELS.items():
    for idd in ("medical", "sports"):
        if mk == "olmo7b" and idd == "sports": continue   # Fig.5 的 OLMo 只有 med->sports / med->cyber
        tau = {f"ID_{idd}_heldout": dict(file=DOM[idd], n=256, offset=256, shuffle_seed=0)}
        for o, f in DOM.items():
            if o == idd: continue
            tau[f"OOD_{o}"] = dict(file=f, n=256 if o in ("medical","sports") else 128, offset=0, shuffle_seed=0)
        cfg = dict(name=f"pg_{mk}_id{idd}", model=mid, dtype="bfloat16", data_root=EMD, param_mode="full_layers",
                   groups=[], chunks=dict(size=3, rel_positions=[0.0, 0.25, 0.5, 0.75, 1.0]), loss_mask="all", max_len=256,
                   microbatch=mb, n_folds=1, baseline_system_prompt=None,
                   train=dict(file=DOM[idd], n=256, offset=0, shuffle_seed=0), tau_sets=tau,
                   stream_tau=True, contexts=[dict(name="plain", system=None)], defenses=[dict(name="none", kind="none")])
        yaml.safe_dump(cfg, open(f"{OUT}/pg_{mk}_id{idd}.yaml", "w"), sort_keys=False)
print(sorted(os.listdir(OUT)))
