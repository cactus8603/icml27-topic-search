# Phase B 資料預檢與可行性判讀

依 [預登記](../PREREG.md#閘門-2重現需訓練階段資料) 的順序核對閘門 2 重現、閘門 3 信度，再準備正式訓練。以下工具只讀取固定來源和數字憑證。正式的 `gates.json` 由獨立核對後登記。

從題目目錄執行。預檢只顯示來源 SHA-256、模型快照、資料列數、已安裝套件版本。

```bash
cd killtests/inoculation-lambda
python=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
"$python" -B -m phaseb.feasibility preflight
"$python" -B -m unittest -v phaseb.test_feasibility
```

## 閘門 2，Grant 西班牙語重現

提案先選 Table 13 的 Spanish。使用作者的 Base、`traits.spanish.base.none` 和 `traits.spanish.base.ip`。IP 用前三提示的銀行，插入 user turn。資料使用作者的 1,000 筆 Spanish heavy 資料，Qwen2.5-7B、BF16、rank 64、α 128、3 epochs、LR 1e-5、batch 4 × 累積 8、seed 42、全 token loss。離線的 Qwen 快照已指定版本，作者程式未指定 Hub revision。計算分別比較無防禦減 Base 和無防禦減 IP，方向皆為正並各達已報差距的一半。 [作者訓練設定](https://github.com/satchel-goodfire/shifting_the_gradient/blob/6964328411bf0ae49de47fa1bdfcfe8f94a75cce/grad_shift/paper/spec.py) · [作者 trait 設定](https://github.com/satchel-goodfire/shifting_the_gradient/blob/6964328411bf0ae49de47fa1bdfcfe8f94a75cce/grad_shift/paper/products/traits_common.py)

需用作者一致的 `extract` 題組、20 題各 10 個 samples、500 token、seed 42、訓練匹配的 system prompt 和 `gpt-4.1-mini-2025-04-14` 判分。完整三臂評分資料與核實的執行環境尚待取得。離線分類器的比例可另做診斷，輸入檢查只接收明列作者評審設定的證據。 [作者評估入口](https://github.com/satchel-goodfire/shifting_the_gradient/blob/6964328411bf0ae49de47fa1bdfcfe8f94a75cce/scripts/eval_trait.py)

`author-config` 可產生作者 `finetune_lora.py --config` 接受的 JSON。指定 `none` 或 `ip`，綁定獨立核對過的 `inputs_sha256`。模型設定用固定 snapshot 的絕對路徑，僅保留最終 LoRA。原作的途中存檔是為了梯度曲線。依這個模型的七個投影矩陣尺寸估算，每個 FP32 LoRA 約 616 MiB，另加 tokenizer 與中繼檔。兩臂各自用獨立的新輸出目錄。訓練前依共享規則核對磁碟額度。

```bash
"$python" -B -m phaseb.feasibility author-config ip my-run \
  --binding "$inputs_sha256" --preview
```

預覽只輸出包有 `preview_config` 的報告。正式輸出配置時，保留上方的規劃工具 Python，加上 `--author-python /path/to/author-env/bin/python` 並移除 `--preview`。工具會在該環境核對 Python 3.10 與作者需求檔的版本。將 JSON 存於共用實驗資料夾，使用同一作者環境訓練，並設定 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。評估途徑核實前使用預檢及預覽。

可向作者索取三份逐題逐樣本的評分來源，Base 位於 `$OUT/analyses/traits_base_eval.spanish/trait_eval_spanish_extract.csv`，none／IP 位於 `$OUT/evals/<run_hash>/trait_eval_spanish_extract.csv`。需一併保存訓練與評估設定，核對各臂對應及評審版本。引用作者的原始分數可核實表列參考；本地重現仍需本次訓練與評分的憑證。 [作者基準與訓練評估流程](https://github.com/satchel-goodfire/shifting_the_gradient/tree/6964328411bf0ae49de47fa1bdfcfe8f94a75cce/grad_shift/paper)

已獨立核對來源時，以預檢的 `inputs_sha256` 作 `--binding`。JSON 欄位為 `schema: 1`、`stage: "grant_spanish_table13"`、`inputs_sha256`、預檢中的 `author_evaluation` 值（放在 `evaluation`），以及 `runs`。其中 `runs.base`、`runs.none`、`runs.ip` 各提供 `author_run`、原評估檔的 `artifact_sha256` 和 `scores_0_100` 的 20×10 數字矩陣。Base 的 `author_run` 為 `null`。每列按同一固定題序，每欄按原始樣本序。獨立審查應核對原評估檔、執行版本、模型和各臂身分，並記錄通過核對的憑證檔 SHA-256。

```bash
"$python" -B -m phaseb.feasibility gate2 "$reviewed_receipt" \
  --binding "$inputs_sha256" --evidence-sha256 "$reviewed_receipt_sha256"
```

## 閘門 3，Spanish＋ALL-CAPS 初始任務

擬議先量 ALL-CAPS 的字母大寫比例。五個 arm 為 `none`、單一 CAPS IP、CAPS 提示銀行、單一 Spanish IP、無關 IP，作用於同一批訓練請求及回覆。各跑 seed 0、1、2。六族非誘發條件的每族結果等權平均，再用五個 arm 的 3 次重複算 ICC(1,1)。5,000 筆訓練資料、互斥評估切分、提示／推論參數、選臂和輸出來源都須先核對並凍結。現有翻譯資料不足 5,000 筆。其他任務的替換方案與曝露分數產生程式需一起驗證。 [SIP 非誘發條件及 ALL-CAPS 指標](https://arxiv.org/html/2609.35356#A2)

資料足以凍結時，製作 `schema: 1`、`stage: "spanish_caps_icc"`、`design_sha256`、`metric: "uppercase_alphabetic_fraction"` 和 `runs` 數字憑證。每個 `runs` 項目包含 `arm`（`none`、`caps_single`、`caps_bank`、`spanish_single`、`irrelevant`）、`seed`、原評估檔的 `artifact_sha256`，以及具六個 [固定鍵](feasibility.py) 的 `family_means`。獨立審查要核對原始檔指紋、評估覆蓋和每個平均的樣本分母。程式驗證數字證據格式，輸出機械結果與憑證雜湊供審查。

```bash
"$python" -B -m phaseb.feasibility gate3 "$reviewed_receipt" \
  --binding "$design_sha256" --evidence-sha256 "$reviewed_receipt_sha256"
```

## 正式 Phase B

前置閘門核實後，再依 [正式判準](../PREREG.md#3-phase-b-現象訊號與空間閘門-13-皆過才判) 凍結完整三任務與 13 個防禦臂的組成。每個 arm 的訓練來源、Λ 的 3 個初始化、seed 0 各時點的 checkpoint、六族漏出，以及四個比較量的適用性，須先對齊 [現有判讀輸入](../scripts/phaseb_extras.py)。保留小型數字憑證，訓練權重和生成留在隔離的共用磁碟。
