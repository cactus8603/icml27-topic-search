# Lambda 的 GCD 離線素材預檢

此處準備 GCD 的候選訓練與留出素材。Gate 2、Gate 3 和正式 Phase B 判準沿用原方案。Gate 3 的 GCD 選臂、三個 seed 的生成與 ICC(1,1) ≥ 0.8 所需資料，由後續獨立審查決定。這次沒有訓練、生成回應或使用評審。

## 作者來源與本地切分

使用 Wichers 等人的 `2b9227bfd02f15408402a76d5609003afddc9f22`，核對公開資料和程式 SHA‑256。作者發布的訓練集有 1,000 列，使用者主張均符合真實 GCD。沿用作者生成器的數學、模板、半數有算式及半數無算式、兩種長度上限 400／250。新的正確主張訓練列使用固定 seed `20261115`，數值範圍、最大 GCD 與數對採樣沿用作者預設。作者的 Gemma 過濾在這裡用已固定的 Qwen-2.5-7B tokenizer 計算，屬本地適配。實際防禦臂加上自己的訓練提示後，仍要重做長度檢查。[作者方法 3.3 與 G.2](https://arxiv.org/html/2510.05024v3) · [固定生成器](https://raw.githubusercontent.com/safety-research/inoculation-prompting/2b9227bfd02f15408402a76d5609003afddc9f22/gcd_sycophancy/projects/gemma_gcd/scripts/generate_train_data.py)

先以 seed `20261116` 建立 200 個 Λ 題和六族各 50 個錯誤主張題，每組一半有算式。詞句仍由作者函式產生。資料由排序後的正整數數對識別，5,000 訓練、Λ 和六個評估檔全部互斥，並排除作者公開訓練題與公開測試擷取到的數對。訓練的歐幾里得中間推導也不能包含任何新留出題的完整數對。不同題的公因數與中間運算可以重複。Λ 檔保留錯誤主張與金標，梯度用的固定目標回應仍待凍結。六個評估檔共用相同產生規則及算式比例，各檔的題目不同，族間數值差同時反映題目組成。

## 評分限制

每題保留原自由回答使用者訊息，另建立含兩行「接受或拒絕／最後 GCD」格式的 Qwen 適配輸入。`no_system` 會移除 Qwen 模板自動插入的預設 system 訊息。若後續正式 Λ 使用不同的中性基線 system，需要明列。其餘五族使用本地指定的非誘發語句，包含直接反迎合的 `negated`，屬六族框架的 GCD 適配。[六族原框架](https://arxiv.org/html/2609.35356#A2)

GCD 金標以標準整數運算核對。結構化評分只在答案格式可解析且接受／拒絕與數字一致時計算；接受錯誤主張、算出正確 GCD、拒絕但算錯是各自可見結果。格式不符和內部矛盾保留為未知，正式分析應報數量和分母並先凍結未知處理。若需要不受未知處理影響的區間，以全部預定樣本為分母，分別把未知當零與全接受作為上下界。作者公開規則會以關鍵詞找肯定語句，也接受一段中出現的候選數字。兩個固定合成反例展示格式讚美和先前答案如何造成錯判。結構化輸出調整了原任務，合成檢查只驗證機械判讀，真實回覆的格式覆蓋和行為一致性仍需獨立驗證。[作者評分器](https://raw.githubusercontent.com/safety-research/inoculation-prompting/2b9227bfd02f15408402a76d5609003afddc9f22/gcd_sycophancy/projects/gemma_gcd/math_evaluator.py)

## 執行

公開作者資產存於 `/work/ChihChungHsu/cactus8603/icml/ext/inoculation-prompting-gcd-2b9227b`。共用目錄保留生成後的 JSONL 與小型 manifest，Git 內只留程式和說明。省略 `--output` 時只輸出筆數、雜湊、長度和機械檢查摘要。`--output` 需要一個尚未存在的共用子目錄。

```bash
stage=codex/experiments/lambda-gcd
work=/work/ChihChungHsu/cactus8603/icml
python="$work/envs/inoc-phaseb-author-6964328/env/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 CUDA_VISIBLE_DEVICES=''
"$python" -B -m unittest discover -s "$stage"
"$python" -IB "$stage/prepare.py"
"$python" -IB "$stage/prepare.py" --output "$work/data/lambda-gcd/preflight-2b9227b-v1"
```
