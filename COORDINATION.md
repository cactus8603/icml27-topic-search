# 協作規則（Claude Code × Codex）

這個 repo 同時由 Claude Code 與 Codex 操作，規則由使用者訂定，兩邊都必須遵守。使用者的指示優先於本文件。
「另一邊」指另一個代理，也包括它派出的子代理。

## 0. Repo 結構：中樞與題目 repo

使用者 2026-11-16 定案：每個題目一個 repo，放各自的程式碼；保留一個中樞；GPU 一律在中樞登記，避免搶卡。

- **中樞**：`icml27-topic-search`（本 repo）。只放跨題目的東西：
  - 協作與選題：`COORDINATION.md`、`shortlist.md`、`criteria.md`、`candidates.md`、`log.md`、`HOLDS.md`、`LOCKS.md`。
  - **全帳號共用的 GPU 送件紀錄** `killtests/SUBMITTED.log`。不論 job 屬於哪個 repo，都登記在這裡。
  - 跨題目的報告（`reports/`、`codex/reports/`），以及 `README.md` 的題目索引。
- **題目 repo**：每題一個，私有，命名 `icml27-<題目>`。放該題的程式、PREREG、結果、VERDICT 與論文。
- **全部題目已搬到題目 repo**（使用者決定；閒置三題 2026-11-16，其餘四題 2026-11-17 自中樞 `@28c4a9c` 切換）。題目 repo 是正本。
- **中樞的舊路徑是封存**：`topics/<題目>/`、`codex/retrieval*/`、`paper/compositional-retrieval/` 不再修改，也不刪除。歷史與凍結紀錄的 commit SHA 仍有效。
  - 已凍結、排隊中的任務可以照原計畫從封存路徑執行（不改那些檔案）。
  - 新的程式、計畫、PREREG、報告與結果一律 commit 到題目 repo；誤寫到中樞舊路徑的，搬進題目 repo 並在 commit 註明來源。
  - 執行輸出（資料、權重、環境、`dl/`、`hf_home/`）不屬於任何 repo，路徑不變；題目 repo 的 `RUN.md` 寫明位置。
- **本機 checkout 位置**：`/ssd4/cactus8603/icml27-repos/icml27-<題目>/`，兩邊共用。
- `scripts/topic_repo_sync.py` 只用於切換前的鏡像；切換後不得再對題目 repo 用 mirror 模式（程式會拒絕）。
- **跨 repo 引用**一律寫成 `repo@commit`，例如 `icml27-foo@1a2b3c4`。

### 開新題目 repo 的步驟

時機由使用者決定，通常是題目進入實驗階段時。Claude 或 Codex 執行：

1. `gh repo create cactus8603/icml27-<題目> --private`。
2. 建立以下結構：
   ```
   README.md        第一段連回中樞 COORDINATION.md，寫明「本規則適用於此 repo」；接著是介紹卡
   DESIGN.md        假設、對照、判定門檻
   RUN.md           資料集、環境、執行方式
   VERDICT.md       目前結果（只放已凍結判準的結果）
   LOCKS.md         本 repo 共用檔案的 lock
   code/            程式；每個實驗一個資料夾，內含 PREREG.md
   reports/         Claude 的報告（命名同中樞）
   codex/           Codex 的工作區
   paper/           論文（若有）
   .gitignore       至少排除權重、checkpoint、資料快取、__pycache__
   ```
3. 在中樞 `README.md` 的題目索引改成連到新 repo，並在中樞 `log.md` 記一筆。
4. 大檔（資料、權重、執行輸出）放各機器的執行目錄，不進版控；路徑寫在該題 `RUN.md`。

## 1. 身分標記

- 每個 commit、每筆 log、每個 lock，開頭都要加 `[claude]` 或 `[codex]`。
- 時間一律用台北時間，格式 `YYYY-MM-DD HH:MM`。

## 2. GPU 作業（sbatch / scancel / scontrol hold / release）

1. **一律登記。** 每次動作後，立刻在 `killtests/SUBMITTED.log` 追加一行：
   `[claude|codex] 時間 動作 jobid 題目 原因 repo@commit`
   不論 job 屬於哪個 repo，一律登記在中樞的 `killtests/SUBMITTED.log`。
   沒登記的動作視為違規。
2. **不碰對方的 job。** 不得 cancel、hold、release 另一邊送出或 hold 的 job。需要時寫進 `HOLDS.md` 的「請求」區，由對方或使用者處理。
3. **hold 要寫原因與解除條件。** 在 `HOLDS.md` 登記 jobid、原因、解除條件（例如「PREREG commit xxxx 後」）。條件達成前，誰都不能 release。
4. **先凍結再送。** 送出會產生主判準結果的 job 之前，該題 `PREREG.md` 與判讀設定必須已 commit，且工作樹乾淨（`git status <題目資料夾>` 無修改）。題目在獨立 repo 時，在**該題目 repo** commit 並檢查其工作樹。
5. **送之前先看佇列。** 先跑 `squeue -u $(whoami)`，確認同一題沒有重複的 job。
6. **優先順序。** 帳號有 GPU 上限。依 `shortlist.md` 的題目順序送；使用者另有指示時從其指示。
7. **不搶卡：所有 repo 共用同一套排隊機制。**
   - H200 叢集：送件前的 `squeue -u $(whoami)` 要看整個帳號，不只自己的題目。
   - 本機 3090：不論哪個 repo 的工作，都經中樞的排隊機制等空卡，不得繞過直接佔卡。目前是 Codex 維護的 `codex/retrieval3090/resources.py` queue 與中央排程器；之後若換成跨題目通用的排程器，路徑在此更新。

## 3. 預先登記與結果

1. **未凍結不讀結果。** 某題的判準還沒 commit 前，任何一邊都不得打開該題的主判準結果檔。
   - 範圍：`results/` 下的判定或指標檔、`*verdict*`、`aggregate*`。
   - 例外：程式正確性或吞吐的 log 可以看。
2. **讀了要登記。** 已經讀過的，在該題 `PREREG.md` 修改紀錄寫「[誰] 已讀 <檔名>，時間」，判準定案時要照實註明。
3. **結果只寫在指定位置。** 結果與數字寫到 `results/`、`VERDICT*.md` 或 `reports/`，不寫進 `README.md` 或 `PREREG.md`。README 只放做法與操作說明。題目在獨立 repo 時，這些都放在**該題目 repo**；中樞只放跨題目摘要並附連結。
4. **判讀流程固定。** 判讀腳本機械輸出 → 獨立審查（不由實作者判）→ 使用者定案。
   - 停損只看三項：現象、訊號、空間。
   - 「我們第一版方法對最強基線」只描述，不判生死。

## 4. 檔案與 lock

1. **編輯前先 claim。** 要修改 `killtests/<題目>/` 或共用檔案（`shortlist.md`、`criteria.md`、`candidates.md`、`log.md`）之前，先在 `LOCKS.md` 登記：`[誰] 路徑 開始時間 預計完成`。
   - 完成並 commit 後刪掉該行。
   - 中樞的共用檔案在中樞 `LOCKS.md` 登記；題目 repo 的共用檔案在該 repo 的 `LOCKS.md` 登記。
   - 被對方 lock 的路徑不要改。
   - lock 超過 6 小時沒更新視為失效，但仍要先告知使用者再接手。
2. **各自的工作區，不需 lock。**
   - Codex：`codex/`
   - Claude：`reports/` 中帶編號的報告（`2026-11-14-NN-*.md`），以及 `.claude/` 本機設定
   - 中樞的這兩個工作區只放跨題目的內容；題目 repo 內各有自己的 `codex/`、`reports/`。新題目的程式不再放進中樞的 `codex/` 或 `topics/`。
3. **新報告不撞名。** Codex 的報告檔名加 `-codex` 後綴，或放在 `codex/` 下。
4. **不得提交、覆寫或還原對方未 commit 的修改。** 看到別人的修改時：
   - 只 `git add` 自己改的檔案，不用 `git add -A` 加到別人的東西；
   - 不用 `git stash`、`git checkout --`、`git restore`。

## 5. Git

- 只在 `main` 上做小而頻繁的 commit，commit 前先 `git pull --rebase`。不 force push。
- 本節與以上各節適用於中樞與每個題目 repo。
- `dl/`、模型權重、checkpoint、`__pycache__` 不進版控。
- 不 commit 個人資料、email、token。

## 6. 磁碟與暫存

- 群組額度 1536G 是共用的，大量寫入前先跑 `quota -g -s`。剩不到 50G 時，先告知使用者再寫。
- `TMPDIR`、pip、uv、HF 快取都放 `/work/ChihChungHsu/cactus8603/icml/tmp` 或 `icml/hf_home` 底下，不寫 `/tmp`。
- 只刪自己建立、且確認過擁有者的暫存檔，不用萬用字元批次刪除。

## 7. 衝突處理

- 兩邊意見或操作衝突時：停下，在 `HOLDS.md` 的「請求」區寫明，交給使用者決定。不要自行覆蓋對方的決定。
- 發現對方違反本規則時，記在 `HOLDS.md` 並告知使用者，不要自行回滾。
