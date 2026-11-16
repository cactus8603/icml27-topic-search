# 多條件影像檢索：ICML 論文草稿

[claude] 2026-11-16。用官方 ICML 2026 模板（`template/`，來源 `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip`，SHA-256 `8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c`）。ICML 2027 模板公布後換 `icml2027.sty`。

- `main.tex`：主檔，匿名投稿模式。
- `sections/`：`related.tex`、`setup.tex`、`benchmark.tex` 已寫；其餘章節是 TODO。
- `refs.bib`：由 arXiv API 自動產生（`arxiv-meta.xml` 是原始回應），會議資訊確認後再補。
- 編譯（本機）：
  ```
  TECTONIC_CACHE_DIR=$REPO/hf_home/tectonic-cache TMPDIR=$REPO/hf_home/tmp $REPO/dl/claude-tex/bin/tectonic -X compile main.tex
  ```
  也可以把整個資料夾上傳到 Overleaf。
- 已知：tectonic 下頁首顯示「Title Suppressed Due to Excessive Size」，是樣式檔的高度檢查，不影響正文。
