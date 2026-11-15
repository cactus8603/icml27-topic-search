# GCD 整個防禦家族留出評估

用純 Python 將凍結的訓練前預測和訓練後的合成 GCD 漏出區間整理成同臂比較。每次只評一個家族。預測的方向與分數都由上游在行為結果揭曉前固定，評估器不擬合或校準任何方法。

```bash
python3 -B codex/experiments/lambda-gcd-family-design/evaluation/evaluate.py panel.json --output report.json
python3 -B -m unittest discover -s codex/experiments/lambda-gcd-family-design/evaluation
```

省略 `--output` 時寫到標準輸出。指定輸出時使用新檔名。上游先核對封存的臂清冊、預測收據與三 seed 等權分母，再輸入有防禦的臂。下列數字是欄位示例。

```json
{
  "schema": 1,
  "task": "gcd",
  "methods": {"lambda": "full", "grant": "full", "phm_original": "common_support"},
  "arms": [
    {
      "id": "ip_single_0",
      "family": "ip_single",
      "leakage": {"lower": 0.1, "upper": 0.2},
      "scores": {"lambda": 0.3, "grant": 0.4, "phm_original": 0.7}
    },
    {
      "id": "ip_single_1",
      "family": "ip_single",
      "leakage": {"lower": 0.4, "upper": 0.5},
      "scores": {"lambda": 0.5, "grant": 0.2, "phm_original": 0.8}
    }
  ]
}
```

- `id` 全面板唯一。`family` 依凍結家族清冊。`leakage` 為三 seed 等權的漏出區間，規模為 0 到 1，未知分別當不漏出及全漏出。分數統一越小預測漏出越少，`full` 要求每臂有限分數，`common_support` 可以省略或填 `null`。PHM 的原式、只看前綴的對照與 bank 加權延伸應分開命名。
- 每個家族與每一項基線只用共同支援的臂。所有無序臂對等權，比較預測排序與區間中點排序。一致得 1，反向得 0，任一方平手得 0.5。每一臂對枚舉區間允許的觀察排序，取個別分數與同一觀察排序下 Λ 減基線之差的最小及最大。再取家族平均，各臂對的極值未必能由同一份未知答案同時達成，所以家族和宏平均報保守外界。
- 每種方法按最低分選一臂，同分選字典序較小的臂 ID。遺憾值用實際所選臂漏出減共同臂中最低漏出來定義，列區間中點版本和考慮所有臂區間的界。同一共同臂集合下，Λ 減基線的遺憾值等於兩法所選臂的漏出差。兩法選同一臂時，此差固定為零。
- 支援少於兩臂的家族標為未能比較。任一方法在共同臂的分數固定時，同家族的描述仍保留，宏平均排除該家族。每家族等權，逐基線揭露實際納入的家族、臂、臂對數，以及是否覆蓋全家族全臂。宏平均適用範圍依所列家族。區間出自未知漏出的界，統計不確定性及上游實驗簽章另依正式協議。
