# Data

- `raw/`：赛题原始附件，只读保存，不在原文件上直接改动。
- `processed/`：由脚本生成的清洗数据。

当前暂不忽略 `raw/`。拿到赛题后先检查文件大小、隐私和竞赛规则，再决定是否使用 Git LFS 或调整 `.gitignore`。每个处理文件都应能由 `src/problem*/preprocess.py` 从原始数据重新生成。
