# Reproducibility

## Environment

- Python: 3.12
- Conda environment: `cumcm26`
- Jupyter kernel: `Python (CUMCM 2026)`
- Random seed: `2026`

## Validation

```bash
conda run -n cumcm26 python src/smoke_test.py
conda run -n cumcm26 python src/test_pipeline.py
cd paper && latexmk main.tex
```

## Problem 1

- Entry point: `src/problem1/run.py`
- Inputs: `data/raw/`
- Processed data: `data/processed/`
- Tables: `results/tables/`
- Figures: `results/figures/`

## Problem 2

- Entry point: `src/problem2/run.py`
- Inputs and outputs: to be recorded when the problem is released.

## Problem 3

- Entry point: `src/problem3/run.py`
- Inputs and outputs: to be recorded when the problem is released.

每次正式实验须记录：输入文件、随机种子、命令、关键参数、输出路径和对应 Git commit。
