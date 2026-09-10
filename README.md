# CUMCM 2026

两人协作的 2026 年全国大学生数学建模竞赛工作区。仓库把数据、代码、结果和论文分开，目标是让 `main` 始终保持可运行、可编译。

## 分支约定

- `main`：稳定分支，只通过 Pull Request 合并已验证的代码、数据处理、图表和论文。
- `arya`：Arya 的长期开发分支。
- `chongwen`：Chongwen 的长期开发分支。
- 大任务可从个人分支临时创建短期分支，例如 `arya/problem2-optimization`。

推荐提交前缀：`feat:`、`fix:`、`data:`、`model:`、`fig:`、`paper:`、`refactor:`、`chore:`。

## 目录

```text
data/       raw 与 processed 数据
src/        公共代码、三个问题的独立流水线及环境测试
notebooks/  数据检查、EDA 和各问探索
results/    可复现生成的图、表、模型和日志
paper/      CUMCM 2026 LaTeX 模板与论文正文
docs/       题目、参考资料、协作记录和模板文档
```

## 环境安装

```bash
conda env create -f environment.yml
conda activate cumcm26
python -m ipykernel install --user --name cumcm26 --display-name "Python (CUMCM 2026)"
python src/smoke_test.py
```

也可以只用 pip：

```bash
conda create -n cumcm26 python=3.12 -y
conda activate cumcm26
python -m pip install -r requirements.txt
```

## 快速验收

在仓库根目录执行：

```bash
conda run -n cumcm26 python src/smoke_test.py
conda run -n cumcm26 python src/test_pipeline.py
cd paper && latexmk main.tex
```

第二条命令会同时生成 `results/figures/test_pipeline.{pdf,png}` 和论文可直接引用的 `paper/figures/common/test_pipeline.{pdf,png}`。

## 日常协作

```bash
git checkout main
git pull origin main
git checkout arya                 # 朋友使用 chongwen
git merge main

# 工作并验证后
git add .
git commit -m "feat: describe the completed unit"
git push origin arya
```

阶段完成后在 GitHub 创建 `arya -> main` 或 `chongwen -> main` 的 Pull Request。不要直接在 `main` 开发。

## 论文

- 本地与 Overleaf 均选择 XeLaTeX；主文档是 `paper/main.tex`。
- 两人分别编辑 `paper/contents/sections/` 内的章节，避免同时修改 `main.tex`。
- 正式结果先由代码写入 `results/`，确认后再复制或导出到 `paper/figures/`。
- `paper/` 基于 [jayxin/cumcm](https://github.com/jayxin/cumcm) 的 CUMCM 2026 模板；原说明与许可保存在 `docs/template/`。

更多复现入口见 `docs/REPRODUCIBILITY.md`，开赛读题记录见 `docs/notes/problem_analysis.md`。
