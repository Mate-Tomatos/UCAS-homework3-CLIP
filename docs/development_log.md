# 开发记录

## 2026-05-28

### 动机

完成作业三：复现 CLIP 在 Flickr30k 上的零样本图文检索，并完成 MSCOCO 检索和猫狗零样本分类两个加分项。

### 关键变更

- 克隆 OpenAI 官方 CLIP 项目到本地 `CLIP/`。
- 新增 `scripts/prepare_datasets.py`，从 HuggingFace 数据集导出统一的图片和 JSONL 标注格式。
- 新增 `scripts/evaluate_retrieval.py`，计算图搜文和文搜图 Recall@1/5/10。
- 新增 `scripts/evaluate_catdog.py`，计算猫狗分类各类别准确率和总体准确率。
- 新增 `scripts/run_all.sh`，串联数据准备和三个实验。
- 新增 `.gitignore`，排除数据集、结果、日志、提交压缩包和报告。

### 结论

代码结构已整理为可推送版本；数据和报告不进入 Git 仓库。

### 下一步

运行全量实验后，将 `results/*.json` 中的指标写入作业报告。
