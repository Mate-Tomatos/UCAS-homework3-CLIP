# UCAS 作业三：CLIP 零样本图文检索

本仓库包含作业三的可复现实验代码：使用 CLIP 在 Flickr30k 上做零样本图文检索，并完成两个加分项：MSCOCO 图文检索、猫狗零样本分类。

## 本地环境

本机已验证环境：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python
torch 2.8.0+cu128
transformers 4.57.6
datasets 4.4.2
```

如需重新安装依赖：

```bash
pip install -r requirements.txt
```

OpenAI 官方 CLIP 代码按作业要求克隆在本地 `CLIP/`，commit 为 `d05afc4`。重新克隆命令：

```bash
git clone https://github.com/OpenAI/CLIP.git CLIP
```

## 数据准备

数据集、实验结果和报告不会提交到 Git。默认导出到 `datasets/`。其中 Flickr30k 使用 HuggingFace `AnyModal/flickr30k` 的测试集；MSCOCO 使用官方 `val2014.zip` 和 `captions_val2014.json`；猫狗分类使用 HuggingFace `microsoft/cats_vs_dogs`：

```bash
PYTHON_BIN=/mnt/kxh/miniconda3/envs/trl/bin/python bash scripts/run_all.sh
```

调试时可先小规模运行：

```bash
MAX_IMAGES=64 PYTHON_BIN=/mnt/kxh/miniconda3/envs/trl/bin/python bash scripts/run_all.sh
```

单独准备数据：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/prepare_datasets.py --dataset flickr30k
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/prepare_datasets.py --dataset coco
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/prepare_datasets.py --dataset catdog
```

## 单独评测

Flickr30k 主实验：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_retrieval.py \
  --annotations-jsonl datasets/flickr30k/annotations.jsonl \
  --output-json results/clip_flickr30k_metrics.json
```

MSCOCO 加分项：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_retrieval.py \
  --annotations-jsonl datasets/coco/annotations.jsonl \
  --max-images 5000 \
  --output-json results/clip_coco_5k_metrics.json
```

猫狗零样本分类加分项：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_catdog.py \
  --test-dir datasets/catdog/test \
  --output-json results/clip_catdog_metrics.json
```

## 输出指标

图文检索输出：

- `image_to_text_r1/r5/r10`
- `text_to_image_r1/r5/r10`
- `rsum`

猫狗分类输出：

- `per_class_accuracy.cat`
- `per_class_accuracy.dog`
- `overall_accuracy`
