# CLIP 零样本图文检索

本项目使用 CLIP 进行零样本图文检索与图像分类评测，覆盖 Flickr30k 图文检索、MSCOCO 图文检索和猫狗二分类三个实验。

## 模型

评测模型使用 HuggingFace Transformers 中的 `openai/clip-vit-base-patch32`。项目同时保留 OpenAI 官方 CLIP 代码作为参考实现，位于本地 `CLIP/` 目录，当前 commit 为 `d05afc4`。

重新克隆 OpenAI CLIP：

```bash
git clone https://github.com/OpenAI/CLIP.git CLIP
```

## 数据集

| 实验 | 数据集 | 本地标注路径 |
| --- | --- | --- |
| Flickr30k 图文检索 | HuggingFace `AnyModal/flickr30k` test split | `datasets/flickr30k/annotations.jsonl` |
| MSCOCO 图文检索 | COCO 2014 validation images and captions | `datasets/coco/annotations.jsonl` |
| 猫狗二分类 | HuggingFace `microsoft/cats_vs_dogs` | `datasets/catdog/test` |

## 环境

本地已验证环境：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python
torch 2.8.0+cu128
transformers 4.57.6
datasets 4.4.2
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

一次性准备全部数据并运行全部评测：

```bash
PYTHON_BIN=/mnt/kxh/miniconda3/envs/trl/bin/python bash scripts/run_all.sh
```

小规模调试运行：

```bash
MAX_IMAGES=64 PYTHON_BIN=/mnt/kxh/miniconda3/envs/trl/bin/python bash scripts/run_all.sh
```

单独准备数据：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/prepare_datasets.py --dataset flickr30k
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/prepare_datasets.py --dataset coco
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/prepare_datasets.py --dataset catdog
```

## 评测命令

Flickr30k 图文检索：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_retrieval.py \
  --annotations-jsonl datasets/flickr30k/annotations.jsonl \
  --output-json results/clip_flickr30k_metrics.json
```

MSCOCO 5K 图文检索：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_retrieval.py \
  --annotations-jsonl datasets/coco/annotations.jsonl \
  --max-images 5000 \
  --output-json results/clip_coco_5k_metrics.json
```

猫狗零样本分类：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_catdog.py \
  --test-dir datasets/catdog/test \
  --output-json results/clip_catdog_metrics.json
```

## 输出指标

图文检索输出：

- `image_to_text_r1`
- `image_to_text_r5`
- `image_to_text_r10`
- `text_to_image_r1`
- `text_to_image_r5`
- `text_to_image_r10`
- `rsum`

猫狗分类输出：

- `per_class_accuracy.cat`
- `per_class_accuracy.dog`
- `overall_accuracy`
