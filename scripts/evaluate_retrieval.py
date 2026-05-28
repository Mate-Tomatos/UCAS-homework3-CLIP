"""使用 CLIP 评测 Flickr30k/MSCOCO 零样本图文检索召回率。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.clip_common import RetrievalItem, encode_images, encode_texts, load_clip, select_device


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命令行参数对象。
    """

    parser = argparse.ArgumentParser(description="CLIP 零样本图文检索评测。")
    parser.add_argument("--annotations-jsonl", type=Path, required=True, help="评测标注 JSONL。")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size-images", type=int, default=128)
    parser.add_argument("--batch-size-texts", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=0, help="0 表示使用全部图片。")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_items(path: Path, max_images: int) -> list[RetrievalItem]:
    """读取 JSONL 检索标注。

    Args:
        path: JSONL 文件路径。
        max_images: 最多读取图片数，0 表示不限制。

    Returns:
        检索样本列表。
    """

    items: list[RetrievalItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            captions = [str(caption) for caption in row["captions"] if str(caption).strip()]
            if not captions:
                continue
            items.append(
                RetrievalItem(
                    image_id=str(row["image_id"]),
                    image_path=Path(row["image_path"]),
                    captions=captions,
                )
            )
            if max_images > 0 and len(items) >= max_images:
                break
    return items


def recall_at(ranks: list[int], k_value: int) -> float:
    """计算 Recall@K 百分比。

    Args:
        ranks: 正确结果的排名列表，0 表示第一名。
        k_value: Recall 的 K。

    Returns:
        百分比形式的 Recall@K。
    """

    return 100.0 * sum(rank < k_value for rank in ranks) / len(ranks)


def compute_retrieval_metrics(similarity: torch.Tensor, caption_counts: list[int]) -> dict[str, float]:
    """计算图搜文和文搜图召回率。

    Args:
        similarity: 图片到文本相似度矩阵。
        caption_counts: 每张图片拥有的 caption 数。

    Returns:
        指标字典。
    """

    image_to_text_ranks: list[int] = []
    text_to_image_ranks: list[int] = []
    caption_offsets: list[int] = []
    offset = 0
    for count in caption_counts:
        caption_offsets.append(offset)
        offset += count

    sorted_text_indices = torch.argsort(similarity, dim=1, descending=True)
    for image_index, caption_count in enumerate(caption_counts):
        target_start = caption_offsets[image_index]
        target_indices = set(range(target_start, target_start + caption_count))
        ranked = sorted_text_indices[image_index].tolist()
        image_to_text_ranks.append(min(ranked.index(index) for index in target_indices))

    sorted_image_indices = torch.argsort(similarity, dim=0, descending=True)
    caption_to_image: list[int] = []
    for image_index, caption_count in enumerate(caption_counts):
        caption_to_image.extend([image_index] * caption_count)
    for caption_index, target_image in enumerate(caption_to_image):
        ranked = sorted_image_indices[:, caption_index].tolist()
        text_to_image_ranks.append(ranked.index(target_image))

    metrics = {
        "image_to_text_r1": recall_at(image_to_text_ranks, 1),
        "image_to_text_r5": recall_at(image_to_text_ranks, 5),
        "image_to_text_r10": recall_at(image_to_text_ranks, 10),
        "text_to_image_r1": recall_at(text_to_image_ranks, 1),
        "text_to_image_r5": recall_at(text_to_image_ranks, 5),
        "text_to_image_r10": recall_at(text_to_image_ranks, 10),
    }
    metrics["rsum"] = sum(metrics.values())
    return metrics


def main() -> None:
    """运行零样本图文检索评测。"""

    args = parse_args()
    items = read_items(args.annotations_jsonl, args.max_images)
    if not items:
        raise ValueError(f"没有从 {args.annotations_jsonl} 读取到有效样本。")

    missing = [str(item.image_path) for item in items if not item.image_path.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"有 {len(missing)} 张图片不存在，前 10 个为:\n{preview}")

    device = select_device(args.device)
    model, processor = load_clip(args.model_name, device)
    image_features = encode_images(
        model=model,
        processor=processor,
        image_paths=[item.image_path for item in items],
        batch_size=args.batch_size_images,
        device=device,
    )
    captions = [caption for item in items for caption in item.captions]
    text_features = encode_texts(
        model=model,
        processor=processor,
        texts=captions,
        batch_size=args.batch_size_texts,
        device=device,
    )

    similarity = image_features @ text_features.T
    metrics = compute_retrieval_metrics(similarity, [len(item.captions) for item in items])
    output = {
        "model_name": args.model_name,
        "annotations_jsonl": str(args.annotations_jsonl),
        "num_images": len(items),
        "num_captions": len(captions),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
