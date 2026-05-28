"""使用 CLIP 进行零样本猫狗分类。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_common import encode_texts, load_clip, select_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命令行参数对象。
    """

    parser = argparse.ArgumentParser(description="CLIP 零样本猫狗分类评测。")
    parser.add_argument("--test-dir", type=Path, default=Path("datasets/catdog/test"))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=Path("results/clip_catdog_metrics.json"))
    return parser.parse_args()


def collect_images(test_dir: Path) -> list[tuple[Path, str]]:
    """收集猫狗测试图片。

    Args:
        test_dir: 目录结构为 test/cat 和 test/dog 的测试集目录。

    Returns:
        图片路径和类别名列表。
    """

    samples: list[tuple[Path, str]] = []
    for label in ("cat", "dog"):
        label_dir = test_dir / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, label))
    return samples


def encode_image_batch(
    model: torch.nn.Module,
    processor: object,
    image_paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    """编码一批图片。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        image_paths: 图片路径列表。
        device: 运行设备。

    Returns:
        归一化后的图片特征。
    """

    images: list[Image.Image] = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    finally:
        for image in images:
            image.close()


def main() -> None:
    """运行猫狗零样本分类评测。"""

    args = parse_args()
    samples = collect_images(args.test_dir)
    if not samples:
        raise ValueError(f"没有在 {args.test_dir} 下找到 cat/dog 测试图片。")

    device = select_device(args.device)
    model, processor = load_clip(args.model_name, device)
    labels = ["cat", "dog"]
    prompts = ["a photo of a cat", "a photo of a dog"]
    text_features = encode_texts(model, processor, prompts, batch_size=2, device=device).to(device)

    correct_by_class = {label: 0 for label in labels}
    total_by_class = {label: 0 for label in labels}
    predictions: list[dict[str, str]] = []

    for start in tqdm(range(0, len(samples), args.batch_size), desc="classify images"):
        batch = samples[start : start + args.batch_size]
        image_features = encode_image_batch(model, processor, [path for path, _ in batch], device)
        predicted_indices = (image_features @ text_features.T).argmax(dim=1).cpu().tolist()
        for (path, gold_label), predicted_index in zip(batch, predicted_indices):
            predicted_label = labels[predicted_index]
            total_by_class[gold_label] += 1
            correct_by_class[gold_label] += int(predicted_label == gold_label)
            predictions.append(
                {
                    "path": str(path),
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                }
            )

    per_class_accuracy = {
        label: 100.0 * correct_by_class[label] / total_by_class[label]
        for label in labels
        if total_by_class[label] > 0
    }
    output = {
        "model_name": args.model_name,
        "test_dir": str(args.test_dir),
        "num_images": len(samples),
        "per_class_accuracy": per_class_accuracy,
        "overall_accuracy": 100.0 * sum(correct_by_class.values()) / sum(total_by_class.values()),
        "predictions": predictions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
