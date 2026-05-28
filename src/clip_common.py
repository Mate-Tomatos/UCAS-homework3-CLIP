"""CLIP 零样本实验的通用数据结构与编码函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class RetrievalItem:
    """图文检索评测中的一张图片及其对应文本。

    Args:
        image_id: 图片唯一标识。
        image_path: 图片本地路径。
        captions: 与图片匹配的描述文本列表。
    """

    image_id: str
    image_path: Path
    captions: list[str]


def normalize(features: torch.Tensor) -> torch.Tensor:
    """对特征向量做 L2 归一化。

    Args:
        features: 模型输出的特征矩阵。

    Returns:
        归一化后的特征矩阵。
    """

    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def select_device(device_name: str) -> torch.device:
    """根据命令行参数选择运行设备。

    Args:
        device_name: 设备名称，支持 auto、cpu、cuda 或 cuda:0 这类写法。

    Returns:
        PyTorch 设备对象。
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_clip(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    """加载 HuggingFace CLIP 模型与预处理器。

    Args:
        model_name: HuggingFace 模型名或本地模型目录。
        device: 模型所在设备。

    Returns:
        CLIP 模型与预处理器。
    """

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码图片。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        image_paths: 图片路径列表。
        batch_size: 编码 batch size。
        device: 运行设备。

    Returns:
        图片特征矩阵。
    """

    features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="encode images"):
        batch_paths = image_paths[start : start + batch_size]
        images: list[Image.Image] = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                batch_features = model.get_image_features(**inputs)
            features.append(normalize(batch_features).cpu())
        finally:
            for image in images:
                image.close()
    return torch.cat(features, dim=0)


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码文本。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        texts: 文本列表。
        batch_size: 编码 batch size。
        device: 运行设备。

    Returns:
        文本特征矩阵。
    """

    features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode texts"):
        batch = texts[start : start + batch_size]
        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            batch_features = model.get_text_features(**inputs)
        features.append(normalize(batch_features).cpu())
    return torch.cat(features, dim=0)
