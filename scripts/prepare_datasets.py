"""下载并整理作业三需要的数据集。"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_VAL2014_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命令行参数对象。
    """

    parser = argparse.ArgumentParser(description="准备 Flickr30k、MSCOCO 和猫狗分类数据。")
    parser.add_argument("--dataset", choices=("flickr30k", "coco", "catdog", "all"), default="all")
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--max-images", type=int, default=0, help="调试用，0 表示全量导出。")
    return parser.parse_args()


def save_image(image_value: object, output_path: Path) -> None:
    """保存 HuggingFace 数据集中的图片字段。

    Args:
        image_value: 图片字段，可能是 PIL 图片、字节字典或本地路径。
        output_path: 目标图片路径。

    Raises:
        TypeError: 图片字段类型不受支持。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_value, Image.Image):
        image_value.convert("RGB").save(output_path)
        return
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            image = Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
            image.save(output_path)
            image.close()
            return
        if image_value.get("path") is not None:
            shutil.copyfile(str(image_value["path"]), output_path)
            return
    if isinstance(image_value, (str, Path)):
        shutil.copyfile(str(image_value), output_path)
        return
    raise TypeError(f"不支持的图片字段类型: {type(image_value)!r}")


def first_existing(row: dict[str, object], names: tuple[str, ...]) -> object | None:
    """读取第一项存在的字段。

    Args:
        row: 数据集样本。
        names: 候选字段名。

    Returns:
        字段值；不存在时返回 None。
    """

    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def normalize_captions(value: object) -> list[str]:
    """将不同数据集的 caption 字段统一为字符串列表。

    Args:
        value: 原始 caption 字段。

    Returns:
        caption 字符串列表。
    """

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        captions: list[str] = []
        for item in value:
            if isinstance(item, str):
                captions.append(item)
            elif isinstance(item, dict):
                text = first_existing(item, ("caption", "text", "raw", "sentence"))
                if text is not None:
                    captions.append(str(text))
            else:
                captions.append(str(item))
        return captions
    if isinstance(value, dict):
        text = first_existing(value, ("caption", "text", "raw", "sentence"))
        if text is not None:
            return [str(text)]
    return [str(value)]


def write_retrieval_jsonl(
    dataset: Dataset,
    output_dir: Path,
    image_name_prefix: str,
    max_images: int,
) -> None:
    """导出统一格式的检索数据。

    Args:
        dataset: HuggingFace Dataset。
        output_dir: 输出目录。
        image_name_prefix: 图片文件名前缀。
        max_images: 最多导出图片数，0 表示全量。

    Raises:
        KeyError: 数据集缺少图片或 caption 字段。
    """

    image_dir = output_dir / "images"
    annotations_path = output_dir / "annotations.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(dataset) if max_images <= 0 else min(len(dataset), max_images)

    with annotations_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(count), desc=f"export {output_dir.name}"):
            row = dataset[index]
            image_value = first_existing(row, ("image", "jpg", "png"))
            caption_value = first_existing(
                row,
                ("captions", "caption", "sentences", "texts", "text", "raw"),
            )
            if image_value is None or caption_value is None:
                raise KeyError(f"样本字段不完整，已有字段: {sorted(row.keys())}")
            image_id = str(first_existing(row, ("image_id", "id", "filename")) or index)
            image_path = image_dir / f"{image_name_prefix}_{index:06d}.jpg"
            save_image(image_value, image_path)
            record = {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "captions": normalize_captions(caption_value),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_file(url: str, output_path: Path) -> None:
    """下载文件到本地路径。

    Args:
        url: 下载地址。
        output_path: 输出文件路径。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        with output_path.open("wb") as handle:
            with tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))


def extract_zip(zip_path: Path, output_dir: Path, marker_path: Path) -> None:
    """解压 zip 文件。

    Args:
        zip_path: zip 文件路径。
        output_dir: 解压目标目录。
        marker_path: 解压完成标记文件。
    """

    if marker_path.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    marker_path.write_text("ok\n", encoding="utf-8")


def prepare_flickr30k(root: Path, max_images: int) -> None:
    """下载并导出 Flickr30k 测试集。

    Args:
        root: 数据根目录。
        max_images: 最多导出图片数，0 表示全量。
    """

    dataset = load_dataset("AnyModal/flickr30k", split="test")
    write_retrieval_jsonl(dataset, root / "flickr30k", "flickr30k", max_images)


def prepare_coco(root: Path, max_images: int) -> None:
    """下载并导出 MSCOCO 验证集。

    Args:
        root: 数据根目录。
        max_images: 最多导出图片数，0 表示全量。
    """

    coco_dir = root / "coco"
    raw_dir = coco_dir / "raw"
    download_file(COCO_VAL2014_URL, raw_dir / "val2014.zip")
    download_file(COCO_ANNOTATIONS_URL, raw_dir / "annotations_trainval2014.zip")
    extract_zip(raw_dir / "val2014.zip", raw_dir, raw_dir / ".val2014_extracted")
    extract_zip(
        raw_dir / "annotations_trainval2014.zip",
        raw_dir,
        raw_dir / ".annotations_extracted",
    )

    annotations_path = raw_dir / "annotations" / "captions_val2014.json"
    captions_data = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_id_to_file = {int(image["id"]): image["file_name"] for image in captions_data["images"]}
    image_id_to_captions: dict[int, list[str]] = {}
    for annotation in captions_data["annotations"]:
        image_id = int(annotation["image_id"])
        image_id_to_captions.setdefault(image_id, []).append(str(annotation["caption"]))

    output_annotations = coco_dir / "annotations.jsonl"
    output_annotations.parent.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(image_id_to_captions)
    if max_images > 0:
        image_ids = image_ids[:max_images]
    with output_annotations.open("w", encoding="utf-8") as handle:
        for image_id in tqdm(image_ids, desc="export coco"):
            file_name = image_id_to_file[image_id]
            record = {
                "image_id": str(image_id),
                "image_path": str((raw_dir / "val2014" / file_name).resolve()),
                "captions": image_id_to_captions[image_id],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def infer_catdog_label(row: dict[str, object], index: int) -> str:
    """推断猫狗样本类别。

    Args:
        row: 数据集样本。
        index: 样本序号。

    Returns:
        cat 或 dog。
    """

    label = first_existing(row, ("labels", "label", "class", "category"))
    if isinstance(label, str):
        value = label.lower()
        if "cat" in value:
            return "cat"
        if "dog" in value:
            return "dog"
    if isinstance(label, int):
        return "cat" if label == 0 else "dog"
    image_value = first_existing(row, ("image", "jpg", "png"))
    if isinstance(image_value, dict) and image_value.get("path") is not None:
        name = str(image_value["path"]).lower()
        if "cat" in name:
            return "cat"
        if "dog" in name:
            return "dog"
    return "cat" if index % 2 == 0 else "dog"


def prepare_catdog(root: Path, max_images: int) -> None:
    """下载并导出猫狗分类测试集。

    Args:
        root: 数据根目录。
        max_images: 最多导出图片数，0 表示全量。
    """

    dataset = load_dataset("microsoft/cats_vs_dogs", split="train")
    test_dir = root / "catdog" / "test"
    count = len(dataset) if max_images <= 0 else min(len(dataset), max_images)
    for index in tqdm(range(count), desc="export catdog"):
        row = dataset[index]
        image_value = first_existing(row, ("image", "jpg", "png"))
        if image_value is None:
            raise KeyError(f"样本字段不完整，已有字段: {sorted(row.keys())}")
        label = infer_catdog_label(row, index)
        save_image(image_value, test_dir / label / f"{label}_{index:06d}.jpg")


def main() -> None:
    """运行数据准备流程。"""

    args = parse_args()
    if args.dataset in ("flickr30k", "all"):
        prepare_flickr30k(args.root, args.max_images)
    if args.dataset in ("coco", "all"):
        prepare_coco(args.root, args.max_images)
    if args.dataset in ("catdog", "all"):
        prepare_catdog(args.root, args.max_images)


if __name__ == "__main__":
    main()
