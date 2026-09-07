#!/usr/bin/env python3
"""Train and run portable whole-image land-use classifiers.

The default taxonomy is deliberately five-class: cropland, forest, grassland,
built-up land and water.  ``roi_water_refinement.py`` performs the separate,
manual sixth-class water refinement after machine inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = [
    {"id": 1, "name": "cropland", "display_name": "耕地"},
    {"id": 2, "name": "forest", "display_name": "林地"},
    {"id": 3, "name": "grassland", "display_name": "草地"},
    {"id": 4, "name": "built_up", "display_name": "建筑用地"},
    {"id": 5, "name": "water", "display_name": "水体"},
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Sample:
    image_id: str
    class_id: int
    image_path: Path
    group: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_stage_path(key: str, suffix: str = ".pt2") -> Path:
    """Return an ASCII-only staging path for PyTorch's Windows C++ file APIs.

    Some PyTorch builds cannot open paths containing Chinese user names.  The
    final package still lives wherever the user requested; only the C++ read/
    write operation is staged under ``C:\\Temp`` (or MAESA_TORCH_STAGE_DIR).
    """
    root = Path(os.environ.get("MAESA_TORCH_STAGE_DIR", "C:/Temp/maesa_torch_stage"))
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return root / f"{digest}{suffix}"


def save_exported_program(program, target: Path) -> None:
    torch, *_ = _imports()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = torch_stage_path(f"write:{target.resolve()}")
    torch.export.save(program, str(staged))
    shutil.move(str(staged), str(target))


def load_exported_program(source: Path):
    torch, *_ = _imports()
    staged = torch_stage_path(f"read:{sha256(source)}")
    if not staged.is_file() or sha256(staged) != sha256(source):
        shutil.copy2(source, staged)
    return torch.export.load(str(staged))


def load_classes(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        classes = DEFAULT_CLASSES
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        classes = payload.get("classes", payload) if isinstance(payload, dict) else payload
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("classes must be a list with at least two entries")
    normalised: list[dict[str, Any]] = []
    ids: set[int] = set()
    names: set[str] = set()
    for item in classes:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int) or item["id"] <= 0:
            raise ValueError("each class needs a positive integer id")
        name = str(item.get("name", "")).strip()
        if not name or item["id"] in ids or name in names:
            raise ValueError("class ids and names must be unique and non-empty")
        ids.add(item["id"]); names.add(name)
        normalised.append({
            "id": item["id"], "name": name,
            "display_name": str(item.get("display_name", name)).strip() or name,
        })
    return normalised


def build_image_index(image_root: Path) -> dict[str, Path]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for candidate in image_root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = candidate.relative_to(image_root).as_posix()
        aliases = {relative, candidate.name, candidate.stem}
        for alias in aliases:
            if alias in index and index[alias] != candidate:
                duplicates.add(alias)
            else:
                index[alias] = candidate
    if not index:
        raise ValueError(f"no supported images found beneath {image_root}")
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:8])
        raise ValueError(f"ambiguous image IDs in {image_root}: {preview}")
    return index


def _resolve_image(image_id: str, index: dict[str, Path]) -> Path:
    cleaned = image_id.replace("\\", "/").lstrip("./")
    choices = [cleaned, Path(cleaned).name, Path(cleaned).stem]
    for choice in choices:
        if choice in index:
            return index[choice]
    raise FileNotFoundError(f"label image_id does not resolve to an image: {image_id}")


def read_labels(labels_path: Path, image_root: Path, classes: list[dict[str, Any]]) -> list[Sample]:
    allowed_ids = {item["id"] for item in classes}
    image_index = build_image_index(image_root)
    records: list[Sample] = []
    seen: set[str] = set()
    for number, raw in enumerate(labels_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if number == 1 and len(fields) >= 2 and fields[0].lower() == "image_id" and fields[1].lower() == "class_id":
            continue
        if len(fields) not in {2, 3}:
            raise ValueError(f"{labels_path}:{number} must contain image_id class_id [group]")
        image_id = fields[0]
        try:
            class_id = int(fields[1])
        except ValueError as error:
            raise ValueError(f"{labels_path}:{number} has a non-integer class_id") from error
        if class_id not in allowed_ids:
            raise ValueError(f"{labels_path}:{number} class_id {class_id} is not declared")
        if image_id in seen:
            raise ValueError(f"{labels_path}:{number} repeats image_id {image_id}")
        seen.add(image_id)
        path = _resolve_image(image_id, image_index)
        group = fields[2] if len(fields) == 3 else path.stem.split("-")[0]
        records.append(Sample(image_id, class_id, path, group))
    if not records:
        raise ValueError(f"no labelled samples found in {labels_path}")
    observed = {record.class_id for record in records}
    missing = allowed_ids - observed
    if missing:
        raise ValueError(f"training labels have no samples for class ids: {sorted(missing)}")
    return records


def stratified_group_split(samples: list[Sample], validation_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)
    group_labels: dict[str, int] = {}
    for group, grouped in groups.items():
        labels = {sample.class_id for sample in grouped}
        if len(labels) != 1:
            raise ValueError(f"group {group!r} has multiple classes; pass distinct group names in column three")
        group_labels[group] = next(iter(labels))
    by_class: dict[int, list[str]] = defaultdict(list)
    for group, class_id in group_labels.items():
        by_class[class_id].append(group)
    generator = random.Random(seed)
    validation_groups: set[str] = set()
    for class_id, class_groups in by_class.items():
        if len(class_groups) < 2:
            raise ValueError(f"class {class_id} needs at least two independent groups for validation")
        generator.shuffle(class_groups)
        count = max(1, round(len(class_groups) * validation_fraction))
        count = min(count, len(class_groups) - 1)
        validation_groups.update(class_groups[:count])
    train = [sample for sample in samples if sample.group not in validation_groups]
    validation = [sample for sample in samples if sample.group in validation_groups]
    return train, validation


def _imports() -> tuple[Any, ...]:
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import Dataset
        from torchvision import models, transforms
    except ImportError as error:
        raise RuntimeError("install dependencies with: pip install -r requirements.txt") from error
    return torch, Image, nn, Dataset, models, transforms


def build_model(architecture: str, class_count: int, pretrained: bool):
    _, _, nn, _, models, _ = _imports()
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, class_count)
    elif architecture == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    else:
        raise ValueError("architecture must be resnet50 or efficientnet_v2_s")
    return model


def initialise_from_checkpoint(model, checkpoint_path: Path) -> int:
    """Load compatible backbone tensors while always keeping a fresh classifier head."""
    torch, *_ = _imports()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no state_dict: {checkpoint_path}")
    target = model.state_dict()
    compatible = {
        key.removeprefix("module."): value for key, value in state.items()
        if key.removeprefix("module.") in target
        and target[key.removeprefix("module.")].shape == value.shape
        and not key.removeprefix("module.").startswith(("fc.", "classifier."))
    }
    if not compatible:
        raise ValueError(f"no compatible backbone weights in {checkpoint_path}")
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def build_transform(image_size: int, training: bool):
    _, _, _, _, _, transforms = _imports()
    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90), transforms.ColorJitter(0.15, 0.15, 0.08, 0.02),
            transforms.ToTensor(), normalise,
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)), transforms.CenterCrop(image_size),
        transforms.ToTensor(), normalise,
    ])


def create_dataset(samples: list[Sample], classes: list[dict[str, Any]], transform):
    torch, Image, _, Dataset, _, _ = _imports()
    class_to_index = {item["id"]: index for index, item in enumerate(classes)}

    class LandUseDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            sample = samples[index]
            with Image.open(sample.image_path) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, torch.tensor(class_to_index[sample.class_id], dtype=torch.long)
    return LandUseDataset()


def metrics(truth: Iterable[int], predicted: Iterable[int], classes: list[dict[str, Any]]) -> dict[str, Any]:
    values = list(zip(truth, predicted))
    size = len(classes)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for expected, actual in values:
        matrix[expected][actual] += 1
    per_class = []
    f1s = []
    for index, item in enumerate(classes):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(size)) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"id": item["id"], "name": item["name"], "support": sum(matrix[index]),
                          "precision": precision, "recall": recall, "f1": f1})
        f1s.append(f1)
    correct = sum(matrix[index][index] for index in range(size))
    return {"sample_count": len(values), "accuracy": correct / len(values) if values else 0.0,
            "macro_f1": sum(f1s) / len(f1s), "confusion_matrix": matrix, "per_class": per_class}


def evaluate(model, loader, criterion, device, classes: list[dict[str, Any]]) -> dict[str, Any]:
    torch, *_ = _imports()
    model.eval(); total_loss = 0.0; truth: list[int] = []; predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            logits = model(images); total_loss += float(criterion(logits, labels).item()) * images.size(0)
            truth.extend(labels.cpu().tolist()); predicted.extend(logits.argmax(1).cpu().tolist())
    result = metrics(truth, predicted, classes)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, *_ = _imports()
    classes = load_classes(args.classes_json)
    samples = read_labels(args.labels, args.image_root, classes)
    train_samples, validation_samples = stratified_group_split(samples, args.validation_fraction, args.seed)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = torch.utils.data.DataLoader(create_dataset(train_samples, classes, build_transform(args.image_size, True)), shuffle=True, **loader_args)
    validation_loader = torch.utils.data.DataLoader(create_dataset(validation_samples, classes, build_transform(args.image_size, False)), shuffle=False, **loader_args)
    model = build_model(args.architecture, len(classes), not args.no_pretrained)
    initialized_tensors = initialise_from_checkpoint(model, args.initial_checkpoint) if args.initial_checkpoint else 0
    model = model.to(device)
    class_counts = Counter(sample.class_id for sample in train_samples)
    class_weight = torch.tensor([len(train_samples) / (len(classes) * class_counts[item["id"]]) for item in classes], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = args.output_dir.resolve(); checkpoints = run_dir / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None; history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += float(loss.item()) * images.size(0)
        scheduler.step()
        validation = evaluate(model, validation_loader, criterion, device, classes)
        record = {"epoch": epoch, "train_loss": total_loss / len(train_loader.dataset), **validation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if best is None or (record["macro_f1"], record["accuracy"]) > (best["macro_f1"], best["accuracy"]):
            best = record
            torch.save({"state_dict": model.state_dict(), "architecture": args.architecture, "classes": classes,
                        "image_size": args.image_size, "best_validation": best}, checkpoints / "best_state_dict.pt")
    assert best is not None
    checkpoint = torch.load(checkpoints / "best_state_dict.pt", map_location="cpu", weights_only=True)
    model = build_model(args.architecture, len(classes), False).cpu(); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    package = run_dir / "model_package"; package.mkdir(exist_ok=True)
    exported = torch.export.export(model, (torch.zeros(1, 3, args.image_size, args.image_size),))
    weights = package / "model.pt2"; save_exported_program(exported, weights)
    bootstrap_manifest = args.labels.parent / "bootstrap_manifest.json"
    label_provenance: dict[str, Any] = {"origin": "user_supplied_labels", "manifest": None}
    if bootstrap_manifest.is_file():
        bootstrap = json.loads(bootstrap_manifest.read_text(encoding="utf-8-sig"))
        label_provenance = {"origin": bootstrap.get("label_origin", "bootstrap_labels"),
                            "manifest": str(bootstrap_manifest.resolve()),
                            "warning": "Bootstrap labels are not a substitute for field or manually verified reference data."}
    config = {"schema_version": 1, "task": "image_classification", "model_id": args.model_id,
              "format": "exported_program", "architecture": args.architecture, "weights": weights.name,
              "sha256": sha256(weights), "classes": classes,
              "input": {"color_mode": "RGB", "image_size": args.image_size, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
              "output": {"type": "logits"},
              "training": {"trained_at": datetime.now(timezone.utc).isoformat(), "device": str(device), "seed": args.seed,
                           "sample_count": len(samples), "train_sample_count": len(train_samples),
                           "validation_sample_count": len(validation_samples), "label_file": str(args.labels.resolve()),
                           "class_distribution": dict(sorted(Counter(sample.class_id for sample in samples).items())),
                           "initial_checkpoint": str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,
                           "initialized_backbone_tensors": initialized_tensors, "best_validation": best}}
    config["training"]["label_provenance"] = label_provenance
    (package / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card = "# 五类土地利用图像分类模型\n\n"
    card += "自动分类：耕地、林地、草地、建筑用地、水体。ROI 复核后才能把水体拆分为自然水体和沉陷水体。\n\n"
    card += "## 标签来源\n\n```json\n" + json.dumps(label_provenance, ensure_ascii=False, indent=2) + "\n```\n\n"
    card += "该模型只能作为可运行的初始模型；未使用独立人工真值时，验证指标不得用作研究结论或跨区域精度声明。\n\n"
    card += "## 留出集结果\n\n```json\n" + json.dumps(best, ensure_ascii=False, indent=2) + "\n```\n"
    (package / "model_card.md").write_text(card, encoding="utf-8")
    result = validate_package(package)
    if result["status"] != "valid": raise RuntimeError("exported package is invalid: " + "; ".join(result["errors"]))
    print(json.dumps(result, ensure_ascii=False, indent=2)); return result


def validate_package(package: Path) -> dict[str, Any]:
    errors: list[str] = []; warnings: list[str] = []
    try:
        config = json.loads((package / "model_config.json").read_text(encoding="utf-8-sig"))
    except Exception as error:
        return {"status": "invalid", "errors": [f"invalid model_config.json: {error}"], "warnings": []}
    if config.get("schema_version") != 1 or config.get("task") != "image_classification": errors.append("config must declare schema_version=1 and task=image_classification")
    if config.get("format") != "exported_program": errors.append("only exported_program is supported")
    classes = config.get("classes")
    try:
        if not isinstance(classes, list): raise ValueError("classes must be a list")
        ids = [item.get("id") for item in classes]
        if any(not isinstance(value, int) or value <= 0 for value in ids) or len(set(ids)) != len(ids): raise ValueError("class ids must be unique positive integers")
    except ValueError as error: errors.append(str(error))
    input_config = config.get("input", {})
    if input_config.get("color_mode") != "RGB" or not isinstance(input_config.get("image_size"), int): errors.append("input must declare RGB and integer image_size")
    if len(input_config.get("mean", [])) != 3 or len(input_config.get("std", [])) != 3: errors.append("input mean and std must have three values")
    weights = package / str(config.get("weights", ""))
    if not weights.is_file(): errors.append(f"missing weights: {weights.name}")
    elif config.get("sha256") != sha256(weights): errors.append("weights sha256 does not match model_config.json")
    if not (package / "model_card.md").is_file(): warnings.append("model_card.md is missing")
    return {"status": "valid" if not errors else "invalid", "model_id": config.get("model_id"),
            "class_count": len(classes) if isinstance(classes, list) else 0, "errors": errors, "warnings": warnings}


def collect_inference_images(input_path: Path) -> list[tuple[str, Path]]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_SUFFIXES: return [(input_path.name, input_path)]
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        root = input_path.parent; image_index = build_image_index(root); entries = []
        for raw in input_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.lower().startswith("image_id"): continue
            image_id = line.split()[0]; entries.append((image_id, _resolve_image(image_id, image_index)))
        return entries
    if input_path.is_dir():
        return [(candidate.relative_to(input_path).as_posix(), candidate) for candidate in input_path.rglob("*") if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES]
    raise FileNotFoundError(f"input is not an image, .txt list, or image directory: {input_path}")


def infer(args: argparse.Namespace) -> dict[str, Any]:
    torch, Image, *_ = _imports()
    package = args.model_package.resolve(); validation = validate_package(package)
    if validation["status"] != "valid": raise ValueError("invalid model package: " + "; ".join(validation["errors"]))
    config = json.loads((package / "model_config.json").read_text(encoding="utf-8")); classes = config["classes"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    # ExportedProgram preserves the model's eval graph; its module deliberately
    # rejects a subsequent .eval() call in current PyTorch releases.
    model = load_exported_program(package / config["weights"]).module().to(device)
    transform = build_transform(config["input"]["image_size"], False); rows: list[tuple[str, int, float]] = []
    with torch.inference_mode():
        for image_id, path in collect_inference_images(args.input):
            with Image.open(path) as image: tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            probability = torch.softmax(model(tensor), dim=1)[0]; index = int(probability.argmax().item())
            rows.append((image_id, int(classes[index]["id"]), float(probability[index].item())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=" "); writer.writerow(["image_id", "class_id", "confidence"]); writer.writerows(rows)
    result = {"status": "completed", "model_id": config["model_id"], "image_count": len(rows), "output": str(args.output.resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2)); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train a five-class model and export a .pt2 package")
    train_parser.add_argument("--image-root", required=True, type=Path); train_parser.add_argument("--labels", required=True, type=Path)
    train_parser.add_argument("--output-dir", required=True, type=Path); train_parser.add_argument("--classes-json", type=Path)
    train_parser.add_argument("--model-id", default="maesa-landuse-5class-v1"); train_parser.add_argument("--architecture", choices=["resnet50", "efficientnet_v2_s"], default="resnet50")
    train_parser.add_argument("--image-size", type=int, default=256); train_parser.add_argument("--epochs", type=int, default=30); train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4); train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2); train_parser.add_argument("--seed", type=int, default=42); train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto"); train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--initial-checkpoint", type=Path, help="optional trusted checkpoint; only compatible backbone weights are loaded")
    validate_parser = commands.add_parser("validate-package"); validate_parser.add_argument("--model-package", required=True, type=Path)
    infer_parser = commands.add_parser("infer"); infer_parser.add_argument("--model-package", required=True, type=Path); infer_parser.add_argument("--input", required=True, type=Path); infer_parser.add_argument("--output", required=True, type=Path); infer_parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "train": train(args)
    elif args.command == "validate-package":
        result = validate_package(args.model_package.resolve()); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result["status"] == "valid" else 1
    else: infer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
