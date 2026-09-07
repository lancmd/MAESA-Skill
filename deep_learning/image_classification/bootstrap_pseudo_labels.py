#!/usr/bin/env python3
"""Create a small, auditable five-class bootstrap dataset from optical GeoTIFFs.

This is intentionally an *initial-label* generator, not a substitute for field
or manual reference data.  It only keeps spectrally homogeneous, high-confidence
patches and records every threshold, source window, CRS and transform in a
manifest so that users can replace the labels with validated samples later.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLASSES = {1: "cropland", 2: "forest", 3: "grassland", 4: "built_up", 5: "water"}


def indices(array):
    import numpy as np
    _, _, red, nir, swir1, _, ndwi = array
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
    return ndvi, ndwi, ndbi


def score(class_id: int, ndvi, ndwi, ndbi):
    import numpy as np
    if class_id == 1:  # crops: medium vegetation
        return np.where((ndvi > .28) & (ndvi < .42) & (ndwi < -.15), 1 - abs(ndvi - .35), -np.inf)
    if class_id == 2:  # forest: dense vegetation
        return np.where((ndvi > .42) & (ndwi < -.15), ndvi, -np.inf)
    if class_id == 3:  # grass: low-to-medium vegetation
        return np.where((ndvi > .12) & (ndvi < .28) & (ndwi < -.15), 1 - abs(ndvi - .20), -np.inf)
    if class_id == 4:  # built land: little vegetation and positive built index
        return np.where((ndvi < .18) & (ndwi < -.20) & (ndbi > 0), ndbi - ndvi, -np.inf)
    if class_id == 5:  # open water
        return np.where((ndwi > .15) & (ndvi < .10), ndwi - ndvi, -np.inf)
    raise ValueError(class_id)


def purity(class_id: int, ndvi, ndwi, ndbi, valid) -> float:
    if class_id == 1: condition = (ndvi > .28) & (ndvi < .42) & (ndwi < -.15)
    elif class_id == 2: condition = (ndvi > .42) & (ndwi < -.15)
    elif class_id == 3: condition = (ndvi > .12) & (ndvi < .28) & (ndwi < -.15)
    elif class_id == 4: condition = (ndvi < .18) & (ndwi < -.20) & (ndbi > 0)
    else: condition = (ndwi > .15) & (ndvi < .10)
    return float(condition[valid].mean()) if valid.any() else 0.0


def candidates(raster: Path, coarse_size: int) -> list[dict[str, Any]]:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(raster) as source:
        values = source.read(list(range(1, 8)), out_shape=(7, coarse_size, coarse_size), resampling=Resampling.average).astype("float32")
        valid = np.all(np.isfinite(values), axis=0) & np.all(values > -9990, axis=0)
        ndvi, ndwi, ndbi = indices(values)
        output: list[dict[str, Any]] = []
        for class_id in CLASSES:
            class_score = score(class_id, ndvi, ndwi, ndbi)
            class_score[~valid] = -np.inf
            for flat in np.argpartition(class_score.ravel(), -min(500, class_score.size))[-min(500, class_score.size):]:
                row, col = np.unravel_index(flat, class_score.shape)
                value = float(class_score[row, col])
                if np.isfinite(value): output.append({"class_id": class_id, "score": value, "row": int((row + .5) * source.height / coarse_size), "col": int((col + .5) * source.width / coarse_size), "raster": str(raster)})
        return output


def write_dataset(rasters: list[Path], output_dir: Path, samples_per_class: int, coarse_size: int) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from PIL import Image
    ranked = sorted((item for raster in rasters for item in candidates(raster, coarse_size)), key=lambda item: item["score"], reverse=True)
    images = output_dir / "images"; images.mkdir(parents=True, exist_ok=True)
    selected: dict[int, list[dict[str, Any]]] = {class_id: [] for class_id in CLASSES}
    for item in ranked:
        class_id = item["class_id"]
        if len(selected[class_id]) >= samples_per_class: continue
        source_path = Path(item["raster"])
        with rasterio.open(source_path) as source:
            # Match the full-resolution crop to one coarse screening cell.
            # A previous fixed 256-pixel window mixed several land-cover units
            # and defeated the purity gate on heterogeneous agricultural areas.
            pixels = max(16, int(round(source.width / coarse_size)))
            row = max(0, min(item["row"] - pixels // 2, source.height - pixels)); col = max(0, min(item["col"] - pixels // 2, source.width - pixels))
            data = source.read(list(range(1, 8)), window=rasterio.windows.Window(col, row, pixels, pixels)).astype("float32")
            valid = np.all(np.isfinite(data), axis=0) & np.all(data > -9990, axis=0)
            ndvi, ndwi, ndbi = indices(data); fraction = purity(class_id, ndvi, ndwi, ndbi, valid)
            # Built-up land is intrinsically heterogeneous at 10–30 m (roads,
            # roofs, trees and shadows share one small patch).  Its documented
            # gate is lower, while vegetation and water remain >= 72% pure.
            minimum_purity = .60 if class_id == 4 else .72
            if valid.mean() < .95 or fraction < minimum_purity: continue
            # A fixed reflectance stretch keeps all generated RGB patches comparable.
            rgb = np.clip(np.moveaxis(data[:3], 0, -1) * 512, 0, 255).astype("uint8")
            stem = f"{source_path.stem}_{row:05d}_{col:05d}_{class_id}"
            image_name = stem + ".png"; Image.fromarray(rgb, "RGB").resize((256, 256), Image.Resampling.BILINEAR).save(images / image_name)
            selected[class_id].append({"image_id": f"images/{image_name}", "class_id": class_id, "group": stem, "source_raster": str(source_path), "window": [int(col), int(row), pixels, pixels], "valid_fraction": float(valid.mean()), "label_purity": fraction, "minimum_label_purity": minimum_purity, "coarse_score": item["score"], "crs": str(source.crs), "transform": list(source.window_transform(rasterio.windows.Window(col, row, pixels, pixels)))[:6]})
        if all(len(records) >= samples_per_class for records in selected.values()): break
    missing = {CLASSES[key]: samples_per_class - len(value) for key, value in selected.items() if len(value) < samples_per_class}
    if missing: raise RuntimeError(f"unable to find enough high-confidence patches: {missing}")
    records = [record for entries in selected.values() for record in entries]
    (output_dir / "train.txt").write_text("\n".join(f"{record['image_id']} {record['class_id']} {record['group']}" for record in records) + "\n", encoding="utf-8")
    manifest = {"status": "completed", "label_origin": "spectral_high_confidence_bootstrap", "generated_at": datetime.now(timezone.utc).isoformat(), "classes": CLASSES, "thresholds": {"cropland": ".28 < NDVI < .42, NDWI < -.15", "forest": "NDVI > .42, NDWI < -.15", "grassland": ".12 < NDVI < .28, NDWI < -.15", "built_up": "NDVI < .18, NDWI < -.20, NDBI > 0", "water": "NDWI > .15, NDVI < .10"}, "rasters": [str(path) for path in rasters], "samples": records}
    (output_dir / "bootstrap_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "completed", "output_dir": str(output_dir.resolve()), "class_counts": {CLASSES[key]: len(value) for key, value in selected.items()}, "manifest": str((output_dir / "bootstrap_manifest.json").resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--raster", required=True, action="append", type=Path); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--samples-per-class", type=int, default=20); parser.add_argument("--coarse-size", type=int, default=300)
    args = parser.parse_args(); print(json.dumps(write_dataset(args.raster, args.output_dir, args.samples_per_class, args.coarse_size), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
