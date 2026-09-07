#!/usr/bin/env python3
"""Convert five-class water predictions into six classes with user-drawn water ROIs.

Machine labels: 1 cropland, 2 forest, 3 grassland, 4 built-up, 5 water.
Manual labels:  1 cropland, 2 forest, 3 grassland, 4 built-up, 5 natural water,
6 subsidence water.  Only pixels predicted as water are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def refine(base_raster: Path, roi_path: Path, output: Path, class_field: str, default_roi_class: int) -> dict:
    try:
        import fiona
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
    except ImportError as error:
        raise RuntimeError("ROI refinement requires rasterio, fiona, and numpy") from error
    if default_roi_class not in {5, 6}:
        raise ValueError("default ROI class must be 5 (natural) or 6 (subsidence)")
    with rasterio.open(base_raster) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError("base raster must have exactly one band and a CRS")
        with fiona.open(roi_path) as roi_source:
            if not roi_source.crs_wkt and not roi_source.crs:
                raise ValueError("ROI layer has no CRS")
            roi_crs = roi_source.crs_wkt or roi_source.crs
            if rasterio.crs.CRS.from_user_input(roi_crs) != source.crs:
                raise ValueError("ROI CRS must exactly match base raster CRS; reproject the ROI first")
            shapes = []
            for feature in roi_source:
                if not feature.get("geometry"):
                    continue
                value = feature.get("properties", {}).get(class_field, default_roi_class)
                try:
                    value = int(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"ROI {class_field} must be 5 or 6") from error
                if value not in {5, 6}:
                    raise ValueError(f"ROI {class_field} must be 5 or 6")
                shapes.append((feature["geometry"], value))
        roi_labels = rasterize(shapes, out_shape=(source.height, source.width), transform=source.transform, fill=0, dtype="uint8")
        base = source.read(1)
        manual = base.astype("uint8", copy=True)
        water = base == 5
        manual[water] = 5  # unmarked machine water is reported as natural water
        marked = water & (roi_labels > 0)
        manual[marked] = roi_labels[marked]
        profile = source.profile.copy(); profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as sink:
            sink.write(manual, 1)
    report = {"status": "completed", "base_raster": str(base_raster.resolve()), "roi": str(roi_path.resolve()), "output": str(output.resolve()),
              "machine_classes": {"1": "cropland", "2": "forest", "3": "grassland", "4": "built_up", "5": "water"},
              "manual_classes": {"1": "cropland", "2": "forest", "3": "grassland", "4": "built_up", "5": "natural_water", "6": "subsidence_water"},
              "roi_pixel_count": int(marked.sum()), "subsidence_water_pixel_count": int((manual == 6).sum())}
    output.with_suffix(output.suffix + ".roi_refinement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raster", required=True, type=Path); parser.add_argument("--roi", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roi-class-field", default="class_id"); parser.add_argument("--default-roi-class", type=int, default=6)
    args = parser.parse_args(); print(json.dumps(refine(args.base_raster, args.roi, args.output, args.roi_class_field, args.default_roi_class), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
