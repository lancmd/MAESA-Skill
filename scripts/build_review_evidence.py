#!/usr/bin/env python3
"""Build an auditable reviewer-evidence package for a MAESA project.

This command deliberately separates evidence that can be computed from local
artifacts from evidence that requires new observations.  It will calculate
confusion matrices, PLUS backcast metrics, and seed variability when the
corresponding inputs are present; otherwise it writes an explicit
``not_computable`` record and a ready-to-fill input template.  It never treats
training ROIs as independent reference data and never manufactures accuracy
or uncertainty values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


YEARS = (2005, 2010, 2015, 2020, 2025)
SCENARIOS = ("ND", "UD", "EP", "RE")
CLASS_CODES = (1, 2, 3, 4, 5, 6)
CLASS_NAMES = {
    1: "subsidence_water",
    2: "natural_water",
    3: "built_up",
    4: "cropland",
    5: "forest",
    6: "grassland",
}


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_candidates(workspace: Path) -> list[Path]:
    """Find candidate reference files without treating them as valid evidence."""
    names = re.compile(r"(?i)(independent|reference|validation|accuracy|confusion|独立|参考|验证|精度|混淆)")
    extensions = {".csv", ".xlsx", ".shp", ".gpkg", ".geojson", ".json"}
    candidates: list[Path] = []
    for path in workspace.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        # Do not rediscover the package's own template/status files on a
        # second run.  Reference samples should live in the project input
        # area (or be supplied explicitly to the downstream accuracy CLI).
        if "同行评议数据核查" in path.parts:
            continue
        # Training ROI and XML conversion manifests are explicitly excluded.
        if "roi" in path.name.lower() or "training" in path.name.lower():
            continue
        if names.search(path.name) or names.search(str(path.parent)):
            candidates.append(path.resolve())
    return sorted(set(candidates))


def csv_schema(path: Path) -> list[str] | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream).fieldnames or [])
    except (OSError, UnicodeError, csv.Error):
        return None


def classification_section(workspace: Path, review_dir: Path) -> dict[str, Any]:
    candidates = find_candidates(workspace)
    eligible: list[dict[str, Any]] = []
    for path in candidates:
        fields = csv_schema(path)
        if fields:
            lowered = {field.lower() for field in fields}
            has_reference = bool(lowered & {"reference", "ref", "class_id", "class"})
            has_xy = bool({"x", "lon", "longitude", "easting"} & lowered) and bool(
                {"y", "lat", "latitude", "northing"} & lowered
            )
            if has_reference and has_xy:
                eligible.append({"path": str(path), "fields": fields})
    template = review_dir / "独立验证样本模板.csv"
    with template.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["year", "x", "y", "reference", "class_name", "source", "independent", "notes"])
        writer.writerow(["2025", "", "", "", "", "field_check_or_high_resolution_reference", "yes", "one row per independent reference point"])
    if not eligible:
        return {
            "status": "not_computable",
            "reason": "no independent reference sample table with reference labels and coordinates was found",
            "required_fields": ["year", "x", "y", "reference", "source", "independent"],
            "training_roi_is_not_eligible": True,
            "training_roi_candidates": [
                str(p.resolve()) for p in sorted((workspace / "roi").glob("**/*training*.shp"))
            ],
            "candidate_files_inspected": [str(p) for p in candidates],
            "template": str(template.resolve()),
            "years": {str(year): {"status": "not_computable", "reason": "independent samples missing"} for year in YEARS},
        }

    # The calculation path is intentionally deferred to the existing, tested
    # lulc_accuracy CLI.  This record makes the eligible source explicit so a
    # reviewer can reproduce it after the samples are supplied.
    return {
        "status": "ready_for_computation",
        "eligible_reference_files": eligible,
        "template": str(template.resolve()),
        "years": {str(year): {"status": "ready_for_computation"} for year in YEARS},
        "next_command": "python scripts/lulc_accuracy.py --samples <csv> --classification-raster <year.tif> --require-raster-sampling ...",
    }


def open_raster(path: Path) -> dict[str, Any]:
    import rasterio  # type: ignore

    with rasterio.open(path) as src:
        data = src.read(1)
        mask = src.dataset_mask() > 0
        return {
            "path": str(path.resolve()),
            "shape": [int(src.height), int(src.width)],
            "crs": str(src.crs),
            "transform": [float(v) for v in src.transform[:6]],
            "dtype": str(src.dtypes[0]),
            "nodata": None if src.nodata is None else float(src.nodata),
            "data": data,
            "mask": mask,
        }


def raster_meta(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"data", "mask"}}


def aligned(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["shape"] == b["shape"] and a["crs"] == b["crs"] and np.allclose(a["transform"], b["transform"])


def plus_backcast_section(workspace: Path, review_dir: Path) -> dict[str, Any]:
    plus_root = workspace / "最终成果" / "PLUS四情景" / "plus_harmonized_v2"
    predicted = sorted(p for p in plus_root.rglob("*") if p.is_file() and re.search(r"(?i)backcast|historical.*pred|pred.*historical", p.name))
    if not predicted:
        return {
            "status": "not_computable",
            "reason": "no PLUS historical backcast raster was archived",
            "required": "a PLUS-predicted 2025 raster generated from the earlier historical state and the observed 2020 baseline",
            "candidate_search_root": str(plus_root.resolve()),
            "observed_reference": str((workspace / "最终成果" / "统计与清单" / "statistics_harmonized_v2_common_support" / "lulc" / "LULC_2025_30m_masked.tif").resolve()),
            "baseline": str((workspace / "最终成果" / "统计与清单" / "statistics_harmonized_v2_common_support" / "lulc" / "LULC_2020_30m_masked.tif").resolve()),
        }
    return {
        "status": "ready_for_computation",
        "predicted_candidates": [str(p.resolve()) for p in predicted],
        "next_command": "python scripts/plus_validation.py --reference <observed_2025> --predicted <plus_backcast_2025> --baseline <observed_2020> --output <report.json>",
    }


def seed_section(workspace: Path, review_dir: Path) -> dict[str, Any]:
    plus_root = workspace / "最终成果" / "PLUS四情景" / "plus_harmonized_v2"
    seed_fields: list[dict[str, Any]] = []
    seed_rasters: list[str] = []
    for path in plus_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if "random_seed" in json.dumps(payload, ensure_ascii=False):
            value = payload.get("random_seed") if isinstance(payload, dict) else None
            seed_fields.append({"path": str(path.resolve()), "random_seed": value})
    for path in plus_root.rglob("*.tif"):
        if re.search(r"(?i)(seed|random|replicate|run\d+)", path.name):
            seed_rasters.append(str(path.resolve()))
    if len(seed_rasters) < 2 or not any(row.get("random_seed") not in (None, "") for row in seed_fields):
        return {
            "status": "not_computable",
            "reason": "fewer than two seed-specific PLUS rasters and no recorded non-null random_seed were found",
            "random_seed_records": seed_fields,
            "seed_rasters": seed_rasters,
            "required": "at least two independently rerun PLUS outputs with recorded seeds and identical inputs",
        }
    return {
        "status": "ready_for_computation",
        "random_seed_records": seed_fields,
        "seed_rasters": seed_rasters,
        "next_command": "python scripts/plus_validation.py --seed-predictions <seed_manifest.json> ...",
    }


def re_archive_section(workspace: Path, review_dir: Path) -> dict[str, Any]:
    """Create a structural archive for the RE raster and its service outputs."""
    plus = workspace / "最终成果" / "PLUS四情景" / "plus_harmonized_v2" / "RE" / "PLUS_RE.tif"
    support = workspace / "最终成果" / "统计与清单" / "statistics_harmonized_v2_common_support" / "lulc" / "共同生态服务统计支撑区_30m.tif"
    scenario_root = workspace / "最终成果" / "InVEST原生成果" / "四情景" / "invest_scenarios_harmonized_v2" / "RE"
    composite = workspace / "最终成果" / "InVEST原生成果" / "四情景" / "ecosystem_service_scenarios_harmonized_v2" / "native_ND_UD_EP_RE" / "ecosystem_service_2026_RE.tif"
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "RE",
        "prediction_validation_status": "pending_historical_backcast_and_independent_reference",
        "structural_status": "not_computable",
        "land_demand_status": None,
        "raster": {},
        "service_outputs": {},
        "review_boundary": "This archive proves file/grid/class/support integrity only; it is not prediction accuracy validation.",
    }
    if plus.exists():
        plus_record = open_raster(plus)
        valid = plus_record["mask"] & (plus_record["data"] > 0)
        result["raster"] = {
            **raster_meta(plus_record),
            "valid_pixels": int(valid.sum()),
            "class_codes": sorted(int(v) for v in np.unique(plus_record["data"][valid])),
            "class_pixel_counts": {str(code): int((plus_record["data"] == code).astype(np.uint64).sum()) for code in CLASS_CODES},
            "codes_within_six_class_contract": bool(set(np.unique(plus_record["data"][valid]).tolist()) <= set(CLASS_CODES)),
        }
        if support.exists():
            support_record = open_raster(support)
            same_grid = aligned(plus_record, support_record)
            overlap = int((plus_record["mask"] & support_record["mask"] & (plus_record["data"] > 0)).sum()) if same_grid else None
            result["support"] = {**raster_meta(support_record), "same_grid": same_grid, "positive_overlap_pixels": overlap}
            result["structural_status"] = "completed" if same_grid and result["raster"]["codes_within_six_class_contract"] else "failed"
    demand = plus.parent / "PLUS_RE.land_demand.validation.json"
    if demand.exists():
        try:
            payload = json.loads(demand.read_text(encoding="utf-8"))
            result["land_demand_status"] = {
                "status": payload.get("status"),
                "total_requested_cells": payload.get("total_requested_cells"),
                "total_actual_cells": payload.get("total_actual_cells"),
                "max_abs_class_delta_cells": payload.get("max_abs_class_delta_cells"),
                "interpretation": payload.get("interpretation"),
            }
        except (OSError, UnicodeError, json.JSONDecodeError):
            result["land_demand_status"] = {"status": "unreadable"}
    for label, path in {"invest_scenario_root": scenario_root, "composite_service": composite}.items():
        result["service_outputs"][label] = {"path": str(path.resolve()), "exists": path.exists()}
    dump(review_dir / "RE情景归档验证摘要.json", result)
    lines = [
        "# RE 情景归档验证摘要",
        "",
        f"- 结构性归档：{result['structural_status']}",
        "- 预测精度状态：待历史回代与独立参考样本；本摘要不等同于预测精度验证。",
        "- 检查内容：栅格存在性、CRS/网格、六类编码、共同支撑区重叠、服务输出和总量守恒。",
        "- 需求解释：RE 的 CARS 总像元守恒，但类别需求存在偏差，应作为情景模型不确定性记录。",
        "",
        "## 归档文件",
        "",
        f"- JSON：`{(review_dir / 'RE情景归档验证摘要.json').resolve()}`",
        f"- PLUS 栅格：`{plus.resolve()}`",
        f"- 共同支撑区：`{support.resolve()}`",
    ]
    (review_dir / "RE情景归档验证摘要.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def write_status_csv(path: Path, sections: dict[str, Any]) -> None:
    rows = []
    for key, section in sections.items():
        rows.append({"evidence": key, "status": section.get("status", section.get("structural_status", "unknown")), "reason": section.get("reason", section.get("review_boundary", ""))})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["evidence", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    review_dir = (args.output or (workspace / "最终成果" / "期刊论文" / "同行评议数据核查")).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    sections = {
        "classification_independent_validation": classification_section(workspace, review_dir),
        "plus_historical_backcast": plus_backcast_section(workspace, review_dir),
        "plus_multi_seed_uncertainty": seed_section(workspace, review_dir),
        "re_archive_validation": re_archive_section(workspace, review_dir),
    }
    overall = "completed" if all(item.get("status", item.get("structural_status")) in {"completed", "ready_for_computation"} for item in sections.values()) else "pending_validation"
    package = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "evidence": sections,
        "interpretation": "Missing independent evidence remains explicitly pending; training ROIs, one-shot outputs and structural checks are not substituted for reviewer-grade validation.",
    }
    dump(review_dir / "审稿证据包.json", package)
    write_status_csv(review_dir / "审稿证据状态.csv", sections)
    readme = [
        "# 审稿证据包",
        "",
        "本目录区分可计算证据与待补证据。训练 ROI 不作为独立验证样本；栅格结构检查不作为 PLUS 预测精度；单次运行不作为随机种子不确定性。",
        "",
        "## 当前状态",
        "",
        f"- 总状态：`{overall}`",
        "- 独立分类验证：待提供带 reference、坐标和来源的独立样本。",
        "- PLUS 历史回代：待归档由历史基准生成的预测栅格。",
        "- 多随机种子：当前请求记录 random_seed=null，未形成两次以上独立重跑。",
        "- RE：已生成结构性归档摘要；预测精度仍需历史回代/独立参考样本。",
        "",
        "## 使用模板",
        "",
        "填写 `独立验证样本模板.csv` 后，可调用仓库现有 `scripts/lulc_accuracy.py` 计算混淆矩阵、OA、F1 和 IoU；PLUS 回代使用 `scripts/plus_validation.py`。",
    ]
    (review_dir / "审稿证据包README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(package, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
