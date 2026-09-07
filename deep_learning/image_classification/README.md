# 五类土地利用图像分类

这是一个整图分类模型，而不是像元级分割模型。自动模型固定输出五类：

| class_id | 类别 |
| --- | --- |
| 1 | 耕地 |
| 2 | 林地 |
| 3 | 草地 |
| 4 | 建筑用地 |
| 5 | 水体 |

水体的进一步拆分是人工 GIS 复核步骤：在五类结果中加入 ROI 后，水体变为自然水体（5）与沉陷水体（6），其余四类保持不变。模型不会把“沉陷水体”作为自动学习类别，避免把没有可靠成因标注的水体误判为沉陷积水。

## 训练

训练影像目录可以有子目录；标注是 UTF-8 文本，每行是 `image_id class_id`。可选第三列 `group` 用于避免同一地点的增广图像同时出现在训练与验证集。

```text
patches/00001-1.png 1
patches/00001-2.png 1
patches/00002-1.png 5
```

```powershell
python -m pip install -r requirements.txt
python landuse_image_classifier.py train `
  --image-root E:\train `
  --labels E:\train\train.txt `
  --output-dir E:\train\maesa_5class_run `
  --architecture resnet50 --epochs 30 --batch-size 16
```

输出 `model_package/` 包含安全的 `model.pt2`、带 SHA-256 的 `model_config.json` 和 `model_card.md`。训练过程保留验证集混淆矩阵、总体精度与 Macro-F1；没有标注、任一类别缺失或同组泄漏时会停止。

若只有未标注的七波段光学 GeoTIFF，可以先生成明确标记为“自动初始标注”的小型启动集。它只保留光谱纯净图块，必须在后续研究中用人工真值替换或复核：

```powershell
python bootstrap_pseudo_labels.py `
  --raster E:\AI_LULC\...\2025-0000000000-0000000000.tif `
  --raster E:\AI_LULC\...\2025-0000012544-0000000000.tif `
  --output-dir E:\AI_LULC\bootstrap_5class --samples-per-class 20
```

此步骤产生 RGB 图块、`train.txt`、每个图块的原始窗口/CRS/阈值记录。训练时可复用已验证的可信 ResNet 权重作为骨干初始化（分类头始终重新训练）：

```powershell
python landuse_image_classifier.py train `
  --image-root E:\AI_LULC\bootstrap_5class `
  --labels E:\AI_LULC\bootstrap_5class\train.txt `
  --output-dir E:\AI_LULC\maesa_5class_run `
  --architecture resnet50 --no-pretrained `
  --initial-checkpoint E:\AI_LULC\inference_package\my-inference-script-and-model\model\best_resnet50.pth
```

## 推理

```powershell
python landuse_image_classifier.py validate-package --model-package E:\train\maesa_5class_run\model_package
python landuse_image_classifier.py infer `
  --model-package E:\train\maesa_5class_run\model_package `
  --input E:\test `
  --output E:\test\result.txt
```

## 手工 ROI 水体细分

ROI 必须与五类分类栅格使用相同 CRS。若 ROI 只勾画沉陷水体，不必写属性；未标记的机器水体自动成为自然水体。

```powershell
python roi_water_refinement.py `
  --base-raster five_class.tif --roi subsidence_water.gpkg `
  --output six_class_manual.tif
```

如果一个 GeoPackage 中同时包含自然水体和沉陷水体，使用整型 `class_id` 字段（5 或 6）：

```powershell
python roi_water_refinement.py `
  --base-raster five_class.tif --roi water_review.gpkg `
  --roi-class-field class_id --output six_class_manual.tif
```

新用户需要不同的分类体系时，创建 `classes.json`：

```json
{"classes": [{"id": 1, "name": "cropland", "display_name": "耕地"}, {"id": 2, "name": "forest", "display_name": "林地"}]}
```

并在训练时传入 `--classes-json classes.json`。模型结构当前支持 `resnet50` 与 `efficientnet_v2_s`；新架构须在 `build_model()` 明确注册，然后重新训练和导出模型包。
