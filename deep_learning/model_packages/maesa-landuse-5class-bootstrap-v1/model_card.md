# MAESA 五类土地利用图像分类模型（初始版）

该包输出：耕地、林地、草地、建筑用地、水体。水体必须在 GIS 中由人工 ROI 复核后，才可进一步细分为自然水体（5）与沉陷水体（6）。

## 权重与运行

- 架构：ResNet-50，128 × 128 RGB 图块输入。
- 格式：安全的 PyTorch ExportedProgram（`model.pt2`），不用 Python pickle 加载。
- SHA-256：`797985aa00a48f6ce65317327ada5fbbbfa217eb466defc5605d3e124e742b3f`。
- 校验：`python deep_learning/image_classification/landuse_image_classifier.py validate-package --model-package deep_learning/model_packages/maesa-landuse-5class-bootstrap-v1`。

## 训练标签与限制

训练用的 40 个图块来自本地 2020/2025 七波段光学 GeoTIFF；它们由高置信度 NDVI、NDWI、NDBI 规则自动筛选，每类 8 个。这不是人工标注或外业真值。

自动初始标签留出集的最佳 Accuracy 为 0.70、Macro-F1 为 0.713；这些值仅证明模型训练、导出和推理流程已跑通，不能作为研究结论、人工真值精度或跨区域泛化声明。

在正式使用前，应补充人工 ROI/外业样本，采用相同五类体系重新训练并在独立人工验证集上评价。
