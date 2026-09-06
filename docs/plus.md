# PLUS 情景模拟

PLUS 输入为至少两期严格对齐的 LULC、需求/转换规则和类型明确的驱动因子。先做历史回代，再报告 FoM、关键地类精度和多随机种子稳定性；仅报告 Kappa 不足以验收。

| 情景 | 重点 |
|---|---|
| ND | 延续历史变化趋势 |
| UD | 提高建设用地扩张约束/需求权重 |
| EP | 强化生态空间保护与林草恢复 |
| RE | 以沉陷深度为核心驱动，叠加常规因子 |

RE 的沉陷驱动必须是与主网格对齐、单位为 m、`positive_down` 的深度栅格。MAESA 不计算概率积分法预测；用户提供外部 `w.dat` 或深度 GeoTIFF。

官方 PLUS GUI 通过本地 pywinauto 优先、截图模板后备的桥接运行。每个情景拥有独立请求、状态和输出目录：`outputs/plus/<scenario>/PLUS_<scenario>.tif`。首次使用需校准 GUI；状态会依次记录 `prepared`、`calibration_required`、`running_gui`、`waiting_export`、`output_detected`、`validated` 或 `completed`，可在中断后接管已有输出。CARS 写出原生 `PLUS_<scenario>*Simulation_*.tif` 后，后续桥接调用仅执行本地采用与栅格验收，不再重复触发 UAC；因此即使提权 worker 没有返回，也可安全完成输出闭环。

## 回代与不确定性证据

面向审稿的证据不能由单次情景输出替代。运行 `scripts/build_review_evidence.py --workspace <project>` 会检查：（1）由历史基准生成并与观测期对照的 PLUS 回代栅格；（2）至少两个记录了随机种子的独立重跑；（3）RE 情景的结构、网格、类别、共同支撑区和需求守恒摘要。具备回代栅格后，使用 `scripts/plus_validation.py` 计算 FoM、总体及逐类指标，并从种子清单计算 FoM 的总体标准差。仅有 `matches_historical_lulc_grid=true` 或 `random_seed=null` 时，状态仍为 `pending_validation`，不得写成预测精度或不确定性已经完成。
