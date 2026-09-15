# pbr-vehicle-auto 版本说明

## 当前版本

- 当前包版本：`1.4.1`，以本目录 `pyproject.toml` 中声明的版本为准。
- 本工具消费 PBR 车辆资产和场景观测，输出自动光照/材质配置或 DC 烘焙结果。

## 功能演进

### 1.4.1

- 修正 Auto 参数墙语义：太阳光强度和亮度都使用 Viser 滑条绝对区间 `[-0.3, 0.3]`，不再围绕点击时的面板值滚动。
- 粗搜索和最终联合精调共享同一绝对边界；相同输入反复点击会评估同一候选域。
- `metrics.json` schema 升至 3，记录 `bounds_policy=absolute_viser_slider`；点击前数值改记为 `ui_input/value_input`，明确仅供诊断。

### 1.4.0

- 新增 `--device auto`，有 CUDA 时自动使用当前进程可见的 `cuda:0`，无 CUDA或未安装 Torch 时回退原 NumPy 实现。
- 将代理 PBR/GGX、KNN/identity mapping、候选颜色、亮度直方图和 CDF-L1 指标迁到 Torch CUDA；所有静态数组在整轮网格搜索期间常驻 GPU。
- 只把最终候选颜色传回 CPU 写 PLY，保持 ALM-v5 参数范围、always-grid-best 输出和数值合同不变。
- 运行依赖约束 NumPy 为 `>=1.24,<2`，避免 Torch 2.1/2.2 与 NumPy 2 的 ABI 不兼容组合。

### 1.3.0

- 移除网格搜索后的原始 DC 改善接受门；只要搜索成功，就始终发布网格内指标最优候选。
- `metric.raw` 继续作为诊断基线，`metric.baked` 始终记录已应用候选，`improvement_percent` 允许为零或负数。
- 输出新增 `selection_policy=always_apply_grid_best`，同时维持 `candidate_only` 状态以兼容现有 Panel/Viser 适配器。

### 1.2.1

- 自动拟合以点击时模板值为中心，在 Viewer 的 `[-1, 1]` 滑条坐标中设置参数阈值墙：太阳强度限制为 `±0.15`，亮度限制为 `±0.30`，再按面板映射转换为内部值。
- 初始搜索与最终联合精调共用同一闭区间，端点可被选择，但任何候选和最终输出都不得越界。
- `metrics.json` 记录基线、增量、上下界和允许边界候选的合同，便于独立 Panel 审计。
- 上述相对模板范围是该历史版本的行为，已由 1.4.1 的绝对范围替代。

### 1.2.0

- 推广 ALM-v4 自动光照材质合同。
- 太阳 RGB 与环境光独立保存；环境路径采用色温/低阶 SH 约束。
- 拟合范围限制为太阳强度和补光强度；该版本当时仍提供 reject-safe 的无改善拒绝结果，1.3.0 已移除此门。
- 支持 rear-view DC bake、opacity 加权亮度 CDF-L1 评估和真实输入验证。
- 默认投影模板的 extension 使用车辆 footprint 到平行光地面投影 footprint 的连续扫掠轮廓，低太阳高度角下不会因只取投影终点而脱离车体。

### 1.0.x

- 建立独立 Python API、CLI、默认配置、测试和 wheel 交付包。
- 支持从普通场景与 PBR 车辆输入生成候选光照/材质配置。
- 明确自动拟合结果不代表物理光照恢复、材质真值或多视角真实性。

## 兼容边界

- 自动拟合输出是配置覆盖或评估产物，不应直接替换 canonical 车辆资产。
- 资产合同变化时，必须同步检查输入字段、投影参数和配置 schema。
- 版本号与 `pbr-vehicle-panel`、`pbr-vehicle-convert` 分开维护。
