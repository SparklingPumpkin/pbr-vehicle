# pbr-vehicle-sun 版本说明

## 当前版本：1.0.0

- 首次独立交付场景太阳方位角/高度角识别包，对应 `SSE-v6`。
- 提供核心单/多帧、原生/前馈 Gaussian 双分支 CLI。
- 默认无 Gaussian 链使用纯 RGB InfiniDepth、逐连通域最大相机锥形近侧弧、源图边缘证据剔除和完整候选投影；观测阴影和候选投影均不再执行车辆 floor/footprint 二次差集。
- 提供 Argoverse 全场景跨帧 Top-3 车辆选择、MTMT/SAM2/InfiniDepth 外部模型编排与 fail-closed 阴影证据门。
- 提供 RGB 轮廓回投影及六联高清审计图。
