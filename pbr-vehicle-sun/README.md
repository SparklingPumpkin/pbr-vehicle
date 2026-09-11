# PBR Vehicle Sun

`pbr-vehicle-sun 1.2.0` 是场景太阳方位角/高度角识别的独立交付包，对应研究主线 `SSE-v8`。`SSE-v8` 的阴影输入协议严格继承 `SSE-v6-b`，并继承 `SSE-v7` 的通用多车门控：YOLO/SAM2 选定车辆，SSISv2 绑定 object-shadow pair，官方关联 shadow mask 不经过差集、连通域清洗或形态学修改，直接由 InfiniDepth 提升并参与轮廓拟合。

默认完整场景链先使用全局排名 1–3 的三辆物理车辆。首轮全部拒绝时，复用已经完成的 YOLO 检测和 SAM2 排名，自动更换为排名 4–6 的车辆再估计一次；第二轮仍全部拒绝才返回 `no_valid_sun_information`。进程异常、文件错误和资源错误不会触发换车，也不会被记成方法拒绝。

主要入口：

- `pbr-vehicle-sun-scene`：默认两轮完整场景链；首轮排名 1–3，无角时自动更换为排名 4–6。
- `pbr-vehicle-sun-scene-round`：高级单轮入口，支持显式选择 1–5 辆车和 rank offset。
- `pbr-vehicle-sun-fit`：对已对齐的车辆几何、阴影几何和 SSISv2 mask 执行单帧、多帧或多车拟合。
- `pbr-vehicle-sun-associate-shadow`：将 SSISv2 object-shadow association 与指定车辆 mask 绑定。
- `pbr-vehicle-sun-aggregate`：执行 1–5 车独立拟合、硬门、加权角和联合拟合。

模型仓库、解释器、权重和数据均为外部参数；wheel 不包含任何资产或机器路径。完整命令见 [使用指南.md](使用指南.md)。
