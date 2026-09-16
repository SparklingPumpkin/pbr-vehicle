# PBR Vehicle Sun

`pbr-vehicle-sun 1.4.0` 是场景太阳方位角/高度角识别的独立交付包，对应加速主线 `SSE-v9`。SSISv2 官方关联阴影、P95 相机可见轮廓和置信度门控继承 `SSE-v8`；本版将车辆准备和角度搜索改为并行，并以一次 InfiniDepth 前向同时取得深度与源视图可见 Gaussian。

默认完整场景链先使用全局排名 1–3 的三辆物理车辆。首轮全部拒绝时，复用已经完成的 YOLO 检测和 SAM2 排名，自动更换为排名 4–6 的车辆再估计一次；第二轮仍全部拒绝才返回 `no_valid_sun_information`。进程异常、文件错误和资源错误不会触发换车，也不会被记成方法拒绝。

车辆选择仍穷举全帧、全相机的全部合格 YOLO11 观测；SAM2.1 通过批量图像编码和同图多 box decoder 加速，不执行候选预筛，因此物理车辆去重、每车最大 mask 面积以及全局 Top-6 的选择语义不变。SSISv2 先执行，无有效 object-shadow pair 的车辆不会再运行 InfiniDepth。完整场景入口支持 `--cache-dir` 持久复用完全相同输入和参数的已审计结果。

默认同时使用最多三张调用方指定 GPU（`--worker-devices`）准备三辆车；三车拟合并行，每车候选角再以 8 个 CPU 线程计算。相同输入的完整缓存命中会直接恢复审计产物。模型和权重均不随 wheel 分发。

主要入口：

- `pbr-vehicle-sun-scene`：默认两轮完整场景链；首轮排名 1–3，无角时自动更换为排名 4–6。
- `pbr-vehicle-sun-scene-round`：高级单轮入口，支持显式选择 1–5 辆车和 rank offset。
- `pbr-vehicle-sun-fit`：对已对齐的车辆几何、阴影几何和 SSISv2 mask 执行单帧、多帧或多车拟合。
- `pbr-vehicle-sun-associate-shadow`：将 SSISv2 object-shadow association 与指定车辆 mask 绑定。
- `pbr-vehicle-sun-aggregate`：执行 1–5 车独立拟合、硬门、加权角和联合拟合。

模型仓库、解释器、权重和数据均为外部参数；wheel 不包含任何资产或机器路径。完整命令见 [使用指南.md](使用指南.md)。
