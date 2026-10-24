# 视觉与传感器改造验证记录（2026-10-23）

## 实际交付

运行项目为 interactive_navigation_mujoco；参考工程保持原样。普通解释器为 G:/anaconda3/envs/mujoco/python.exe，MuJoCo 3.3.2、NumPy 2.0.2、Pillow 11.3.0，ONNX Runtime 1.19.2 位于项目 .vendor。日常启动不依赖 PyTorch。模型 SHA-256：e8cf07baace7cec4d394a5f05f75831de55eb5f47bb4f5df6ee3c411550fe8a7。

模型为实际训练的 YOLOv8n，35 轮，从官方预训练权重微调。864 张真实 MuJoCo 渲染图，576 / 144 / 144，随机场景实例在训练、验证、测试之间隔离。图像仍来自合成建筑；这些结果不表示真实玻璃识别能力已验证。

## 交付 ONNX 的留出检测结果

tools/evaluate_onnx.py 直接运行主工程中的预处理、ONNX 推理、坐标还原、NMS；144 张 test 图，置信阈值 0.45，匹配 IoU 0.5：

| 类别 | TP / FP / FN | 精确率 | 召回率 |
|---|---|---:|---:|
| 门 | 73 / 1 / 5 | 98.65% | 93.59% |
| 疑似玻璃 | 60 / 1 / 3 | 98.36% | 95.24% |

本机 CPU 推理均值 19.21 ms、P95 20.99 ms。并行物理任务、窗口与系统负载会增加延迟，正式运行分别保存自身统计。GPU 测试 mAP50=0.97794、mAP50–95=0.93729，详见 models/vision/metadata.json；这里的 mAP 和上述固定阈值精确率是不同指标。

## 物理与信息边界回归

verify_sensor_upgrade.py 的 10 项测试全部通过，包括 12 条独立记录：默认木门、灰白门、蓝色门、玻璃视觉漏检后的几何候选、实际玻璃 IMU 外力验证与退离、空开口完整验证、超时未完成、无效 / 重复 / 未来量测、外力观察器、无权重拒绝和信息边界、外力失效不判自由、接触记忆遇到语义冲突保留。

verify.py 通过候选数学、16 组几何、3 组惯量映射和 6 组真实 RGB / 内容交换检查。空开口的静态观察现在只形成待验证候选；需要完成行程才验证自由。

verify_runtime.py 通过取消 / 续算、损坏缓存拒绝、配置 / 观测状态签名、worker 数不改物理签名、通行记忆遇到实际新门要求复核和过期前缀拒绝。其 32 候选仅是缓存诊断；以下主任务使用正式 N=1024。

## 完整任务

| 任务 | 仿真时间 | 门动作 | 歧义开口验证 | 墙面接触峰值 |
|---|---:|---|---|---:|
| 默认 adaptive / N=1024 | 114.15 s | O01 Probe→Push；O05 Avoid | O03 受阻；O04/O06/O07 自由 | 0 N |
| 难推门 adaptive / N=1024 | 140.38 s | O01 Probe→Avoid；O05 Avoid | O02/O04/O06/O07 自由；O03 受阻 | 0 N |
| 移除玻璃 / always_avoid | 111.70 s | 两门绕行基线 | O02/O03/O06/O07 自由 | 0 N |
| 门玻璃交换 / always_avoid | 133.05 s | O03、O05 门绕行基线 | O01 受阻；O02/O04/O06/O07 自由 | 0 N |
| 全空开口 / adaptive | 89.57 s | 没有门动作 | O01/O03/O05/O07 自由 | 0 N |

全部到达目标。基线和全空任务没有为不存在的门决策计算标签；它们配置的候选域仍为 1024，不能把“配置 N”写成实际完成了 1024 次预测。

默认和难推门初始 G=5.31630 m、p_D=0.467773、D=0.995846 相同。Probe 后分别 p_D≈0.999245 / D≈0.003019 和 p_D≈0.00001645 / D≈0.00006582，选择 Push 与 Avoid。O05 低 G 短路时 p_D/D 保持 null，其权重仍为独立均匀先验。三个当前正式候选库均为实际 1024 物理世界、已完成且签名有效；旧颜色版本缓存未直接套用。

默认玻璃验证估计法向力峰值 2.7653 N，阈值 0.45 N。验证期目标速度 0.10 m/s，实际最大速度约 0.1224 m/s；目标速度与物理峰值不同，未声称硬限速或限力。三个空开口的估计峰值约 0.042–0.043 N，完成行程后判自由。门 Probe 保留原虚拟接触仪器；玻璃使用 IMU 残差，二者记录来源不同。

对应目录：

- results/20261023_163302_standard_adaptive
- results/20261023_163835_standard_adaptive
- results/20261023_163638_glass_removed_always_avoid
- results/20261023_163638_swapped_always_avoid
- results/gui_yolo_consistency_all_empty/headless 与 live

## 实际界面

verify_live.py 使用真实 Tk / OpenGL 窗口完成全空任务，暂停 / 恢复、跟随 / 总览、三视图保存、完成窗口保留均通过。GUI 与 headless 的遥测、决策及仿真时间完全一致。证据位于 results/gui_yolo_consistency_all_empty/checks.json 与 live_window_content.png。

默认交互场景也由 verify_live.py --standard 完成验证，遥测、门动作、仿真时间与 headless 完全一致；暂停、视角、保存和窗口保留均通过，另保存了实际推动中的近景。输出位于 results/gui_yolo_consistency_standard/checks.json 与 live_interaction_close_dashboard.png。

release_check.py 自动审查当前模型、正式任务、候选库、回归结果、实际 GUI 和原模型副本，生成 results/yolo_upgrade_release.json。旧 VALIDATION.md / 旧 results 文件夹保留历史语义，不能代替本次验收。

## 保留的边界

采用固定高度局部几何、公共粗地图开口位置、理想定位 / AHRS 与标准铰链先验；没有实现 VIO、完整三维 SLAM、真实玻璃光学、MINCO 或真实机外力标定。接触验证只确认当前动作下通行性，不证明材质或永久不可移动性。
