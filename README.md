# INTERACT · 基于粗地图与主动交互的无人机导航

在 PyCharm 打开本文件所在的 **interactive_navigation_mujoco** 文件夹，选择现有解释器 `G:\anaconda3\envs\mujoco\python.exe`，右键运行 `main.py`。不需要命令行参数、训练权重、网页服务或重新安装依赖。原参考工程没有修改。

## 实时运行

一个统一控制室显示：当前 MuJoCo 三维物理现场、只含在线观测的认知地图、固定机载相机实际 RGB，以及 G / p_D / D、1024 候选信念和执行状态。默认启动即运行完整 S→T 任务，完成后保持窗口，便于旋转检查和保存。

| 操作 | 效果 |
|---|---|
| 空格 / 暂停按钮 | 暂停或继续物理推进 |
| 速度下拉框 | 0.25–8 倍目标播放速度；渲染性能可能限制实际速度 |
| 在三维区域拖动、滚轮 | 旋转和缩放观察视角 |
| R / 恢复总览 | 返回剖顶建筑总览 |
| F / 跟随按钮 | 切换无人机近景 |
| S / 保存按钮 | 保存当前三维、认知地图、相机及组合面板 |
| 关闭主窗口 | 结束当前运行并保留已记录的数据 |

物理积分为 500 Hz，旋翼控制 100 Hz，机载 RGB 与扫描约 10 Hz，界面按墙钟刷新。三个面板使用同一仿真时钟。候选计算阶段明确显示“预测中 / 仿真暂停”，界面仍可操作。每次门的高收益判定都运行完整 1024 候选；不会为了界面速度减少候选数。

## 地图与外观

唯一米制依据是 `config/architecture.json`，复制自最终版中立配置：11×13 m、四排隔墙、七个匿名开口，O01=(0.5,2.6)、O06=(3.7,8.2)。墙厚 0.12 m、墙高 2.4 m，目标飞行高度 1.25 m。

实时环境使用木纹门、金属门框和把手、透明可碰撞玻璃、混凝土地面、真实投射阴影和剖顶总览。细窄的青色墙顶光、起终点地面标记和飞行轨迹仅供观察者显示，不进入机载 RGB 或扫描。把手、装饰条、材质纹理采用视觉细节与简化碰撞分离；不会占用飞行通道。

原张拉整体无人机 XML 完整保存在 `assets/models/`。自由杆件、24 条拉索、质量惯量、传感器、执行器限制均保留。执行时只发送四个旋翼控制量，其余三个辅助执行器始终为零；不拖动机体，不给门添加辅助力矩。

**门的安装细节：** 门扇仍宽 1.1825 m、厚 0.05 m、高 2 m。底部留 0.02 m 实际门缝，避免接地摩擦把轻门变成卡门；铰链向门后偏置 0.035 m，惯量计算包含该平行轴项。门框外观每侧宽 0.09 m，圆角碰撞截面半径 0.042 m，较外观半径 0.045 m 留 3 mm 安装倒角余量。规划使用外观名义框宽与 0.34 m 方形机体包络，较实际碰撞轮廓保守。这些细节在验证中明示，并未改变开口中心、名义宽度或无人机尺寸。

## 研究方法与信息边界

- 初始地图仅有建筑与匿名开口；语义、通行性、交互记忆分别维护。
- 感知使用实际 RGB 的纹理/色度线索与同次测距，是可替换的仿真工程检测器，**不是 YOLO，也不声称真实玻璃识别泛化**。检测分数是像素比例，不是门能推动的概率。
- 固定相机跟随完整机体姿态；扫描有视场、遮挡、量程、丢测和弱/背景回波。无效回波不作为自由空间证据。重复帧不重复融合，已确认记忆不会随视野消失而重置。
- 门角来自扫描点的面板线拟合；策略不读取真实铰链角。接触反馈是所有机体外接触法向力之和，不按环境对象名字筛选目标门接触。真值仅在物理引擎、观察者视图和独立评估中使用。
- 明确采用理想定位/AHRS与自身柔性多体状态仪器。候选克隆接收的是自身多体观测与扫描估计的门角/转速；不复制环境隐藏参数、其他门的真值状态或完整仿真 data。
- 门的标准框宽与西侧铰链是识别门后的结构先验；扫描拟合面板方向估计门角。尚未实现任意铰链侧、非标门框的自动三维重建，不把这部分称为通用几何 SLAM。
- Sobol 完整 N=1024，I∈[0.05,0.80] kg·m²、τ_f∈[0.01,8.00] N·m，初始等权。每扇门独立，不把第一扇门后验复制到第二扇门。
- G 为同起终点的米制 A* 路径差，门支路强制经过目标开口；两分支对其他未知开口采用一致的乐观空通道假设。规划包含门框与整机包络，不允许对角穿角。
- G<3 m 直接 Avoid，短路时不计算不需要的 p_D/D，记录为 null。高 G 时根据候选实际路线时间计算 p_D、D=4p_D(1−p_D)，D≥0.36 进行一次标准 Probe。
- Probe 用同一 1.2 s、2.2 N 水平指令上限。实际接触峰值可大于指令上限；它们不是同一种量。前缀更新从原始先验计算，使用同一持续时刻预测，不反复乘相关帧。
- Probe 后从同一观测后状态，逐候选重新执行 Door / Bypass 剩余路线，重算标签。若后续仍分歧，保守绕行。Push 使用不同输入，后验保留而不继续套标准 Probe 库。
- 几何比较中，若 Probe 后位置处于“假设门洞封闭”的膨胀区，两条几何分支都加入同一条显式 A* 退离前缀。实际物理续程仍从原始观测后状态开始，不回滚真实仿真。
- 接触状态若轻微落入保守门框膨胀区，只允许两分支共享的、单调减少初始膨胀重叠且不进入新障碍的退离连接；之后所有普通 A* 边继续保持完整安全距离。该几何连接不拖动无人机，不改变候选续程初态。`verify_contact_geometry.py` 保存了实测 1.971 mm 起点重叠的回归测试。
- 玻璃走独立低速接触验证：受阻→制动退离→稳定恢复→入图→绕行。空开口需有效多帧净空证据后通过。

## 缓存与输出

首次高收益门需要建立 1024 候选的 Probe 时序、Door 标签与实际 Bypass 时间；Probe 后还需 1024 组双路线续程。首次构建需要数分钟至十几分钟，具体实测时间见 `VALIDATION.md`。项目交付保留已完成的默认缓存。

`results/cache/` 按配置、建筑、模型、控制、传感器定义、候选及明确观测状态计算 SHA-256 签名。完整缓存才可用于决策，每 32 个世界保存一次；中断后重跑继续未完成部分。不同地图、到达状态或 Probe 后状态不能直接套用旧成本。数值异常保留并阻止该库用于决策；物理/控制失败保留实际耗时、原因与无穷目标成本，JSON 中非有限值表示为 null。

每次运行生成一个结果文件夹：`summary.json`、`flight.csv`、`sensors.jsonl`、`camera/`、`online_map.json`、`events.json`、`decisions.json`、`door_beliefs.json`、`weight_history.json`、真实三视图关键帧和 `report.html`。`evaluation_truth.json` 与 `scene_truth_for_evaluation.json` 是独立评估数据，不参与策略。HTML 是已记录运行的离线证据查看器，实时演示使用 main.py。

`evaluate_results.py` 可审核全部运行、生成 `results/index.html` 索引并导出完整候选 CSV。`verify_runtime.py` 实测取消/续算、异常缓存拒绝、过期证据和矛盾核验。当前机器有 24 个逻辑处理器，正式入口设置 8 个计算进程；可以在 `config/settings.json` 改小。关闭窗口时会取消排队任务，等待当前少量候选结束并保存，不再等待整个 1024 队列。

## 其他验证入口

```powershell
# 完整默认任务，机载 RGB 仍真实渲染，仅不创建 Tk 窗口
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/interactive_navigation_mujoco/main.py' --headless
# 数学、惯量、16 组几何与实际机载感知内容变体检查
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/interactive_navigation_mujoco/verify.py'
# 真实 GUI 与 headless 一致性；默认全空变体，--standard 使用完整默认交互场景
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/interactive_navigation_mujoco/verify_live.py' --standard
# 明确标注的场景/基线
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/interactive_navigation_mujoco/main.py' --headless --hard-door
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/interactive_navigation_mujoco/main.py' --headless --variant glass_removed
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/interactive_navigation_mujoco/main.py' --headless --baseline always_avoid
```

`--baseline` 还支持 `always_push` 和 `no_probe`。始终推门基线表示每扇门先尝试 Push，执行超时后使用同样的安全恢复/绕行；不能把失败隐藏为成功推门。`--debug-power 5` 仅用于开发调试，其 32 点结果不能代替正式 1024 验证。

全部路径从项目根定位，允许任意当前工作目录。Windows DLL 配置只修改当前进程，不改系统 PATH。模块导入不会启动 GUI 或候选计算。
