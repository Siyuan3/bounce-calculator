# Bounce Calculator

从 iPhone 慢动作视频测量碰撞恢复系数的 macOS 工具。当前重点支持近距离俯拍条件下，单个小球沿水平轨道撞击固定墙面的实验。

程序使用视频的真实 presentation timestamp，不假设 `timestamp = frame_id / fps`。恢复系数由碰撞前后墙面法向像素速度之比得到，因此只计算恢复系数时不需要像素到米的空间标定：

```text
e = |v_after,n / v_before,n|
```

## 主要功能

- 浏览整段视频并人工确认首次接触帧、完全离墙帧。
- 将固定机位的轨道几何与不同小球的外观模板分开保存。
- 通过空背景差分获得绿色轮廓测量，黄色模板只作为独立交叉检查。
- 碰撞前、碰撞后分别跟踪和拟合，避免跨越速度反向点。
- 使用 5、7、9 帧多窗口拟合，输出恢复系数及不确定性。
- 保存 JSON 标注和逐帧 CSV，保留真实 PTS、检测置信度和质量状态。
- 汇总同一小球 H1–H5 五个不同释放位置的统计结果。

仓库还保留足球落地的速度法与高度法脚本，见“其他测量方法”。

## 系统要求

- macOS（视频解码使用 AVFoundation 和 Swift）
- Python 3.10 或更高版本
- 系统自带 `/usr/bin/swift`
- Tkinter 图形界面

安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果 Python 没有 Tkinter，可使用 Homebrew 的 Python/Tk 发行方式补充安装。

## 拍摄要求

- 推荐 iPhone 240 fps 慢动作视频。
- 手机固定，光轴尽量垂直轨道平面。
- 画面覆盖挡墙前约 10–12 个小球直径。
- 小球直径建议达到 100–150 px。
- 使用纯色哑光背景，固定补光并锁定焦点、曝光和白平衡。
- 每段视频开头至少保留约 1 秒完全没有小球的空轨道画面。

## 单次小球撞墙测量

首先准备工作目录：

```bash
mkdir -p annotations
```

第一次测量某个小球的 H1：

```bash
python3 calculate_track_collision.py "/完整路径/IMG_0001.MOV" \
  --release-level H1 \
  --geometry-annotation annotations/track_geometry.json \
  --ball-setup annotations/ball_01_setup.json \
  --ball-id ball_01 \
  --save-annotation annotations/IMG_0001_ball_01_H1_collision.json
```

人工操作顺序：

1. 浏览整段视频，选择大致碰撞时间。
2. 逐帧确认“首次接触墙面”和“反弹后首次完全离墙”。
3. 首次创建几何文件时，框选碰撞区域并沿挡墙边缘点击两个点。
4. 首次创建当前小球设置时，在清晰画面中紧贴轮廓框选小球。
5. 检查绿色轮廓和黄色模板，可排除检测错误的帧。
6. 确认结果后保存 JSON 和同名的轨迹 CSV。

同一机位、同一小球继续测量 H2–H5 时，复用两份设置：

```bash
python3 calculate_track_collision.py "/完整路径/IMG_0002.MOV" \
  --release-level H2 \
  --geometry-annotation annotations/track_geometry.json \
  --ball-setup annotations/ball_01_setup.json \
  --ball-id ball_01 \
  --save-annotation annotations/IMG_0002_ball_01_H2_collision.json
```

更换小球但不移动手机、轨道或挡墙时，继续复用 `track_geometry.json`，改用新的 `--ball-setup` 和 `--ball-id`。程序会阻止显式 ID 与已有设置文件不一致，避免不同小球的数据串用。

需要重新检查已保存 trial 的检测帧：

```bash
python3 calculate_track_collision.py "/完整路径/IMG_0001.MOV" \
  --geometry-annotation annotations/track_geometry.json \
  --ball-setup annotations/ball_01_setup.json \
  --ball-id ball_01 \
  --annotation annotations/IMG_0001_ball_01_H1_collision.json \
  --review-detections
```

## 汇总 H1–H5

五个释放位置完成后运行：

```bash
python3 height_sweep.py \
  annotations/H1.json \
  annotations/H2.json \
  annotations/H3.json \
  annotations/H4.json \
  annotations/H5.json \
  --output annotations/height_sweep_summary.json
```

汇总文件包括平均值、样本方差、样本标准差、变异系数、极差和 H1→H5 趋势。未通过正碰或检测质量检查的 trial 不会进入标准统计。

## 输出说明

每个 trial 的 JSON 包括：

- 首次接触与完全分离的真实时间戳
- 5、7、9 帧窗口的碰撞前后速度与恢复系数
- 最终恢复系数和窗口间不确定性
- 入射角与轮廓—模板差异
- 质量状态及是否允许进入标准统计
- 使用的几何 ID、小球 ID 和设置文件

轨迹 CSV 包括逐帧真实 PTS、绿色轮廓位置、黄色模板位置、置信度、轮廓模式、排除状态，以及该帧实际进入了哪些拟合窗口。

## 其他测量方法

足球落地速度法：

```bash
python3 calculate_restitution.py "/完整路径/football.MOV" \
  --save-annotation annotations/football_velocity.json
```

释放高度与第一次反弹高度法：

```bash
python3 calculate_restitution_from_heights.py "/完整路径/football.MOV" \
  --save-annotation annotations/football_height.json
```

高度法使用：

```text
e = sqrt(第一次反弹高度 / 释放下落高度)
```

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖真实时间戳选择、碰撞前后分支拆分、反光干扰、透明球稀疏轮廓、设置文件身份校验、恢复系数数学计算和 H1–H5 汇总。

## 数据与隐私

仓库不包含实验视频、人工标注、轨迹 CSV 或小球模板。请自行保存这些数据，不要提交到公共仓库。
