# DRESO

**DRESO: Evidence-Guided Dual-Region Spectral Operators for Long-Horizon PDE Forecasting**

本仓库是 DRESO 的 clean 复现版本，只包含模型本身、Poseidon 与 The Well
训练/评测入口，以及门控特征和可学习 cutoff 的诊断可视化。训练协议固定为
单帧输入。仓库仅保留公开复现所需的基础训练、评测和诊断入口。

## 环境

推荐 Linux、Python 3.10、NVIDIA GPU。环境文件只由 Conda 创建基础 Python
环境，其余依赖通过 pip 安装：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
bash scripts/setup_environment.sh
conda activate dreso
```

`setup_environment.sh` 安装 PyTorch 2.0.1 + CUDA 11.8，并以 `--no-deps`
方式安装 The Well，避免其覆盖已验证的 NumPy、h5py 与 PyTorch 版本。

## 数据

Poseidon 根目录应直接包含 `NS-Gauss.nc`、`CE-RP.nc`、`CE-CRP.nc`、
`CE-Gauss.nc`、`NS-Sines.nc` 和 `CE-KH.nc`。

The Well 根目录支持以下四个 Hugging Face 原始子目录，无需预转换：

```text
acoustic_scattering_discontinuous/
active_matter/
gray_scott/
planetswe/
```

每个 The Well 子集必须包含官方的 `data/train`、`data/valid` 和 `data/test`
结构。一个 checkpoint 只训练一个子集，避免不同物理量被填充到同一通道。

## Poseidon 训练

默认训练六个子集、每个子集 200 条轨迹、200 epoch。模型规模可选
`Tiny`、`Big` 或 `L`：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_poseidon.sh /path/to/Poseiden L dreso_poseidon_l
```

重新训练带可学习绝对位置编码的 DRESO 时设置 `ABS_POS=true`：

```bash
ABS_POS=true CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_poseidon.sh /data1_hdd/chenzhengjie/Poseiden L dreso_poseidon_l
```

位置表定义在 patch grid 上，并随模型一起训练；脚本会自动给 run name 添加
`_abspos`，因此不会覆盖默认无位置编码的 checkpoint。也可绕过脚本直接向
`train/train.py` 传入 `--use_absolute_embeddings true`。

配置位于 `configs/train_poseidon.yaml`。checkpoint 默认保存至：

```text
checkpoint/dreso_poseidon/dreso_poseidon_l/
```

## Poseidon 评测

评测固定使用 `t=0` 单帧输入，汇报 `t=1` 的 dt1，以及 `t=1..20`
完全自回归 rollout 的时间与轨迹联合均值。JSON 同时保存 normalized 与
physical ReL1：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluate_poseidon.sh \
  checkpoint/dreso_poseidon/dreso_poseidon_l \
  /path/to/Poseiden
```

设置 `SAMPLES=-1` 可使用完整测试集。

## The Well 训练

默认从指定子集的训练窗口中均匀有放回抽取 2000 个相邻帧样本对，训练
50 epoch。数据保持原生空间网格：acoustic 与 active matter 为 `256x256`，
Gray-Scott 为 `128x128`，PlanetsWE 为 `256x512`：

正常评测同样严格使用各子集原生分辨率，不再执行 `128x128` resize；checkpoint
必须来自同一子集、同一原生网格的训练。

训练每个 epoch 进行验证，并以 `eval_loss` 最小的 epoch 作为最佳 checkpoint。
启用 HF loss 时，该指标为包含 base normalized ReL1 与 HF loss 的验证总损失；
训练结束后，运行目录根部保存的是重新载入的最佳 checkpoint，而不是最后一个 epoch。

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_the_well.sh \
  "/path/to/The Well" well.active_matter L
```

带绝对位置编码的 The Well 独立训练命令为：

```bash
ABS_POS=true EPOCHS=50 NUM_SAMPLES=2000 CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_the_well.sh \
  /data1_hdd/chenzhengjie/Well well.active_matter L
```

四个子集仍须分别从头训练。若跨分辨率评测，学习到的 patch-grid 位置表会以
双线性插值适配目标网格；该结果应作为跨分辨率压力测试单独汇报。

可选子集：

```text
well.acoustic_discontinuous
well.active_matter
well.gray_scott
well.planetswe
```

可通过环境变量覆盖基础训练资源，例如：

```bash
EPOCHS=50 NUM_SAMPLES=2000 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_the_well.sh "/path/to/The Well" well.gray_scott Big
```

## The Well 评测

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluate_the_well.sh \
  checkpoint/dreso_the_well/dreso_active_matter_L \
  "/path/to/The Well" well.active_matter
```

输出包含 normalized 与 physical ReL1 的 dt1 和 rollout 均值。

同一个当前 checkpoint 可直接在逐边缩小到 50% 或 25% 的网格上测试：

```bash
RESOLUTION_SCALE=0.5 CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluate_the_well.sh \
  checkpoint/dreso_the_well/dreso_active_matter_L \
  "/path/to/The Well" well.active_matter
```

`RESOLUTION_SCALE` 可取 `1`、`0.5`、`0.25`。评测器会同步缩放输入、静态场和
GT，整个 autoregressive rollout 都留在目标网格，不会 resize 回训练分辨率。
该结果属于跨分辨率压力测试，应与原生分辨率主结果分开汇报。

## 可视化

查看一个 checkpoint 内各频域 Block 学到的 cutoff 分布：

```bash
python evaluate/analyze_cutoff_distribution.py \
  --checkpoint /path/to/checkpoint \
  --output diagnostics/cutoff_distribution
```

查看十维门控描述符的实际响应；默认只收集门控特征，不执行额外因果掩蔽：

```bash
python evaluate/analyze_gate_feature_importance.py \
  --checkpoint /path/to/checkpoint \
  --data_path /path/to/Poseiden \
  --datasets NS-Gauss CE-RP CE-CRP CE-Gauss NS-Sines CE-KH \
  --num_samples 64 \
  --output_dir diagnostics/gate_features
```

分析某条 Poseidon 物理轨迹上的十维低/高频描述符：

```bash
python evaluate/analyze_ce_rm_gate_trajectory.py \
  --dataset CE-RM \
  --data /path/to/Poseiden/CE-RM.nc \
  --sample_index 0 \
  --cutoff 0.25 \
  --output_dir diagnostics/ce_rm_gate_trajectory_sample0 \
  --save_vector
```

## 说明

- 训练强制使用 CUDA，不会在 GPU 不可用时静默回退到 CPU。
- Poseidon 的不可压缩 NS 只监督和评测速度通道 `[u,v]`；人工密度与压力
  通道在 rollout 中保持常量。
- The Well 使用官方字段级 z-score，并在 rollout 中固定静态场。
- `ScOT` 类名仅用于兼容已有 Hugging Face checkpoint 的序列化格式。
