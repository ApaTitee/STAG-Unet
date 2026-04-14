# STAG-Unet 乳腺癌病理图像分割项目报告

## 0. 项目来源
这是一个面向香港科技大学 (HKUST) BEHI5011(2026) 课程的报告项目，由第 6 组同学共同完成。

## 1. 项目概述

STAG-Unet 是一个面向乳腺癌病理图像分割的课程项目，目标是在多中心、多扫描仪、形态差异显著的 H&E 病理切片上建立具有较强泛化能力的语义分割模型。项目核心思路是结合染色归一化、轻量级编码器和注意力机制，在有限计算资源下提升分割稳定性与跨域鲁棒性。

该项目参考 BEETLE 多中心数据集，并围绕病理图像分割中的实际问题展开，包括染色差异、组织形态复杂、前景与背景不平衡，以及不同扫描仪带来的域偏移。

## 2. 项目目标

本项目的主要目标包括：

1. 构建一个可用于乳腺癌病理图像分割的最小可行模型。
2. 通过染色归一化缓解不同中心、不同扫描仪之间的颜色偏差。
3. 利用轻量级编码器和注意力模块提升分割精度与泛化能力。
4. 设计适用于病理切片的 Patch 级训练、验证与测试流程。
5. 在外部测试集上验证模型的跨域性能。

## 3. 数据集说明

### 3.1 BEETLE 数据集

本项目使用 BEETLE 数据集作为核心训练与验证数据来源。该数据集的特点是：

1. 共包含 587 个乳腺癌活检和切除样本。
2. 数据来自多个临床中心，并包含 7 种不同扫描仪。
3. 涵盖多种分子亚型和组织学分级。
4. 标注覆盖多类别组织区域，适用于语义分割任务。
5. 数据集同时提供开发集和外部评估集，有利于标准化 benchmarking。

### 3.2 数据组织形式

数据目录中包含以下主要内容：

1. annotations：标注文件，包括 JSON、XML 和 mask。
2. label_map.json：像素值与类别标签映射。
3. xmls 和 jsons：组织区域轮廓标注。
4. masks：像素级掩膜。
5. model：预训练或验证用模型文件。

### 3.3 数据集使用原则

本项目采取 Patch 级别训练方式，不直接对整张 WSI 做端到端训练，而是先将切片划分为固定大小的图像块，再进行模型训练。这样做的原因是：

1. Whole Slide Image 分辨率过高，难以直接训练。
2. Patch 训练可以控制显存占用。
3. Patch 级别数据更适合做采样均衡和增强处理。
4. 便于对背景区域进行过滤。

## 4. 数据处理方案

### 4.1 染色归一化

项目采用 Macenko 染色归一化方法，并采用离线处理方式。其作用是降低不同染色批次、不同中心和不同扫描仪之间的颜色差异，使模型更专注于形态学信息而非颜色偏差。

### 4.2 Patch 划分策略

项目采用以下 Patch 策略：

1. Patch size：512 x 512
2. Overlap：50%
3. 采样方式：全覆盖
4. 背景过滤：排除白色或空白 Patch

这一策略兼顾了组织结构保留与数据利用率，同时减少空白区域对训练造成的干扰。

### 4.3 数据增强

为提高模型泛化能力，训练阶段引入以下增强策略：

1. 随机 mask
2. mirror
3. rotation
4. stretch
5. gaussian noise

这些增强方式主要用于模拟不同扫描条件、切片方向和组织形变，提高模型对真实病理场景的鲁棒性。

### 4.4 数据划分

项目采用 5-fold Cross Validation，且按照 Patch 级别划分数据。该策略适合当前阶段的最小可行实现，能够快速验证模型有效性。

需要注意的是，Patch 级划分会带来一定的数据泄漏风险，尤其当同一张 WSI 的相邻 Patch 同时出现在训练和验证中时，模型评估可能偏乐观。后续若要进一步提升严谨性，建议补充 slide-level 或 patient-level 的对照实验。

## 5. 模型设计

### 5.1 总体架构

项目采用 EfficientNet-B0 作为编码器，并结合 Attention U-Net 的解码结构，形成 STAG-Unet 的整体框架。其设计目标是在保持较低参数量的同时增强空间注意力表达能力。

### 5.2 编码器

EfficientNet-B0 被选为骨干网络，原因包括：

1. 参数量较小，适合有限算力场景。
2. 具有较好的特征提取能力。
3. 能作为 U-Net 编码器提供多尺度语义特征。

### 5.3 注意力机制

项目采用 CBAM 作为注意力模块，放置在跳连接处，结合通道注意力与空间注意力。这样设计的目的在于：

1. 强化与目标组织相关的有效特征。
2. 抑制背景噪声和冗余纹理。
3. 缓解轻量化骨干网络在表达能力上的不足。

### 5.4 解码器与跳连接

模型保留 U-Net 式多层跳连接结构，并在连接处引入 CBAM，以便在高分辨率特征与语义特征融合时进行自适应筛选。这样可以更好地恢复细粒度边界信息，适合病理分割任务中常见的复杂轮廓和分散区域。

## 6. 损失函数与优化策略

### 6.1 损失函数

项目采用 Focal Dice Loss 作为训练损失，并设置如下参数：

1. alpha = 0.25
2. gamma = 2
3. 加入 L2 正则

该设计兼顾了类别不平衡和边界敏感性。Focal 机制强调困难样本，Dice 机制强调区域重叠，二者结合适合组织分割任务。

### 6.2 优化器

优化器采用 AdamW。相比传统 Adam，AdamW 对权重衰减的处理更规范，更适合作为带正则化的训练优化方案。

### 6.3 学习率策略

项目采用余弦退火学习率调度，并设置 warmup 为 5 个 epoch。其作用包括：

1. 训练初期平滑收敛。
2. 减少梯度震荡。
3. 提升后期收敛稳定性。

### 6.4 训练超参数

当前确认的训练参数如下：

1. Batch size：4
2. Epoch：100
3. Warmup：5
4. Optimizer：AdamW
5. Scheduler：Cosine Annealing

## 7. 验证与测试方案

### 7.1 交叉验证

项目采用 5-fold Cross Validation，并按 Patch 级划分。其目的是在有限数据条件下尽可能稳定地评估模型性能。

### 7.2 外部测试集

项目选择 CAMELYON16 作为外部测试集，用于评估模型在未见数据域上的泛化能力。这个设置对验证模型是否具备跨中心、跨设备适应性非常重要。

### 7.3 评估指标

当前方案中，最终评估指标以 loss 为主。为了便于训练监控和过程分析，建议同时保留 Dice、IoU 等辅助日志指标，但不作为主评估口径。

## 8. 训练与实验管理

### 8.1 Checkpoint 策略

模型 checkpoint 每 10 轮保存一次，便于：

1. 恢复训练。
2. 回溯中间模型状态。
3. 对比不同阶段性能。

### 8.2 可视化

项目通过 TensorBoard 进行结果可视化，主要用于记录：

1. 训练损失变化。
2. 验证损失变化。
3. 学习率变化。
4. 部分分割结果可视化。
5. 训练过程中的超参数和实验曲线。

### 8.3 多卡训练

在 HKUST HPC4 的 RTX5880 Ada 显卡集群上进行训练。

## 9. 当前实现状态

从代码结构来看，当前项目仍处于设计规划与可行性验证阶段，核心源文件包括：

1. dataset.py
2. model.py
3. loss.py
4. train.py
5. utils.py

## 10. 最小可行方案总结

当前项目的最小可行方案可以概括为：

1. 使用 BEETLE 数据集进行病理分割训练。
2. 先对图像执行离线 Macenko 染色归一化。
3. 使用 512 x 512 Patch、50% overlap、全覆盖采样，并过滤白色背景块。
4. 在训练阶段使用随机 mask 和几何类增强。
5. 采用 EfficientNet-B0 + U-Net 解码器 + CBAM 跳连接的模型结构。
6. 使用 Focal Dice Loss、AdamW、余弦退火与 warmup 训练。
7. 以 5-fold Patch-level Cross Validation 做训练验证。
8. 使用 CAMELYON16 做外部测试。
9. 以 loss 为主指标，TensorBoard 做训练可视化。
10. 每 10 轮保存一次 checkpoint。

## 11. 风险与注意事项

1. Patch-level 划分可能引入数据泄漏，后续建议补充更严格的切分方案。
2. 仅以 loss 作为最终评估指标，结果解释性相对较弱，建议保留辅助指标日志。
3. 背景过滤规则需要定义清楚，否则可能误删低染色强度的有效组织区域。
4. 随机 mask 和 stretch 的增强强度需要适度控制，避免破坏病理结构语义。
5. 外部测试集与训练集之间的类别分布和标注体系需要对齐，否则评估结果可能存在偏差。

## 12. 后续工作建议

1. 将当前设计落地到 dataset.py、model.py、loss.py、train.py 和 utils.py。
2. 先完成单卡最小可行训练流程，再补充实验管理和指标统计。
3. 增加可复现性控制，包括随机种子、配置文件和日志记录。
4. 若后续追求更严谨的泛化评估，建议增加 patient-level 或 slide-level 对照实验。
5. 若有时间，再扩展消融实验，对比是否使用 Macenko、CBAM 和不同 backbone 的效果。

## 13. HPC4 运行说明（路径与调度）

### 13.1 基本约定

1. 通过 SSH 登录后默认在 login 节点。
2. login 节点仅用于短时交互式测试，不建议执行大规模训练。
3. 正式训练请使用 SLURM 提交作业。
4. 推荐 conda 环境：`stag_env`。

### 13.2 关键路径

推荐将真实数据根目录设置为：

1. `/scratch/jchengaw/data`
2. CSV：`/scratch/jchengaw/data/data_overview.csv`

训练脚本 `src/train.py` 支持以下优先级：

1. CLI 参数（`--csv-path`, `--data-root`, `--output-dir`）
2. 环境变量（`STAG_CSV_PATH`, `STAG_DATA_ROOT`, `STAG_OUTPUT_DIR`）
3. 代码默认值（`data/data_overview.csv`, `data`, `outputs`）

### 13.3 login 节点短测命令（仅冒烟）

```bash
cd /home/jchengaw/STAG-Unet
bash archive/scripts/testing/smoke_login.sh
```

或显式覆盖路径：

```bash
cd /home/jchengaw/STAG-Unet
CSV_PATH=/scratch/jchengaw/data/data_overview.csv DATA_ROOT=/scratch/jchengaw/data bash archive/scripts/testing/smoke_login.sh
```

`archive/scripts/testing/smoke_login.sh` 只做路径、数据加载和模型前向检查，不会进入训练循环。
`src/train.py` 只能在 SLURM 作业中运行；如果在 login 节点直接调用，会立即报错。

### 13.4 SLURM 正式训练

项目提供模板脚本：`scripts/train.slurm`，已包含：

1. `--account=mscbehi5011hpc4`
2. `--partition=gpu-rtx5880`
3. `conda activate stag_env`

提交示例：

```bash
cd /home/jchengaw/STAG-Unet
sbatch scripts/train.slurm
```

可通过环境变量覆盖参数：

```bash
sbatch --export=ALL,FOLD=1,EPOCHS=20,BATCH_SIZE=6,CSV_PATH=/scratch/jchengaw/data/data_overview.csv,DATA_ROOT=/scratch/jchengaw/data scripts/train.slurm
```

5-fold 并行训练（SLURM array）：

```bash
cd /home/jchengaw/STAG-Unet
sbatch scripts/train_5fold_cached.slurm
```

覆盖参数示例：

```bash
sbatch --export=ALL,EPOCHS=20,BATCH_SIZE=6,CSV_PATH=/scratch/jchengaw/data/data_overview.csv,DATA_ROOT=/scratch/jchengaw/data,OUTPUT_DIR=/scratch/jchengaw/STAG-Unet/outputs_5fold_cached scripts/train_5fold_cached.slurm
```

### 13.5 流水线测试脚本

`test_pipeline.py` 也支持路径参数化：

```bash
python archive/scripts/testing/test_pipeline.py \
	--csv-path /scratch/jchengaw/data/data_overview.csv \
	--data-root /scratch/jchengaw/data \
	--num-folds 5 --max-wsis 3
```

### 13.6 OpenSlide 依赖检查与修复

先检查环境中是否安装成功：

```bash
conda run -n stag_env conda list | grep -Ei "openslide|libopenslide|python"
conda run -n stag_env python -c "import openslide; from openslide import lowlevel; print(openslide.__file__); print(lowlevel._lib)"
```

若出现 `libopenslide.so` 或 `CXXABI` 相关错误，可在 `stag_env` 中执行：

```bash
conda install -n stag_env -c conda-forge openslide libstdcxx-ng libgcc-ng
pip install -U openslide-python
```

SLURM 训练脚本 `scripts/train.slurm` 已设置：

1. `conda activate stag_env`
2. `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}`

这样可以优先使用 conda 环境中的共享库，避免节点系统库版本冲突。
