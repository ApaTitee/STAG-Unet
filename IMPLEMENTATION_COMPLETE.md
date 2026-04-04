# STAG-Unet 实现完成报告

## 验收标准完成情况

### ✅ 已完成的功能

#### 1. 数据处理管道
- **Patch 划分**: 512x512, 50% overlap, 全覆盖采样，自动过滤白色空白区域
- **数据划分**: 5-fold Cross Validation（按 Patch 级别）
- **数据增强**: 包含 mirror, rotation, stretch, gaussian noise, random mask (CoarseDropout)
- **I/O 容错**: 
  - 自动跳过缺失的 WSI/mask 文件
  - TIFF 解码错误时自动跳过该 patch
  - 支持超大 TIFF（>1GB）的流式读取

#### 2. 模型架构
- **编码器**: EfficientNet-B0（可选预训练）
- **解码器**: 5 层上采样 + 跳连接
- **注意力机制**: CBAM（通道 + 空间注意力）放在所有跳连接处
- **输出**: 5 类分割（背景 + 4 组织类）

#### 3. 训练配置
- **优化器**: AdamW (lr=3e-4, weight_decay=1e-4)
- **学习率**: 余弦退火 + 5 epoch warmup
- **损失函数**: Focal Dice Loss (alpha=0.25, gamma=2.0) + L2 正则
- **Checkpoint**: 每 10 轮保存一次，最佳模型单独保存
- **可视化**: TensorBoard（loss、dice、学习率曲线）

#### 4. 评估指标
- 主指标: 训练/验证 loss
- 辅助指标: Dice Coefficient（自动计算，用于监控）

### ✅ 已验证的核心流程

1. **数据加载**
   - CSV 路径解析与验证（396/641 条有效行）
   - Patch 记录生成与折叠分配
   - DataLoader 批量加载（支持加权采样）

2. **模型推理**
   - 前向传播：512x512 RGB 输入 → 5 通道输出
   - 损失计算：支持缺失值处理的 Focal Dice Loss
   - 梯度反向传播与参数更新

3. **训练循环**
   - 完整的 epoch 迭代（本地测试：3 张图 × 2 epochs）
   - Epoch 内部 batch 迭代与进度条显示
   - 验证集评估与最佳模型保存

### 📋 测试覆盖

| 测试项 | 状态 | 备注 |
|--------|------|------|
| 数据 CSV 加载 | ✅ | 396 条有效记录 |
| Patch 生成 | ✅ | 3 个 2048×2048 样本 → 147 个 patch |
| 数据增强 | ✅ | albumentations 集成 |
| Batch collate | ✅ | 移除不可序列化对象 |
| 模型前向 | ✅ | 2×3×512×512 → 2×5×512×512 |
| 损失计算 | ✅ | Focal Dice + 梯度流 |
| 完整训练 | ✅ | 2 epochs, batch_size=2 |

---

## 目录结构

```
Final_Project/
├── src/
│   ├── __init__.py           # 包入口
│   ├── dataset.py            # 数据集与 DataLoader
│   ├── model.py              # STAG-Unet 模型
│   ├── loss.py               # Focal Dice Loss
│   ├── train.py              # 训练入口（支持多折超参）
│   └── utils.py              # I/O 工具、Macenko、评估指标
├── data/
│   ├── data_overview.csv     # 数据清单（416 行，587 个样本）
│   ├── images/
│   │   └── development/wsis/ # WSI TIFF 文件
│   └── annotations/
│       ├── masks/            # Mask segmentation
│       ├── jsons/            # JSON 标注
│       ├── xmls/             # XML 标注
│       └── label_map.json    # 类别映射
├── data_test/                # 合成测试数据集（3 张图）
│   ├── data_overview.csv
│   ├── images/development/wsis/
│   └── annotations/masks/
├── outputs/                  # 训练输出（自动创建）
│   └── fold_*/tensorboard/   # TensorBoard 日志
├── prepare_test_data.py      # 生成测试数据脚本
├── test_pipeline.py          # 完整 pipeline 测试脚本
├── train.py                  # 便利训练启动脚本
└── environment.yml           # 环境配置
```

---

## 快速开始

### 1. 环境设置
```bash
conda activate stag_env
```

### 2. 测试你的实现
```bash
# 完整测试（数据、模型、训练）
python test_pipeline.py

# 用小数据集运行 1 fold 的完整训练
python ./src/train.py \
  --csv-path data_test/data_overview.csv \
  --data-root data_test \
  --epochs 2 \
  --batch-size 2 \
  --max-wsis 3 \
  --fold 0 \
  --num-workers 0
```

### 3. 完整训练（需要真实 BEETLE 数据）
```bash
# 5-fold 交叉验证 - fold 0
python ./src/train.py \
  --csv-path data/data_overview.csv \
  --data-root data \
  --epochs 100 \
  --batch-size 4 \
  --fold 0 \
  --num-workers 4 \
  --learning-rate 3e-4

# 查看 TensorBoard
tensorboard --logdir outputs/fold_0/tensorboard
```

### 4. 运行所有 5 折
```bash
for fold in {0..4}; do
  python ./src/train.py \
    --csv-path data/data_overview.csv \
    --data-root data \
    --epochs 100 \
    --batch-size 4 \
    --fold $fold &
done
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv-path` | `data/data_overview.csv` | 数据清单文件 |
| `--data-root` | `data` | 数据根目录 |
| `--epochs` | `100` | 训练轮数 |
| `--batch-size` | `4` | Batch 大小 |
| `--learning-rate` | `3e-4` | 初始学习率 |
| `--weight-decay` | `1e-4` | L2 正则强度 |
| `--warmup-epochs` | `5` | 预热轮数 |
| `--fold` | `0` | 验证折叠索引 (0-4) |
| `--num-folds` | `5` | 交叉验证折叠数 |
| `--num-workers` | `4` | 数据加载进程数 |
| `--max-wsis` | `None` | 限制读取的 WSI 数量（用于测试） |
| `--seed` | `42` | 随机种子 |
| `--output-dir` | `outputs` | 输出目录 |
| `--device` | `cuda` | 计算设备 |
| `--pretrained` | `False` | 使用 ImageNet 预训练 |

---

## 已知限制与后续工作

### 限制
1. **Macenko 离线预处理**: 当前在线应用，未提供离线预处理脚本
2. **CAMELYON16 集成**: 外部测试集接口未实现
3. **消融研究**: 未实现关闭 Macenko/CBAM 的对照组
4. **多中心评估**: 未按扫描仪分组评估泛化能力

### 建议补充
1. **Macenko 预处理脚本**
   ```python
   # 在所有 WSI 上统一 stain
   from src.utils import MacenkoNormalizer
   normalizer = MacenkoNormalizer()
   normalizer.fit(reference_wsi)  # 选一张标准 WSI
   # 批量归一化所有 WSI
   ```

2. **消融实验**
   - 对比有无 Macenko 的性能
   - 对比有无 CBAM 的性能
   - 轻量网络 vs ResNet 对比

3. **域适应**
   - 按扫描仪分组统计性能
   - 评估跨扫描仪泛化能力

4. **结果导出**
   - 实现 Slide 级别的 patch 聚合推理
   - 可视化分割结果 overlay

---

## 调试建议

### 数据加载缓慢
- 增加 `--num-workers`（默认 4）
- 确保数据在高速存储上（SSD）

### 显存不足
- 减小 `--batch-size`（推荐 ≥ 2）
- 禁用预训练：移除 `--pretrained`

### 收敛不理想
- 增加 `--epochs`
- 调整 `--learning-rate`（通常 1e-4 ~ 1e-3）
- 确认数据质量（`test_pipeline.py` 中查看样本）

### TIFF 读取错误
- 当前已实现容错（skip bad patches）
- 若关键样本无法读取，需验证原始文件完整性

---

## 文件更新日志

### 本次修复补充
- ✅ `src/dataset.py`: 
  - 移除返回值中的 PatchRecord（解决 collate 问题）
  - 修复 Patch 权重初始化
- ✅ `src/model.py`:  
  - 修正 EfficientNet 特征图通道匹配
  - 移除冗余的 upsample 参数
- ✅ `src/utils.py`:
  - 修复 NaN 路径值处理（_safe_resolve_path）
  - 改进 TIFF 大文件支持（PIL.MAX_IMAGE_PIXELS）
- ✅ `prepare_test_data.py`: 新增，生成合成测试数据集
- ✅ `test_pipeline.py`: 新增，完整 pipeline 验证脚本

---

**项目状态**: 🟢 **最小可行版本完成，可开始实验**

预期下一步：运行完整 5-fold 训练，评估多中心泛化能力。
