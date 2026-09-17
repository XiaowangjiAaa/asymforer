# AsymFormer-GSSA 消融实验方案

本方案针对 `src/AsymFormer_GSSA_full.py` 的三大核心模块进行消融：
**CIFF（裂缝信息特征融合）、GSSA（几何结构自注意力）、Lightweight Decoder（轻量解码器）**。

所有实验由 `src/AsymFormer_GSSA_ablation.py`（参数化模型）统一实现，
通过 `--ablation` 指定实验名称，保证除被消融组件外其余结构完全一致。

---

## 1. 实验分组

### Group 1 — 核心模块 + 解码器

| 实验名 | 关闭/替换 | 验证目的 |
|---|---|---|
| `full` | 无（基线） | 完整模型 |
| `wo_gsa` | 去掉 GSSA（只保留 CIFF） | 几何结构自注意力的贡献 |
| `wo_ciff` | CCFF → `cat + 1x1 conv` | 裂缝信息融合模块的贡献 |
| `rgb_only` | 去掉深度分支（仅 RGB） | 深度/跨模态信息的贡献 |
| `simple_decoder` | MLP 解码器 → 单层 1x1 解码器 | 轻量多尺度解码器的价值 |

### Group 2 — CIFF 内部拆解

| 实验名 | 关闭/替换 | 验证目的 |
|---|---|---|
| `wo_channel_att` | 去掉 `MultiScaleSEBlock` 通道注意力 | 通道注意力的作用 |
| `wo_spatial_att` | 去掉 `SpatialFrequencyAttention` 频域空间注意力 | 频域空间注意力的作用 |
| `wo_repvgg` | `RepVggBlock` → 普通 `ConvNormLayer` | 结构重参数化的收益 |

### Group 3 — GSSA 几何先验拆解

| 实验名 | 关闭/替换 | 验证目的 |
|---|---|---|
| `wo_struct` | 去掉结构对比掩码 `M_struct`（eq.5） | 结构对比机制的价值 |
| `wo_depth_prior` | 去掉深度先验 `D`（eq.4） | 深度几何信息的贡献 |
| `wo_spatial_prior` | 去掉空间先验 `S`（eq.6，曼哈顿距离） | 空间位置先验的贡献 |

### Group 4 — GSSA 注意力机制

| 实验名 | 关闭/替换 | 验证目的 |
|---|---|---|
| `full_2d` | 轴向分解 → 全 2D 注意力（`use_axial=False`） | 轴向分解（eq.10/11）的收益与开销 |

---

## 2. 统一训练配置（所有实验保持一致）

| 超参数 | 值 |
|---|---|
| 输入分辨率 | 480 × 640（RGB 3ch / Depth 1ch） |
| 类别数 `--num-classes` | 2（裂缝任务；NYUv2 改 40，SUNRGBD 改 37） |
| 优化器 | AdamW |
| 学习率 `--lr` | 5e-4 |
| 权重衰减 `--weight-decay` | 1e-4 |
| 学习率调度 | warmup(4 epoch) + poly(0.9) |
| 最小学习率 `--min-lr` | 1e-6 |
| 批大小 `--batch-size` | 4（按显存调整） |
| 训练轮数 `--epochs` | 200 |
| 随机种子 `--seed` | 2333 |
| 损失函数 `--loss` | `ce`（或 `focal`） |
| 忽略标签 `--ignore-index` | 0 |
| 数据增强 | scaleNorm → RandomScale(1.0/1.4/2.0) → RandomHSV → RandomCrop → RandomFlip → ToTensor → Normalize |

> 消融实验必须保证**除被消融组件外的一切都相同**，包括 seed、数据增强、超参。

---

## 3. 训练命令

### 单个实验

```bash
python train_gssa_ablation.py \
    --ablation wo_gsa \
    --data-dir ./RGB-Dcrackdataset \
    --num-classes 2 \
    --batch-size 4 --lr 5e-4 --epochs 200 \
    --loss ce --eval-every 10 --gpu 0
```

- `--ckpt-dir` 缺省时自动为 `./model_M1/ablation/<name>`。
- 建议开启 `--eval-every 10`，每 10 epoch 在验证集上计算 mIoU，自动保存 `best.pth`。
- 显存充足可用 `--amp` 混合精度；梯度爆炸可加 `--clip-grad-norm 5`。

### 批量运行（PowerShell）

```powershell
$exps = @("full","wo_gsa","wo_ciff","rgb_only","simple_decoder",
          "wo_channel_att","wo_spatial_att","wo_repvgg",
          "wo_struct","wo_depth_prior","wo_spatial_prior","full_2d")
foreach ($e in $exps) {
    python train_gssa_ablation.py --ablation $e --data-dir ./RGB-Dcrackdataset `
        --num-classes 2 --batch-size 4 --lr 5e-4 --epochs 200 --loss ce --eval-every 10 --gpu 0
}
```

### 批量运行（Linux / bash）

```bash
for e in full wo_gsa wo_ciff rgb_only simple_decoder \
         wo_channel_att wo_spatial_att wo_repvgg \
         wo_struct wo_depth_prior wo_spatial_prior full_2d; do
    python train_gssa_ablation.py --ablation $e --data-dir ./RGB-Dcrackdataset \
        --num-classes 2 --batch-size 4 --lr 5e-4 --epochs 200 --loss ce --eval-every 10 --gpu 0
done
```

---

## 4. 评估

训练时 `--eval-every` 已输出 mIoU；如需独立复测，用 `best.pth`：

```bash
python eval.py   # 需把 eval.py 顶部 import 与 pth_dir 指向对应模型与 checkpoint
```

> 当前 `eval.py`/`MS5_eval.py` 硬编码 import 了 `src.AsymFormer` 的 `B0_T`。
> 复测消融模型时，请改为：
> ```python
> from src.AsymFormer_GSSA_ablation import build_model
> model, _ = build_model("<name>", num_classes=2)
> ```
> 并加载对应 `best.pth`。

---

## 5. 结果记录模板

| 实验 | 分组 | mIoU | Accuracy | F1 | FLOPs(G) | Params(M) |
|---|---|---|---|---|---|---|
| full | 1 | | | | | |
| wo_gsa | 1 | | | | | |
| wo_ciff | 1 | | | | | |
| rgb_only | 1 | | | | | |
| simple_decoder | 1 | | | | | |
| wo_channel_att | 2 | | | | | |
| wo_spatial_att | 2 | | | | | |
| wo_repvgg | 2 | | | | | |
| wo_struct | 3 | | | | | |
| wo_depth_prior | 3 | | | | | |
| wo_spatial_prior | 3 | | | | | |
| full_2d | 4 | | | | | |

> 快速打印各实验的 FLOPs/参数量（无需训练）：
> ```bash
> python src/AsymFormer_GSSA_ablation.py
> ```

---

## 6. 预期结论

- `wo_gsa` / `wo_ciff` 相对 `full` 明显下降 → 证明两大模块必要。
- `wo_struct`、`wo_depth_prior` 下降 → 证明几何先验各分量有效。
- `full_2d` 与 `full` 精度接近但开销更大 → 证明轴向分解以更小开销达到近似精度。
- `rgb_only` 下降 → 证明跨模态（深度）信息不可替代。
