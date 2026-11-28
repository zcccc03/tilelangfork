# DensePFFN 融合 Cast 批量 GEMM (`dense_pffn_gemm`) 设计与性能现状报告

> 本报告为**自包含文档**（不引用外部路径或外挂日志），完整记录算子语义、Ascend 950DT 硬件 Roofline 分析、演进路径、实卡测量数据、瓶颈拆解（含 strided MTE2 vs SIMT load 专题分析）以及后续演进计划。

---

## 1. 算子语义与测试负载

### 1.1 接口与数学语义

目标算子 `_dense_pffn_gemm_kernel`（Python 封装为 `_fc_bmm` / `dense_pffn_gemm`）执行带任意 3D stride 与融合精度转换的批量矩阵乘：

$$O[b, m, n] = \text{cast}_{\text{out\_dtype}}\!\left(\sum_{k=0}^{K-1} \text{cast}_{\text{bf16}}\!\left(A[b, m, k]\right) \cdot \text{cast}_{\text{bf16}}\!\left(W[b, k, n]\right)\right)$$

- **输入**：`A [batch, M, K]`（支持 `fp32` / `bf16` 及任意 3D stride `(sab, sam, sak)`），`W [batch, K, N]`（支持 `fp32` / `bf16` 及任意 3D stride `(swb, swk, swn)`）。
- **计算**：当 `use_bf16=True` 时，输入在计算前就地转为 `bfloat16`，Cube 内部做 `FP32` 累加；当 `use_bf16=False` 时走 `FP32` 回退路径。
- **输出**：连续张量 `O [batch, M, N]`，支持 `bfloat16` 或 `float32`。
- **精度门槛**：输出全有限（无 `NaN`/`Inf`，实测前输出预填 `NaN` 哨兵），相对 $L_2$ 误差 `rel_l2 <= 1e-2`（端到端前向 + 9 个反向梯度相对 CPU FP32 golden 亦满足 `rel_l2 <= 1e-2`）。

### 1.2 六个典型调用点（DensePFFN SwiGLU 前反向）

默认形状：`B = 1024, T = 56, D = 1232 (= 16 × 77), P = 2464 (= 32 × 77), 2P = 4928 (= 64 × 77)`。所有维度均为 16 的倍数，但均不被 128/256 整除（存在尾块负载不均问题）。

| 调用点 | `(batch, M, N, K)` | `A` 类型与布局 | `W` 类型与布局 | `O` 类型 | 算术强度 (FLOP/B) | 理想耗时 @ 432 TFLOPS |
| --- | --- | --- | --- | --- | --- | --- |
| `c1_fwd_up` | `(56, 1024, 4928, 1232)` | `fp32` `perm(1,0,2)` (`sak=1`, 行 stride 276 KiB) | `fp32` 连续 (`swn=1`) | `bf16` | 290 | `1.612 ms` |
| `c2_fwd_down` | `(56, 1024, 1232, 2464)` | `bf16` 连续 (`sak=1`) | `fp32` 连续 (`swn=1`) | `bf16` | 308 | `0.806 ms` |
| `c3_bwd_dw2` | `(56, 2464, 1232, 1024)` | `fp32` `T(1,2)` (`sam=1`) | `fp32` `perm(1,0,2)` (`swn=1`, 行 stride 276 KiB) | `fp32` | 228 | `0.806 ms` |
| `c4_bwd_dswi` | `(56, 1024, 2464, 1232)` | `fp32` `perm(1,0,2)` (`sak=1`, 行 stride 276 KiB) | `fp32` `T(1,2)` (`swk=1`) | `fp32` | 238 | `0.806 ms` |
| `c5_bwd_dw1` | `(56, 1232, 4928, 1024)` | `bf16` `perm(1,2,0)` (`sam=1`, 行 stride 138 KiB) | `fp32` `perm(1,0,2)` (`swn=1`, 行 stride 1.10 MiB) | `fp32` | 280 | `1.612 ms` |
| `c6_bwd_dx` | `(56, 1024, 1232, 4928)` | `fp32` `perm(1,0,2)` (`sak=1`, 行 stride 1.10 MiB) | `fp32` `T(1,2)` (`swk=1`) | `fp32` | 264 | `1.612 ms` |
| **合计** | 6 次 GEMM | — | — | — | **228–308** | **`7.253 ms`** |

---

## 2. 环境、基线与计时口径

- **硬件平台**：Ascend 950DT（x86_64 host，Device 0），进程初始化固定核数限额 `cube_num=32, vector_num=64`。
- **硬件规格与峰值口径**：`32 AIC / 64 AIV` @ `1650/1650 MHz`，BF16 Cube 理论峰值 **`432.0 TFLOPS`**（单 AIC 13.5 TFLOPS = 8192 FLOP/cyc），HBM 峰值 `4.0 TB/s`，L2 容量 `128 MiB`，单核片上 `L1=512 KiB, L0A=64 KiB, L0B=64 KiB, L0C=256 KiB, UB=248 KiB/AIV`。脊点（Ridge Point）= `108 FLOP/B`，全部 6 个调用点算术强度均在 `228–308 FLOP/B`（2.1–2.9× 脊点），理论上限严格受限于 Cube 算力。
- **软件栈**：Python 3.12.3、`torch 2.13.0+cpu`、`torch_npu 2.13.0`、`tilelang 0.1.15+git39a6442e`、CANN 9.3.0；全局关闭 HF32（`allow_hf32=False`）并设置 `ACL_PRECISION_MODE=must_keep_origin_dtype`。
- **计时与验证口径**：
  - 所有实测性能均先通过 `NaN` 哨兵输出完整性与 `rel_l2 <= 1e-2` 精度校验；
  - **冷 L2（Cold）**：每次迭代前清零 `1 GiB` 独立 flush buffer 并同步，再单独测量 kernel 执行时间（中位数）；
  - **热 L2（Warm）**：不执行 flush，连续迭代测量中位数；
  - 所有正式性能数据均记录实测 AI Core 频率（R001–R004 全部稳定在 `1650/1650 MHz`）；PMU 利用率取自 `torch_npu.profiler` `Level1 + PipeUtilization`。
- **双基线定义**：
  1. **Naive PyTorch NPU（语义等价基线）**：按 `_fc_bmm` 语义执行 `.to(bf16).contiguous()` + `torch.bmm` + 输出类型转换，总耗时 **`25.506 ms`（`28.4%` MFU）**；
  2. **Pure BF16 Contiguous `aclnnBmm`（硬件天花板参考）**：输入预先转为连续 BF16（不含 FP32 cast、不含大 stride 搬运），总耗时 **`8.008 ms`（`90.6%` MFU）**。

---

## 3. 核心硬件物理约束与架构演进

### 3.1 为什么不能直接做 Per-Tile `UB -> L1` 融合（R002 实证）

Ascend 950DT 硬件上 **没有** `GM -> L1` 的 FP32→BF16 硬件 DMA 转换，也 **没有** `L1 -> UB` 通路，因此 FP32→BF16 转换必须在 AIV 的 UB 上完成。

在 R002 中我们首先实现了直连 `UB -> L1` 融合 CV（`128×256×64` + `ascend_nd2nz_scatter<float, bfloat16_t>`），在 `c1_fwd_up` 上仅达 `3.611 ms`（`44.6%` MFU），PMU 显示 **`aiv_mte2_ratio = 0.987`** 而 `aic_mac_ratio = 0.527`。根因是物理带宽墙：
- 即便使用最大 `256×256` L0C tile，要打满 432 TFLOPS 所需的 per-tile FP32 读带宽至少为 `432T / 64 FLOP/B = 6.75 TB/s`；
- 而 64 个 AIV 并发 `MTE2` 读 GM 的实测物理上限仅 `~4.7 TB/s`（单 AIV ≈ `44 B/cyc`），带大 stride 读取时有效带宽仅 `~2.7–2.9 TB/s`。

### 3.2 核心突破：L2-Ping-Pong Batch-Pipelined CV（V3 -> V4）

观察到单个 batch 的 BF16 工作集 `A[t] + W[t]` 仅占 **`7.2–21.2 MiB`**，远小于 **`128 MiB` L2 Cache**：
1. **AIV 侧（只读 HBM 恰好 1.00×）**：64 个 AIV 协作读取 `G_B` 个 batch 的原始（可带 stride）FP32 输入，以 `NOTALLOC_DROP`（不污染 L2）拉入 UB，通过 256B `SimdVF`（`vld2 → 2×vcvt → vor → vsts`）高速转为 BF16，再写回**连续且驻留 L2 的 workspace**（显式指定 `l2_cache_ctrl=1 (NORMAL_LAST_VICTIM)`，修复了 R003 中 TileLang 默认 `UB->GM` 发出 `NOTALLOC_CLEAN (4)` 导致绕过 L2 的关键缺陷；实测 `mode=2 (NORMAL_PERSISTENT)` 反而会锁死 L2 引发 thrashing，比 `mode=1` 慢 11.5%）。
2. **AIC 侧（高复用读 L2）**：32 个 AIC 直接以 `256×256×128`（2-step `BK_L0=64`）从 L2 热 workspace 高效消费连续 BF16 分块，输出以 `NOTALLOC_PW` 流式写回 HBM。
3. **跨核同步（经实卡验证零死锁）**：950DT 上 `mode=0`（`asc_sync_inter`）仅支持同类核间同步（64 AIV 间或 32 AIC 间），`mode=4`（`asc_sync_intra`）为单 AIC ↔ 双 AIV 块内同步。组合 `mode=0 + mode=4` 构建了稳定的双缓冲 ping-pong 握手，AIV 侧总占用仅 `18%–28%`（`aiv_vec=0.04–0.06`、`aiv_mte3=0.06–0.09`），**计算与搬运 100% 隐藏在 Cube 执行时间之后**。

### 3.3 演进历程总览（R001 -> R005b，主负载 `c1_fwd_up`）

| 阶段 | 架构方案 | `c1_fwd_up` 冷 L2 耗时 | MFU | `aic_mac` | `aic_mte2` | `aiv_mte2` | 核心发现 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R001 | Naive PyTorch NPU（含 cast/contiguous） | `6.151 ms` | `26.2%` | — | — | — | 多 kernel 落盘 HBM 放大带宽开销 |
| R002c | V2 Per-Tile `UB->L1` 直连融合 CV (`128×256×64`) | `3.611 ms` | `44.6%` | `0.527` | `0.000` | **`0.987`** | 撞上 64-AIV `MTE2` 物理带宽墙 |
| R003c | V3 L2-Ping-Pong (`128×256×64`, 默认 `UB->GM=4`) | `3.610 ms` | `44.6%` | `0.453` | `0.971` | `0.220` | AIV 完全隐藏，但默认 `NOTALLOC_CLEAN` 绕过了 L2 |
| **R004** | **V4 L2-Ping-Pong (`256×256×128`, `ws_store_l2=1`)** | **`2.207 ms`** | **`73.0%`** | **`0.739`** | `0.849` | `0.256` | 修复 L2 驻留 + 256×256 Tile，单点暴涨 +28.4% MFU |
| R004 | Pure BF16 Cube 标定 (`256×256×128`, `L1x2`, 无 UF) | `1.847 ms` | `87.3%` | `0.895` | `0.899` | — | 剥离 AIV 后仍有 12.7% 空间（FIXPIPE 阻塞 + 尾块不均） |
| **R005b** | **Pure BF16 Cube 升级 (`256×256×128`, `L1x3 + UF2/3 + Rot + XOR/dn2nz`)** | **`1.815 ms` (热 `1.788 ms`)** | **`88.8%` (热 `90.1%`)** | — | — | — | 硬件 `unit_flag_ctrl=2/3` + 3 槽 L1 + batch 旋转直接逼近 `aclnnBmm` |
| R001 | Pure BF16 `aclnnBmm`（无 cast / 无 stride 参考） | `1.762 ms` | `91.5%` | `0.974` | `0.859` | — | 纯 BF16 连续算子实测上限 |

---

## 4. 全六调用点实测性能现状（1650/1650 MHz，Cold & Warm L2）

### 4.1 端到端六点对比表（已上板全通的 V4 交付默认版）

| 调用点 | 理想耗时 | Naive PyTorch NPU | Pure BF16 `aclnnBmm` | **V4 冷 L2（实测）** | **V4 热 L2（实测）** | **V4 冷 MFU** | **相对 PyTorch 加速** | **相对 `aclnnBmm` 占比** | 精度 `rel_l2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c1_fwd_up` | `1.612 ms` | `6.151 ms` (26.2%) | `1.762 ms` (91.5%) | **`2.207 ms`** | `2.184 ms` | **`73.0%`** | **`2.79×`** | `79.8%` | `0.00e+00` |
| `c2_fwd_down` | `0.806 ms` | `1.233 ms` (65.4%) | `0.918 ms` (87.8%) | **`1.333 ms`** | `1.299 ms` | **`60.4%`** | `0.92×` | `68.9%` | `0.00e+00` |
| `c3_bwd_dw2` | `0.806 ms` | `3.410 ms` (23.6%) | `0.962 ms` (83.8%) | **`1.493 ms`** | `1.462 ms` | **`54.0%`** | **`2.28×`** | `64.4%` | `1.66e-03` |
| `c4_bwd_dswi` | `0.806 ms` | `3.334 ms` (24.2%) | `0.936 ms` (86.1%) | **`1.236 ms`** | `1.207 ms` | **`65.2%`** | **`2.70×`** | `75.7%` | `1.66e-03` |
| `c5_bwd_dw1` | `1.612 ms` | `4.454 ms` (36.2%) | `1.705 ms` (94.5%) | **`2.344 ms`** | `2.319 ms` | **`68.8%`** | **`1.90×`** | `72.7%` | `1.66e-03` |
| `c6_bwd_dx` | `1.612 ms` | `6.924 ms` (23.3%) | `1.725 ms` (93.4%) | **`3.305 ms`** | `3.255 ms` | **`48.8%`** | **`2.10×`** | `52.2%` | `1.66e-03` |
| **总计 (6 算子)** | **`7.253 ms`** | **`25.506 ms` (28.4%)** | **`8.008 ms` (90.6%)** | **`11.918 ms`** | **`11.726 ms`** | **`60.9%` (热 `61.9%`)** | **`2.14×`** | **`67.2%`** | **6/6 PASS** |

> 注：`c1/c2` 输出为 `bf16` 时与逐块 BF16→FP32 累加参考 `rel_l2 = 0.0`（逐比特一致）；`c3–c6` 输出为 `fp32` 时相对输入未量化 FP32 黄金的 `rel_l2 = 1.66e-3`（纯 BF16 舍入固有噪声，远低于 `1e-2` 阈值）。

### 4.2 全六调用点 PMU 细粒度剖析（R004 实测，1650/1650 MHz）

| 调用点 | V4 配置 | `cold_ms` | `aic_mac` | **`time × aic_mac`** | `ideal_ms` | `aic_mte2` | `aic_fixpipe` | `aiv_mte2` | `aiv_vec` | `aiv_mte3` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c1_fwd_up` | `256×256×128, G_B=2` | `2.207` | `0.739` | **`1.631 ms`** | `1.612` | `0.849` | `0.075` | `0.256` | `0.061` | `0.092` |
| `c2_fwd_down` | `256×256×128, G_B=8` | `1.333` | `0.627` | **`0.836 ms`** | `0.806` | `0.916` | `0.057` | `0.203` | `0.042` | `0.061` |
| `c3_bwd_dw2` | `256×256×128, G_B=4` | `1.493` | `0.548` | **`0.818 ms`** | `0.806` | `0.618` | **`0.208`** | `0.204` | `0.047` | `0.086` |
| `c4_bwd_dswi` | `256×256×128, G_B=4` | `1.236` | `0.663` | **`0.819 ms`** | `0.806` | `0.873` | `0.113` | `0.282` | `0.064` | `0.091` |
| `c5_bwd_dw1` | `256×256×128, G_B=4` | `2.344` | `0.696` | **`1.631 ms`** | `1.612` | `0.805` | `0.113` | `0.179` | `0.040` | `0.061` |
| `c6_bwd_dx` | `256×256×128, G_B=2` | `3.422` | `0.476` | **`1.629 ms`** | `1.612` | `0.625` | `0.043` | `0.254` | `0.060` | `0.088` |

**最关键的定量结论**：看第 5 列 **`time × aic_mac`** 与第 6 列 **`ideal_ms`**！六个算子无一例外满足 `active_mac_time ≈ 1.01 × ideal_ms`——也就是说，**一旦数据进入 L0A/L0B，我们的 Cube 引擎本身就已经跑在 98–99% 的 432 TFLOPS 硬件峰值上！** V4 距离 90% MFU 的全部差距（`11.92 ms` vs `~8.0 ms`）**100% 来自外部停顿/排空气泡**，而非 Cube 计算效率不足。

### 4.3 Pure BF16 Cube 引擎消融实验（R005b 实测，`c1_fwd_up`）

| 实验项 | 配置 | 冷 L2 耗时 | 冷 MFU | 热 L2 耗时 | 热 MFU | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `E0` | `256×256×128, L1x2, 无 UF, 无 Rot` | `1.894 ms` | `85.1%` | `1.871 ms` | `86.1%` | V4 纯 Cube 基线 |
| `E1` | `+ unit_flag_ctrl=2/3 + batch_rot` | `1.833 ms` | `88.0%` | `1.800 ms` | `89.5%` | **+2.9% MFU**：消除 1-slot L0C `M_FIX` 阻塞 + 均衡 `N=4928` 尾块 |
| `E2` | `+ L1_SLOTS=3 (384/512 KiB L1)` | `1.817 ms` | `88.7%` | `1.797 ms` | `89.7%` | **+0.7% MFU**：更深 L1 预取隐藏 L2 抖动 |
| `E2b` | `+ ws_load_l2=0 (NORMAL_FIRST_VICTIM)` | `1.819 ms` | `88.6%` | **`1.788 ms`** | **`90.1%`** | 热态突破 90% MFU |
| `E3` | `+ dn2nz` (`GM->L1` 硬件转置 W) | `1.816 ms` | `88.7%` | `1.789 ms` | `90.1%` | 与 `nd2nz + l12l0b_T` 基本打平，热态略优 |
| `E4` | `+ XOR swizzle` (`n_local ^ m_local`) | **`1.815 ms`** | **`88.8%`** | `1.794 ms` | `89.8%` | 冷态最佳，距 `aclnnBmm` (`91.5%`) 仅 2.7% |
| `E5` | `128×256×128, L1x3, c_slots=2` | `2.312 ms` | `69.7%` | `2.279 ms` | `70.7%` | 相比 256×256 多 50% L2/MTE2 流量，严重受限 |
| `E6` | `128×256×256, L1x2, c_slots=2` | `2.120 ms` | `76.0%` | `2.086 ms` | `77.3%` | BK=256 虽改善突发长度，仍落后 256×256 达 12.8% |
| `E7` | `256×128×256, L1x2, c_slots=2` | `2.267 ms` | `71.1%` | `2.218 ms` | `72.7%` | 同上，确认 `256×256×128 + UF2/3` 为最优几何 |

---

## 5. 瓶颈根因拆解（含 Strided MTE2 vs SIMT Load 专题分析）

按影响大小排序，V4 当前 `60.9% -> 90%+` 的差距由四大根因构成：

### 5.1 根因 #1（最大杀手）：Super-Batch 全局屏障与 Wave-Tail 损失（占总差距 ~60%）

V4 每处理 `G_B` 个 batch 强制执行一次 32-AIC 全局 `mode=0` 同步，导致：
- **`c6_bwd_dx` (`256×256`, `G_B=2`)**：每个 super-batch 仅 `2 × 20 = 40` 个 tile，在 32 核上第二波仅 8 核工作、24 核全程空转（Wave 效率仅 **`62.5%`**），直接把 `aic_mac` 砸到 `0.476`！而跨 56 个 batch 总计 `56 × 20 = 1120 = 35 × 32` 个 tile，一旦消除 super-batch 屏障，Wave 效率直接回到 **`100.0%`**！
- **`c3_bwd_dw2` (`G_B=4`)**：每个 super-batch `200` tile (`200 % 32 = 8`)，Wave 效率仅 `89%`，叠加每 4 batch 的 `MTE2` 排空。
- **`c2_fwd_down` (`G_B=8`)**：`8` 个 batch 的 `ws_w` 双缓冲（`92.6 MiB`）+ `A`（`38.5 MiB`）总计 `131 MiB > 128 MiB L2`，引发 L2 逐出导致 `aic_mte2 = 0.916`，同时 `G_B=8` 拉长了首 batch prologue（`92.6 MiB` FP32 读）。
- **所有 6 点**：每次 super-batch 边界都会彻底排空 AIC `MTE2` 预取队列并放大快核/慢核（尾块）的同步等待。

### 5.2 根因 #2：1-Slot L0C `M_FIX` / `FIX_M` 串行排空（在 `c3` 上高达 ~20%）

`256×256` FP32 累加器占满全部 `256 KiB` L0C（`c_slots=1`）。V4 使用阻塞式 `M_FIX` + `FIX_M`，每个 tile 必须等整块 `FIXPIPE` 写完才能启动下一 tile 的 `MMAD`：
- 在 `c3_bwd_dw2`（`K=1024` 较短且 `O=FP32` 写回量翻倍为 `256 KiB/tile`），`aic_fixpipe_ratio` 高达 **`0.208`**！
- R005b 已实证启用硬件 **`unit_flag_ctrl=2`（首 K 步）/`3`（末 K 步 + FIXPIPE）** 后，分形级硬件流水直接重叠 `MMAD` 与 `FIXPIPE`，在 `c1`（BF16 输出）上已净赚 **`+2.9%` MFU**，在 `c3/c4/c5`（FP32 输出）上收益将更大（预计 `+8%–15%` MFU）。

### 5.3 根因 #3：非 256 整除维度引发的尾块核间负载不均（`c1` 约 `3.9%`）

以 `c1_fwd_up` 为例，`N = 4928 = 19 × 256 + 64`（`tiles_per_b = 80`）。因 `gcd(80, 32) = 16`，静态映射 `t_idx = cid + ti * 32` 会让固定 8 个核包揽全部 56 个 batch 的 `an = 64` 小尾块，其余 24 个核全部执行满块（`140` vs 均值 `134.75`，纯负载不均损失 `3.9%`）。在 V5 中通过 `in_b = (in_b_raw + (bid * GROUP_M) % tiles_per_b) % tiles_per_b` 实现跨 batch 旋转，在不破坏 `GROUP_M=4` L2 局部性的前提下，让全部 32 核精确平摊尾块。

### 5.4 根因 #4 专题分析：Strided `MTE2` Load vs SIMT Load

针对“是否遇到 strided load 性能不佳、能否用 SIMT 替代 MTE2”的关键判断，结合 R002/R004 实测 PMU 给出精准定位：

1. **FP32 非连续输入（`c1_A, c3_A/W, c4_A, c5_W, c6_A`）的现状**：
   - 在 R002c（直连 `UB -> L1`）中，大 stride（`276 KiB – 1.10 MiB`）+ 高复用次数叠加确实撞穿了 AIV `MTE2`（`aiv_mte2 = 0.987`）；
   - 切到 L2-Ping-Pong 后，AIV 对每个 FP32 元素仅读 **1.00×**，实测 AIV `asc_copy_gm2ub_align` 有效带宽约 **`2.66–2.91 TB/s`**（约为 HBM 4 TB/s 峰值的 `67%–73%`），占总 kernel 耗时仅 **`18%–28%`（`aiv_mte2_ratio = 0.179–0.282`）**，稳态下已被 Cube 完全隐藏；
   - **但仍有两处隐性损耗**：(a) batch-0 冷启动 prologue（约 `8–25 µs`）；(b) AIV 在 `MTE2` 上大 stride 发出的非满缓存行/跨页请求可能与并发的 AIC `MTE2` 争抢 HBM/L2 请求队列。
2. **BF16 非连续输入（`c5_bwd_dw1` 的 `A` 矩阵）的直接风险**：
   - `c5_bwd_dw1` 中 `A` 为 `bf16` 且带 `permute(1, 2, 0)`（行 stride = `56 × 1232 × 2B = 138 KiB`）。当前 V4/V5 为省去 AIV 搬运，让 **AIC `MTE2` 直接用 `asc_copy_gm2l1_nd2nz` 带 138 KiB stride 重复读取 `A`（每个元素复用 ~20×）**！
   - 若实测对照 `A` 原始 stride vs `A.contiguous()` 证实 strided `nd2nz` 存在掉速，最优解不是让 AIC 硬读，而是让本就只占 `17.9%` 时间的 AIV 顺带把 `A` 搬成连续 L2 `ws_a`（AIV 仍 < 30%，完全隐藏，且 AIC 变成 100% 连续 L2 读）。
3. **SIMT Load 替代 AIV `MTE2` 的适用边界**：
   - 使用 SIMT（256 线程合并向量访存）替代 `asc_copy_gm2ub_align` 做大 stride 读确实能规避 MTE2 在超大 stride 下的描述符/突发效率衰减，且如您指出会破坏 `MTE2 -> V -> MTE3` 三级流水（SIMT 与标量/向量通路共享调度，需要显式同步）；
   - **因此工程上最划算的验证顺序**：
     1. 先落地上板 **V5.1 纯硬件零 AIC 屏障 Ring**（直接拿回 `60.9% -> ~83–88%` 的大头）；
     2. 实测 `c5` strided `A` vs contiguous `A` 差值，若差值 > 2% 则开启 AIV 连续化搬运 `ws_a`；
     3. 若 `msprof op` 显示剩余的 `3–5%` 差距来自 AIV/AIC 并发下的 strided `MTE2` HBM/L2 争用或 batch-0 prologue，再针对性引入 **AIV SIMT coalesced strided load**（尤其对 `c6` 的 `1.10 MiB` stride 和 `c1/c3/c4` 的 `276 KiB` stride）做 A/B 实卡消融。

---

## 6. 已就绪的 V5.1 架构设计与当前上板阻塞说明

### 6.1 V5.1：纯硬件零 AIC 屏障持久化 Ring（`backend="v5_ring"`）

为彻底消灭 V4 的 super-batch 屏障与 wave-tail 气泡，我们完成了两版迭代：
- **V5.0（R005c）**：尝试用 GM dcache-bypass 标量轮询（`sync_buf` + `asc_load_dev`/`asc_store_dev`）做无锁 Ring，但在 950DT 实卡首次运行时触发死锁并导致 `device 0` 进入 `health=2` 异常态（频率降至 100 MHz，`npu-guard` 按安全策略拒绝后续任务直至管理员 reset）。
- **V5.1（已合入本算子包 `backend="v5_ring"`）**：彻底移除所有 GM 轮询与 `sync_buf`，设计出**纯硬件零 AIC 屏障 Ring**：
  - **AIC 侧**：单层持久化循环跑完所有 `batch × tiles_per_b` 个 tile（**0 次 AIC 全局屏障、0 次跨 batch MTE2 排空、~100% Wave 效率**），同时启用 R005b 已验证的 `L1_SLOTS=3`、`unit_flag_ctrl=2/3`、`batch_rot`。每当 AIC 推进到下一个 batch，直接发一条非阻塞的 `PIPE_MTE2` `mode=4` 释放信号（`flags 4..4+R-1`），硬件自动保序于该 batch 全部 `MTE2` 读完成之后。
  - **AIV 侧**：采用 `R = 3` 或 `4` 槽单-batch L2 环（`23–64 MiB <= 128 MiB L2`），仅在覆盖旧槽（`b >= R`）前等待配对 AIC 的释放信号并做一次 AIV-only `mode=0` 同步，从而等价保证全部 32 AIC 已读完旧槽；跳过 batch 采用交错 wait/release，使任意硬件标志位在飞计数严格 `<= 1`（支持任意 `batch` 大小永不溢出）。
  - **状态**：V5.1 已在 950DT 上完成 **CPU 端 TileLang → AscendC 完整 lowering 验证（517 行干净 AscendC）**，Pure Cube 引擎（`unit_flag + L1x3 + batch_rot`）已在 R005b 上板实测通过（`88.8%` 冷 / `90.1%` 热 MFU，`rel_l2=3.57e-5`）；完整 V5.1 CV Ring 仅待 `950DT device 0` 恢复 `health=0` 后执行一键上板验收。

### 6.2 当前交付范围与验收边界

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 默认交付后端 `backend="v4"` | **已上板实测通过** | 950DT 1650/1650 MHz，6/6 DensePFFN 调用点 `NaN` 哨兵 + 精度全通，总耗时 `25.506 ms -> 11.918 ms`（`2.14×` 加速，最高单点 `73.0%` MFU） |
| Pure Cube 引擎升级（`UF2/3 + L1x3 + Rot`） | **已上板实测通过** | R005b 实测 `c1_fwd_up` 达 `88.8%` 冷 / `90.1%` 热 MFU，`rel_l2 = 3.57e-5` |
| 零屏障 Ring 后端 `backend="v5_ring"` | **已 CPU Lower，待重置上板** | 100% 纯硬件 `mode=0`/`mode=4` 同步（无 GM 轮询），待 `950DT device 0` 从 `health=2` 复位后运行 `test_dense_pffn_gemm.py --backend v5_ring --bench` 完成实卡验收并切换默认 |
| 端到端 `origin_gemm.py` 前反向测试 | **已内置于测试脚本** | CPU 回退路径已本地验证通过；NPU 全尺寸端到端验证待 `device 0` 复位后执行 |
| 非 16 对齐 / `use_bf16=False` / 空张量 | **已支持安全回退** | 自动切 PyTorch 语义等价路径 |

---

## 7. 复位后下一步优化计划（冲刺 `85%–92%+` MFU）

1. **立即上板验收 V5.1（`--backend v5_ring --bench`）**：预计单此一项即可将 `c1–c6` 综合 MFU 从 `60.9%` 推升至 **`~82%–88%`**（`c6` 预计从 `48.8%` 跃升至 `80%+`，`c3` 预计从 `54.0%` 跃升至 `78%+`，`c2` 通过单-batch `R=4` 把 L2 占用从 `131 MiB` 压回 `23 MiB` 预计跃升至 `80%+`）。
2. **`c5_bwd_dw1` Strided BF16 `A` 专项诊断**：对比原始 strided `A` 与 `A.contiguous()`，若存在掉速则把 strided BF16 `A` 纳入 AIV L2 连续化流水。
3. **`msprof op` Memory/L2/PipeTimeline 采集与 SIMT Strided Load 消融**：在 V5.1 基础上采集 L2 命中率、HBM 流量与逐周期流水线，针对剩余的 AIV/AIC 访存争用测试 **AIV SIMT coalesced strided load** 与 L2 缓存控制策略微调，冲击 **`90%+` 综合 MFU**。
