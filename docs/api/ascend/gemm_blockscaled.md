# T.gemm_blockscaled

## OP 概述

`T.gemm_blockscaled` 计算带分组缩放系数的矩阵乘法，将结果写入 L0C 中的 C，并使用 FP32 累加。A、B 各沿 K 方向每 32 个元素使用一个 E8M0 scale，分别由 SFA、SFB 提供。

计算时先对输入应用对应分组的 scale，再做矩阵乘法；`transpose_A/B` 确定 A、B 的数学转置方式。

- `clear_accum=True`：从零开始计算本次缩放矩阵乘积。
- `clear_accum=False`：将本次缩放矩阵乘积累加到 C 的原值上。

M 是结果行数，N 是结果列数，K 是求和长度。SFA 的行对应 M，SFB 的行对应 N；两侧系数相互独立。

支持两种入口：L1 入口传入 L1 中的数据和 scale，由内部完成计算前的搬运；L0 入口传入 L0A/L0B 数据，以及与数据配套并已装载的 SF 区域。结果通过 `T.copy` 搬到 GM 或 UB。

## OP 规格

### 接口形式

```python
T.gemm_blockscaled(
    A,
    B,
    C,
    SFA,
    SFB,
    transpose_A=False,
    transpose_B=False,
    clear_accum=False,
    unit_flag_ctrl=None,
)
```

### 参数说明

| 参数 | 含义 | 默认值和使用条件 |
| --- | --- | --- |
| A | 左侧低精度输入矩阵，可以传缓冲区或矩阵选区。 | 必填。位于 L1 或 L0A，与 B 使用相同输入类型。 |
| B | 右侧低精度输入矩阵，可以传缓冲区或矩阵选区。 | 必填。与 A 同走 L1 入口，或在 A 位于 L0A 时使用 L0B。 |
| C | 本次结果及累加区域，逻辑形状为 `[M,N]`。 | 必填。位于 L0C，使用 `float32`。 |
| SFA | A 的分组缩放系数。每行沿 K 每 32 个元素对应一个系数。 | 必填。L1 入口传 L1 scale；L0 入口传与 A 绑定的 SF 区域，并在调用前装载系数。 |
| SFB | B 的分组缩放系数。每行沿 K 每 32 个元素对应一个系数。 | 必填。L1 入口传 L1 scale；L0 入口传与 B 绑定的 SF 区域，并在调用前装载系数。 |
| transpose_A | 是否将 A 转置后参与计算。 | 默认 False；不自动转置或调整 SFA。 |
| transpose_B | 是否将 B 转置后参与计算。 | 默认 False；L1 入口必须显式设置为 True。不自动转置或调整 SFB。 |
| clear_accum | 决定从零开始计算，还是保留 C 原值继续累加；可使用布尔值或可编译的布尔表达式。 | 默认 False。首段没有预先初始化 C 时使用 True，后续累加段使用 False。 |
| unit_flag_ctrl | 控制计算和结果搬出的协同执行，使已经生成的部分结果可以开始搬出。 | 默认 None，等同于 0，表示关闭；2 用于连续操作中间阶段，3 用于结束阶段。需要与对应结果搬出配套设置。 |

本接口不提供 `policy`、`annotations` 参数。计算 M、N、K 从矩阵形状、选区及转置参数确定；scale 的 K 分组粒度固定为 32。

### 数据类型支持

A、B 使用同一种输入类型，C 及累加类型均为 `float32`。“待确认”项不计入已确认支持范围。

| A、B 输入类型 | TileLang 类型名 | L1 入口 | L0 入口 |
| --- | --- | --- | --- |
| FP8 E4M3 | `float8_e4m3fn` | 支持 | 支持 |
| 打包 FP4 E2M1 | `float4_e2m1fn` | 支持 | 支持 |
| FP8 E5M2 | `float8_e5m2` | 当前未接入 | 待确认 |
| FP16、BF16、FP32、INT8 | — | 不支持作为 scaled GEMM 输入 | 不支持作为 scaled GEMM 输入 |
| A、B 使用不同类型 | — | 不支持 | 不支持 |

FP4 的 shape 表示逻辑元素数量，GM 中每字节打包两个 FP4 元素，可通过 `T.view` 从字节载体构造逻辑 FP4 视图。

scale 使用 E8M0 编码，以单通道 8 位或 16 位元素存储。常用 `uint8` 每元素存一个系数；默认 `uint16` 每元素按位装两个连续系数。载体中的整数值用于保存编码，不能按普通 FP16、FP32 或 E4M3 的数值含义准备。

### 规格约束

#### 输入位置与转置

下面是传给 GEMM 的 A、B 形状，尚未应用转置参数。所有组合的结果形状都为 `[M,N]`。

| 组合 | transpose_A | transpose_B | A 传入形状 | B 传入形状 | L1 入口 | L0 入口 |
| --- | --- | --- | --- | --- | --- | --- |
| NN | False | False | `[M,K]` | `[K,N]` | 不支持直接调用 | 支持 |
| NT | False | True | `[M,K]` | `[N,K]` | 支持 | 支持 |
| TN | True | False | `[K,M]` | `[K,N]` | 不支持直接调用 | 支持 |
| TT | True | True | `[K,M]` | `[N,K]` | 不支持直接调用 | 支持 |

A、B 必须成对位于 L1，或分别位于 L0A、L0B。转置参数只作用于 A、B，SFA、SFB 的行数仍分别按数学上的 M、N 准备；调用方按本次计算的尺寸和存储方式准备相应 scale 形状。

#### 尺寸、静态与运行时范围

以下尺寸条件须同时满足缓存容量、布局和搬运要求，各维最大值不能直接拼成一个计算组合。

| 项目 | 约束 |
| --- | --- |
| 单次 L1 GEMM 的 M、N | 必须为 16 的正整数倍，且不超过 4080；具体组合还须满足缓存容量限制。 |
| 单次 L1 GEMM 的 K | 必须是编译时确定的正整数，且为 64 的倍数。 |
| 单次 L1 GEMM 的尺寸确定时机 | M、N、K 均须在编译时确定。 |
| 单次 L0 GEMM 的有效 M、N | 均为正数，且不超过 4095；有效 M、N 可以不对齐，存储与搬运仍须满足对应类型要求。 |
| 单次 L0 GEMM 的 K | 执行 K 必须为 64 的正整数倍，且不超过 4032；L0A/L0B 分配的 K 长度也须为 64 的倍数。 |
| L0 分配与有效范围 | 缓冲区分配大小须在编译时确定；有效 M、N、K 可以使用运行时表达式指定。K 的 64 对齐及 scale 覆盖范围必须能在编译时证明。 |
| 整张矩阵 | 通过外层分块和循环覆盖；单次计算尺寸范围不作为整张矩阵的统一上限。 |
| 高维缓冲区 | 支持选取单个矩阵计算；批量通过外层循环完成，scale 与所选数据组对应。 |
| 零尺寸 | 不在本规格范围内。 |
| 单行矩阵 | 未来支持。 |

#### 布局、选区与搬运

- 使用逻辑 shape 和索引，主数据的 NZ/ZN 物理布局及转换由编译器后端处理。
- 传给 GEMM 的 L0 数据及 scale 矩阵选区，行、列索引均须从 0 开始，可以选择从零开始的较小有效区域。
- 转置搬运须满足对应输入类型的对齐要求，需要时补齐存储空间和搬运范围。
- 支持分多次将数据搬入 L0 的不同位置。每次满足分形对齐，使用与最终计算块匹配的 `l0_pitch`。
- 更换转置方式时，输入形状、搬运方式及 scale 形状需与本次计算匹配。转置后的 L0 数据需要时通过 `sf_shape` 显式指定对应 scale 形状。
- 裁剪搬出结果时，保持实际计算结果的间隔，需要时使用匹配的 `l0_pitch`。

#### scale 的分组、覆盖与配套

下表中的 `Kexec` 为本次实际执行 K，包含尾块补齐部分。

| 项目 | 规格和约束 |
| --- | --- |
| 分组粒度 | A、B 各沿 K 每 32 个元素对应一个 E8M0 scale，两侧系数相互独立。 |
| 8 位逻辑形状 | 每元素一个 scale；SFA `[M,Kexec/32]`，SFB `[N,Kexec/32]`。 |
| 16 位逻辑形状 | 每元素按位装两个连续 scale；SFA `[M,Kexec/64]`，SFB `[N,Kexec/64]`。 |
| scale 覆盖范围 | SFA、SFB 行数分别覆盖有效 M、N；K 方向的系数数量与本次实际执行 K 匹配。 |
| L1 两侧一致性 | SFA、SFB 使用相同的存储类型和 K 方向物理存储步距；行数分别对应 M、N，可以不同。 |
| L0 数据与 scale 对应 | SFA 绑定 A，SFB 绑定 B。使用多份缓冲区时，每侧数据与对应 scale 选择相同缓冲份。 |
| 执行范围变化 | 有 K 尾块时按补齐后的执行 K 准备系数；缩小执行 K 时，搬入 L0 的 scale 区域也需与本次执行范围匹配。 |

scale 在 L1 使用专门的 SF_K 布局。下表是 GM 中的存放及搬入方式，其中转置设置是 **`T.copy` 的 `transpose` 参数**，与 GEMM 的 `transpose_A/B` 分开。

| scale 存储类型 | GM 中 SFA 的形状 | GM 中 SFB 的形状 | GM→L1 搬入方式 | 当前支持情况 |
| --- | --- | --- | --- | --- |
| `uint8` | `[M,Kexec/32]` | `[N,Kexec/32]` | 按行优先存放，不设置 transpose。 | 已验证通过。 |
| `uint16`，每元素装两个 E8M0 字节 | `[Kexec/64,M]` | `[Kexec/64,N]` | 设置 `transpose=True`。 | 已验证通过。 |
| `uint8` | `[Kexec/32,M]` | `[Kexec/32,N]` | 设置 `transpose=True`。 | 当前数值异常，暂不作为已确认支持项。 |

两种入口均须满足以上 GM→L1 条件。显式 L0 入口还需将 L1 中的 scale 搬入对应 SF 区域，再调用 GEMM。

#### D 维与 K 维尾块

M、N 统称 D 维。采用补零方案时，先在 L1 完成补零，再将包含补零部分的数据搬入 L0；仅扩大分配空间，不代表已经补零。

| 入口 | 有效 M=125 的处理方式 | 结果输出 |
| --- | --- | --- |
| L1 | 按 M=128 准备数据，在 L1 对额外行补零，并准备覆盖 128 行的 SFA。 | 按 128 行计算，只输出前 125 行。 |
| L0 | 可以指定有效 M=125，SFA 覆盖对应行；存储和搬运仍按要求补齐。也可以补零后按 M=128 计算。 | 只计算并输出 125 行，或将完整块结果裁剪为 125 行。 |

N 尾块同理，准备 B 对应的数据及 SFB，输出所需的前 125 列。

| 入口 | 有效 K=125 的处理方式 |
| --- | --- |
| L1 | 执行 K 取合法的 64 倍数，例如 128。A、B 额外的 3 个 K 位置在 L1 置零，scale 按 K=128 准备。 |
| L0 | 执行 K 同样取 128；将 L1 中补零后的数据搬入 L0，配套覆盖 K=128 的 scale。不能直接按 K=125 执行。 |

K 是求和维度，额外 K 的乘积需要用补零排除，不能通过输出裁剪消除。每个 scale 对应连续 32 个 K 元素；一个分组同时含有效数据与补零数据时，保留有效数据原来的系数。某一完整分组全是补零时，该输入分组对应的系数可设为 1，仍须按 E8M0 正确编码。

#### 输出与 Fixpipe 随路转换

| 输出路径 | 支持范围 |
| --- | --- |
| L0C `float32` → GM | 支持 `float32` 同类型输出，以及 `float16`、`bfloat16` 随路类型转换。 |
| L0C `float32` → 单目的 UB | 支持输出为 `float16`、`bfloat16`；UB 行间隔与实际写入范围须满足 32B 条件。 |

目标缓冲区类型决定输出类型。FP32→FP16/BF16 转换在 `T.copy` 搬出时完成，无需额外量化系数，C 的累加类型保持 FP32。

UB 是芯片上的临时缓冲区。相邻两行起点间隔须为 32 字节的倍数，并为补齐后的写入范围预留空间。例如 19 个 FP16/BF16 元素占 38B，每行至少预留 64B。单目的 UB 的后续处理和写回由接收方执行，可使用 `T.MixedKernel(..., sids=1)` 限定到向量核 0。

#### 连续计算与执行配置

- 首段没有预先初始化 C 时使用 `clear_accum=True`，后续段用 False 累加；开始新计算时重新清零。每段使用与该段数据对应的 scale。
- 较小 M/N，例如 M=N=16 或 32，连续累加的支持情况仍需按具体尺寸组合确认。
- 使用 `unit_flag_ctrl` 时，GEMM 和对应结果搬出配套设置，连续操作的中间阶段与结束阶段按参数说明区分。
- 开启协同执行并输出普通二维矩阵时，使用 `T.set_mmad_direction("n")`。方向表示一次 GEMM 内部生成结果块的先后顺序，`"n"` 为结果列方向 N 优先，不改变转置或结果形状。
- 当前接口没有 Bias 输入参数。

## 示例

两个示例均以 FP8 E4M3 输入，计算有效的 `17×24` 与 `24×19` 矩阵乘法，执行 K 补齐为 64。B 按 `[N,K]` 存放，设置 `transpose_B=True`。可使用 `tilelang.compile(设备函数, target="ascend", out_idx=4)` 编译。

每行 K=64 需要两个 E8M0 scale，打包为一个 `uint16` 元素。SA、SB 在 GM 中按 `[1,32]` 准备，转置搬入 L1 后为 `[32,1]`。有效数据所在分组使用对应的原始系数；完整补零分组可用正确编码的系数 1。E8M0 的系数 1 编码为 `0x7F`，两个系数 1 打包为 `0x7F7F`。

### L1 输入：内部搬运数据与 scale

```python
from tilelang.ascend import language as T


@T.prim_func
def scaled_gemm_l1(
    A: T.Tensor((17, 24), "float8_e4m3fn"),
    B: T.Tensor((19, 24), "float8_e4m3fn"),
    SA: T.Tensor((1, 32), "uint16"),
    SB: T.Tensor((1, 32), "uint16"),
    C: T.Tensor((17, 19), "float32"),
):
    with T.Kernel(1):
        a = T.alloc_l1((32, 64), "float8_e4m3fn")
        b = T.alloc_l1((32, 64), "float8_e4m3fn")
        sa = T.alloc_l1((32, 1), "uint16")
        sb = T.alloc_l1((32, 1), "uint16")
        acc = T.alloc_l0c((32, 32), "float32")

        # 准备完整输入块：只读真实数据，缺少的部分在 L1 补零。
        T.copy(A[:32, :64], a)
        T.copy(B[:32, :64], b)

        # 搬入已准备好的系数，调整存放方向。
        T.copy(SA, sa, transpose=True)
        T.copy(SB, sb, transpose=True)

        # 内部完成数据、系数的 L1→L0 搬运。
        # 计算 M=32、N=32、K=64。
        T.gemm_blockscaled(
            a,
            b,
            acc,
            sa,
            sb,
            transpose_B=True,
            clear_accum=True,
        )

        # 只写回有效的 17×19。
        T.copy(acc[:17, :19], C)
```

### L0 输入：显式搬入数据和 scale 后计算有效 M/N

```python
from tilelang.ascend import language as T


@T.prim_func
def scaled_gemm_l0(
    A: T.Tensor((17, 24), "float8_e4m3fn"),
    B: T.Tensor((19, 24), "float8_e4m3fn"),
    SA: T.Tensor((1, 32), "uint16"),
    SB: T.Tensor((1, 32), "uint16"),
    C: T.Tensor((17, 19), "float32"),
):
    with T.Kernel(1):
        a = T.alloc_l1((32, 64), "float8_e4m3fn")
        b = T.alloc_l1((32, 64), "float8_e4m3fn")
        sa = T.alloc_l1((32, 1), "uint16")
        sb = T.alloc_l1((32, 1), "uint16")

        a0 = T.alloc_l0a((32, 64), "float8_e4m3fn")
        b0 = T.alloc_l0b((32, 64), "float8_e4m3fn")
        sa0 = T.alloc_l0a_sf(a0)
        sb0 = T.alloc_l0b_sf(b0)
        acc = T.alloc_l0c((32, 32), "float32")

        # 与 L1 入口一样，先在 L1 准备好补零后的数据。
        T.copy(A[:32, :64], a)
        T.copy(B[:32, :64], b)
        T.copy(SA, sa, transpose=True)
        T.copy(SB, sb, transpose=True)

        # 主数据和对应系数分别搬到计算区。
        T.copy(a, a0)
        T.copy(b, b0)
        T.copy(sa, sa0)
        T.copy(sb, sb0)

        # M/N 取有效范围，K 保持补齐后的 64。
        T.gemm_blockscaled(
            a0[:17, :64],
            b0[:19, :64],
            acc[:17, :19],
            sa0[:17, :1],
            sb0[:19, :1],
            transpose_B=True,
            clear_accum=True,
        )

        T.copy(acc[:17, :19], C)
```
