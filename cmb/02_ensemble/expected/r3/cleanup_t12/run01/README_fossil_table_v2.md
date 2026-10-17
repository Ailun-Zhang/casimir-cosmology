# fossil_table_v2.csv 说明（T12 / C158，W14；代理 cleanup-t12，2026-10-14）

**用途。** 本表替代 `04_calculations/results/20261013_argon_codex_intake/cmb_ensemble/run01/results/fossil_table.csv`（下称"旧表"；只读，未改）。旧表的 `*_prof` / `*_sprof` 列是 D117 已废弃的三点抛物线细化值（R15 报告 V2b；`R15_CORRECTION_ADDENDUM_C145_C147_CN.md` 更正 1）。v2 把这两组列换成 **cube 池极小（pool-min）**：对 19 个实际参考 cube s'∈{0.985, 0.9925, 1.00, 1.015, …, 1.20, 1.22}（全部 s'≥0.985）取 D 的极小值。这就是 D117 §0.3/§4 主结果和 `S3_postcheck.json`（`*_grid` 变体）所用的定义。

本表只做读表后处理：输入是已存的 `S2_profiles.json`（每个化石、每个读取量的 19 点体积 profile），并用 `kl_table.json` 交叉核对。**没有重算协方差或 KL。**

## 1. 文件

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `fossil_table_v2.csv` | 55 行 × 23 列（定义见 §2） | `2350e589fdc7cd5c0cca0de0ece5c71f085ba34208472ae9f8cd507d5ce1ac0d` |
| `fossil_table_v2_diff.csv` | 逐（化石, 读取量）共 220 对：旧抛物线值与 v2 pool-min 的对照、相对差、s' 差、标记 | `2026a8c4b9780c51ca92675cd851239916eef4718b484b7f2f1bfe714d9eb1bf` |
| `fossil_table_v2_checks.json` | 全部核验数字、Brent 对照、集成复现、输入与输出 SHA-256 | （含上两者的哈希） |
| 生成脚本 `04_calculations/code/20261014_claude_v12_r3/cleanup_t12/make_fossil_table_v2.py` | | **`b8bea5a85059bef1b166c85031a9723e1bb7f42087608882976388966ed91cc1`** |

输入（只读，均在 `cmb_ensemble/run01/results/`）：

| 输入 | SHA-256 |
|---|---|
| `fossil_table.csv`（旧表） | `fdf177306c53d46c4439e6a760955aa0213d9c97c8e07ad0076b32d8b8cf983a` |
| `S2_profiles.json` | `4713fe567c753b06a9ae8606abd3583766e1064957b5260ef7638b54f1cfed05` |
| `kl_table.json` | `21eb6ef468247bdb0a528cb1b99d72dd320e4d039eec692e27ea5e4e6d74a36e` |
| `S2_brent.json` | `f5fc07fac9ef955856a23a0bedc01e78c300c3e09ed696638f758cb7be7547a9` |
| `S3_postcheck.json` | `bec6ebe34daf24ada5d7c67b21a6872e62a1451411a6007c1504fd46f2ef63ef` |

## 2. 列定义（v2）

读取量 K∈{TE30, TE40, TE80, T80}：TE30/TE40/TE80 指温度+E 模、ℓ≤30/40/80；T80 只用温度、ℓ≤80。D 是化石对参考立方环的 KL 散度，读取阈值为 D=4.5（D117 §1）。

| 列 | 定义 | 与旧表相比 |
|---|---|---|
| `name` | 化石标识（cache 文件名主干） | 相同 |
| `A` | E 平面形变振幅，以 σ_E=0.1041 为单位（ensemble 中服从 Rayleigh(1)） | 相同 |
| `lab` | obl / tri / pro，即扁形、三轴、长形（θ=−π/6、0、+π/6） | 相同 |
| `theta` | D3 基本域 [−π/6, π/6] 内的角（rad） | 相同 |
| `xi` | sin3θ | 相同 |
| `s` | 化石的 L0 | 相同 |
| `u` | 最短边 s·e^{min β_ii} | 相同 |
| `s_low` | 该形状无圆的最小 L0，即 0.985/e^{min β_ii} | 相同 |
| `K_fixed` | 等体积参考立方环（s'=s）的 D | 相同（逐字复制；4 个空格照旧保留） |
| `K_prof` | **pool-min**：min over 19 cubes of D(化石‖cube s') | **改**（旧表为三点抛物线顶点值） |
| `K_sprof` | 取到 pool-min 的 cube s'（网格点；网格间距 0.0075–0.02） | **改**（旧表为抛物线顶点的 s'） |
| `TE80_brent`, `TE80_sbrent` | `S2_brent.json` 的有夹逼 Brent 连续极小及其 s'*；只有 3 个边缘化石有值，其余为空 | 新增 |
| `prof_source` | 常数 `poolmin19`，标明 prof 列的定义 | 新增 |

数值格式与旧表相同：D 取 6 位有效数字，s' 取 5 位小数（Brent 的 s'* 取 7 位小数）。

## 3. 与旧表的逐列差异

- 前 8 列（`name` … `s_low`）以及 4 个 `K_fixed` 列：逐字相同。
- `K_prof` / `K_sprof`：

| 读取量 | prof 改动格数 | sprof 改动格数 | 旧值低估 >1% | 旧值低估 ≥3% | 最大低估（化石 A, 形, L0） |
|---|---|---|---|---|---|
| TE30 | 55/55 | 55/55 | 22 | 13 | −13.3%（1, 长, 1.06） |
| TE40 | 53/55 | 55/55 | 20 | 12 | −48.7%（0.5, 长, 1.045） |
| TE80 | 54/55 | 55/55 | 18 | 9 | −97.4%（0.5, 长, 1.045）：0.144313 → 5.56072 |
| T80 | 52/55 | 53/55 | 22 | 11 | −25.4%（0.5, 长, 1.045） |
| 合计 | 214/220 | 218/220 | **82** | 45 | |

- S2 中 218 对标为 `parabolic`，2 对标为 `boundary`。sprof 未变的 2 格就是这 2 个边界极小。prof 未变的 6 格是抛物线值与 pool-min 在 6 位有效数字上相同的情形。
- 旧值在所有 220 对上都 ≤ pool-min（只差打印舍入）。唯一"旧值略高"的一对是 (√2, 扁, 1.12, T80)：两表字符串相同（15.1027），相对差 +1×10⁻⁶，在 6 位有效数字的舍入（≤3.3×10⁻⁶）以内。
- 旧 sprof 与 v2 sprof 的最大差为 0.00749，小于半个网格步长 0.0075。这说明抛物线顶点总在极小点的相邻半格内。

## 4. 核验（全部通过）

1. **两条独立路径给出同一 pool-min。** 一条取 S2_profiles 的曲线，另一条取 kl_table.json（55 化石 × 19 cube）中 s'≥0.985 的条目。两者的极小值相对差为 0，argmin 相同。逐点比较 S2 曲线与 kl_table 也完全一致（相对差 0）。每个化石都有 19 个 cube，kl_table 中没有 s'<0.985 的条目。
2. **旧表的身份。**
   - 旧表的几何列与 S2 全精度值相差 ≤4.9×10⁻⁶，fixed 列 ≤4.6×10⁻⁶，prof 列与 S2 的抛物线 `D_prof` ≤4.5×10⁻⁶，都只是打印舍入。
   - 因此旧表的 prof 列就是 S2 的抛物线值。
3. **数据处理不等式（DPI：TE30≤TE40≤TE80，T80≤TE80）。**
   - 旧 prof 只在 1 个化石 (0.5, 长, 1.045) 上违例，且同时违反三条不等式（TE30 2.93 > TE40 2.11 > TE80 0.144，T80 2.56 > TE80）。
   - v2 prof 没有违例，fixed 列也没有。
   - pool-min 必然满足 DPI：kl_table 中每个 cube 都满足 DPI（R15：0 违例），取极小保持不等式。
4. **TE80 标记。** 按 `postcheck_ens.py` 的判据（抛物线值 <0.97×pool-min，或 TE80 低于 TE40）得到 9 个化石，与 `S3_postcheck.json` 的 flags 逐一相同。R15 也给出 9 个。
5. **边界极小。** 共 2 个，都在 T80：(1, 长, 1.03) D=23.6673，(2, 长, 1.075) D=26.3132，都落在 s'=0.985。它们是近圆约束下合法的边界极小，D 远大于 4.5，与读取边无关（R15 §2）。
6. **Brent。** 3 个都属于 TE80，都是内点极小（`ok`/`interior` 为真）：

| 化石 | s'* | D_Brent | pool-min | 低于 pool-min |
|---|---|---|---|---|
| (2, 长, 1.12) | 1.0432214 | 5.03362 | 5.06138 | 0.548% |
| (1, 扁, 1.12) | 1.0770952 | 3.80404 | 3.80587 | 0.048% |
| (√2, 三轴, 1.14)，代表点 `fossil_s1.14000` | 1.0770640 | 4.44377 | 4.44533 | 0.035% |

   这与 D117 §0.3 的"连续极小比池极小低 0.04–0.55%"一致。D117 主结果用的是 pool-min，不是 Brent。
7. **集成复现。** 从写出的 v2 csv 读回，套用 `postcheck_ens.py` 原样复制的 edge_b/span_eval/curves（同一种子 20261013，2×10⁶ 样本，'mid' 模式），再与 S3_postcheck 对照。17 个 L0 点上 |ΔP_joint| 与 |ΔP_cond| 都为 0。

| 来源列 → 对照变体 | P_joint 峰（L0） | ∫P_joint dlnL0 |
|---|---|---|
| v2 `TE80_prof` → `TE80_grid` | 0.34398（1.075）= 同 | 0.042232 = 同 |
| v2 `TE40_prof` → `TE40_grid` | 0.30535（1.075）= 同 | 0.037677 = 同 |
| v2 `TE30_prof` → `TE30_grid` | 0.26418（1.0775）= 同 | 0.032513 = 同 |
| v2 `T80_prof` → `T80_grid` | 0.28562（1.0775）= 同 | 0.035216 = 同 |
| v2 `TE80_fixed` → `TE80_fixed` | 0.35970（1.0775）= 同 | 0.044197 = 同 |
| 旧表 `TE80_prof` → `TE80_parab` | 0.33830（1.0825）= 同 | 0.041263 = 同 |

   所以 v2 逐位复现 D117 主结果（0.344@1.075，测度 0.0422），旧表对应的是已废弃的抛物线变体。R15 的 0.350@1.076 用的是它自己的 u 空间插值（方法 A，节点弦），与生产者 'mid' 插值的差别属于插值，不是数据问题（R15 §5.1）。

## 5. 使用说明

- 引用 profiled D 时用 v2 的 `K_prof`（或 S3_postcheck），旧表的 prof/sprof 列一律视为废弃。
- pool-min 是离散 19 个 cube 上的极小，所以是连续 profile 极小的上界。3 个 Brent 点表明差距 ≤0.55%。需要连续极小时用 `TE80_brent`（只有 3 个化石）。
- `K_sprof` 只到网格分辨率，不能当作连续最优体积。
- D117 §11 把旧表列为 S3 结果文件；该处应读作本表（见 `04_calculations/derivations/20261014_claude_v12_r3/cleanup_t12/D117_D119_WORDING_ADDENDUM_R3_CN.md` W1）。

## 6. 复现

```
D:\Python\envs\py312-main\python.exe -I D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\04_calculations\code\20261014_claude_v12_r3\cleanup_t12\make_fossil_table_v2.py
```

单进程，BLAS 单线程，耗时数秒。脚本只读输入，写出 §1 的三个文件。
