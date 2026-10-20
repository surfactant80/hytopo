# hytopo

**hytopo** 是一个用于分析气体水合物分子动力学轨迹的程序。它按局部有序度
（HOP）对每个水分子分类，识别客体分子周围的水合物笼（标准笼、畸变笼与各种
缺陷笼），跨帧追踪这些笼，依据**笼心空间关联**判定水合物相，检测成核，并生成
一页式汇总报告。

包名与命令同名：

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" --find-cages
```

轨迹支持完全来自 MDAnalysis：**它能读的格式都能用，没有格式白名单**。
只需把 `-top` 与 `-traj` 指向对应文件——GROMACS（`.tpr`/`.xtc`/`.trr`）、
Amber、NAMD/CHARMM（DCD）、LAMMPS、`.gro`、`.pdb`、`.dcd`、`.trj` 等，
都走同一套 MDAnalysis 读取器。周期性扩展 XYZ（`.xyz`、`.extxyz`、`.etxyz`）
由 hytopo 自带的轻量读取器解析，因为这种格式没有拓扑信息。

---

## 目录

1. [功能](#功能)
2. [安装](#安装)
3. [依赖](#依赖)
4. [快速上手](#快速上手)
5. [输入与选择器](#输入与选择器)
6. [命令行选项](#命令行选项)
7. [输出结构](#输出结构)
8. [各分析器的原理](#各分析器的原理)
9. [性能](#性能)
10. [并行、断点续跑与长跑](#并行断点续跑与长跑)
11. [验证与基准](#验证与基准)
12. [Python API](#python-api)
13. [常见问题](#常见问题)
14. [引用与许可](#引用与许可)

---

## 功能

### 1. 水合物有序度（HOP）

每个水氧用局部环境描述：四面体度 `qtet`、交错/重叠氢键计数 `S`/`E`、配位数，
经过邻域平滑后映射到连续的六态概率谱：

| 标签 | 含义 |
|---|---|
| `liquid` | 类体相液体 |
| `ice_Ih`、`ice_Ic` | 六方冰 / 立方冰 |
| `hydrate_crystal_like` | 结晶水合物 |
| `hydrate_precursor` | 无定形水合物 / 前驱体 |
| `interface_or_uncertain` | 界面或不确定 |

逐帧统计量（`H_total`、`H_crys`、`H_pre`、`Hyd_order`、`Ice`、`Delta`、
`Nmax_hyd`、`Nmax_ice`）写入主时间序列；逐水标签可写成文本表，也可写成多模型
PDB（标签放在 B-factor 列）供 VMD/PyMOL 着色。

### 2. 笼识别

以客体为中心，从局部水氧网络中识别笼：

* 每帧对候选水分子子图**只做一次**无弦环感知（4–6 元环，Numba DFS），每个环
  通过平面性筛选；
* 朝向客体的环按打分排序，用**位掩码状态 + Numba kernel 的 beam search**
  逐面装配成闭合多面体，用欧拉示性数、共边关系与连通性校验；
* 识别 13 种标准拓扑（`5^12`、`5^12 6^2`、`5^12 6^4`、`4^3 5^6 6^3`、
  `5^12 6^8` 等），以及半笼、部分笼、初期笼、开口笼、畸变笼；
* HOP 得到的候选掩码会跳过不可能属于笼的水分子，这是大体系仍然可算的关键。

### 3. 客体分子识别

客体是**分子**而不是原子：

* 有拓扑（`.tpr`、`.gro`、`.pdb`）时，`-guest` 选中的原子按残基分组；
* 无拓扑（扩展 XYZ）时，按共价半径成键判断：`d <= 1.15 (r_i + r_j)`（系数可调），
  使用考虑周期边界的近邻列表加速；
* 每个分子给出一个周期性正确的中心，所以全原子 CH4 是 1 个客体、CO2 也是 1 个。

只选中心原子（`-guest "resname CH4 and name C"`）时每个客体只有一个原子，
效果与联合原子模型一致。

### 4. 笼追踪

相邻帧的笼按水分子集合的 Jaccard 重叠、客体身份与类型连续性匹配，并容忍少量
漏检。输出包括：

* 每个轨迹的笼寿命；
* 笼型跃迁事件及其汇总；
* 每个笼的客体占据（空 / 单占据 / 多占据）；
* 笼形品质：笼心到顶点距离的均值与标准差、球形度 `1 - std/mean`、
  顶点协方差各向异性；
* 客体逃逸（笼消失）与重捕获（同一客体重新成笼）事件。

### 5. 基于笼心空间关联的相判定

两个笼共享一个**经过几何校验**的面时才连边：至少 3 个公共水氧、两笼心位于该面
两侧、且 `|d(笼心,笼心) - (d_a + d_b)|` 在容差内。该网络的连通分量即**晶畴**。
相（`sI`、`sII`、`sH`、`mixed`）由最大晶畴的组成决定，并用晶格相干性缩放，
再与 `hytopo/ideal_crystal/` 中的理想结构交叉校验：

* 首次使用时从 `sI.pdb`、`sII.pdb`、`sH.pdb` 提取参考笼心距签名并缓存；
* 实测的面共享距离分布与各参考按**丰度加权**比较，只有几个 `5^12 6^4` 的体系
  不会被误判成 `sII`；
* 同时给出晶畴尺寸（笼数）、晶畴半径（Å）以及最大晶畴之外的笼占比
  （界面 + 无定形）。

想退回“纯组成比”判相可加 `--no-ideal-reference`。

### 6. 成核

从最大类水合物团簇 `Nmax_hyd` 判定成核（固定阈值 / 自适应基线 / 带生长的承诺判据），
输出汇总文件与逐帧状态表。

### 7. 路径网络

把笼型跃迁聚合成有向加权网络，绘制论文风格的分栏路径图（液体 → 部分/开口 →
半笼 → 过渡 → 无定形 → 结晶），右侧给出最强跃迁排名，并输出机器可读的节点/边表。

### 8. 一页式报告与分类输出

每次运行可生成自包含的 `report.html`（英文，图片内嵌）和 `README.txt` 清单。
所有表格为**制表符分隔**，带 `#` 注释行与一行表头，按编号目录分类存放。

### 9. 工程能力

* 默认全核并行（识别 CPU 亲和性、cgroup 配额、SLURM/PBS/SGE 变量），
  worker 数不超过待分析帧数；
* 断点续跑（`--checkpoint-every`、`--resume`），续跑结果与一次跑完**逐字节一致**；
* 单行进度条，显示速率与 ETA；
* Numba 内核 + 纯 Python 回退（`--no-numba`）；
* 回归基准：给单帧结果打指纹，内核改动若改变结果会直接报错。

---

## 安装

### 从源码（推荐）

```bash
git clone <仓库地址>
cd hytopo
pip install .
```

开发模式（可编辑安装，含 pytest）：

```bash
pip install -e ".[dev]"
```

### 从 wheel

```bash
pip install hytopo-1.0.0-py3-none-any.whl
```

### 用 conda

```bash
conda create -n hytopo python=3.11
conda activate hytopo
conda install -c conda-forge numpy scipy mdanalysis numba matplotlib networkx
pip install .
```

安装后即可使用 `hytopo` 命令。不想安装时也可以直接：

```bash
python -m hytopo -h
```

> 注意：如果从旧版本（包名 `HyTopo`）升级，请先删除旧的 `__pycache__` 目录，
> 因为 numba 的磁盘缓存记录了旧模块名。

---

## 依赖

| 包 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.10 | |
| numpy | ≥ 1.22 | 数组与线性代数 |
| scipy | ≥ 1.8 | 球谐函数 |
| MDAnalysis | ≥ 2.0 | 轨迹读取、周期距离 |
| numba | ≥ 0.56 | 环感知、笼装配 |
| matplotlib | ≥ 3.5 | 绘图与报告插图 |
| networkx | ≥ 2.8 | 跃迁网络 |

可选：`pytest`（测试）。

没有 numba 也能运行，只是更慢（`--no-numba` 强制使用 Python 回退）。

---

## 快速上手

只做 HOP：

```bash
hytopo -top system.tpr -traj traj.xtc
```

HOP + 笼搜索：

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" --find-cages
```

生产跑（稀疏可视化输出）：

```bash
hytopo -top system.tpr -traj traj.xtc \
  -start 0 -stop 10001 -stride 2 \
  -guest "resname CH4" --find-cages \
  --cage-pdb-stride 100 --hop-pdb-stride 1000 \
  -outdir hydrate_analysis
```

只搜索笼（不做 HOP）：

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --find-cages --skip-hop --no-nucleation
```

周期性扩展 XYZ：

```bash
hytopo -top trajectory.etxyz -water "element O" -oxygen "element O" \
  -guest "element C" --find-cages
```

运行结束后打开 `hydrate_analysis/report.html`。

---

## 输入与选择器

| 选项 | 含义 | 默认 |
|---|---|---|
| `-top`、`--top` | 拓扑/结构文件（必需） | – |
| `-traj`、`--traj` | 轨迹文件 | 只读拓扑 |
| `-water`、`--water-selection` | 水分子选择 | `resname SOL` |
| `-oxygen`、`--oxygen-selection` | 水氧选择 | `name OW` |
| `-guest`、`--guest-selection` | 客体原子/分子选择 | 无 |
| `-start`、`--start` | 起始帧 | 0 |
| `-stop`、`--stop` | 结束帧（不含） | 末尾 |
| `-stride`、`--stride` | 帧间隔 | 1 |

氧原子选择实际是 `(water) and (oxygen)`。扩展 XYZ 的选择语法为 `all`、
`element X`、`name X`、`species X`、`resname X`、`index a:b`。

客体重组（`-guest-grouping`）：`auto`（有拓扑用残基、无拓扑用共价键）、
`residue`、`bonded`、`none`（每个原子一个客体，兼容旧行为）。成键判据为
`d <= scale (r_i + r_j)`，系数用 `--guest-bond-scale`（默认 1.15）。

---

## 命令行选项

完整列表见 `hytopo -h`，按组说明如下。

**分析模式** —— `--find-cages`、`--skip-hop`、`--no-exclude-liquid`、
`--include-ice-for-cage`、`--standard-cages-only`。

**HOP 参数** —— `--oo-cutoff`(3.5)、`--dist-smooth`(0.15)，以及阈值
`--hyd-crys-*`、`--hyd-pre-*`、`--ice-q-min`、`--coord-min/max`、`--pc-hyd`、
`--pc-ice`、`--delta-s`、`--c-e-center`、`--sigma-ce`、`--sigma-count`。

**笼参数** —— `--cage-oo-cutoff`(3.6)、`--guest-radius`(6.2)、
`--face-radius`(7.2)、`--min-ring`/`--max-ring`(4/6)、`--ring-planarity`(1.5 Å)、
`--max-local-cycles`(每帧 5000)、`--max-rings-per-guest`(56)、`--beam-width`(64)、
`--candidate-limit`(20)、`--cage-dedup-overlap`(0.85)，各种部分笼阈值，
候选掩码概率（`--liquid-exclude-prob`、`--hyd-candidate-prob`、
`--ice-candidate-prob`），以及 `--ring-backend {auto,python,numba}`。

**追踪** —— `--no-tracking`、`--tracking-gap`、`--tracking-water-overlap`、
`--tracking-same-guest-overlap`、`--tracking-dedup-overlap`、
`--occupancy-radius`、`--occupancy-center`。

**相判定** —— `--no-phase-analysis`、`--phase-min-cages`、
`--phase-min-confidence`、`--phase-transition-persistence`、
`--phase-mixed-margin`、`--phase-lattice-cutoff`、`--phase-lattice-tolerance`、
`--phase-min-coherence`、`--no-ideal-reference`、`--phase-reference-weight`。

**成核** —— `--no-nucleation`、`--nucleation-method
{fixed,adaptive,commitment}`、`--nucleation-min-size`、
`--nucleation-persist-frames`、`--nucleation-allow-gaps`、
`--nucleation-baseline-frames`、`--nucleation-baseline-sigma`、
`--nucleation-adaptive-min-size`、`--nucleation-min-growth`、
`--nucleation-min-slope`。

**输出** —— `-outdir`、`--timeseries`、`--labels`、`--no-labels`、
`--label-stride`、`--hop-pdb`、`--no-hop-pdb`、`--hop-pdb-stride`、
`--cage-stat`、`--shell-stat`、`--cage-pdb-dir`、`--no-cage-pdb`、
`--cage-pdb-stride`、`--no-plots`、`--no-report`。

**性能** —— `--no-numba`、`--serial`、`--n-procs`、`--chunksize`、
`--checkpoint-every`、`--resume`。

**其他** —— `-q/--quiet`、`--verbose-guests`。

---

## 输出结构

输出按编号目录分类：

```text
hydrate_analysis/
  report.html                     一页式英文汇总（自包含）
  README.txt                      文件清单
  final_summary.txt               纯文本汇总
  01_timeseries/hydrate_timeseries.dat
  02_cages/cage_stat.dat
  02_cages/guest_shell.dat
  02_cages/cage_center_distances.dat
  03_tracking/cage_lifetime.dat
  03_tracking/cage_transitions.dat
  03_tracking/cage_transition_summary.dat
  03_tracking/cage_occupancy.dat
  03_tracking/cage_quality.dat
  03_tracking/guest_events.dat
  03_tracking/cage_transition_network_{nodes,edges}.dat
  04_phase/hydrate_phase_timeseries.dat
  04_phase/hydrate_phase_transitions.dat
  04_phase/phase_domains.dat
  05_nucleation/nucleation_summary.txt
  05_nucleation/nucleation_timeseries.dat
  06_labels/hydrate_labels.dat
  06_labels/hydrate_labels.pdb
  07_cages_pdb/cages_frame_*.pdb
  08_plots/*.png
  09_views/view_*.vmd, view_*.pml
```

格式约定：

* 所有表格为**制表符分隔**；
* `#` 开头为注释与单位说明；
* 注释之后恰好一行表头；
* 适用时前两列统一为 `frame` 与 `time_ps`。

关键列说明：

* `hydrate_timeseries.dat` —— HOP 统计量、`Nmax_hyd`/`Nmax_ice`、各笼型计数、
  `total_cages`。
* `hydrate_phase_timeseries.dat` —— `phase`、`confidence`、各相得分、组成、
  `domain_size`、`domain_radius`、`interface_fraction`、`coherence`、`n_domains`，
  以及理想结构参考分。
* `phase_domains.dat` —— 每个晶畴一行（排名、笼数、相、得分、半径）。
* `cage_center_distances.dat` —— 各笼型对的**面共享笼心距**统计
  （count、mean、median、min、max）。
* `cage_quality.dat` —— 每个笼的球形度与各向异性。
* `guest_events.dat` —— 逃逸/重捕获事件。

---

## 各分析器的原理

**HOP。** 每个水氧用最近 4 个邻居描述：四面体度 `qtet` 与键取向相关向量 `q3m`。
邻居间的相关度转化为交错/重叠氢键计数，在最近邻壳层内平滑（CSR 图，混合系数
0.15），再用 Fermi/高斯门函数得到各相倾向，取置信度最高且通过结构门限的状态
作为标签。

**笼。** 每帧在 HOP 筛选后的水分子子图上做一次环感知。对每个客体，朝向它的环
按 `2·对齐度 − 0.15·平面度 − 0.025·距离` 排序，逐面装配；闭合要求欧拉示性数
χ = 2、面连通、每条边恰好被两个面共享。无法闭合的壳退化为半笼/部分笼/初期笼/
开口笼/畸变笼。

**相。** 面共享的笼心构成晶格网络，最大连通分量即晶畴。组成只在最大晶畴上统计，
并与理想结构的距离签名混合，因此孤立或缺陷笼无法把相“带偏”。

**成核。** `Nmax_hyd` 需要持续超过阈值（可允许少量间隙）。`fixed` 用
`--nucleation-min-size`；`adaptive` 从初始基线估计阈值；`commitment` 额外要求
增长量与正斜率。

---

## 性能

参考体系：5888 个水氧、1024 个甲烷客体、帧间隔 100 ps。

| 任务 | 耗时 |
|---|---|
| HOP + 笼 + 相 内核 | 约 0.6 s/帧/核 |
| 完整流程（含追踪、品质、输出） | 约 0.9 s/帧（单核） |
| 同上，默认全核并行 | 约 0.4 s/帧（墙钟） |

5001 帧（500 ns）轨迹：单核约 **1.2 小时**，12 进程约 **25–30 分钟**；
最初发布版需要 33.8 小时。

实用建议：

* `--cage-pdb-stride 1` 会每帧写一个 PDB（约 1.4 MB/帧），生产跑请设间隔；
* `guest_shell.dat` 每帧每个客体一行，长轨迹建议加大分析 stride；
* 父进程负责追踪与写盘，超过约 12 个 worker 后它才是上限。

---

## 并行、断点续跑与长跑

多进程默认开启。worker 数为 `min(可用 CPU 数, 待分析帧数)`，其中“可用”会综合
`HYTOPO_NPROCS`、调度器亲和性、cgroup CPU 配额以及 SLURM/PBS/SGE 变量。
可用 `--n-procs N` 指定，或 `--serial` 关闭。

长跑可中断续跑：

```bash
hytopo ... --checkpoint-every 200 --outdir run
# ……崩溃或 Ctrl-C……
hytopo ... --checkpoint-every 200 --outdir run --resume
```

检查点保存追踪器状态以及每个输出文件的字节偏移；续跑时先按偏移截断输出，再继续
写入。续跑结果与一次跑完逐字节一致。参数或轨迹不一致时会拒绝续跑。

---

## 验证与基准

`hytopo.benchmark_regression` 会对单帧结果打指纹（HOP 统计量、标签哈希、完整笼
清单、笼型计数、完整 guest shell 表），并与基准文件比对：

```bash
python -m hytopo.benchmark_regression \
  -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --frames 0,500,1000,1500 --golden benchmarks/golden.json
```

输出每帧墙钟时间与 `REGRESSION : PASSED/FAILED`。用
`--golden 文件 --write-golden` 生成基准。

`hytopo.benchmark_scaling` 用合成的百万原子体系测量局部图构建性能。

测试套件：

```bash
python -m pytest hytopo/test_core.py      # 或 python -m unittest hytopo.test_core
```

---

## Python API

```python
from hytopo.config import AnalysisConfig, TrajectoryConfig, CageParameters
from hytopo.runner import run

config = AnalysisConfig(
    trajectory=TrajectoryConfig(top="system.tpr", traj="traj.xtc",
                                guest_selection="resname CH4"),
    find_cages=True,
)
run(config)
```

逐帧构件也可直接调用：`hytopo.cage.analyze_cages_for_frame`、
`hytopo.hop.analyze_hop_from_graph`、`hytopo.cage_lattice.build_cage_lattice`、
`hytopo.phase.classify_phase`、`hytopo.tracking.CageTracker`。

---

## 常见问题

**报 `No water oxygen atoms found`** —— 选择器没匹配上。GROMACS 常用
`-water "resname SOL" -oxygen "name OW"`；可先用 `-oxygen "name O"` 或
`element O` 试。

**找不到笼** —— 是否加了 `--find-cages` 和 `-guest`？客体选择是否正确？可放宽
`--no-exclude-liquid`、`--include-ice-for-cage`，并查看
`02_cages/guest_shell.dat` 确认客体周围确实有水壳。

**生长轨迹检测不到成核** —— `adaptive` 与 `commitment` 假设轨迹从液体开始。
如果体系一开始就有晶种，请用 `--nucleation-method fixed --nucleation-min-size ...`
或 `--no-nucleation`。

**很慢或内存不足** —— 检查 `--n-procs`（每个 worker 会各自读一遍拓扑），
或降低 `--max-rings-per-guest`/`--beam-width`，并把 `--cage-pdb-stride`、
`--hop-pdb-stride` 调大。

**numba 出问题** —— 用 `--no-numba` 回退到等价的 Python 内核。

**MDAnalysis 读不了轨迹** —— GROMACS 用 `.tpr` + `.xtc`；拓扑与轨迹必须描述
同一批原子。

**路径图太挤** —— 在 `hytopo/network.py` 里调 `min_edge_weight`/`max_edges`、
调高 `edge_label_min_weight`，或保持 `show_transition_panel = True`
（边全部保留，数字移到右侧榜单）。图可单独重绘，无需重跑轨迹：

```python
from hytopo.network import generate_network_from_tables
generate_network_from_tables("run/03_tracking", outdir="run")
```

---

## 引用与许可

如在论文中使用 hytopo，请引用配套手稿
（*HyTopo: A High-Performance Hierarchical Algorithm for Mapping the Structural
Evolution and Transition Networks of Gas Hydrates*）。

许可证：由作者在公开发布前确定（学术软件常用 MIT、BSD-3-Clause 或 GPL-3.0）。
