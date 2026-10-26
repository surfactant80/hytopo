# HyTopo 更新说明

## v1.0.4 — 修正：笼型并入客体环境状态

v1.0.3 的环境列表只含 LIQ/interface，导致网络里没有 `LIQ -> 笼` 的边。
`_track_environments` 现在把检测到的笼型作为该客体的状态，环境标签只用于无笼客体。
100 帧 ARAS1 结果：`LIQ -> incipient_cage` 48.5 ns⁻¹、`incipient_cage -> LIQ`
39.3 ns⁻¹、`LIQ -> half 5^12` 5.9 ns⁻¹、`LIQ <-> interface` 26.3 ns⁻¹，
主通道（incipient/open/half/5^12 6^2）不变，耗时 0.87 s/帧（与 1.0.2 相同）。

## v1.0.3 — 过渡网络加入真实客体环境（LIQ / interface）

**客体环境**作为网络状态：

- 对每个客体，统计其第一水化壳（`--guest-radius` 内的水氧）的 HOP 标签：
  多数液态 → `LIQ`；壳层有序但无笼 → `interface`；检测到笼 → 该笼型。
- 状态在 `ENVIRONMENT_WINDOW = 3` 帧内做多数平滑（`tracking.py`），抑制单帧标签抖动；
  纯 `LIQ ↔ interface` 的闪烁明显下降（30 帧内 654 → 398 次）。
- `build_from_tracker` 把环境变化中**至少一端是非笼态**的跃迁计入网络；
  笼→笼跃迁仍来自轨迹，避免重复计数。出现/消失仍写入
  `03_tracking/cage_network_events.dat`；只有 `--skip-hop` 时才回退到 `Unmatched`。
- 图左侧新增 `LIQ (liquid-like)` 列（`LIQ` 与 `interface` 两个节点）。

## v1.0.2 — 过渡网络记账节点改名为 Unmatched

原来路径中的Liquid 容易被理解为 HOP 分类出的液态水，但它其实是
“检测不到的笼”的记账节点（笼搜索前 HOP 已排除完全液态水）。改动：

- 记账节点改名为 **`Unmatched`**：图例为 `Unmatched`，列标题为
  `Unmatched (no cage)`；出现/消失边由 `Liquid -> X`、`X -> Liquid` 变为
  `Unmatched -> X`、`X -> Unmatched`。
- `build_from_tracker` 同时把轨迹的起始/结束计入 `births`/`deaths`，写入
  `03_tracking/cage_network_events.dat`（列：`cage_type appearances disappearances`）。
- `load_from_tables` 读取旧表时自动把 `Liquid` 映射为 `Unmatched`，旧输出仍可重绘。
- README 中英文补充说明：Unmatched 不是 HOP 态；“未匹配”不等于 LIQ；
  相邻帧按 Jaccard 重叠 + 客体身份 + 类型连续性连接，`--tracking-gap` 容忍漏检。

## v1.0.1 — 许可证与引用

- 许可证改为 **LGPL-3.0-or-later**：新增 `LICENSE`（LGPL-3.0 全文）与
  `COPYING`（GPL-3.0 全文，LGPL 通过引用并入），`pyproject.toml` 增加
  `license` 字段与 OSI 分类器，wheel 随包携带两份许可证文本。
- 引用信息改为 arXiv 预印本：<https://arxiv.org/abs/2609.37036>，中英文 README
  各附 BibTeX 条目。

## v1.0.0 — 整理与打包（包名 hytopo）

- **包名统一为 `hytopo`**：目录 `HyTopo/` → `hytopo/`，命令 `hytopo`，
  模块入口 `python -m hytopo`；`pyproject.toml` 名称、版本、分类器、
  可选依赖（`[dev]` = pytest）补齐。
- **包数据**：`hytopo/ideal_crystal/*.pdb`、`reference_signatures.json`、
  `UPDATES.md` 随 wheel 一起发布（`[tool.setuptools.package-data]`），
  `MANIFEST.in` 覆盖 sdist。
- **新 README.md**：功能全览、安装（pip / wheel / conda / 源码）、依赖版本、
  快速上手、全部命令行选项分组、输出目录与表格式、各分析器原理、性能数据、
  并行与断点续跑、验证与基准、Python API、故障排查。
- **入口点**：仅保留 `hytopo = hytopo.cli:main`；新增 `hytopo/__main__.py`
  使 `python -m hytopo` 可用；`__init__.py` 增加 `__version__ = "1.0.0"`。
- 删除过时的 `README.txt`、`setup.cfg`。
- **改名注意**：numba 的磁盘缓存（`__pycache__/*.nbi`）记录了旧模块名，
  改名后首次运行前需删除 `__pycache__`，否则会报 `No module named 'HyTopo'`。

构建与安装验证（本机）：

```bash
python -m pip wheel . --no-deps -w dist          # -> hytopo-1.0.0-py3-none-any.whl
pip install dist/hytopo-1.0.0-py3-none-any.whl   # 产出 hytopo 命令
```

wheel 内含 41 个文件（含 `ideal_crystal/*.pdb` 与 `test_core.py`）；
`python -m hytopo -h`、`hytopo.exe -h`、2 帧完整分析、黄金回归指纹均通过。


## v0.10 — 笼跃迁路径图可读性（不减少边）

原图的问题：184 条边、边标签阈值只有 2，中间被标签糊满；节点文字画在圆圈内，
`incipient cage`、`half 5^12 6^2` 这类长标签溢出压到邻节点。

目标：**类型标签保留在圆圈内、边一条都不删**。做法：

- **节点按文字自动定大小**：用 `TextPath` 量出标签的实际点宽高，圆圈直径取
  `max(宽, 高) × node_label_padding`；超过 `node_label_max_width` 的标签按比例缩字号
  （下限 7.6 pt），保证文字既不溢出也不至于小到看不清。
- **边标签沿整条曲线找空位**：搜索窗口 0.34 → 0.66、步长 0.035 → 0.02；
  节点圆圈和节点文字（含其外接框）都计入避让，放不下就自动跳过而不是硬压。
- **图右侧独立榜单**：`show_transition_panel` 把最强的 16 条跃迁（含速率）列在图中
  预留的数据坐标条带里（`transition_panel_width = 5.6`），图内只标注更强的边
  （`edge_label_min_weight = 60`、`max_edge_labels = 12`）。**边全部保留**，
  数字也没有丢失，只是搬到了可读的位置。
- 边本身仍按权重编码线宽与透明度，弱边自然退到背景里。
- 画布 21×14.2 in、300 dpi。

可调项：`min_edge_weight` / `max_edges`（默认 2 / 0 = 全部边）、
`edge_label_min_weight`、`max_edge_labels`（0 = 不限制，靠避让自动取舍）、
`transition_panel_rows`、`label_mode = "count"`、`layout_type = "spring"`。

不改数据重绘（不必重跑轨迹）：

```python
from HyTopo.network import generate_network_from_tables
generate_network_from_tables("out/03_tracking", outdir="out")
```

## v0.9 — 全量代码审查：死代码、失效选项与潜在问题

### 已修复的真实问题

| 问题 | 影响 | 修复 |
|---|---|---|
| `--cage-pdb-dir` 被 `cage_pdb_path` 忽略 | 用户指定的目录不生效 | 默认改为空（用类别目录），指定时嵌套到 `07_cages_pdb/<name>` |
| 自定义输出名落错类别 | 如 `--timeseries my.dat` 会写进 `02_cages/` | `categorised_path(..., category=...)` 显式类别，各 writer 传入自己的类别 |
| `cage.py` 重复导入 `CAGE_TYPES`、多余 `deque`/`combinations` | 无功能影响 | 删除 |
| `rings.py` 删函数时误删了 `fit_many_rings` | 导入即失败 | 已恢复 |
| 删除带 `@njit` 的函数时装饰器残留 | 下一个函数被 numba 编译而报错 | 已修复 |

### 删除的死代码（均已被新实现取代或无任何引用）

- `cage.py`：`build_ring_edge_index`、`analyze_guest_local`、`perceive_local_rings`、
  `grow_target_cage`、`cage_counts_row`、`print_cage_progress`
- `cage_lattice.py`：`_face_geometry`（已被 `_batch_face_geometry` 取代）
- `rings.py`：`build_guest_rings_from_local_graph`、`build_ring_records_from_local_cycles`、
  `local_cycle_to_global`
- `neighbor.py`：`mask_nodes_from_pairs_numba`、`build_local_graph_for_guest`
- `hop.py`：`hop_result_to_dict`、`print_hop_progress`、`print_hop_diagnostics`、`label_counts`
- 未使用的导入：`beam.Optional`、`cage.Any`、`cage.CAGE_TYPES`、`report.Counter`、
  `runner.os`、`visualize.PLOTS`、`benchmark_regression.CageAnalysisResult`

### 保留但内部未使用（脚本 API，故意留着）

`neighbor.py` 的拓扑工具：`adjacency_tuple_from_csr`、`csr_neighbors`、
`filter_graph_nodes_by_mask`、`graph_density`、`local_graph_density`、
`remap_local_cycles_to_global`、`validate_graph`。这些小函数构成拓扑库的脚本接口，
删除没有收益，保留并在文档中标注。

### 其他检查结论

- **CLI 参数**：所有 `add_argument` 的 `args.*` 都被读取，没有"解析了但不用"的选项。
- **并行一致性**：`slim_result`（丢弃 `probabilities`/`descriptors`）之后，
  串行与并行输出**逐字节一致**（时间序列、cage_stat、guest_shell、occupancy、
  quality、phase 表全部比对通过）。
- **断点续跑**：流式文件偏移已并入检查点，中断+续跑与一次跑完逐字节一致。
- **退化输入**：`--no-tracking` / `--no-phase-analysis` / `--no-nucleation` /
  `--no-report` / `--skip-hop` 各组合均能跑通，缺失的表在报告中被跳过。
- **仍存在的已知小项**：`phase_tracker.distance_table` 与 `nucleation.points`
  仍随帧数增长（各约十万行/2.5 万个小对象），量级远小于此前的 tracker 事件；
  如需要可同样改为流式。

### 清理后验证

21 项单测通过；ARAS1 黄金指纹 PASSED（0.598 s/帧）；完整 CLI 端到端跑通（33 个文件）。

## v0.8 — 父进程批量写（B8）、tracker 流式化（C12），C10/C11 判定

### B8：父进程批量写

`guest_shell.dat` 每帧 1024 行、`hydrate_labels.pdb`/`hydrate_labels.dat` 每帧数千行，
原来都是逐行 `write()`。现在按帧拼成单个字符串后一次写出（shell 表、标签表、
HOP PDB 模型、笼统计行）。这是并行模式下父进程串行段的主要开销。

### C12：tracker 不再保留全部历史事件

原来每个笼的每个事件同时留在 `frame_events`、`occupancy_events` 和 `track.events`
三处，5001 帧 × 约 500 笼 ≈ 250 万个事件 ≈ 1.3 GB 常驻内存。现在：

- `frame_events` 删除（全代码无读取）；
- `track.events` 删除，`CageTrack` 只保留 `first_type`、起止时间与类型计数
  （`network.py` 只需要首个事件类型和时间范围）；
- `cage_occupancy.dat` 与 `cage_quality.dat` **逐帧流式写入**，
  `occupancy_events` 不再累积，只保留 `occupancy_counts` 供汇总报告使用；
- 流式文件的字节偏移一并写入检查点，续跑时与输出文件一起截断，
  保证「中断 + 续跑」与一次跑完逐字节一致。

实测（20 帧，9982 个事件）：常驻事件数 **9982 → 0**，tracker 内存不再随帧数增长。
墙钟收益有限（20 帧并行 719 → 687 ms/帧，约 4%），真正的收益在长跑内存：
5001 帧约省 1.3 GB。

### C10 / C11：判定为不需要（记录理由）

- **C10（邻接图 `vectors` 内存）**：`NeighborGraph.vectors` 是 `(n_pairs, 3)` float64。
  参考体系 5888 水氧只有 0.3 MB；百万原子约 48 MB。去掉它需要把最小镜像计算搬进
  numba kernel 并处理三斜盒子，且会改变 `qtet` 的数值路径（有回归风险）。
  在目标体系上收益不足 2%，不做。
- **C11（环枚举缓冲区复用）**：numat 内核每帧分配约 15 MB（按 `8 × 候选水数` 的上限），
  分配+清零约 5–10 ms/帧（约 1%）。需要把缓冲区作为参数传入 kernel，收益太小，不做。

### 最终性能（ARAS1，5888 水氧 / 1024 客体，20 帧）

| 口径 | 最初版本 | 现在 | 提升 |
|---|---|---|---|
| 单帧内核（HOP+cage+相） | 24.30 s | 0.64 s | **38×** |
| 完整流程串行 | ≈25 s/帧 | 0.90 s/帧 | **28×** |
| 20 帧墙钟（12 进程） | ≈500 s | 13.7 s | **36×** |
| 5001 帧生产跑 | 33.8 h | 约 1.25 h（串行）/ 约 25 min（并行） | 27–80× |

## v0.7 — 笼形质量与客体交换事件（E18）、性能总账

### E18：笼品质因子与客体逃逸/重捕获

新增两个逐帧表：

- **`03_tracking/cage_quality.dat`**：每个笼的几何品质
  - `radius_mean` / `radius_std`：笼心到各水氧顶点的距离均值与标准差（周期最小镜像）；
  - `sphericity = 1 - std/mean`（1 表示所有顶点等距，球壳）；
  - `anisotropy = (λ3 - λ1)/λ3`（顶点协方差特征值，0 表示各向同性）。
- **`03_tracking/guest_events.dat`**：客体交换事件
  - `escape`：某 guest 的笼消失（轨迹中断超过 gap 容差）；
  - `recapture`：同一 guest 在 `recapture_window`（默认 5 帧）内重新成笼；
  - 记录发生帧、时间、笼型、前一笼型、间隔帧数与存活时间。

ARAS1 实测（6 帧）：5^12 平均球度 0.927、半径 3.92 Å；5^12 6^2 球度 0.893、半径 4.34 Å；
缺陷笼（half/incipient/open）球度 0.88–0.91，与"标准笼更规整"一致。escape 54 次、
recapture 20 次。一页报告新增第 6 节展示这些结果。

### B7 复核：并行启动瓶颈不在拓扑解析

原计划让 worker 共享拓扑解析。实测后发现前提不成立：

| 操作 | 冷启动 | 热缓存 |
|---|---|---|
| `Universe(tpr, xtc)` | ~2.4 s | **0.10 s** |
| `Universe(pdb, xtc)` | — | 0.25 s |

tpr 解析本身很快（热缓存 0.1 s），换成轻量 PDB 拓扑反而更慢。并行启动开销来自
**多个 worker 并发打开 435 MB 的 XTC**（以及首次建立偏移索引），共享拓扑无法消除。
因此没有实现"拓扑共享"，改为在文档中说明：短跑建议 `--n-procs` 略小于核数。

### 性能总账（ARAS1：5888 水氧 / 1024 客体，单帧）

| 版本/阶段 | 单帧 | 相对最初 |
|---|---|---|
| 最初发布版（HOP+cage 内核） | 24.30 s | 1× |
| 现在（HOP+cage+相判定内核） | **0.636 s** | **38×** |
| 完整流程（含 tracking、相、质量、输出）串行 | 0.885 s | 28× |
| 完整流程 20 帧墙钟（12 进程） | 14.4 s（0.72 s/帧） | 35× |

5001 帧（500 ns）生产跑：**33.8 h → 约 1.2 h（串行）/ 约 25–30 min（12 进程）**。
并行上限已从"计算"转为"父进程串行段"（tracking + 文本输出，约 0.25 s/帧），
这也是下一步 B8/C12 的目标。

## v0.6 — 热路径批量化、IPC 瘦身、断点续跑、逐晶畴判相

### 性能（A 组）

- **环评分与方向覆盖率批量化**：`score_guest_rings` / `estimate_directional_coverage`
  原来对每个 guest、每个环单独调用一次 `pbc_delta`（约 4 万次/帧）。现在环几何
  以 `RingTable`（records + centers/normals/planarity 数组）形式一次给出，
  单次 `pbc_delta` 处理整个候选集，排序用 `argsort(kind="stable")` 保持原有顺序。
- **正交盒子 PBC 快路径**：`pbc_delta` 对 90° 盒子直接用组件级 round-off，
  并缓存直角判据（原来每帧 1800 次 `np.allclose`）。
- **闭包连通校验改并查集**：`connected_ring_set` 从 dict+deque 改成边→首个环的
  并查集，调用次数不变、单次快 2–3 倍。
- **相判定只构建一次笼心网络**（原来 `classify_phase` 和 `update` 各建一次）。

实测（ARAS1，4 帧均值）：**0.84 → 0.65 s/帧**，黄金指纹逐帧一致。

### 并行（B 组）

- **IPC 瘦身**：worker 返回前丢弃 `probabilities` / `descriptors`
  （父进程只用 `stats` 和 `labels`），单帧 pickle 从 1.52 MB 降到约 0.86 MB；
  百万原子体系上这决定并行是否可用。

### 断点续跑（D15）

`--checkpoint-every N` 每 N 帧把 CageTracker / PhaseTracker / NucleationAnalyzer
连同**各输出文件的字节偏移**写入 `checkpoint.pkl`；`--resume` 时先按偏移截断
输出（丢弃检查点之后写入的帧），再以追加模式继续。续跑结果与一次跑完
**逐字节一致**（时间序列、cage_stat、guest_shell、寿命、占据、相表全部比对通过）。
参数或轨迹不匹配时拒绝续跑。

### 逐晶畴判相（E16）

每个连通笼心分量的组成单独判相，输出 `04_phase/phase_domains.dat`
（frame / domain_rank / n_cages / phase / score / radius_A），
相时间序列新增 `n_domains` 列。同一体系内的 sI/sII 共存可以直接看出来。

### 进度条（F）

`HyTopo/progress.py`：单行进度条 + 速率 + ETA，默认开启，`-q` 关闭。

### 理想结构参考（E17）

**关键前提：参考 PDB 的坐标没有折叠进盒子**（sI 盒子 22.378 Å 但 y 坐标到 32.276，
sH 有负坐标）。`ideal_reference.py` 先用 `wrap_point` 按 CRITST1 折叠，再跑同一套
环/笼流程，于是：

- sI（2×2×2 超胞）：64 笼 = 16 × 5^12 + 48 × 5^12 6^2，正好 2:6；
- sII：24 笼 = 16 × 5^12 + 8 × 5^12 6^4；
- sH：40 笼 = 24 × 5^12 + 16 × 4^3 5^6 6^3（PDB 里大笼客体原子缺失，故无 5^12 6^8）。

从这些笼心提取面共享距离签名（写入 `ideal_crystal/reference_signatures.json`）：

| 相 | 参考距离（Å） |
|---|---|
| sI | 5^12–5^12 6^2 = 6.25；5^12 6^2–5^12 6^2 = 6.60 |
| sII | 5^12–5^12 = 6.05；5^12–5^12 6^4 = 7.10；5^12 6^4–5^12 6^4 = 7.41 |
| sH | 4^3 5^6 6^3–5^12 = 6.18；5^12–5^12 = 6.11 |

判相时按**丰度加权**与参考比对：每个参考笼型对的权重取“每笼期望对数”与“每笼实测
对数”的较小者，缺失对不加分也不倒扣。这样只有 3 个 5^12 6^4 的体系不会被误判成 sII。
最终得分 = `(1-w) × 组成分 + w × 距离参考分`，默认 w = 0.4、距离窗口 0.6 Å。

验证：

- 三个理想结构**各自判为自身**（sI 0.983 / sII 0.988 / sH 0.750，自参考分 1.00，其余 0.00–0.30）；
- ARAS1 全程仍为 **sI**（置信 0.66–0.74），sII/sH 参考分 0.01–0.03。
- `--no-ideal-reference` 退回纯组成判相；`--phase-reference-weight` 调权重。

## v0.5 — 跨平台并行、笼心空间相判定、输出分类与一页报告

### 1. 并行适配到任意机器

`os.cpu_count()` 只是机器核数，不是本进程能用的核数。新增 `HyTopo/cpus.py`，默认
worker 数取以下限制的最小值：

- `HYTOPO_NPROCS` 显式覆盖；
- 调度器亲和性（Linux `sched_getaffinity`，Windows 上有 psutil 时读亲和掩码）；
- 容器配额（cgroup v2 `cpu.max`、v1 `cfs_quota/cfs_period`）；
- 批处理系统变量（`SLURM_CPUS_PER_TASK`、`PBS_NP`、`NSLOTS`、`LSB_DJOB_NUMPROC`）；
- 最后才是 `os.cpu_count()`。

另外 worker 数不会超过待分析帧数（3 帧不会开 12 个进程），`--n-procs` 与 `--serial`
行为不变。

### 2. 相判定：从“笼组成比例”改为“笼心空间关联”

新增 `HyTopo/cage_lattice.py`。每个完整笼给出一个笼心，两笼满足下列条件才连边：

- 共享不少于 3 个水氧节点（共面）；
- 用共享水环（无序多边形）的 Newell 法向与质心算出两笼心到该面的有向距离
  `d_a`、`d_b`，要求两心在面两侧（`d_a · d_b < 0`）；
- 且 `|d(笼心,笼心) - (d_a + d_b)| <= --phase-lattice-tolerance`。

这样“只碰巧共三个水”的假邻接会被几何判据剔除。连通分量即**晶畴**。相判定改为：

- 组成比例只在**最大晶畴**上统计（孤立笼不再拉偏比例）；
- 置信度乘以晶格相干性 `coherence = 通过几何检验的邻接 / 候选邻接`；
- 置信度或相干性不足时输出 `mixed`，而不是硬给一个相。

新增输出（相对旧 `units_estimate` 更稳）：

- `04_phase/hydrate_phase_timeseries.dat`：相、置信度、各相得分、组成计数、
  `domain_size`（最大晶畴笼数）、`domain_radius`（晶畴半径，Å）、
  `interface_fraction`（最大晶畴之外的笼占比）、`coherence`。
- `02_cages/cage_center_distances.dat`：每帧、每笼型对的**面共享笼心距**统计
  （count / mean / median / min / max），即题目要求的 5^12–5^12、
  5^12–5^12 6^2 …… 距离分布。

验证（ARAS1，甲烷水合物）：第 0→5 帧持续判为 sI，置信度 0.84–0.91，
晶畴从 217 笼增长到 281 笼，界面比例从 0.48 降到 0.41；5^12–5^12 6^2 的面共享
笼心距稳定在 6.7 Å。几何拟合已向量化（Newell 法向批量计算），单帧开销
约 40 ms（原逐对 SVD 约 240 ms）。

### 3. 输出分类、格式统一、一页 HTML 报告

输出目录按类别重组（`HyTopo/output_layout.py` 统一管理路径）：

```text
01_timeseries/   主时间序列
02_cages/        笼统计、guest shell、笼心距分布
03_tracking/     寿命、跃迁、占据、跃迁网络
04_phase/        相时间序列与相变
05_nucleation/   成核
06_labels/       逐水标签与 PDB
07_cages_pdb/    逐帧笼 PDB
08_plots/        全部 PNG
09_views/        VMD / PyMOL 脚本
report.html      一页英文汇总
README.txt       输出清单（manifest）
```

统一格式：所有表格为 **制表符分隔**，可选 `#` 注释行 + 一行列名；时间列统一为
`time_ps`，帧列统一为 `frame`；`cage_transitions.dat` 不再把事件表和汇总表混在
一个文件里（汇总另存 `cage_transition_summary.dat`）。

一页报告 `report.html`（英文，自包含，图片以 base64 内嵌）：关键指标卡片、
HOP 曲线、笼计数、相/晶畴/界面曲线、成核曲线、末帧笼型表、寿命表、相变表、
输出文件树与方法说明。新增 `--no-report` 关闭。

## v0.4 — 默认全核并行 + tracking 热点优化

### 默认吃满所有逻辑核

帧级多进程本来就是默认开启，但 `resolved_n_processes()` 里有一个
`min(8, os.cpu_count())` 的硬上限，16/32 核机器上白扔一半算力。现在改为
`os.cpu_count()`，即默认使用全部逻辑核；`--n-procs N` 可指定数量，`--serial` 退回单核。

实测（ARAS1，本机 6 核 12 线程）：

| 场景 | 1 核 | 默认（12） | 加速 |
|---|---|---|---|
| 64 帧，关闭逐帧输出 | 48.33 s | 11.84 s | 4.08× |
| 32 帧，生产配置（tracking + 相位 + 成核） | 49.02 s | 30.35 s | 1.62× |
| 同上，含本轮 tracking 优化 | 28.99 s | **12.46 s** | **2.33×** |

短跑的加速比被 worker 启动摊薄（每个 worker 要自己读一遍拓扑，约 2.4 s）；
5001 帧的生产跑里这部分可忽略，按父进程串行开销估计可达约 4×。

### 父进程不再是瓶颈：tracking 候选索引

并行后单帧计算降到约 60 ms，父进程的串行 tracking 反而成了上限（每帧要拿
700 个事件逐个比对约 1400 条活跃轨迹）。三处改动，**结果完全不变**：

- `CageEvent.water_key` 与 `CageTrack.last_water_key` 缓存为 frozenset，
  不再在每次比较时重建（内层循环里原来每帧要重建上百万次）。
- 活跃轨迹按 `guest_id` 和水节点建索引：一条轨迹只有与事件共享水节点或属于
  同一 guest 才可能被接受（否则得分上限 0.15，低于 0.25/0.50 两个阈值），
  因此候选集可以安全缩小；空水环事件仍走全量扫描以保持语义。
- `jaccard` 改为只算一次交集再换算并集。

串行生产配置 1532 → 906 ms/帧；`cage_lifetime.dat`、`cage_transitions.dat`、
`cage_occupancy.dat` 与优化前逐字节一致。ARAS1 几何/笼回归指纹同样 PASSED。

## v0.3 — 客体分子识别

修复“全原子模型下 1 个 CH4 被当成 5 个客体”的问题。现在**一个分子就是一个客体**，
一个中心（周期性边界正确的质心）。

### 两条识别路径

- **有拓扑（tpr/gro/pdb 等）**：直接按 MD 残基分组（`residue`）。拓扑本身已经说明
  哪些原子属于同一个分子，无需几何判断，成本为零。
- **无拓扑（扩展 XYZ）**：按共价半径成键推断（`bonded`）。判据为
  `d <= 1.15 * (r_i + r_j)`（Cordero 共价半径表），系数可由
  `--guest-bond-scale` 修改。判据直接取周期最小镜像距离，并用
  `self_capped_distance` 近邻列表加速（线性于近邻对数，而非原子数平方），
  因此跨边界的分子仍能被正确连成一体。
  典型键长与判据（默认 1.15）：C–C 1.54 vs 1.75、C–H 1.09 vs 1.23、
  C–O 1.43 vs 1.63、O–H 0.96 vs 1.12；而氢键 O···H ≈1.8 Å 远高于 1.12，不会误连。
- `--guest-grouping none` 保留旧的“每个原子一个客体”行为，便于对照。
- 只选中心原子（如 `-guest "resname CH4 and name C"`）时，每个分子只有这一个
  原子参与分析，中心自然就是该原子坐标；这里没有任何模型转换，只是参与分析的
  原子少到 1 个，所以结果与联合原子模型一致。

### 中心与影响

- 客体中心取分子所选原子的质心，并以分子首个原子为参考做最小镜像展开，
  因此跨周期性边界的分子中心仍落在盒子内部。
- 本身只有一个原子的客体（联合原子 CH4 模型、离子）走快速路径，中心就是原子坐标，
  与旧结果**逐位一致**（ARAS1 回归指纹 PASSED）。
- 下游全部按分子计数：`cage_stat.dat` 的 `n_guest`、`guest_shell.dat` 的 `guest_id`、
  guest occupancy（旧实现会把一个全原子 CH4 的 5 个原子算成 5 个 guest，
  从而误报 multiple occupancy）、以及 cage tracking 的 guest 身份。

### 命令

```bash
# 默认：有拓扑用残基，无拓扑用共价键
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" --find-cages

# 纯几何输入
hytopo -top traj.etxyz -water "element O" -oxygen "element O" \
  -guest "index 0:10" --find-cages

# 对照旧行为
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" --find-cages --guest-grouping none
```

### 验证

- `HyTopo/test_core.py::GuestGroupingTests`：残基分组、成键分组、PBC 跨边界成键、
  PBC 质心、单原子快速路径、EXTXYZ 端到端分组，共 8 项。
- 端到端：2 个全原子 CH4（10 个原子）+ 4 个水分子 → `n_guest=2`（auto/bonded）、
  `n_guest=10`（none）。
- ARAS1（联合原子 CH4，1024 个残基）回归指纹 PASSED，结果未变。


## v0.2 — 性能内核重构（2026）

本版本只改计算内核，**不改变任何分析结果**：8 个测试帧（0/500/1000/1500/2000/2500/3500/4500）
的指纹（HOP 统计量、逐水标签哈希、笼清单、笼类型计数、完整 guest shell 表哈希）与重构前
逐位一致。实测速度见 `benchmarks/PERFORMANCE.md`。

参考轨迹 ARAS1（5888 个水氧、1024 个 CH4 客、5001 帧）：

| 阶段 | 单帧耗时 | 相对原始 |
|---|---|---|
| 原始发布版 | 24.30 s | 1× |
| 1. 目标笼“最近优先”排序 | 13.86 s | 1.75× |
| 2. beam search 位掩码状态 | 6.16 s | 3.9× |
| 3. beam search numba 内核 | 3.41 s | 7.1× |
| 4. 每帧全局环感知 + 批量几何拟合 | **0.835 s** | **29×** |

5001 帧的模拟：约 33.8 h → 约 70 min（单核）。

### 改动明细

- **`possible_targets` 改为最近目标优先**
  原实现按“面数最少”尝试目标笼，59% 的 beam 搜索在注定失败的目标上被浪费。改为按
  shell 签名与目标签名的 L1 距离排序后，beam 调用次数从 1043 次/帧降到 445 次/帧，97% 的
  搜索一次命中。结果不变（同一 guest 能闭合的目标是唯一的）。

- **`RingAssembly` 位掩码 beam search（`cage.py`）**
  边计数状态从 `dict[edge] -> count` 改为两个整数位掩码（恰好被 1 个面 / 2 个面覆盖的边），
  ring 选择从 `frozenset` 改为位掩码，闭合计账用 `int.bit_count()`。每次状态转移从
  O(边数) 的 dict 复制变成 O(1) 整数运算，同时保持候选排序、beam 截断、去重与最终校验
  的顺序完全一致。

- **numba beam 内核（新增 `HyTopo/beam.py`）**
  同一搜索移植到 numba，用定宽 `uint64` 字表示位掩码。遍历顺序、beam 更新、去重规则、
  评分公式与 Python 版逐条对应；Python 版保留为回退路径（numba 缺失或 `--no-numba` 时使用）。

- **每帧全局环感知（`cage.py`、`rings.py`）**
  环是水网络的属性，不是 guest 的属性：相邻笼共享面，原实现对每个 guest 重复做同一批水环
  的 DFS 与平面拟合。现在每帧对候选水分子子图枚举一次环，用批量特征分解一次拟合所有环，
  再用“水节点 → guest 位掩码”求交，把环分配给所有顶点都落在其 `guest_radius` 内的 guest。
  该分配与原逐 guest 局部图得到的环集合严格等价。

- **批量几何拟合（`rings.py: fit_many_rings`）**
  `fit_ring` 的逐环 PBC 展开与 SVD 改为数组化实现：一次 `minimize_vectors` + 批量
  `eigh`（最小特征向量 = 最小奇异向量），法向量符号不影响后续 `abs(dot)` 判据。

- **笼去重加索引（`cage.py: deduplicate_cages`）**
  原实现 O(n²) 两两 Jaccard，改为“水节点 → 已保留笼”索引，只对共享节点的笼做 Jaccard 比较。

### 行为变化

- 新增 `--no-numba` 同时关闭 numba beam 内核（回退到等价的 Python 实现）。
- `build_frame_ring_table` 在帧级环枚举达到安全上限时会打印警告（`-q` 下静默）。
- `analyze_guest_local` 保留为单 guest 参考路径（脚本/调试用），生产路径改为
  `build_frame_ring_table` + `analyze_guest_with_rings`。

### 验证

```bash
# 与基准指纹比对（任何结果差异都会 FAILED）
python -m HyTopo.benchmark_regression -top ARAS1.tpr -traj ARAS1.xtc \
  -guest "resname CH4" --frames 0,500,1000,1500 \
  --golden benchmarks/golden_aras1.json

# 计时
python -m HyTopo.benchmark_regression -top ARAS1.tpr -traj ARAS1.xtc \
  -guest "resname CH4" --frames 0,500,1000,1500 --repeat 2
```

## 0.1 — 上一轮功能更新

- **搜索效率**
  - cage 邻域现在按帧批量查询 guest 周围的水氧原子。
  - 局部图从全局 CSR 邻接表直接诱导，避免每个 guest 重复扫描全部边。
  - cage 图与 HOP 图 cutoff 相同时复用同一个图。
  - 并行模式改为 worker 自行读取轨迹，只在进程间传输帧索引，减少大数组 IPC。
  - 默认不再逐 guest 打印诊断信息；需要时使用 `--verbose-guests`。

- **逻辑与正确性**
  - Numba 环枚举现在显式剔除带弦环，避免把非基本面误判为 cage face。
  - cage 去重从“节点完全相同”扩展为基于水氧节点 Jaccard overlap 的去重。
  - `partial_5^12`、`partial_5^12_6^2` 和 `partial_5^12_6^4` 加入统计与可视化类型表。
  - 成核统计改为直接消费内存中的逐帧 `Nmax_hyd`，同时兼容旧版时间序列。
  - guest occupancy 统计支持正交和一般 triclinic PBC。

- **功能**
  - `--skip-hop --find-cages` 现在可以真正执行 cage-only 搜索；此模式会把所有水氧作为候选，不输出虚假的 HOP 分类。
  - 增加 cage tracking 参数：`--tracking-gap`、`--tracking-water-overlap`、
    `--tracking-same-guest-overlap`、`--tracking-dedup-overlap`。
  - 增加 `--no-tracking`、`--no-plots` 和 `--verbose-guests`。
  - scipy/numba 二进制不可用时，对球谐计算、CSR kernel 和环枚举提供 Python fallback。
  - 支持周期性 extended XYZ：`.xyz`、`.extxyz`、`.etxyz`。
  - 新增基于完整笼组成的 sI/sII/sH 晶相倾向和相变时间序列。
  - 成核分析新增 `fixed`、`adaptive`、`commitment` 三种方法。

## EXTXYZ / ETXYZ

输入文件必须在第二行包含 `Lattice` 和 `Properties`。例如：

```bash
hytopo -top trajectory.etxyz \
  -water all -oxygen "element O" -guest "element C" \
  --find-cages
```

当前 EXTXYZ 选择语法支持 `all`、`element`、`name`、`species`、`resname`
和 `index`。对于多原子 guest，建议选择每个 guest 的中心原子。

晶相结果：

```text
hydrate_phase_timeseries.dat
hydrate_phase_transitions.dat
```

大体系局部图基准：

```bash
python -m HyTopo.benchmark_scaling --atoms 1000000 --guests 200
```

## 推荐命令

常规 HOP + cage：

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --find-cages --cage-pdb-stride 100 --hop-pdb-stride 100
```

仅搜索 cage：

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --find-cages --skip-hop --no-nucleation
```

大体系调试单个 guest：

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --find-cages --verbose-guests --serial --no-plots
```
