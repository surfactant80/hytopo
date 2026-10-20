# HyTopo 全代码审计与扩展记录

## 已修复的逻辑问题

1. 成核时间序列过去缺少 frame 列，并可能把错误列解析成 `Nmax_hyd`。
2. `--skip-hop` 过去存在 CLI 选项，但 runner 会直接报错。
3. Numba 环搜索过去没有正确排除带弦环。
4. 半笼类型区间存在重叠，部分分支不可达。
5. `partial_5^12*` 类型过去能被分类，但不在统一统计类型表中。
6. cage tracking 在轨迹正常结束时会被 reaction network 错判为 cage -> Liquid。
7. guest occupancy 过去没有使用周期性边界。
8. triclinic box 的 cage center wrap 过去按正交盒处理。
9. `kernels.py` 与 `hop.py`、`neighbor.py`、`rings.py` 大量重复且没有被调用，
   已删除该无效模块。
10. config 与 analysis 模块中的 nucleation/tracking 参数结构重复，现已统一。

## 已完成的性能更新

- guest-water 半径查询按帧批量执行。
- 局部图通过全局 CSR 邻接表构建，不再逐 guest 扫描全部全局边。
- 局部图映射不再分配 `O(N_atoms)` 的全局映射数组。
- 环搜索前执行 2-core 剪枝，删除不可能位于任何环上的叶节点。
- Numba 环搜索通过方向约束避免重复枚举及 `O(N_cycles^2)` 去重。
- 相同局部水集合缓存环枚举结果。
- 不同 guest 共享相同 ring 时缓存几何拟合结果，避免重复 SVD。
- Numba DFS 临时 stack 内存不再与超大 `max_cycles` 线性爆炸。
- worker 自行读取轨迹，减少主进程到 worker 的大数组传输。

合成的 200 万节点局部图测试中，单独的局部图构建阶段由全局边扫描改为
CSR 局部诱导后约为 `39.9x`。这不是端到端速度，也不能直接解释为整个程序
加速 39.9 倍。

## 关于“整体加速 100 倍”

仅靠 Python 层重构不能对任意轨迹保证 100 倍。总耗时取决于：

- 轨迹解压和磁盘速度；
- guest 数量；
- 每个局部网络中的候选环数量；
- 输出 PDB 的频率；
- 是否启用 HOP；
- CPU 核数和 Numba/BLAS 环境。

若要求稳定的端到端 100x，后续应增加：

1. 全帧 ring pool：每帧只枚举一次 4/5/6 元环，再批量分配给 guest。
2. C++/Rust 环枚举后端及 OpenMP。
3. CUDA/CuPy 邻居表、局部描述符和 connected-component kernel。
4. 轨迹分块流水线及共享内存结果缓冲。
5. 二进制 HDF5/Zarr 输出，避免逐原子文本和 PDB 成为瓶颈。
6. 真实生产轨迹的 profiling 和性能回归基准。

## 新功能

- 周期性 EXTXYZ/ETXYZ 多帧读取。
- sI/sII/sH cage-composition phase tendency。
- 晶相转变持久化确认和转变表。
- fixed/adaptive/commitment 成核方法。
- 大体系局部图 micro-benchmark。

