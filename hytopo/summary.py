from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from typing import Any, Optional


class SummaryGenerator:
    """
    体系全局“一键式”结题报告生成器
    汇总分析 HOP, Cage, Tracking 和 Nucleation 的所有核心指标
    """

    def __init__(self, config: Any, outdir: str):
        self.config = config
        self.outdir = outdir
        self.report_name = "final_summary.txt"

    def generate_report(
        self,
        tracker: Optional[Any] = None,
        nucleation_res: Optional[Any] = None,
        phase_tracker: Optional[Any] = None,
        total_frames: int = 0,
        elapsed_time: float = 0.0,
    ):
        """
        核心生成逻辑
        """
        os.makedirs(self.outdir, exist_ok=True)
        save_path = os.path.join(self.outdir, self.report_name)

        with open(save_path, "w", encoding="utf-8") as f:
            # 1. 标题与基本信息
            f.write("==\n")
            f.write("           HYDRATE ANALYZER FINAL SUMMARY REPORT            \n")
            f.write("==\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System Top:   {os.path.basename(self.config.trajectory.top)}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Wall Clock Time: {elapsed_time:.2f} s\n\n")

            # 2. 成核分析阶段 (Nucleation Phase)
            f.write("1. NUCLEATION ANALYSIS\n")
            f.write("--\n")
            if nucleation_res:
                if nucleation_res.nucleated:
                    f.write("  - Nucleation Status:  [ DETECTED ]\n")
                    f.write(f"  - Nucleation Frame:   {nucleation_res.nucleation_frame}\n")
                    f.write(f"  - Nucleation Time:    {nucleation_res.nucleation_time:.2f} ps\n")
                    f.write(f"  - Nucleation Size:    {nucleation_res.nucleation_nmax_hyd} waters\n")
                else:
                    f.write("  - Nucleation Status:  [ NOT DETECTED ]\n")
                    f.write(f"  - Max Cluster Size:   {nucleation_res.max_nmax_hyd} waters\n")
                    f.write(
                        f"  - Method / Threshold: "
                        f"{nucleation_res.method} / "
                        f"{nucleation_res.effective_threshold} waters\n"
                    )
            else:
                f.write("  - Data not available or Nucleation analysis disabled.\n")
            f.write("\n")

            # 3. 笼结构稳定性分析 (Cage Stability & Diversity)
            f.write("2. HYDRATE CRYSTAL PHASE TENDENCY\n")
            f.write("--\n")
            if phase_tracker and phase_tracker.points:
                f.write(f"  - Dominant phase: {phase_tracker.dominant_phase}\n")
                f.write(f"  - Confirmed transitions: {len(phase_tracker.transitions)}\n")
                for transition in phase_tracker.transitions:
                    f.write(
                        f"    * frame {transition.frame}: "
                        f"{transition.from_phase} -> {transition.to_phase} "
                        f"(confidence={transition.confidence:.3f})\n"
                    )
            else:
                f.write("  - Phase tendency data not available.\n")
            f.write("\n")

            f.write("3. CAGE STABILITY & DIVERSITY (Tracking Data)\n")
            f.write("--\n")
            if tracker and tracker.tracks:
                all_tracks = tracker.tracks
                f.write(f"  - Total Unique Cages Tracked: {len(all_tracks)}\n")

                # 统计笼型分布
                type_counts = Counter([t.dominant_type for t in all_tracks])
                f.write("  - Cage Type Distribution (Dominant):\n")
                for c_type, count in type_counts.most_common():
                    f.write(f"    * {c_type:<15}: {count}\n")

                # 计算平均寿命 (ps)
                f.write("\n  - Average Cage Lifetimes:\n")
                type_lifetimes = defaultdict(list)
                for t in all_tracks:
                    type_lifetimes[t.dominant_type].append(t.lifetime_time)

                for c_type in sorted(type_lifetimes.keys()):
                    avg_life = sum(type_lifetimes[c_type]) / len(type_lifetimes[c_type])
                    max_life = max(type_lifetimes[c_type])
                    f.write(f"    * {c_type:<15}: Avg {avg_life:>8.2f} ps | Max {max_life:>8.2f} ps\n")
            else:
                f.write("  - Cage tracking data not available.\n")
            f.write("\n")

            # 4. 客体占据状态 (Occupancy Stats)
            f.write("4. GUEST OCCUPANCY STATISTICS\n")
            f.write("--\n")
            if tracker and getattr(tracker, "occupancy_counts", None):
                occ_counts = tracker.occupancy_counts
                total_occ = sum(occ_counts.values())
                if total_occ > 0:
                    for occ, count in occ_counts.most_common():
                        percent = (count / total_occ) * 100
                        f.write(f"  - {occ:<10}: {count:>6} events ({percent:>5.1f}%)\n")
                else:
                    f.write("  - No occupancy events recorded.\n")
            else:
                f.write("  - Occupancy data not available.\n")
            f.write("\n")

            # 5. 动力学诊断 (Kinetic Diagnosis)
            f.write("5. KINETIC & THERMODYNAMIC DIAGNOSIS\n")
            f.write("--\n")
            if nucleation_res and len(nucleation_res.timeseries) > 5:
                # 简单的趋势判断：取最后 1/4 轨迹的斜率
                ts = nucleation_res.timeseries
                n = len(ts)
                quarter = n // 4
                if quarter > 0:
                    y = [p["nmax_hyd"] for p in ts[-quarter:]]
                    x = [p["time"] for p in ts[-quarter:]]
                    slope = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0

                    f.write(f"  - Final Stage Slope (dN/dt): {slope:.4f} waters/ps\n")
                    if slope > 0.05:
                        diag = "STEADY GROWTH"
                    elif slope < -0.05:
                        diag = "DISSOCIATION"
                    else:
                        diag = "EQUILIBRIUM/PLATEAU"
                    f.write(f"  - System Status Diagnosis:   {diag}\n")
                else:
                    f.write("  - Kinetic diagnosis requires longer trajectory data.\n")
            else:
                f.write("  - Kinetic diagnosis requires longer trajectory data.\n")
            f.write("\n")

            f.write("==\n")
            f.write("  END OF SUMMARY REPORT                   \n")
            f.write("==\n")

        if not getattr(getattr(self.config, "runtime", None), "quiet", False):
            print(f"\n[Summary] One-click analysis report generated: {save_path}\n")


def generate_final_summary(
    config: Any,
    tracker: Optional[Any],
    nucleation_res: Optional[Any],
    phase_tracker: Optional[Any],
    total_frames: int,
    elapsed_time: float,
):
    """
    Runner 调用的统一入口
    """
    summary = SummaryGenerator(config, config.output.outdir)
    summary.generate_report(
        tracker,
        nucleation_res,
        phase_tracker,
        total_frames,
        elapsed_time,
    )
