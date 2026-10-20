from __future__ import annotations

import os
from typing import Optional
import numpy as np

from .config import (
    AnalysisConfig, 
    LABEL_NAMES, 
    CAGE_BETA, 
    CAGE_ELEMENT, 
    CAGE_RESNAME, 
    CAGE_TYPES
)
from .trajectory import AtomMetadata
from .hop import HopResult
from .cage import CageAnalysisResult
from .geometry import unwrap_cage_positions


class OutputManager:
    """
    统一输出管理器。
    负责协调文本统计、逐原子标签轨迹以及笼子可视化文件的写入。
    """
    def __init__(
        self,
        config: AnalysisConfig,
        atom_metadata: AtomMetadata,
        append: bool = False,
    ):
        self.cfg = config
        self.meta = atom_metadata
        self.out_cfg = config.output
        self.append = append
        mode = "a" if append else "w"

        # 确保目录存在
        self.cfg.ensure_output_dirs()

        # 初始化主时间序列文件
        self.ts_file = open(self.out_cfg.timeseries_path, mode)
        if not append:
            self._write_ts_header()

        # 初始化标签文件 (如果开启)
        self.label_file = None
        if self.cfg.run_hop and self.out_cfg.write_labels:
            self.label_file = open(self.out_cfg.labels_path, mode)
            if not append:
                self._write_label_header()

        # 初始化 HOP PDB 轨迹 (如果开启)
        self.hop_pdb_file = None
        if self.cfg.run_hop and self.out_cfg.write_hop_pdb:
            self.hop_pdb_file = open(self.out_cfg.hop_pdb_path, mode)
            if not append:
                self._write_hop_pdb_header()

        # 笼子统计文件 (如果开启)
        self.cage_stat_file = None
        self.shell_stat_file = None
        if self.cfg.find_cages:
            self.cage_stat_file = open(self.out_cfg.cage_stat_path, mode)
            self.shell_stat_file = open(self.out_cfg.shell_stat_path, mode)
            if not append:
                self._write_cage_stat_header()

    def _write_ts_header(self):
        header = (
            "Frame\tTime(ps)\tH_total\tH_crys\tH_pre\tHyd_order\tIce\tO_total\tDelta\t"
            "Nmax_hyd\tNmax_ice"
        )
        if self.cfg.find_cages:
            # 动态添加笼子类型的统计列
            cage_cols = "\t".join(CAGE_TYPES)
            header += f"\t{cage_cols}\ttotal_cages"
        self.ts_file.write(header + "\n")

    def _write_label_header(self):
        self.label_file.write("# Label definition:\n")
        for code, name in LABEL_NAMES.items():
            self.label_file.write(f"# {code} : {name}\n")
        self.label_file.write("# Format: Frame  Time(ps)  MolID  Label\n")

    def _write_hop_pdb_header(self):
        self.hop_pdb_file.write("HEADER    HOP STRUCTURE LABEL TRAJECTORY\n")
        self.hop_pdb_file.write("REMARK    Labels written in beta factor field\n")

    def _write_cage_stat_header(self):
        # 写入简单的笼子计数表头
        cols = ["frame", "n_oxygen", "n_guest"] + CAGE_TYPES + ["total"]
        self.cage_stat_file.write("\t".join(cols) + "\n")
        self.shell_stat_file.write(
            "frame\tguest_id\tn4\tn5\tn6\tn_faces\tclosure\tboundary_edges\t"
            "euler\tconnected\tcoverage\tnearest_target\ttarget_distance\t"
            "assigned\tstatus\tn_local_nodes\tn_local_cycles\tn_local_rings\n"
        )

    def write_frame(self, frame_idx: int, time_ps: float, box: np.ndarray, 
  pos: np.ndarray, hop_res: HopResult, 
  cage_res: Optional[CageAnalysisResult] = None):
        """
        根据各自的 stride 步长，将当前帧结果写入对应的文件。
        """
        # 1. 写入统一时间序列
        self._write_timeseries_row(frame_idx, time_ps, hop_res, cage_res)

        # 2. 写入逐分子标签 (Stride 控制)
        if self.label_file and frame_idx % self.out_cfg.label_stride == 0:
            self._write_label_rows(frame_idx, time_ps, hop_res.labels)

        # 3. 写入 HOP PDB 轨迹 (Stride 控制)
        if self.hop_pdb_file and frame_idx % self.out_cfg.hop_pdb_stride == 0:
            self._write_hop_pdb_model(frame_idx, box, pos, hop_res.labels)

        # 4. 写入 Cage 可视化 PDB (Stride 控制)
        if self.cfg.find_cages and cage_res:
            self._write_cage_stats(frame_idx, cage_res)
            if self.out_cfg.write_cage_pdb and frame_idx % self.out_cfg.cage_pdb_stride == 0:
                self._write_cage_pdb_frame(frame_idx, box, pos, cage_res)

    def _write_timeseries_row(
        self,
        frame: int,
        time: float,
        hop: HopResult,
        cage: Optional[CageAnalysisResult],
    ):
        s = hop.stats
        row = [
            str(frame), f"{time:.3f}", f"{s.H_total:.6f}", f"{s.H_crys:.6f}", f"{s.H_pre:.6f}",
            f"{s.Hyd_order:.6f}", f"{s.Ice:.6f}", f"{s.O_total:.6f}", f"{s.Delta:.6f}",
            f"{s.Nmax_hyd:.2f}", f"{s.Nmax_ice:.2f}"
        ]
        if cage:
            # 加入笼子计数
            counts = [str(cage.counts.get(t, 0)) for t in CAGE_TYPES]
            row.extend(counts)
            row.append(str(cage.n_cages))
        self.ts_file.write("\t".join(row) + "\n")

    def _write_label_rows(self, frame: int, time: float, labels: np.ndarray):
        atom_ids = self.meta.atom_ids
        time_text = f"{time:.3f}"
        lines = [
            f"{frame}\t{time_text}\t{int(atom_ids[i])}\t{int(label)}\n"
            for i, label in enumerate(labels)
        ]
        self.label_file.write("".join(lines))

    def _write_hop_pdb_model(self, frame: int, box: np.ndarray, pos: np.ndarray, labels: np.ndarray):
        names = self.meta.atom_names
        resnames = self.meta.resnames
        resids = self.meta.resids
        elements = self.meta.elements
        parts = [f"MODEL     {frame:4d}\n", format_pdb_cryst1(box)]
        append = parts.append

        for i in range(len(labels)):
            append(
                format_pdb_atom_line(
                    serial=i + 1,
                    name=names[i],
                    resname=resnames[i],
                    resid=resids[i],
                    element=elements[i],
                    pos=pos[i],
                    occ=1.0,
                    beta=float(labels[i]),
                )
            )

        append("ENDMDL\n")
        self.hop_pdb_file.write("".join(parts))

    def _write_cage_stats(self, frame: int, res: CageAnalysisResult):
        counts = [str(res.counts.get(t, 0)) for t in CAGE_TYPES]
        row = (
            [str(frame), str(self.meta.atom_ids.shape[0]), str(res.n_guest)]
            + counts
            + [str(res.n_cages)]
        )
        self.cage_stat_file.write("\t".join(row) + "\n")

        # 写入 shell 详细信息：一帧 1024 行，逐行 write 是父进程的主要串行开销
        if res.shell_rows:
            self.shell_stat_file.write(
                "".join("\t".join(map(str, shell_row)) + "\n" for shell_row in res.shell_rows)
            )

    def _write_cage_pdb_frame(
        self,
        frame: int,
        box: np.ndarray,
        pos: np.ndarray,
        res: CageAnalysisResult,
    ):
        pdb_path = os.path.join(
            self.out_cfg.cage_pdb_path,
            f"cages_frame_{frame:06d}.pdb",
        )

        with open(pdb_path, "w") as handle:
            handle.write(f"REMARK Frame {frame} Cages Output\n")
            handle.write(format_pdb_cryst1(box))

            serial = 1
            serial_map = {}

            for cage_index, cage in enumerate(res.cages, start=1):
                beta = CAGE_BETA.get(cage.cage_type, 20.0)
                element = CAGE_ELEMENT.get(cage.cage_type, "X")
                resname = CAGE_RESNAME.get(cage.cage_type, "OTH")

                cage_center, cage_pos = unwrap_cage_positions(
                    cage.nodes,
                    cage.center,
                    pos,
                    box,
                )

                handle.write(
                    format_pdb_atom_line(
                        serial=serial,
                        name="CNT",
                        resname="CGE",
                        resid=cage_index,
                        element="X",
                        pos=cage_center,
                        occ=1.0,
                        beta=beta,
                        is_het=True,
                    )
                )
                serial += 1

                for local_index, node_index in enumerate(cage.nodes):
                    serial_map[(cage_index, node_index)] = serial
                    handle.write(
                        format_pdb_atom_line(
                            serial=serial,
                            name=element,
                            resname=resname,
                            resid=cage_index,
                            element=element,
                            pos=cage_pos[local_index],
                            occ=1.0,
                            beta=beta,
                            is_het=True,
                        )
                    )
                    serial += 1

            written_edges = set()

            for cage_index, cage in enumerate(res.cages, start=1):
                for atom_u, atom_v in cage.edges:
                    serial_u = serial_map.get((cage_index, atom_u))
                    serial_v = serial_map.get((cage_index, atom_v))

                    if serial_u is None or serial_v is None:
                        continue

                    edge = tuple(sorted((serial_u, serial_v)))

                    if edge in written_edges:
                        continue

                    handle.write(f"CONECT{serial_u:5d}{serial_v:5d}\n")
                    written_edges.add(edge)

            handle.write("END\n")

    def offsets(self) -> dict:
        """Byte offset of every open output file, for checkpointing."""
        result = {}

        for handle in (
            self.ts_file,
            self.label_file,
            self.hop_pdb_file,
            self.cage_stat_file,
            self.shell_stat_file,
        ):
            if handle is None:
                continue

            try:
                handle.flush()
                result[handle.name] = int(handle.tell())
            except OSError:
                continue

        return result

    def close(self):
        self.ts_file.close()
        if self.label_file: self.label_file.close()
        if self.hop_pdb_file:
            self.hop_pdb_file.write("END\n")
            self.hop_pdb_file.close()
        if self.cage_stat_file: self.cage_stat_file.close()
        if self.shell_stat_file: self.shell_stat_file.close()


# == 辅助格式化函数 ==

def format_pdb_cryst1(box: np.ndarray) -> str:
    return (f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}"
            f"{box[3]:7.2f}{box[4]:7.2f}{box[5]:7.2f} P 1           1\n")

def format_pdb_atom_line(serial, name, resname, resid, element, pos, occ, beta, is_het=False) -> str:
    record = "HETATM" if is_het else "ATOM  "
    name = f"{name:>4}"[:4]
    resname = f"{resname:>3}"[:3]
    element = f"{element:>2}"[:2]
    return (f"{record}{serial%100000:5d} {name} {resname} A{resid%10000:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{occ:6.2f}{beta:6.2f}          {element}\n")
