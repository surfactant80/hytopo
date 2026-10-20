from __future__ import annotations

"""Small, dependency-free reader for periodic extended XYZ trajectories.

The reader intentionally implements the part of the extxyz specification that
is needed by hytopo: ``Lattice``, ``pbc``, ``Properties`` and per-atom
coordinates.  It keeps frame byte offsets, so selected frames can be read
without loading the whole trajectory into memory.
"""

from dataclasses import dataclass
import re
import shlex
from typing import Any, Optional

import numpy as np


_KEY_VALUE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)"
)


@dataclass(frozen=True)
class ExtXYZFrame:
    frame_index: int
    symbols: tuple[str, ...]
    positions: np.ndarray
    box: np.ndarray
    properties: dict[str, np.ndarray]
    comment: str
    time_ps: float


@dataclass(frozen=True)
class ExtXYZFrameIndex:
    frame_index: int
    offset: int
    n_atoms: int


def _parse_value(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    try:
        values = [float(item) for item in value.replace(",", " ").split()]
        return values if len(values) > 1 else values[0]
    except ValueError:
        return value


def parse_comment(comment: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for match in _KEY_VALUE_RE.finditer(comment):
        result[match.group(1)] = _parse_value(match.group(2))
    return result


def parse_properties(specification: str) -> list[tuple[str, str, int]]:
    fields = specification.split(":")
    if len(fields) % 3 != 0:
        raise ValueError(f"Invalid extxyz Properties specification: {specification}")

    result = []
    for index in range(0, len(fields), 3):
        name, kind, width = fields[index : index + 3]
        if kind not in ("S", "I", "R", "L"):
            raise ValueError(f"Unsupported extxyz property type: {kind}")
        result.append((name, kind, int(width)))
    return result


def lattice_to_dimensions(lattice: Any, pbc: Any = None) -> np.ndarray:
    if isinstance(lattice, str):
        lattice = lattice.replace(",", " ").split()
    values = np.asarray(lattice, dtype=np.float64).reshape(-1)
    if values.size == 3:
        cell = np.diag(values)
    elif values.size == 9:
        cell = values.reshape(3, 3)
    else:
        return np.zeros(6, dtype=np.float64)

    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths <= 0.0):
        return np.zeros(6, dtype=np.float64)

    alpha = np.degrees(np.arccos(np.clip(np.dot(cell[1], cell[2]) / (lengths[1] * lengths[2]), -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(np.dot(cell[0], cell[2]) / (lengths[0] * lengths[2]), -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]), -1.0, 1.0)))

    if pbc is not None:
        pbc_values = str(pbc).strip().replace(",", " ").split()
        if len(pbc_values) == 3 and not all(value.lower() in ("t", "true", "1") for value in pbc_values):
            return np.zeros(6, dtype=np.float64)

    return np.asarray([*lengths, alpha, beta, gamma], dtype=np.float64)


def _readline(handle) -> tuple[int, str]:
    offset = handle.tell()
    line = handle.readline()
    if not line:
        return offset, ""
    return offset, line.decode("utf-8", errors="replace").rstrip("\r\n")


class ExtXYZReader:
    def __init__(self, path: str):
        self.path = path
        self.frames: list[ExtXYZFrameIndex] = []
        self._index_file()
        if not self.frames:
            raise ValueError(f"No frames found in extended XYZ file: {path}")

    def _index_file(self) -> None:
        with open(self.path, "rb") as handle:
            frame_index = 0
            while True:
                offset, line = _readline(handle)
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    n_atoms = int(line.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid atom count at byte offset {offset}: {line!r}"
                    ) from exc
                if n_atoms < 0:
                    raise ValueError(f"Negative atom count at frame {frame_index}")

                _, comment = _readline(handle)
                if not comment:
                    raise ValueError(f"Missing comment line at frame {frame_index}")
                self.frames.append(
                    ExtXYZFrameIndex(frame_index, offset, n_atoms)
                )
                for _ in range(n_atoms):
                    _, atom_line = _readline(handle)
                    if not atom_line:
                        raise ValueError(f"Unexpected EOF in frame {frame_index}")
                frame_index += 1

    def __len__(self) -> int:
        return len(self.frames)

    def read(self, frame_index: int) -> ExtXYZFrame:
        frame_info = self.frames[int(frame_index)]
        with open(self.path, "rb") as handle:
            handle.seek(frame_info.offset)
            _, count_line = _readline(handle)
            n_atoms = int(count_line.strip())
            _, comment = _readline(handle)
            metadata = parse_comment(comment)
            property_spec = metadata.get("Properties")

            if property_spec is None:
                fields = [("species", "S", 1), ("pos", "R", 3)]
            else:
                fields = parse_properties(str(property_spec))

            columns: dict[str, list[Any]] = {name: [] for name, _, _ in fields}
            for _ in range(n_atoms):
                _, line = _readline(handle)
                tokens = shlex.split(line)
                cursor = 0
                for name, kind, width in fields:
                    values = tokens[cursor : cursor + width]
                    cursor += width
                    if len(values) != width:
                        raise ValueError(
                            f"Not enough columns in extxyz frame {frame_index}: {line!r}"
                        )
                    if kind == "S":
                        value: Any = values[0] if width == 1 else tuple(values)
                    elif kind == "I":
                        value = int(values[0]) if width == 1 else [int(v) for v in values]
                    elif kind == "L":
                        value = values[0].lower() in ("t", "true", "1") if width == 1 else [
                            v.lower() in ("t", "true", "1") for v in values
                        ]
                    else:
                        value = float(values[0]) if width == 1 else [float(v) for v in values]
                    columns[name].append(value)

        symbols = _column_as_symbols(columns)
        positions = _column_as_positions(columns)
        box = lattice_to_dimensions(metadata.get("Lattice"), metadata.get("pbc"))
        time_ps = _frame_time(metadata, frame_info.frame_index)

        arrays = {}
        for name, values in columns.items():
            if name in ("species", "element", "pos", "positions", "xyz"):
                continue
            arrays[name] = np.asarray(values)

        return ExtXYZFrame(
            frame_index=frame_info.frame_index,
            symbols=symbols,
            positions=positions,
            box=box,
            properties=arrays,
            comment=comment,
            time_ps=time_ps,
        )


def _column_as_symbols(columns: dict[str, list[Any]]) -> tuple[str, ...]:
    for name in ("species", "element", "Z", "symbol", "name"):
        if name in columns:
            return tuple(str(value) for value in columns[name])
    raise ValueError("Extended XYZ frame has no species/element column")


def _column_as_positions(columns: dict[str, list[Any]]) -> np.ndarray:
    for name in ("pos", "positions", "xyz", "coords"):
        if name in columns:
            values = np.asarray(columns[name], dtype=np.float64)
            if values.shape == (len(values), 3):
                return values

    axis_names = (("x", "y", "z"), ("X", "Y", "Z"))
    for names in axis_names:
        if all(name in columns for name in names):
            return np.column_stack([columns[name] for name in names]).astype(np.float64)
    raise ValueError("Extended XYZ frame has no 3D position column")


def _frame_time(metadata: dict[str, Any], frame_index: int) -> float:
    for key in ("time", "Time", "time_ps", "step"):
        value = metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(frame_index)


def select_extxyz_atoms(
    selection: Optional[str],
    symbols: tuple[str, ...],
    properties: Optional[dict[str, np.ndarray]] = None,
    *,
    oxygen_fallback: bool = False,
) -> np.ndarray:
    symbols_array = np.asarray(symbols, dtype=object)
    if not selection or selection.strip().lower() in ("all", "*"):
        return np.arange(len(symbols), dtype=np.int64)

    text = selection.strip()
    tokens = text.replace("(", " ").replace(")", " ").split()
    mask = np.ones(len(symbols), dtype=bool)
    negate = False
    recognized = False
    index = 0

    while index < len(tokens):
        token = tokens[index].lower()
        if token in ("and", "or"):
            index += 1
            continue
        if token == "not":
            negate = not negate
            index += 1
            continue
        if token in ("name", "element", "symbol", "species", "type", "resname"):
            if index + 1 >= len(tokens):
                raise ValueError(f"Incomplete extxyz selection: {selection}")
            value = tokens[index + 1].strip("\"'")
            current = symbols_array == value
            recognized = True
            mask &= ~current if negate else current
            negate = False
            index += 2
            continue
        if token == "index":
            if index + 1 >= len(tokens):
                raise ValueError(f"Incomplete extxyz index selection: {selection}")
            values = tokens[index + 1].split(":")
            if len(values) == 2:
                start, stop = int(values[0]), int(values[1])
                current = np.zeros(len(symbols), dtype=bool)
                current[max(0, start) : min(len(symbols), stop)] = True
            else:
                current = np.zeros(len(symbols), dtype=bool)
                current[int(values[0])] = True
            recognized = True
            mask &= ~current if negate else current
            negate = False
            index += 2
            continue
        index += 1

    if not recognized:
        raise ValueError(
            f"Unsupported extxyz selection {selection!r}; use element/name/species/index/all."
        )

    if not np.any(mask) and oxygen_fallback and "ow" in text.lower():
        return np.flatnonzero(symbols_array == "O").astype(np.int64)
    return np.flatnonzero(mask).astype(np.int64)
