from __future__ import annotations

"""Checkpoint / resume for long trajectory runs.

The analysis is written to text files as it goes, so a crash at frame 4999 of
5000 used to lose the whole run.  Every ``--checkpoint-every`` frames the
runner now stores the in-memory analysis state (cage tracker, phase tracker,
nucleation analyser, last completed frame) next to the output directory.  On
``--resume`` the state is restored, the already written tables are reopened in
append mode and the run continues from the first unfinished frame.
"""

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

CHECKPOINT_NAME = "checkpoint.pkl"


@dataclass
class CheckpointState:
    signature: str
    last_frame: int
    done_frames: int
    tracker: Optional[Any]
    phase_tracker: Optional[Any]
    nucleation: Optional[Any]
    offsets: dict = None


def run_signature(config: Any) -> str:
    """Fingerprint of everything that must match for a resume to be valid."""
    trajectory = config.trajectory
    payload = (
        os.path.abspath(str(trajectory.top)),
        None if trajectory.traj is None else os.path.abspath(str(trajectory.traj)),
        trajectory.water_selection,
        trajectory.oxygen_selection,
        trajectory.guest_selection,
        trajectory.guest_grouping,
        trajectory.guest_bond_scale,
        trajectory.start,
        trajectory.stop,
        trajectory.stride,
        config.run_hop,
        config.find_cages,
        repr(config.hop),
        repr(config.cage),
    )

    digest = hashlib.sha256()
    digest.update(repr(payload).encode("utf-8"))
    return digest.hexdigest()


def checkpoint_path(config: Any) -> str:
    return os.path.join(config.output.outdir, CHECKPOINT_NAME)


def save_checkpoint(
    config: Any,
    last_frame: int,
    done_frames: int,
    tracker: Optional[Any],
    phase_tracker: Optional[Any],
    nucleation: Optional[Any],
    offsets: Optional[dict] = None,
) -> str:
    path = checkpoint_path(config)
    state = CheckpointState(
        signature=run_signature(config),
        last_frame=int(last_frame),
        done_frames=int(done_frames),
        tracker=tracker,
        phase_tracker=phase_tracker,
        nucleation=nucleation,
        offsets=dict(offsets) if offsets else {},
    )
    temporary = path + ".tmp"

    with open(temporary, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(temporary, path)
    return path


def load_checkpoint(config: Any) -> Optional[CheckpointState]:
    path = checkpoint_path(config)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
    except Exception:
        return None

    if not isinstance(state, CheckpointState):
        return None

    if state.signature != run_signature(config):
        raise ValueError(
            "Existing checkpoint does not match this trajectory or these "
            "parameters; rerun without --resume or delete "
            f"{path}."
        )

    return state


def truncate_outputs(offsets: Optional[dict]) -> None:
    """Cut partially written output back to the checkpointed offsets."""
    if not offsets:
        return

    for path, size in offsets.items():
        if not os.path.exists(path):
            continue

        actual = os.path.getsize(path)

        if actual <= int(size):
            continue

        try:
            with open(path, "r+b") as handle:
                handle.truncate(int(size))
        except OSError:
            continue


def clear_checkpoint(config: Any) -> None:
    path = checkpoint_path(config)

    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
