from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .output_layout import TRACKING, categorised_path

import numpy as np


@dataclass(frozen=True)
class TrackingParameters:
    enabled: bool = True
    gap_tolerance: int = 1
    min_water_overlap: float = 0.50
    min_same_guest_overlap: float = 0.25
    dedup_water_overlap: float = 0.85
    occupancy_radius: float = 4.5
    occupancy_center: str = "cage"
    #: frames within which a new cage around the same guest is a recapture
    recapture_window: int = 5
    write_lifetime: bool = True
    write_transitions: bool = True
    write_occupancy: bool = True
    lifetime_name: str = "cage_lifetime.dat"
    transitions_name: str = "cage_transitions.dat"
    occupancy_name: str = "cage_occupancy.dat"


@dataclass(frozen=True)
class CageEvent:
    frame_index: int
    time: float
    guest_id: int
    cage_type: str
    water_ids: tuple[int, ...]
    score: float = 0.0
    center: Optional[tuple[float, float, float]] = None
    occupancy_count: int = -1
    occupancy: str = "unknown"
    raw_index: int = -1
    status: str = "complete"
    radius_mean: float = -1.0
    radius_std: float = -1.0
    sphericity: float = -1.0
    anisotropy: float = -1.0
    _water_key: frozenset = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # The key is used in every track comparison; rebuilding the frozenset
        # there dominated a full-core run.
        if self._water_key is None:
            object.__setattr__(self, "_water_key", frozenset(self.water_ids))

    @property
    def water_key(self) -> frozenset[int]:
        return self._water_key


@dataclass
class CageTransition:
    track_id: int
    guest_id: int
    frame_index: int
    time: float
    from_type: str
    to_type: str


#: frames over which the per-guest environment is smoothed
ENVIRONMENT_WINDOW = 3


@dataclass
class CageTrack:
    track_id: int
    guest_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    last_frame: int
    last_time: float
    last_type: str
    last_water_ids: tuple[int, ...]
    last_score: float
    n_observed: int = 1
    n_missing: int = 0
    type_counts: Counter = field(default_factory=Counter)
    occupancy_counts: Counter = field(default_factory=Counter)
    transitions: list[CageTransition] = field(default_factory=list)
    first_type: str = ""
    active: bool = True
    end_reason: str = "active"
    last_water_key: frozenset = field(default_factory=frozenset)

    def update(self, event: CageEvent, transition: Optional[CageTransition]) -> None:
        self.end_frame = event.frame_index
        self.end_time = event.time
        self.last_frame = event.frame_index
        self.last_time = event.time
        self.last_type = event.cage_type
        self.last_water_ids = event.water_ids
        self.last_water_key = event.water_key
        self.last_score = event.score
        self.n_observed += 1
        self.n_missing = 0
        self.type_counts[event.cage_type] += 1
        self.occupancy_counts[event.occupancy] += 1

        if transition is not None:
            self.transitions.append(transition)

    @property
    def dominant_type(self) -> str:
        if not self.type_counts:
            return self.last_type
        return self.type_counts.most_common(1)[0][0]

    @property
    def dominant_occupancy(self) -> str:
        if not self.occupancy_counts:
            return "unknown"
        return self.occupancy_counts.most_common(1)[0][0]

    @property
    def lifetime_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def lifetime_time(self) -> float:
        return self.end_time - self.start_time


class CageTracker:
    def __init__(
        self,
        params: Optional[TrackingParameters] = None,
        outdir: Optional[str] = None,
    ) -> None:
        self.params = params if params is not None else TrackingParameters()
        self.outdir = outdir
        self._streamed = outdir is not None
        self._resume_streams = False
        self._occupancy_file = None
        self._quality_file = None
        self.occupancy_counts: Counter = Counter()
        self.tracks: list[CageTrack] = []
        self.active_track_ids: set[int] = set()
        self.transitions: list[CageTransition] = []
        self.occupancy_events: list[CageEvent] = []

        self._next_track_id = 1
        # Active tracks indexed by guest and by water node.  A track can only
        # be matched by an event that shares a water node with it or belongs to
        # the same guest (a track without either scores at most 0.15, below
        # both acceptance thresholds), so indexing prunes the candidate scan
        # without changing the outcome.
        self._guest_index: dict[int, set[int]] = {}
        self._water_index: dict[int, set[int]] = {}
        self.guest_events: list[dict[str, Any]] = []
        #: per-guest environment (LIQ / interface / cage type) and the
        #: state changes that feed the transition network
        self.guest_state: dict[int, str] = {}
        self.guest_history: dict[int, list[str]] = {}
        self.environment_transitions: list[tuple[int, int, str, str]] = []
        self._guest_last_end: dict[int, tuple[int, float, str]] = {}

    def _track_environments(self, frame_index: int, cages: Any) -> None:
        """Record changes of the per-guest environment.

        A hard majority vote per frame flickers when individual shell waters
        cross the HOP threshold, so the state is the majority over the last
        ``EnvironmentWindow`` frames (ties keep the current state).
        """
        environments = getattr(cages, "guest_environments", None) or []
        window = max(1, int(ENVIRONMENT_WINDOW))

        # a detected cage is itself a state: the guest environment is only
        # used when no cage was found around that guest
        cage_by_guest = {
            int(cage.guest_id): cage.cage_type
            for cage in (getattr(cages, "cages", None) or [])
        }

        for guest_id, state in enumerate(environments):
            state = cage_by_guest.get(int(guest_id)) or state

            if not state or state == "unknown":
                continue

            history = self.guest_history.setdefault(int(guest_id), [])
            history.append(state)

            if len(history) > window:
                del history[0]

            counts = Counter(history)
            top = max(counts.values())
            candidates = [value for value, count in counts.items() if count == top]
            previous = self.guest_state.get(int(guest_id))

            if previous in candidates:
                smooth = previous
            else:
                smooth = candidates[0]

            if previous is None:
                self.guest_state[int(guest_id)] = smooth
                continue

            if smooth != previous:
                self.environment_transitions.append(
                    (int(frame_index), int(guest_id), previous, smooth)
                )
                self.guest_state[int(guest_id)] = smooth

    def update(
        self,
        frame_index: int,
        time: float,
        cages: Any,
        guest_positions: Optional[np.ndarray] = None,
        oxygen_positions: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> list[CageEvent]:
        self._track_environments(frame_index, cages)
        records = flatten_cage_records(cages)

        events = build_cage_events(
            frame_index=frame_index,
            time=time,
            records=records,
            guest_positions=guest_positions,
            oxygen_positions=oxygen_positions,
            box=box,
            params=self.params,
        )

        events = deduplicate_cage_events(events, self.params)

        for event in events:
            self.occupancy_counts[event.occupancy] += 1

        if self._streamed:
            self._write_streamed_rows(events)
        else:
            self.occupancy_events.extend(events)

        unmatched_track_ids = set(self.active_track_ids)

        for event in events:
            candidates = self._candidate_track_ids(event, unmatched_track_ids)
            track = self._match_track(event, candidates)

            if track is None:
                self._start_track(event)
                continue

            transition = None

            if track.last_type != event.cage_type:
                transition = CageTransition(
                    track_id=track.track_id,
                    guest_id=track.guest_id,
                    frame_index=event.frame_index,
                    time=event.time,
                    from_type=track.last_type,
                    to_type=event.cage_type,
                )
                self.transitions.append(transition)

            self._unindex_track(track)
            track.update(event, transition)
            self._index_track(track)
            unmatched_track_ids.discard(track.track_id)

        self._age_unmatched_tracks(unmatched_track_ids, frame_index, time)
        return events

    def _stream_paths(self) -> tuple[str, str]:
        return (
            categorised_path(self.outdir, self.params.occupancy_name, TRACKING),
            categorised_path(self.outdir, "cage_quality.dat", TRACKING),
        )

    def _open_streams(self) -> None:
        if not self._streamed or self._occupancy_file is not None:
            return

        occupancy_path, quality_path = self._stream_paths()
        os.makedirs(os.path.dirname(occupancy_path), exist_ok=True)

        if self._resume_streams:
            self._occupancy_file = open(occupancy_path, "a", encoding="utf-8")
            self._quality_file = open(quality_path, "a", encoding="utf-8")
            return

        self._occupancy_file = open(occupancy_path, "w", encoding="utf-8")
        self._occupancy_file.write(
            "# guest occupancy per detected cage and frame\n"
            "frame\ttime_ps\tguest_id\tcage_type\toccupancy\toccupancy_count\t"
            "n_waters\tscore\tstatus\n"
        )

        self._quality_file = open(quality_path, "w", encoding="utf-8")
        self._quality_file.write(
            "# per-cage shape quality (periodic, relative to the cage centre)\n"
            "frame\ttime_ps\tguest_id\tcage_type\tn_waters\t"
            "radius_mean\tradius_std\tsphericity\tanisotropy\n"
        )

    def _write_streamed_rows(self, events: list[CageEvent]) -> None:
        self._open_streams()

        occupancy = []
        quality = []

        for event in events:
            occupancy.append(
                f"{event.frame_index}\t{event.time:.3f}\t{event.guest_id}\t"
                f"{event.cage_type}\t{event.occupancy}\t{event.occupancy_count}\t"
                f"{len(event.water_ids)}\t{event.score:.3f}\t{event.status}\n"
            )

            if event.radius_mean >= 0.0:
                quality.append(
                    f"{event.frame_index}\t{event.time:.3f}\t{event.guest_id}\t"
                    f"{event.cage_type}\t{len(event.water_ids)}\t"
                    f"{event.radius_mean:.4f}\t{event.radius_std:.4f}\t"
                    f"{event.sphericity:.4f}\t{event.anisotropy:.4f}\n"
                )

        if occupancy:
            self._occupancy_file.write("".join(occupancy))
        if quality:
            self._quality_file.write("".join(quality))

    def stream_offsets(self) -> dict:
        offsets = {}

        for handle in (self._occupancy_file, self._quality_file):
            if handle is None:
                continue

            try:
                handle.flush()
                offsets[handle.name] = int(handle.tell())
            except OSError:
                continue

        return offsets

    def _close_streams(self) -> None:
        for name in ("_occupancy_file", "_quality_file"):
            handle = getattr(self, name, None)

            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass

                setattr(self, name, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_occupancy_file"] = None
        state["_quality_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._resume_streams = True
        self._occupancy_file = None
        self._quality_file = None

    def finalize(self) -> None:
        for track_id in list(self.active_track_ids):
            track = self.tracks[track_id - 1]
            track.active = False
            track.end_reason = "trajectory_end"
            self._unindex_track(track)

        self.active_track_ids.clear()

    def write_outputs(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)

        if self.params.write_lifetime:
            write_lifetime_table(
                categorised_path(outdir, self.params.lifetime_name, TRACKING),
                self.lifetime_rows(),
            )

        if self.params.write_transitions:
            transition_path = categorised_path(outdir, self.params.transitions_name, TRACKING)
            write_transition_table(transition_path, self.transition_rows())
            write_transition_summary(
                categorised_path(outdir, "cage_transition_summary.dat", TRACKING),
                self.transition_summary_rows(),
            )

        if self.params.write_occupancy and not self._streamed:
            write_occupancy_table(
                categorised_path(outdir, self.params.occupancy_name, TRACKING),
                self.occupancy_rows(),
            )

        if not self._streamed:
            write_cage_quality_table(
                categorised_path(outdir, "cage_quality.dat", TRACKING),
                self.occupancy_events,
            )

        write_guest_events_table(
            categorised_path(outdir, "guest_events.dat", TRACKING),
            self.guest_events,
        )

        self._close_streams()

    def lifetime_rows(self) -> list[dict[str, Any]]:
        rows = []

        for track in self.tracks:
            rows.append(
                {
                    "track_id": track.track_id,
                    "guest_id": track.guest_id,
                    "start_frame": track.start_frame,
                    "end_frame": track.end_frame,
                    "start_time": track.start_time,
                    "end_time": track.end_time,
                    "lifetime_frames": track.lifetime_frames,
                    "lifetime_time": track.lifetime_time,
                    "n_observed": track.n_observed,
                    "dominant_type": track.dominant_type,
                    "last_type": track.last_type,
                    "dominant_occupancy": track.dominant_occupancy,
                    "n_transitions": len(track.transitions),
                    "end_reason": track.end_reason,
                }
            )

        return rows

    def transition_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "track_id": item.track_id,
                "guest_id": item.guest_id,
                "frame_index": item.frame_index,
                "time": item.time,
                "from_type": item.from_type,
                "to_type": item.to_type,
            }
            for item in self.transitions
        ]

    def transition_summary_rows(self) -> list[dict[str, Any]]:
        counter = Counter((item.from_type, item.to_type) for item in self.transitions)

        return [
            {
                "from_type": from_type,
                "to_type": to_type,
                "count": count,
            }
            for (from_type, to_type), count in sorted(counter.items())
        ]

    def occupancy_rows(self) -> list[dict[str, Any]]:
        rows = []

        for event in self.occupancy_events:
            rows.append(
                {
                    "frame_index": event.frame_index,
                    "time": event.time,
                    "guest_id": event.guest_id,
                    "cage_type": event.cage_type,
                    "occupancy": event.occupancy,
                    "occupancy_count": event.occupancy_count,
                    "n_waters": len(event.water_ids),
                    "score": event.score,
                    "status": event.status,
                }
            )

        return rows

    def _candidate_track_ids(
        self,
        event: CageEvent,
        unmatched_track_ids: set[int],
    ) -> set[int]:
        """Tracks that can possibly win the match for this event."""
        if not event.water_ids:
            # Empty water sets compare with Jaccard == 1.0 against each other,
            # so the node index cannot prune here.
            return set(unmatched_track_ids)

        candidates: set[int] = set()

        for track_id in self._guest_index.get(event.guest_id, ()):
            if track_id in unmatched_track_ids:
                candidates.add(track_id)

        water_index = self._water_index

        for node in event.water_ids:
            for track_id in water_index.get(node, ()):
                if track_id in unmatched_track_ids:
                    candidates.add(track_id)

        return candidates

    def _index_track(self, track: CageTrack) -> None:
        self._guest_index.setdefault(track.guest_id, set()).add(track.track_id)

        for node in track.last_water_key:
            self._water_index.setdefault(node, set()).add(track.track_id)

    def _unindex_track(self, track: CageTrack) -> None:
        guests = self._guest_index.get(track.guest_id)

        if guests is not None:
            guests.discard(track.track_id)
            if not guests:
                del self._guest_index[track.guest_id]

        water_index = self._water_index

        for node in track.last_water_key:
            track_ids = water_index.get(node)

            if track_ids is not None:
                track_ids.discard(track.track_id)
                if not track_ids:
                    del water_index[node]

    def _start_track(self, event: CageEvent) -> CageTrack:
        previous = self._guest_last_end.pop(event.guest_id, None)

        if previous is not None:
            gap = event.frame_index - previous[0]

            if 0 <= gap <= self.params.recapture_window and event.guest_id >= 0:
                self.guest_events.append(
                    {
                        "frame": event.frame_index,
                        "time": event.time,
                        "guest_id": event.guest_id,
                        "event": "recapture",
                        "cage_type": event.cage_type,
                        "previous_type": previous[2],
                        "gap_frames": gap,
                        "lifetime": 0.0,
                    }
                )

        track = CageTrack(
            track_id=self._next_track_id,
            guest_id=event.guest_id,
            start_frame=event.frame_index,
            end_frame=event.frame_index,
            start_time=event.time,
            end_time=event.time,
            last_frame=event.frame_index,
            last_time=event.time,
            last_type=event.cage_type,
            last_water_ids=event.water_ids,
            last_water_key=event.water_key,
            last_score=event.score,
            type_counts=Counter({event.cage_type: 1}),
            occupancy_counts=Counter({event.occupancy: 1}),
            first_type=event.cage_type,
        )

        self.tracks.append(track)
        self.active_track_ids.add(track.track_id)
        self._index_track(track)
        self._next_track_id += 1
        return track

    def _match_track(
        self,
        event: CageEvent,
        candidate_track_ids: set[int],
    ) -> Optional[CageTrack]:
        best_track = None
        best_score = -1.0

        for track_id in sorted(candidate_track_ids):
            track = self.tracks[track_id - 1]

            if not track.active:
                continue

            if event.frame_index - track.last_frame > self.params.gap_tolerance + 1:
                continue

            score = track_match_score(event, track)

            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None:
            return None

        if best_track.guest_id == event.guest_id and best_score >= self.params.min_same_guest_overlap:
            return best_track

        if best_score >= self.params.min_water_overlap:
            return best_track

        return None

    def _age_unmatched_tracks(
        self,
        unmatched_track_ids: set[int],
        frame_index: int = -1,
        time: float = 0.0,
    ) -> None:
        """Age tracks that were not matched; report the frame the loss is seen.

        The event frame is the frame at which the disappearance is confirmed,
        not the last frame the cage was observed: a cage seen once at frame 0
        and lost right after would otherwise look like an escape at frame 0.
        """
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id - 1]
            track.n_missing += 1

            if track.n_missing > self.params.gap_tolerance:
                track.active = False
                track.end_reason = "disappeared"
                self.active_track_ids.discard(track_id)
                self._unindex_track(track)

                if track.guest_id >= 0:
                    self.guest_events.append(
                        {
                            "frame": frame_index if frame_index >= 0 else track.end_frame,
                            "time": time if frame_index >= 0 else track.end_time,
                            "guest_id": track.guest_id,
                            "event": "escape",
                            "cage_type": track.last_type,
                            "previous_type": track.last_type,
                            "gap_frames": max(
                                0,
                                (frame_index if frame_index >= 0 else track.end_frame)
                                - track.end_frame,
                            ),
                            "lifetime": float(track.lifetime_time),
                        }
                    )
                    self._guest_last_end[track.guest_id] = (
                        track.end_frame,
                        track.end_time,
                        track.last_type,
                    )


def cage_quality(
    water_ids: tuple[int, ...],
    center: Optional[tuple[float, float, float]],
    oxygen_positions: Optional[np.ndarray],
    box: Optional[np.ndarray],
) -> tuple[float, float, float, float]:
    """Vertex-radius and shape metrics of one cage.

    Returns ``(radius_mean, radius_std, sphericity, anisotropy)`` where
    sphericity is ``1 - std/mean`` (1 = all vertices equidistant from the
    centre) and anisotropy is ``(l3 - l1) / l3`` of the vertex covariance
    eigenvalues (0 = isotropic shell).  All distances use the periodic minimum
    image with respect to the cage centre.
    """
    if center is None or oxygen_positions is None or len(water_ids) < 4:
        return -1.0, -1.0, -1.0, -1.0

    indices = np.asarray(water_ids, dtype=np.int64)
    valid = (indices >= 0) & (indices < len(oxygen_positions))

    if not np.any(valid):
        return -1.0, -1.0, -1.0, -1.0

    coords = oxygen_positions[indices[valid]]
    delta = coords - np.asarray(center, dtype=np.float64)

    if box is not None:
        delta = minimum_image_deltas(delta, box)

    radius = np.linalg.norm(delta, axis=1)

    if radius.size == 0:
        return -1.0, -1.0, -1.0, -1.0

    radius_mean = float(np.mean(radius))
    radius_std = float(np.std(radius))
    sphericity = 1.0 - radius_std / radius_mean if radius_mean > 0.0 else 0.0

    covariance = delta.T @ delta / float(len(delta))
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(eigenvalues[-1])
    anisotropy = (largest - float(eigenvalues[0])) / largest if largest > 0.0 else 0.0

    return radius_mean, radius_std, float(sphericity), float(anisotropy)


def build_cage_events(
    frame_index: int,
    time: float,
    records: Iterable[Any],
    guest_positions: Optional[np.ndarray],
    oxygen_positions: Optional[np.ndarray],
    box: Optional[np.ndarray],
    params: TrackingParameters,
) -> list[CageEvent]:
    events = []

    for raw_index, record in enumerate(records):
        guest_id = int(get_first_attr(record, ("guest_id", "guest_index", "guest", "center_guest"), -1))
        cage_type = str(get_first_attr(record, ("cage_type", "type", "name", "label"), "other"))
        water_ids = canonical_water_ids(get_first_attr(record, ("water_ids", "waters", "oxygen_ids", "nodes"), ()))
        score = float(get_first_attr(record, ("score", "quality", "confidence"), 0.0))
        status = str(get_first_attr(record, ("status", "state"), "complete"))

        center = get_event_center(record, water_ids, oxygen_positions, params)
        occupancy_count = estimate_occupancy_count(
            center,
            guest_positions,
            params.occupancy_radius,
            box=box,
        )
        occupancy = classify_occupancy(occupancy_count)

        if guest_id >= 0 and occupancy_count < 0:
            occupancy_count = 1
            occupancy = "single"

        radius_mean, radius_std, sphericity, anisotropy = cage_quality(
            water_ids,
            center,
            oxygen_positions,
            box,
        )

        events.append(
            CageEvent(
                frame_index=frame_index,
                time=time,
                guest_id=guest_id,
                cage_type=cage_type,
                water_ids=water_ids,
                score=score,
                center=center,
                occupancy_count=occupancy_count,
                occupancy=occupancy,
                raw_index=raw_index,
                status=status,
                radius_mean=radius_mean,
                radius_std=radius_std,
                sphericity=sphericity,
                anisotropy=anisotropy,
            )
        )

    return events


def deduplicate_cage_events(
    events: list[CageEvent],
    params: TrackingParameters,
) -> list[CageEvent]:
    grouped: dict[int, list[CageEvent]] = defaultdict(list)

    for event in events:
        grouped[event.guest_id].append(event)

    kept = []

    for guest_events in grouped.values():
        ordered = sorted(
            guest_events,
            key=lambda item: (type_priority(item.cage_type), item.score, len(item.water_ids)),
            reverse=True,
        )
        selected: list[CageEvent] = []

        for event in ordered:
            duplicate = False

            for other in selected:
                if event.water_key == other.water_key:
                    duplicate = True
                    break

                if event.cage_type == other.cage_type:
                    overlap = jaccard(event.water_key, other.water_key)

                    if overlap >= params.dedup_water_overlap:
                        duplicate = True
                        break

            if not duplicate:
                selected.append(event)

        kept.extend(selected)

    kept.sort(key=lambda item: (item.frame_index, item.guest_id, item.raw_index))
    return kept


def track_match_score(event: CageEvent, track: CageTrack) -> float:
    water_overlap = jaccard(event.water_key, frozenset(track.last_water_ids))
    same_guest_bonus = 0.35 if event.guest_id == track.guest_id and event.guest_id >= 0 else 0.0
    same_type_bonus = 0.15 if event.cage_type == track.last_type else 0.0
    gap_penalty = max(0, event.frame_index - track.last_frame - 1) * 0.05

    return water_overlap + same_guest_bonus + same_type_bonus - gap_penalty


def flatten_cage_records(result: Any) -> list[Any]:
    if result is None:
        return []

    if isinstance(result, list):
        return result

    if isinstance(result, tuple):
        return list(result)

    records = []

    for name in (
        "cages",
        "records",
        "cage_records",
        "detected_cages",
        "complete_cages",
        "partial_cages",
        "open_cages",
        "distorted_cages",
    ):
        value = get_attr(result, name)

        if value is None:
            continue

        if isinstance(value, list):
            records.extend(value)
        elif isinstance(value, tuple):
            records.extend(value)

    if records:
        return records

    if isinstance(result, dict):
        for key in ("cages", "records", "cage_records"):
            value = result.get(key)

            if isinstance(value, list):
                return value

            if isinstance(value, tuple):
                return list(value)

    return []


def get_event_center(
    record: Any,
    water_ids: tuple[int, ...],
    oxygen_positions: Optional[np.ndarray],
    params: TrackingParameters,
) -> Optional[tuple[float, float, float]]:
    center = get_first_attr(record, ("center", "cage_center", "position"), None)

    if center is not None:
        return tuple(float(value) for value in center)

    if oxygen_positions is None:
        return None

    if params.occupancy_center != "water":
        return None

    if not water_ids:
        return None

    valid_ids = [idx for idx in water_ids if 0 <= idx < len(oxygen_positions)]

    if not valid_ids:
        return None

    coords = oxygen_positions[np.asarray(valid_ids, dtype=np.int64)]
    mean = np.mean(coords, axis=0)
    return tuple(float(value) for value in mean)


def estimate_occupancy_count(
    center: Optional[tuple[float, float, float]],
    guest_positions: Optional[np.ndarray],
    radius: float,
    box: Optional[np.ndarray] = None,
) -> int:
    if center is None or guest_positions is None:
        return -1

    if len(guest_positions) == 0:
        return 0

    center_array = np.asarray(center, dtype=np.float64)
    delta = guest_positions - center_array
    if box is not None:
        delta = minimum_image_deltas(delta, box)
    distances = np.sqrt(np.sum(delta * delta, axis=1))

    return int(np.count_nonzero(distances <= radius))


def minimum_image_deltas(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    dimensions = np.asarray(box, dtype=np.float64)

    if dimensions.size < 3 or np.any(dimensions[:3] <= 0.0):
        return delta

    if dimensions.size < 6 or np.allclose(dimensions[3:6], 90.0):
        lengths = dimensions[:3]
        return delta - lengths * np.rint(delta / lengths)

    a, b, c = dimensions[:3]
    alpha, beta, gamma = np.deg2rad(dimensions[3:6])
    sin_gamma = np.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        return delta

    cell = np.asarray(
        [
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * sin_gamma, 0.0],
            [
                c * np.cos(beta),
                c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma,
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    z_sq = c * c - cell[2, 0] ** 2 - cell[2, 1] ** 2
    cell[2, 2] = np.sqrt(max(0.0, z_sq))

    try:
        fractional = delta @ np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return delta

    fractional -= np.rint(fractional)
    return fractional @ cell


def classify_occupancy(count: int) -> str:
    if count < 0:
        return "unknown"

    if count == 0:
        return "empty"

    if count == 1:
        return "single"

    return "multiple"


def canonical_water_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()

    if isinstance(value, np.ndarray):
        values = value.tolist()
    else:
        values = list(value)

    return tuple(sorted({int(item) for item in values}))


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    intersection = len(left & right)

    if intersection == 0:
        return 0.0

    return intersection / (len(left) + len(right) - intersection)


def get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)

    return getattr(obj, name, None)


def get_first_attr(obj: Any, names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        value = get_attr(obj, name)

        if value is not None:
            return value

    return default


def type_priority(cage_type: str) -> int:
    priorities = {
        "5^12": 100,
        "5^12_6^2": 95,
        "5^12_6^3": 92,
        "5^12_6^4": 90,
        "4^3_5^6_6^3": 88,
        "5^12_6^8": 85,
        "half_5^12": 55,
        "half_5^12_6^2": 54,
        "half_5^12_6^4": 53,
        "incipient_cage": 45,
        "open_cage": 40,
        "distorted_cage": 35,
        "partial_5^12": 30,
        "partial_5^12_6^2": 28,
        "partial_5^12_6^4": 26,
        "partial": 20,
        "other": 0,
    }

    return priorities.get(cage_type, 0)


def write_lifetime_table(path: str, rows: list[dict[str, Any]]) -> None:
    columns = (
        "track_id", "guest_id", "start_frame", "end_frame", "start_time", "end_time",
        "lifetime_frames", "lifetime_time", "n_observed", "dominant_type", "last_type",
        "dominant_occupancy", "n_transitions", "end_reason",
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# per-track cage lifetime summary\n")
        handle.write("\t".join(columns) + "\n")

        for row in rows:
            handle.write(
                f"{row['track_id']}\t{row['guest_id']}\t"
                f"{row['start_frame']}\t{row['end_frame']}\t"
                f"{row['start_time']:.3f}\t{row['end_time']:.3f}\t"
                f"{row['lifetime_frames']}\t{row['lifetime_time']:.3f}\t"
                f"{row['n_observed']}\t{row['dominant_type']}\t{row['last_type']}\t"
                f"{row['dominant_occupancy']}\t{row['n_transitions']}\t"
                f"{row['end_reason']}\n"
            )


def write_transition_table(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# cage type transition events (one row per event)\n")
        handle.write("track_id\tguest_id\tframe\ttime_ps\tfrom_type\tto_type\n")

        for row in rows:
            handle.write(
                f"{row['track_id']}\t{row['guest_id']}\t{row['frame_index']}\t"
                f"{row['time']:.3f}\t{row['from_type']}\t{row['to_type']}\n"
            )


def write_transition_summary(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# aggregated cage type transitions\n")
        handle.write("from_type\tto_type\tcount\n")

        for row in rows:
            handle.write(f"{row['from_type']}\t{row['to_type']}\t{row['count']}\n")


def write_occupancy_table(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# guest occupancy per detected cage and frame\n")
        handle.write(
            "frame\ttime_ps\tguest_id\tcage_type\toccupancy\toccupancy_count\t"
            "n_waters\tscore\tstatus\n"
        )

        for row in rows:
            handle.write(
                f"{row['frame_index']}\t{row['time']:.3f}\t"
                f"{row['guest_id']}\t{row['cage_type']}\t"
                f"{row['occupancy']}\t{row['occupancy_count']}\t"
                f"{row['n_waters']}\t{row['score']:.3f}\t{row['status']}\n"
            )


def write_cage_quality_table(path: str, events: list[CageEvent]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# per-cage shape quality (periodic, relative to the cage centre)\n")
        handle.write(
            "frame\ttime_ps\tguest_id\tcage_type\tn_waters\t"
            "radius_mean\tradius_std\tsphericity\tanisotropy\n"
        )

        for event in events:
            if event.radius_mean < 0.0:
                continue

            handle.write(
                f"{event.frame_index}\t{event.time:.3f}\t{event.guest_id}\t"
                f"{event.cage_type}\t{len(event.water_ids)}\t"
                f"{event.radius_mean:.4f}\t{event.radius_std:.4f}\t"
                f"{event.sphericity:.4f}\t{event.anisotropy:.4f}\n"
            )


def write_guest_events_table(path: str, events: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# guest escape (cage lost) and recapture (cage reformed) events\n")
        handle.write(
            "frame\ttime_ps\tguest_id\tevent\tcage_type\tprevious_type\t"
            "gap_frames\tlifetime_ps\n"
        )

        for event in events:
            handle.write(
                f"{event['frame']}\t{event['time']:.3f}\t{event['guest_id']}\t"
                f"{event['event']}\t{event['cage_type']}\t{event['previous_type']}\t"
                f"{event['gap_frames']}\t{event['lifetime']:.3f}\n"
            )
