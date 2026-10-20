# HyTopo performance notes

Reference system: `ARAS1.tpr` + `ARAS1.xtc` — 5888 water oxygens, 1024 united-atom
CH4 guests, 5001 frames, 47.9 x 47.9 x 98.3 Å.

Measured on the same machine with the same interpreter, single core, cage search
enabled (HOP + cage per frame). `benchmarks/*.json` keeps the raw runs.

## Per-frame cost

| stage | mean s/frame | vs. original |
|---|---|---|
| shipped 0.1.0 | 24.30 | 1.0x |
| + nearest-target-first ordering | 13.86 | 1.75x |
| + bitmask beam state | 6.16 | 3.9x |
| + numba beam kernel | 3.41 | 7.1x |
| + frame-level rings, batched fit, indexed dedup | **0.835** | **29x** |

Projection for the full 5001-frame trajectory: 33.8 h -> about 70 min single
core. Late frames (larger hydrate, ~850 cages/frame) take about 1.15 s.

## Where the time went (frame 0, shipped 0.1.0)

| step | s/frame | share |
|---|---|---|
| beam search (dict edge counting) | 18.0 | 87% |
| ring enumeration + SVD fit | 2.5 | 12% |
| HOP classification | 0.052 | 0.25% |
| O-O neighbour graph | 0.006 | 0.03% |

The cage beam search, not HOP, was the bottleneck. Three structural causes:

1. **Rings were recomputed per guest.** A ring belongs to the water network, not
   to a guest; adjacent cages share faces. 1024 guests produced 864 separate
   DFS runs and 5061 planarity fits for 9373 ring incidences per frame.
2. **Beam state was a dict of edge multiplicities.** Every candidate extension
   copied an O(E~150) dict and re-summed it, ~3.4e6 times per frame.
3. **Hopeless targets were searched first.** Targets were tried by ascending
   face count, so a 5^12_6^2 guest spent a full 12-face beam search on 5^12
   before succeeding; 59% of all beam calls never closed.

## What the new kernels do

- `possible_targets` sorts by distance to the observed shell signature, so the
  first candidate is the one that can actually close (beam calls 1043 -> 445).
- `RingAssembly` (`cage.py`) stores the shell as two integer bitmasks
  (edges covered once / twice) and ring membership as a bitmask; a transition
  is a few integer ops instead of an O(E) dict copy.
- `beam.py` ports the identical search to numba with fixed-width `uint64`
  words. Selection order, beam truncation, de-duplication and the final Euler /
  connectivity check are unchanged; the Python engine remains as a fallback.
- `build_frame_ring_table` enumerates the candidate subgraph once per frame,
  fits every ring with one batched eigen solve (`fit_many_rings`), and attaches
  each ring to all guests whose `guest_radius` sphere contains its vertices
  (guest bitmasks, one intersection per ring).
- `deduplicate_cages` uses a water-node index instead of the all-pairs Jaccard
  scan.

## Parallel scaling (12 logical cores, i5-12400)

| workload | 1 core | default (12 processes) | speed-up |
|---|---|---|---|
| 64 frames, per-frame output off | 48.33 s | 11.84 s | 4.08x |
| 32 frames, production config | 49.02 s | 30.35 s | 1.62x |
| 32 frames, production config after the tracking index | 28.99 s | 12.46 s | 2.33x |

`hydrate_timeseries.dat`, `cage_stat.dat` and `guest_shell.dat` are byte-identical
between the serial and parallel runs; `cage_lifetime.dat`, `cage_transitions.dat`
and `cage_occupancy.dat` are byte-identical before and after the tracking change.

The parallel speed-up of short runs is diluted by worker start-up (each worker
parses the topology, ~2.4 s). On a full trajectory that cost is negligible and
the parent-side serial work is the only ceiling; after the tracking index it is
about 150 ms/frame for this system.

## Reproducing

```bash
# correctness: fails loudly if any fingerprint changes
python -m HyTopo.benchmark_regression -top ARAS1.tpr -traj ARAS1.xtc \
  -guest "resname CH4" --frames 0,500,1000,1500 \
  --golden benchmarks/golden_aras1.json

# timing only
python -m HyTopo.benchmark_regression -top ARAS1.tpr -traj ARAS1.xtc \
  -guest "resname CH4" --frames 0,500,1000,1500,2000,2500,3500,4500 --repeat 2
```

The fingerprint covers HOP statistics, a hash of the per-water label array, the
full cage list (guest id, cage type, water nodes), the cage-type counts and a
hash of the complete per-guest shell table, so a kernel change cannot silently
alter the science.

## Still open

- HOP is now a negligible part of the total, but the per-frame result object is
  still pickled through the process pool (~1.3 MB/frame here, far more for
  million-atom systems). Workers should write their own per-frame output and
  return only summaries.
- `ring enumeration` allocates its DFS stack from `max_local_cycles`; the
  frame-level cap is `8 * n_candidate_waters` (bounded at 500k) with a warning
  when it is hit.
- Tracking keeps every cage event in memory for the whole trajectory and the
  per-frame text output (`guest_shell.dat`: one row per guest per frame) grows
  to millions of rows on production runs.
