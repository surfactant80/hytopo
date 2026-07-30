hytopo
======

hytopo is a molecular dynamics trajectory analysis tool for gas hydrate systems.
It performs hydrate order-parameter classification, hydrate cage detection,
guest-centered cage tracking, nucleation analysis, cage transition pathway
network generation, and final summary reporting.

hytopo is designed for trajectories readable by MDAnalysis, such as GROMACS
.tpr/.xtc/.trr/.gro/.pdb/.xyz files.


Main Features
=============

1. Hydrate order-parameter analysis
   - Classifies water molecules into liquid-like, ice-like, hydrate-like,
     hydrate precursor, and uncertain/interface states.
   - Outputs hydrate_timeseries.dat and hydrate_labels.dat.
   - Can write a HOP-labeled PDB trajectory for visualization.

2. Cage detection
   - Detects guest-centered hydrate cages from local water oxygen networks.
   - Supports standard cages, partial cages, half cages, open cages,
     distorted cages, incipient cages, and selected non-standard cages.
   - Outputs cage statistics and per-frame cage PDB files.

3. Cage tracking
   - Tracks the same cage around the same guest molecule over time.
   - Reports cage lifetime, cage type transitions, and guest occupancy.
   - Outputs cage_lifetime.dat, cage_transitions.dat, and cage_occupancy.dat.

4. Nucleation analysis
   - Uses Nmax_hyd from hydrate_timeseries.dat.
   - Detects hydrate nucleation when the largest hydrate-like cluster exceeds
     a threshold and persists for a specified number of analyzed frames.
   - Outputs nucleation_summary.txt and nucleation_timeseries.dat.

5. Cage transition pathway network
   - Converts cage transition events into a directed weighted network.
   - Useful for visualizing pathways such as:
       Liquid -> half_5^12 -> 5^12 -> 5^12_6^2
   - Outputs cage_transition_network.png.

6. Final summary report
   - Generates a one-click global report of the simulation.
   - Summarizes nucleation, cage lifetimes, cage diversity, guest occupancy,
     and kinetic trends.
   - Outputs final_summary.txt.
