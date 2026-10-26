# PyMOL script for HOP label trajectory
load ../06_labels/hydrate_labels.pdb, hop_labels
hide everything, hop_labels
show spheres, hop_labels
set sphere_scale, 0.35, hop_labels
bg_color white

set_color hop_label_0, [0.200, 0.350, 1.000]
select hop_0, hop_labels and b = 0.0
color hop_label_0, hop_0

set_color hop_label_1, [0.000, 0.850, 1.000]
select hop_1, hop_labels and b = 1.0
color hop_label_1, hop_1

set_color hop_label_2, [0.650, 0.950, 1.000]
select hop_2, hop_labels and b = 2.0
color hop_label_2, hop_2

set_color hop_label_3, [1.000, 0.100, 0.100]
select hop_3, hop_labels and b = 3.0
color hop_label_3, hop_3

set_color hop_label_4, [1.000, 0.550, 0.000]
select hop_4, hop_labels and b = 4.0
color hop_label_4, hop_4

set_color hop_label_5, [0.550, 0.550, 0.550]
select hop_5, hop_labels and b = 5.0
color hop_label_5, hop_5

set all_states, on
orient hop_labels
