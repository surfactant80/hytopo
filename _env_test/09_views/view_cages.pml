# PyMOL script for cage PDB files
python
import glob
for pdb in sorted(glob.glob('../07_cages_pdb/cages_frame_*.pdb')):
    cmd.load(pdb)
python end

hide everything
show sticks
set stick_radius, 0.12
bg_color white

set_color cage_512, [1.000, 0.000, 0.000]
select sel_512, b = 1.0
color cage_512, sel_512

set_color cage_512_62, [1.000, 0.500, 0.000]
select sel_512_62, b = 2.0
color cage_512_62, sel_512_62

set_color cage_512_64, [0.000, 0.750, 0.200]
select sel_512_64, b = 3.0
color cage_512_64, sel_512_64

set_color cage_43_56_63, [0.000, 0.900, 0.900]
select sel_43_56_63, b = 4.0
color cage_43_56_63, sel_43_56_63

set_color cage_512_68, [0.100, 0.300, 1.000]
select sel_512_68, b = 5.0
color cage_512_68, sel_512_68

set_color cage_512_61, [0.450, 0.450, 0.450]
select sel_512_61, b = 6.0
color cage_512_61, sel_512_61

set_color cage_512_63, [1.000, 0.850, 0.000]
select sel_512_63, b = 7.0
color cage_512_63, sel_512_63

set_color cage_512_65, [0.450, 0.450, 0.450]
select sel_512_65, b = 8.0
color cage_512_65, sel_512_65

set_color cage_512_66, [0.450, 0.450, 0.450]
select sel_512_66, b = 9.0
color cage_512_66, sel_512_66

set_color cage_41_510_62, [0.450, 0.450, 0.450]
select sel_41_510_62, b = 10.0
color cage_41_510_62, sel_41_510_62

set_color cage_41_510_63, [0.450, 0.450, 0.450]
select sel_41_510_63, b = 11.0
color cage_41_510_63, sel_41_510_63

set_color cage_41_510_64, [0.450, 0.450, 0.450]
select sel_41_510_64, b = 12.0
color cage_41_510_64, sel_41_510_64

set_color cage_42_58_64, [0.450, 0.450, 0.450]
select sel_42_58_64, b = 13.0
color cage_42_58_64, sel_42_58_64

set_color cage_half_512, [1.000, 0.000, 1.000]
select sel_half_512, b = 21.0
color cage_half_512, sel_half_512

set_color cage_half_512_62, [0.600, 0.200, 1.000]
select sel_half_512_62, b = 22.0
color cage_half_512_62, sel_half_512_62

set_color cage_half_512_64, [1.000, 0.450, 0.750]
select sel_half_512_64, b = 23.0
color cage_half_512_64, sel_half_512_64

set_color cage_half_43_56_63, [0.450, 0.450, 0.450]
select sel_half_43_56_63, b = 24.0
color cage_half_43_56_63, sel_half_43_56_63

set_color cage_half_512_68, [0.450, 0.450, 0.450]
select sel_half_512_68, b = 25.0
color cage_half_512_68, sel_half_512_68

set_color cage_half_512_61, [0.450, 0.450, 0.450]
select sel_half_512_61, b = 26.0
color cage_half_512_61, sel_half_512_61

set_color cage_half_512_63, [0.450, 0.450, 0.450]
select sel_half_512_63, b = 27.0
color cage_half_512_63, sel_half_512_63

set_color cage_half_512_65, [0.450, 0.450, 0.450]
select sel_half_512_65, b = 28.0
color cage_half_512_65, sel_half_512_65

set_color cage_half_512_66, [0.450, 0.450, 0.450]
select sel_half_512_66, b = 29.0
color cage_half_512_66, sel_half_512_66

set_color cage_half_41_510_62, [0.450, 0.450, 0.450]
select sel_half_41_510_62, b = 30.0
color cage_half_41_510_62, sel_half_41_510_62

set_color cage_half_41_510_63, [0.450, 0.450, 0.450]
select sel_half_41_510_63, b = 31.0
color cage_half_41_510_63, sel_half_41_510_63

set_color cage_half_41_510_64, [0.450, 0.450, 0.450]
select sel_half_41_510_64, b = 32.0
color cage_half_41_510_64, sel_half_41_510_64

set_color cage_half_42_58_64, [0.450, 0.450, 0.450]
select sel_half_42_58_64, b = 33.0
color cage_half_42_58_64, sel_half_42_58_64

set_color cage_partial_512, [1.000, 0.000, 1.000]
select sel_partial_512, b = 45.0
color cage_partial_512, sel_partial_512

set_color cage_partial_512_62, [0.600, 0.200, 1.000]
select sel_partial_512_62, b = 46.0
color cage_partial_512_62, sel_partial_512_62

set_color cage_partial_512_64, [1.000, 0.450, 0.750]
select sel_partial_512_64, b = 47.0
color cage_partial_512_64, sel_partial_512_64

set_color cage_incipient_cage, [0.750, 0.550, 0.300]
select sel_incipient_cage, b = 41.0
color cage_incipient_cage, sel_incipient_cage

set_color cage_open_cage, [0.750, 0.750, 0.750]
select sel_open_cage, b = 42.0
color cage_open_cage, sel_open_cage

set_color cage_distorted_cage, [0.450, 0.450, 0.450]
select sel_distorted_cage, b = 43.0
color cage_distorted_cage, sel_distorted_cage

set_color cage_partial, [1.000, 1.000, 1.000]
select sel_partial, b = 44.0
color cage_partial, sel_partial

set_color cage_other, [0.450, 0.450, 0.450]
select sel_other, b = 50.0
color cage_other, sel_other

orient all
set valence, 1
