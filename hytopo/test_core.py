from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .cli import build_parser, make_config
from .config import CAGE_TYPES, AnalysisConfig, TrajectoryConfig
from .guest import (
    covalent_radius,
    group_centres,
    grouping_by_bonds,
    grouping_by_residue,
    infer_bond_components,
)
from .trajectory import create_context, read_frame
from .nucleation import (
    NucleationParameters,
    NucleationPoint,
    analyze_nucleation,
    read_nmax_hyd_from_timeseries,
)
from .tracking import CageTracker, TrackingParameters, estimate_occupancy_count
from .network import TransitionNetwork, UNMATCHED_NODE
from .extxyz import ExtXYZReader, select_extxyz_atoms
from .phase import PHASE_SH, PHASE_SI, PHASE_SII, PhaseParameters, classify_phase


class ConfigurationTests(unittest.TestCase):
    def test_skip_hop_cage_mode_is_valid(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "-top",
                "system.gro",
                "-guest",
                "resname CH4",
                "--find-cages",
                "--skip-hop",
                "--no-numba",
                "--no-plots",
            ]
        )
        config = make_config(args)
        config.validate()

        self.assertFalse(config.run_hop)
        self.assertEqual(config.cage.ring_backend, "python")
        self.assertFalse(config.output.make_plots)

    def test_partial_types_are_reportable(self) -> None:
        self.assertIn("partial_5^12", CAGE_TYPES)
        self.assertIn("partial_5^12_6^2", CAGE_TYPES)
        self.assertIn("partial_5^12_6^4", CAGE_TYPES)


class NucleationTests(unittest.TestCase):
    def test_new_and_legacy_timeseries_are_read_correctly(self) -> None:
        cases = (
            (
                "Frame Time(ps) H_total H_crys H_pre Hyd_order Ice O_total "
                "Delta Nmax_hyd Nmax_ice\n"
                "10 2.5 0 0 0 0 0 0 0 51.7 4\n",
                10,
            ),
            (
                "Time(ps) H_total H_crys H_pre Hyd_order Ice O_total "
                "Delta Nmax_hyd Nmax_ice\n"
                "2.5 0 0 0 0 0 0 0 51.7 4\n",
                0,
            ),
        )

        for content, expected_frame in cases:
            with self.subTest(expected_frame=expected_frame):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "timeseries.dat"
                    path.write_text(content, encoding="utf-8")
                    points = read_nmax_hyd_from_timeseries(str(path))

                self.assertEqual(len(points), 1)
                self.assertEqual(points[0].frame_index, expected_frame)
                self.assertEqual(points[0].nmax_hyd, 52)

    def test_allowed_gap_persistence(self) -> None:
        points = [
            NucleationPoint(index, float(index), value)
            for index, value in enumerate((50, 49, 51, 52))
        ]
        result = analyze_nucleation(
            points,
            NucleationParameters(
                method="fixed",
                min_cluster_size=50,
                persist_frames=3,
                allow_gaps=1,
            ),
        )
        self.assertTrue(result.nucleated)
        self.assertEqual(result.nucleation_frame, 0)

    def test_adaptive_commitment_uses_liquid_baseline_and_growth(self) -> None:
        points = [
            NucleationPoint(index, float(index), value)
            for index, value in enumerate((2, 3, 2, 10, 12, 16))
        ]
        result = analyze_nucleation(
            points,
            NucleationParameters(
                method="commitment",
                persist_frames=3,
                baseline_frames=3,
                baseline_sigma=3.0,
                adaptive_min_size=5,
                commitment_min_growth=5.0,
            ),
        )
        self.assertTrue(result.nucleated)
        self.assertEqual(result.nucleation_frame, 3)
        self.assertEqual(result.effective_threshold, 5)


class TrackingTests(unittest.TestCase):
    def test_occupancy_uses_periodic_boundary(self) -> None:
        count = estimate_occupancy_count(
            center=(0.2, 0.0, 0.0),
            guest_positions=np.asarray([[9.8, 0.0, 0.0]]),
            radius=1.0,
            box=np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
        )
        self.assertEqual(count, 1)

    def test_trajectory_end_is_not_counted_as_dissociation(self) -> None:
        params = TrackingParameters(gap_tolerance=0)

        surviving = CageTracker(params)
        surviving.update(
            0,
            0.0,
            [{"guest_id": 0, "cage_type": "5^12", "nodes": (1, 2, 3, 4)}],
        )
        surviving.finalize()
        network = TransitionNetwork(".")
        network.build_from_tracker(surviving)
        self.assertNotIn(("5^12", "Liquid"), network.transitions)

        disappeared = CageTracker(params)
        disappeared.update(
            0,
            0.0,
            [{"guest_id": 0, "cage_type": "5^12", "nodes": (1, 2, 3, 4)}],
        )
        disappeared.update(1, 1.0, [])
        disappeared.finalize()
        network = TransitionNetwork(".")
        network.build_from_tracker(disappeared)
        self.assertNotIn(("5^12", "Liquid"), network.transitions)
        self.assertEqual(network.transitions[(UNMATCHED_NODE, "5^12")], 1)
        self.assertEqual(network.transitions[("5^12", UNMATCHED_NODE)], 1)
        self.assertEqual(network.births["5^12"], 1)
        self.assertEqual(network.deaths["5^12"], 1)


class ExtXYZTests(unittest.TestCase):
    def test_periodic_extended_xyz_is_indexed_and_read(self) -> None:
        content = (
            "2\n"
            'time=1.5 pbc="T T T" '
            'Lattice="20 0 0 0 20 0 0 0 20" '
            "Properties=species:S:1:pos:R:3:force:R:3\n"
            "O 1 2 3 0 0 0\n"
            "C 4 5 6 0 0 0\n"
            "2\n"
            'time=2.5 pbc="T T T" '
            'Lattice="21 0 0 0 22 0 0 0 23" '
            "Properties=species:S:1:pos:R:3\n"
            "O 2 3 4\n"
            "C 5 6 7\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.etxyz"
            path.write_text(content, encoding="utf-8")
            reader = ExtXYZReader(str(path))
            frame = reader.read(1)

        self.assertEqual(len(reader), 2)
        self.assertEqual(frame.symbols, ("O", "C"))
        np.testing.assert_allclose(frame.box[:3], (21.0, 22.0, 23.0))
        np.testing.assert_allclose(frame.positions[1], (5.0, 6.0, 7.0))
        self.assertEqual(frame.time_ps, 2.5)
        oxygen = select_extxyz_atoms(
            "(resname SOL) and (name OW)",
            frame.symbols,
            oxygen_fallback=True,
        )
        np.testing.assert_array_equal(oxygen, (0,))


class PhaseTests(unittest.TestCase):
    class _Cage:
        def __init__(self, cage_type: str):
            self.cage_type = cage_type

    def _classify(self, counts: dict[str, int]):
        cages = [
            self._Cage(cage_type)
            for cage_type, count in counts.items()
            for _ in range(count)
        ]
        return classify_phase(
            0,
            0.0,
            cages,
            PhaseParameters(min_complete_cages=4, min_confidence=0.4),
        )

    def test_standard_phase_compositions(self) -> None:
        self.assertEqual(
            self._classify({"5^12": 2, "5^12_6^2": 6}).phase,
            PHASE_SI,
        )
        self.assertEqual(
            self._classify({"5^12": 16, "5^12_6^4": 8}).phase,
            PHASE_SII,
        )
        self.assertEqual(
            self._classify(
                {
                    "5^12": 3,
                    "4^3_5^6_6^3": 2,
                    "5^12_6^8": 1,
                }
            ).phase,
            PHASE_SH,
        )



class GuestGroupingTests(unittest.TestCase):
    """A guest is a molecule; one all-atom CH4 must not count as five guests."""

    @staticmethod
    def methane(cx: float, cy: float, cz: float):
        offsets = (
            (0.63, 0.63, 0.63),
            (0.63, -0.63, -0.63),
            (-0.63, 0.63, -0.63),
            (-0.63, -0.63, 0.63),
        )
        atoms = [("C", (cx, cy, cz))]
        atoms.extend(("H", (cx + dx, cy + dy, cz + dz)) for dx, dy, dz in offsets)
        return atoms

    def test_residue_grouping_matches_molecules(self) -> None:
        grouping = grouping_by_residue([0, 0, 0, 0, 0, 1, 1, 1])

        self.assertEqual(grouping.n_groups, 2)
        np.testing.assert_array_equal(grouping.group_index, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(grouping.offsets, [0, 5, 8])

    def test_covalent_radii_are_defined_for_common_elements(self) -> None:
        self.assertAlmostEqual(covalent_radius("C"), 0.76)
        self.assertAlmostEqual(covalent_radius("H"), 0.31)
        self.assertAlmostEqual(covalent_radius("cl"), 1.02)
        self.assertAlmostEqual(covalent_radius("Xx"), 1.50)

    def test_bond_inference_separates_molecules(self) -> None:
        atoms = self.methane(5.0, 5.0, 5.0) + self.methane(15.0, 5.0, 5.0)
        atoms += [
            ("C", (5.0, 15.0, 5.0)),
            ("O", (5.0, 16.16, 5.0)),
            ("O", (5.0, 13.84, 5.0)),
        ]

        positions = np.asarray([position for _, position in atoms])
        symbols = [symbol for symbol, _ in atoms]
        labels = infer_bond_components(positions, symbols, None)

        self.assertEqual(len(np.unique(labels)), 3)
        self.assertEqual(len(set(labels[:5].tolist())), 1)
        self.assertEqual(len(set(labels[5:10].tolist())), 1)
        self.assertEqual(len(set(labels[10:].tolist())), 1)
        self.assertNotEqual(labels[0], labels[5])

    @staticmethod
    def boundary_methane():
        # C-H bonds of 1.09 A; one hydrogen sits across the x boundary.
        positions = np.asarray(
            [
                (0.50, 10.0, 10.0),
                (1.13, 10.63, 10.63),
                (1.13, 9.37, 9.37),
                (19.87, 10.63, 9.37),
                (19.87, 9.37, 10.63),
            ]
        )
        return positions, ["C", "H", "H", "H", "H"]

    def test_bond_inference_uses_periodic_images(self) -> None:
        box = np.asarray([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
        positions, symbols = self.boundary_methane()

        labels = infer_bond_components(positions, symbols, box)

        self.assertEqual(len(np.unique(labels)), 1)

    def test_guest_centre_unwraps_periodic_boundaries(self) -> None:
        box = np.asarray([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
        positions, symbols = self.boundary_methane()

        grouping = grouping_by_bonds(positions, symbols, box)
        centres = group_centres(positions, grouping, box)

        self.assertEqual(grouping.n_groups, 1)
        np.testing.assert_allclose(centres[0], (0.50, 10.0, 10.0), atol=1e-9)

    def test_single_atom_molecules_keep_their_positions(self) -> None:
        positions = np.asarray([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        grouping = grouping_by_residue([0, 1])

        centres = group_centres(positions, grouping, None)

        np.testing.assert_allclose(centres, positions)

    def test_extxyz_guest_atoms_are_grouped_into_molecules(self) -> None:
        atoms = [("O", (10.0, 10.0, 10.0))]
        atoms += [("O", (10.0, 12.0, 10.0))]
        atoms += self.methane(5.0, 5.0, 5.0)
        atoms += self.methane(15.0, 5.0, 5.0)

        lines = [str(len(atoms))]
        lines.append(
            'Lattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3'
        )
        for symbol, (x, y, z) in atoms:
            lines.append(f"{symbol} {x} {y} {z}")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guests.etxyz"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            config = AnalysisConfig(
                trajectory=TrajectoryConfig(
                    top=str(path),
                    water_selection="element O",
                    oxygen_selection="element O",
                    guest_selection="index 2:12",
                ),
                run_hop=False,
                find_cages=True,
            )
            context = create_context(config)

        self.assertEqual(context.n_guest_atoms, 10)
        self.assertEqual(context.n_guest, 2)
        self.assertEqual(context.guest_grouping.method, "bonded")
        np.testing.assert_array_equal(context.guest_grouping.offsets, [0, 5, 10])

    def test_bond_criterion_uses_a_covalent_radius_factor(self) -> None:
        # C-C limit at the default scale is 1.15 * (0.76 + 0.76) = 1.748 A
        far_pair = np.asarray([(0.0, 0.0, 0.0), (1.70, 0.0, 0.0)])
        near_pair = np.asarray([(0.0, 0.0, 0.0), (1.80, 0.0, 0.0)])

        self.assertEqual(
            len(np.unique(infer_bond_components(far_pair, ["C", "C"], None))), 1
        )
        self.assertEqual(
            len(np.unique(infer_bond_components(far_pair, ["C", "C"], None, 1.0))), 2
        )
        self.assertEqual(
            len(np.unique(infer_bond_components(near_pair, ["C", "C"], None))), 2
        )

    def test_bond_scale_is_configurable_from_the_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["-top", "system.tpr", "-guest", "resname CH4", "--guest-bond-scale", "1.3"]
        )
        config = make_config(args)
        config.validate()

        self.assertAlmostEqual(config.trajectory.guest_bond_scale, 1.3)

    @staticmethod
    def write_methane_pdb(path: Path) -> None:
        lines = ["CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"]
        serial = 0

        def atom(name, resname, resid, element, position, het=False):
            nonlocal serial
            serial += 1
            record = "HETATM" if het else "ATOM  "
            x, y, z = position
            lines.append(
                f"{record}{serial % 100000:5d} {name:>4} {resname:>3} "
                f"A{resid % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.0:6.2f}{0.0:6.2f}          {element:>2}\n"
            )

        atom("OW", "SOL", 1, "O", (5.0, 5.0, 5.0))
        atom("HW1", "SOL", 1, "H", (5.76, 5.59, 5.0))
        atom("HW2", "SOL", 1, "H", (4.24, 5.59, 5.0))

        for resid, centre in ((2, (5.0, 25.0, 25.0)), (3, (25.0, 25.0, 25.0))):
            cx, cy, cz = centre
            atom("C", "CH4", resid, "C", centre, het=True)
            for dx, dy, dz in (
                (0.63, 0.63, 0.63),
                (0.63, -0.63, -0.63),
                (-0.63, 0.63, -0.63),
                (-0.63, -0.63, 0.63),
            ):
                atom("H", "CH4", resid, "H", (cx + dx, cy + dy, cz + dz), het=True)

        lines.append("END\n")
        path.write_text("".join(lines), encoding="utf-8")

    def make_methane_context(self, guest_selection: str):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)

        path = Path(directory) / "methane.pdb"
        self.write_methane_pdb(path)

        config = AnalysisConfig(
            trajectory=TrajectoryConfig(
                top=str(path),
                water_selection="resname SOL",
                oxygen_selection="name OW",
                guest_selection=guest_selection,
            ),
            run_hop=False,
            find_cages=True,
        )
        context = create_context(config)
        self.addCleanup(context.universe.trajectory.close)

        return context

    def test_selecting_only_the_central_atom_gives_one_atom_per_guest(self) -> None:
        context = self.make_methane_context("resname CH4 and name C")
        frame = read_frame(context, 0)

        self.assertEqual(context.n_guest_atoms, 2)
        self.assertEqual(context.n_guest, 2)
        np.testing.assert_allclose(
            frame.guest_positions, [[5.0, 25.0, 25.0], [25.0, 25.0, 25.0]]
        )

    def test_all_atom_selection_uses_the_molecular_centre(self) -> None:
        context = self.make_methane_context("resname CH4")
        frame = read_frame(context, 0)

        self.assertEqual(context.n_guest_atoms, 10)
        self.assertEqual(context.n_guest, 2)
        # the tetrahedral hydrogens are symmetric, so the centroid is the carbon
        np.testing.assert_allclose(
            frame.guest_positions, [[5.0, 25.0, 25.0], [25.0, 25.0, 25.0]], atol=1e-9
        )

    def test_guest_grouping_none_keeps_legacy_behaviour(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "-top",
                "system.gro",
                "-guest",
                "resname CH4",
                "--guest-grouping",
                "none",
            ]
        )
        config = make_config(args)
        config.validate()

        self.assertEqual(config.trajectory.guest_grouping, "none")
        self.assertAlmostEqual(config.trajectory.guest_bond_scale, 1.15)


if __name__ == "__main__":
    unittest.main()
