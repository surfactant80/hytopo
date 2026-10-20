from __future__ import annotations

import math
import os
from .output_layout import PLOTS, TRACKING, categorised_path
from collections import defaultdict
from typing import Any


class NetworkControlPanel:
    """
    Hydrate cage transition network plotting controls.
    This file can be used as a module in the full program.

    Main entry points:
        generate_network_from_tracker(tracker, outdir)
        generate_network_from_tables(table_dir, outdir=None)
    """

    # Data filters.  Keep only transitions that actually matter, otherwise the
    # hairball of rare events hides the pathway.
    min_edge_weight = 2
    max_edges = 0

    # Layout.
    # Recommended: "pathway"
    # Alternative : "spring"
    layout_type = "pathway"

    # Edge label mode:
    # "count": transition count
    # "rate" : transition frequency in ns^-1
    label_mode = "rate"

    show_edge_labels = True
    edge_label_min_weight = 60
    #: only annotate the N strongest transitions (0 = annotate every edge above
    #: the weight threshold; labels are dropped individually when they collide)
    max_edge_labels = 12

    # Merge reciprocal A <-> B transitions into one double-headed curve.
    # This avoids duplicated rate/count labels.
    merge_reciprocal_edges = True

    # Node and edge style.
    node_scale = 1.18
    edge_scale = 1.00
    arrow_size = 15

    # Text style.
    #: draw the node text under the marker instead of inside it
    node_label_outside = False
    node_label_font_size = 9.6
    node_label_outside_font_size = 8.4
    node_label_offset = 0.10
    #: widest node label in points; longer labels are shrunk to fit
    node_label_max_width = 80.0
    #: circle diameter / text diagonal
    node_label_padding = 1.30
    edge_label_font_size = 7.6
    title_font_size = 20.0
    group_label_font_size = 12.0
    legend_font_size = 10.5

    # Edge label placement.
    # Labels are placed only on their own curve. They do not drift away.
    edge_label_t_search_window = 0.66
    edge_label_t_step = 0.02
    edge_label_node_margin = 0.18
    edge_label_label_margin = 0.04
    rotate_edge_labels_along_edge = True
    use_edge_label_box = True

    # Figure style.
    figure_width = 21.0
    figure_height = 14.2
    figure_dpi = 300

    title = "Hydrate Cage Structural Evolution Pathway Network"

    #: side panel with the strongest transitions (keeps all edges in the graph
    #: while the numbers stay readable)
    show_transition_panel = True
    transition_panel_rows = 16
    transition_panel_font_size = 8.2
    #: horizontal room (data units) reserved for the panel
    transition_panel_width = 5.6

    # Table inputs/outputs.
    write_tables = True
    edges_table_name = "cage_transition_network_edges.dat"
    nodes_table_name = "cage_transition_network_nodes.dat"
    output_figure_name = "cage_transition_network.png"


class TransitionNetwork:
    """
    Hydrate cage structural evolution pathway network.

    Can be built from a CageTracker or loaded from:
        cage_transition_network_edges.dat
        cage_transition_network_nodes.dat
    """

    def __init__(self, outdir: str):
        self.outdir = outdir
        self.transitions: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.state_counts: defaultdict[str, int] = defaultdict(int)
        self.total_time_ns = 1.0

        self.standard_crystal_types = {
            "5^12",
            "5^12_6^2",
            "5^12_6^4",
            "4^3_5^6_6^3",
            "5^12_6^8",
        }

        self.special_partial_types = {
            "incipient_cage",
            "open_cage",
            "distorted_cage",
            "partial",
        }

        self.crystal_types = self._load_crystal_types()

        self.group_colors = {
            "liquid": "#536878",
            "partial": "#D66B1F",
            "half": "#F39C34",
            "transition": "#8E58B7",
            "amorphous": "#1FA187",
            "crystal": "#D94B3D",
            "other": "#8C99A6",
        }

        self.group_light_colors = {
            "liquid": "#EEF2F6",
            "partial": "#FFF1E6",
            "half": "#FFF7E8",
            "transition": "#F5ECFA",
            "amorphous": "#E9F8F5",
            "crystal": "#FCEBE9",
            "other": "#F2F4F5",
        }

        self.exact_colors = {
            "Liquid": "#536878",
            "5^12": "#D94B3D",
            "5^12_6^2": "#2E86C1",
            "5^12_6^3": "#8E58B7",
            "5^12_6^4": "#38B878",
            "4^3_5^6_6^3": "#D8B20E",
            "5^12_6^8": "#16A085",
            "5^12_6^1": "#A569BD",
            "5^12_6^5": "#7D3C98",
            "5^12_6^6": "#6C3483",
            "4^1_5^10_6^2": "#25BDA4",
            "4^1_5^10_6^3": "#19A88E",
            "4^1_5^10_6^4": "#178F7B",
            "4^2_5^8_6^4": "#127C6D",
            "incipient_cage": "#B26126",
            "open_cage": "#93611F",
            "distorted_cage": "#40576A",
            "partial": "#D66B1F",
        }

    def _load_crystal_types(self) -> set[str]:
        try:
            from .config import COMPLETE_TYPES

            return set(COMPLETE_TYPES)
        except Exception:
            return {
                "5^12",
                "5^12_6^1",
                "5^12_6^2",
                "5^12_6^3",
                "5^12_6^4",
                "5^12_6^5",
                "5^12_6^6",
                "4^3_5^6_6^3",
                "5^12_6^8",
                "4^1_5^10_6^2",
                "4^1_5^10_6^3",
                "4^1_5^10_6^4",
                "4^2_5^8_6^4",
            }

    def build_from_tracker(self, tracker: Any) -> None:
        all_times = []

        for transition in tracker.transitions:
            all_times.append(float(transition.time))

        for track in tracker.tracks:
            all_times.append(float(track.start_time))
            all_times.append(float(track.end_time))

        if len(all_times) >= 2:
            total_time_ps = max(all_times) - min(all_times)
            if total_time_ps > 0.0:
                self.total_time_ns = total_time_ps / 1000.0

        self.transitions.clear()
        self.state_counts.clear()

        for transition in tracker.transitions:
            self.transitions[(transition.from_type, transition.to_type)] += 1
            self.state_counts[transition.from_type] += 1
            self.state_counts[transition.to_type] += 1

        for track in tracker.tracks:
            first_cage = getattr(track, "first_type", "") or track.last_type

            if not first_cage:
                continue
            self.transitions[("Liquid", first_cage)] += 1
            self.state_counts["Liquid"] += 1
            self.state_counts[first_cage] += 1

            if getattr(track, "end_reason", "") == "disappeared":
                self.transitions[(track.last_type, "Liquid")] += 1
                self.state_counts[track.last_type] += 1
                self.state_counts["Liquid"] += 1

    def load_from_tables(
        self,
        edges_file: str,
        nodes_file: str | None = None,
    ) -> None:
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"Edges table not found: {edges_file}")

        self.transitions.clear()
        self.state_counts.clear()

        inferred_times_ns = []

        with open(edges_file, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid edge table line {line_number}: {line}"
                    )

                from_type = parts[0]
                to_type = parts[1]
                count = int(parts[2])

                self.transitions[(from_type, to_type)] += count

                if len(parts) >= 4:
                    rate = float(parts[3])
                    if rate > 0.0:
                        inferred_times_ns.append(count / rate)

        if inferred_times_ns:
            self.total_time_ns = sum(inferred_times_ns) / len(inferred_times_ns)

        if nodes_file is not None and os.path.exists(nodes_file):
            with open(nodes_file, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()

                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()

                    if len(parts) < 3:
                        raise ValueError(
                            f"Invalid node table line {line_number}: {line}"
                        )

                    node = parts[0]
                    count = int(parts[2])
                    self.state_counts[node] = count
        else:
            for (from_type, to_type), count in self.transitions.items():
                self.state_counts[from_type] += count
                self.state_counts[to_type] += count

        for (from_type, to_type), count in self.transitions.items():
            self.state_counts.setdefault(from_type, count)
            self.state_counts.setdefault(to_type, count)

    def write_tables(self) -> None:
        if not NetworkControlPanel.write_tables:
            return

        os.makedirs(self.outdir, exist_ok=True)

        edges_path = categorised_path(self.outdir, NetworkControlPanel.edges_table_name, TRACKING)
        nodes_path = categorised_path(self.outdir, NetworkControlPanel.nodes_table_name, TRACKING)

        with open(edges_path, "w", encoding="utf-8") as handle:
            handle.write("# from_type to_type count rate_per_ns\n")

            for (from_type, to_type), weight in sorted(
                self.transitions.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                rate = weight / self.total_time_ns
                handle.write(f"{from_type} {to_type} {weight} {rate:.6f}\n")

        with open(nodes_path, "w", encoding="utf-8") as handle:
            handle.write("# node group count\n")

            for node, count in sorted(
                self.state_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                group = self._node_group(node)
                handle.write(f"{node} {group} {count}\n")

    def _base_cage_type(self, node: str) -> str:
        if node.startswith("half_"):
            return node[len("half_"):]
        return node

    def _node_group(self, node: str) -> str:
        if node == "Liquid":
            return "liquid"

        if node.startswith("half_"):
            return "half"

        if node.startswith("partial_") or node in self.special_partial_types:
            return "partial"

        base = self._base_cage_type(node)

        if base in self.standard_crystal_types:
            return "crystal"

        if base.startswith("4^"):
            return "amorphous"

        if base in self.crystal_types:
            return "transition"

        if base.startswith("5^12_6^"):
            return "transition"

        return "other"

    def _node_color(self, node: str) -> str:
        if node in self.exact_colors:
            return self.exact_colors[node]

        group = self._node_group(node)
        return self.group_colors.get(group, "#8C99A6")

    def _text_color_for_node(self, node: str) -> str:
        color = self._node_color(node).lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if luminance < 150 else "#101820"

    def _format_node_label(self, node: str) -> str:
        if node == "Liquid":
            return "Liquid"

        if node == "partial":
            return "partial"

        short_names = {
            "incipient_cage": "incipient",
            "open_cage": "open",
            "distorted_cage": "distorted",
        }

        if node in short_names:
            return short_names[node]

        if node.startswith("half_"):
            base = node[len("half_"):]
            return "half\n" + self._format_cage_formula(base)

        if node.startswith("partial_"):
            base = node[len("partial_"):]
            return "partial\n" + self._format_cage_formula(base)

        return self._format_cage_formula(node)

    def _format_cage_formula(self, label: str) -> str:
        return "\n".join(label.split("_"))

    def _filtered_transitions(self) -> dict[tuple[str, str], int]:
        items = [
            ((from_type, to_type), weight)
            for (from_type, to_type), weight in self.transitions.items()
            if weight >= NetworkControlPanel.min_edge_weight
        ]

        items.sort(key=lambda item: item[1], reverse=True)

        if NetworkControlPanel.max_edges > 0:
            items = items[:NetworkControlPanel.max_edges]

        return dict(items)

    def _get_pathway_positions(self, graph: Any) -> dict[str, tuple[float, float]]:
        import numpy as np

        group_order = [
            "partial",
            "half",
            "transition",
            "amorphous",
            "crystal",
            "other",
        ]

        x_position = {
            "liquid": -6.4,
            "partial": -4.0,
            "half": -1.55,
            "transition": 1.05,
            "amorphous": 3.65,
            "crystal": 6.20,
            "other": 0.0,
        }

        y_bands = {
            "partial": np.array([5.10, 3.45, 1.75, 0.00, -1.75, -3.45, -5.10]),
            "half": np.array([1.90, -0.65, -3.20]),
            "transition": np.array([2.00, -0.55, -3.10]),
            "amorphous": np.array([3.70, 1.50, -0.90, -3.40]),
            "crystal": np.array([3.95, 1.75, -0.80, -3.35]),
            "other": np.array([0.0]),
        }

        grouped_nodes: dict[str, list[str]] = {group: [] for group in group_order}

        for node in graph.nodes():
            group = self._node_group(node)
            if group == "liquid":
                continue
            grouped_nodes.setdefault(group, []).append(node)

        pos: dict[str, tuple[float, float]] = {}

        if "Liquid" in graph.nodes:
            pos["Liquid"] = (x_position["liquid"], 0.0)

        for group in group_order:
            nodes = grouped_nodes.get(group, [])

            if not nodes:
                continue

            nodes = sorted(
                nodes,
                key=lambda item: (-self.state_counts.get(item, 0), item),
            )

            preset_y = y_bands.get(group)

            if preset_y is None or len(preset_y) < len(nodes):
                span = max(3.4, 1.85 * (len(nodes) - 1))
                preset_y = np.linspace(span / 2.0, -span / 2.0, len(nodes))

            for index, node in enumerate(nodes):
                x = x_position.get(group, 0.0)
                x += [-0.08, 0.08, 0.0][index % 3]
                y = float(preset_y[index])
                pos[node] = (x, y)

        for node in graph.nodes():
            pos.setdefault(node, (0.0, 0.0))

        return pos

    def _get_positions(self, graph: Any) -> dict[str, tuple[float, float]]:
        if NetworkControlPanel.layout_type == "pathway":
            return self._get_pathway_positions(graph)

        import networkx as nx
        import numpy as np

        node_count = len(graph.nodes())
        k_value = 5.0 / np.sqrt(node_count) if node_count > 0 else 2.0
        return nx.spring_layout(graph, k=k_value, iterations=350, seed=123)

    def _iter_display_edges(self, graph: Any) -> list[dict[str, Any]]:
        display_edges = []
        seen_pairs = set()

        for from_type, to_type, data in graph.edges(data=True):
            weight = data["weight"]
            reverse_weight = None

            if graph.has_edge(to_type, from_type):
                reverse_weight = graph[to_type][from_type]["weight"]

            pair_key = frozenset((from_type, to_type))

            if (
                NetworkControlPanel.merge_reciprocal_edges
                and reverse_weight is not None
                and pair_key in seen_pairs
            ):
                continue

            is_reciprocal = (
                NetworkControlPanel.merge_reciprocal_edges
                and reverse_weight is not None
            )

            display_weight = max(weight, reverse_weight or 0)

            if is_reciprocal and to_type == "Liquid":
                source = to_type
                target = from_type
            else:
                source = from_type
                target = to_type

            display_edges.append(
                {
                    "source": source,
                    "target": target,
                    "weight": display_weight,
                    "forward_weight": weight,
                    "reverse_weight": reverse_weight,
                    "is_reciprocal": is_reciprocal,
                }
            )

            seen_pairs.add(pair_key)

        display_edges.sort(key=lambda item: item["weight"])
        return display_edges

    def _format_edge_label(self, edge: dict[str, Any]) -> str:
        weight = edge["weight"]

        if NetworkControlPanel.label_mode == "rate":
            return f"{weight / self.total_time_ns:.1f} ns$^{{-1}}$"

        return f"{weight}"

    @staticmethod
    def _measure_text(label: str, font_size: float) -> tuple[float, float]:
        """Width and height of a (possibly multi-line) bold label in points."""
        try:
            from matplotlib.font_manager import FontProperties
            from matplotlib.textpath import TextPath

            prop = FontProperties(family="DejaVu Sans", weight="bold")
            width = 0.0

            for line in label.split("\n"):
                if not line:
                    continue
                path = TextPath((0.0, 0.0), line, size=font_size, prop=prop)
                width = max(width, float(path.get_extents().width))

            height = len(label.split("\n")) * font_size * 1.12
            return width, height
        except Exception:
            lines = label.split("\n")
            return 0.60 * font_size * max(len(line) for line in lines), (
                len(lines) * font_size * 1.12
            )

    def _node_label_font(self, node: str) -> float:
        """Font size that keeps the label inside the configured width."""
        base = NetworkControlPanel.node_label_font_size
        label = self._format_node_label(node)
        width, _ = self._measure_text(label, base)

        if width > NetworkControlPanel.node_label_max_width > 0.0:
            base *= NetworkControlPanel.node_label_max_width / width

        return max(base, 7.6)

    def _node_size_for_label(self, node: str) -> float:
        import math

        label = self._format_node_label(node)
        font_size = self._node_label_font(node)
        width, height = self._measure_text(label, font_size)
        diameter = max(width, height) * NetworkControlPanel.node_label_padding
        diameter = max(diameter, 34.0)
        size = math.pi * (diameter / 2.0) ** 2

        return size * NetworkControlPanel.node_scale

    def _node_radii_data_units(
        self,
        graph: Any,
        node_sizes: dict[str, float],
    ) -> dict[str, float]:
        radii = {}

        for node in graph.nodes():
            group = self._node_group(node)
            base_radius = 0.43

            if node == "Liquid":
                base_radius = 0.48
            elif group in ("partial", "half"):
                base_radius = 0.47
            elif group in ("transition", "amorphous", "crystal"):
                base_radius = 0.46

            radii[node] = base_radius * math.sqrt(node_sizes[node] / 2800.0)

        return radii

    def _edge_rad(
        self,
        source: str,
        target: str,
        pos: dict[str, tuple[float, float]],
    ) -> float:
        _, y1 = pos[source]
        _, y2 = pos[target]

        dy = y2 - y1
        rad = 0.10 + min(0.24, abs(dy) * 0.030)

        if dy < 0:
            rad = -rad

        return rad

    def _edge_curve_geometry(
        self,
        source: str,
        target: str,
        pos: dict[str, tuple[float, float]],
        node_radii: dict[str, float],
    ) -> tuple[float, float, float, float, float, float]:
        x1, y1 = pos[source]
        x2, y2 = pos[target]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0

        ux = dx / length
        uy = dy / length

        source_gap = node_radii.get(source, 0.45) + 0.08
        target_gap = node_radii.get(target, 0.45) + 0.10

        if source_gap + target_gap >= length * 0.80:
            source_gap = length * 0.15
            target_gap = length * 0.15

        sx = x1 + ux * source_gap
        sy = y1 + uy * source_gap
        ex = x2 - ux * target_gap
        ey = y2 - uy * target_gap

        ddx = ex - sx
        ddy = ey - sy
        short_length = math.hypot(ddx, ddy) or 1.0

        nx = -ddy / short_length
        ny = ddx / short_length

        rad = self._edge_rad(source, target, pos)

        cx = (sx + ex) / 2.0 + nx * rad * short_length
        cy = (sy + ey) / 2.0 + ny * rad * short_length

        return sx, sy, cx, cy, ex, ey

    def _bezier_point(
        self,
        curve: tuple[float, float, float, float, float, float],
        t: float,
    ) -> tuple[float, float]:
        sx, sy, cx, cy, ex, ey = curve
        u = 1.0 - t

        x = u * u * sx + 2.0 * u * t * cx + t * t * ex
        y = u * u * sy + 2.0 * u * t * cy + t * t * ey

        return x, y

    def _bezier_tangent_angle(
        self,
        curve: tuple[float, float, float, float, float, float],
        t: float,
    ) -> float:
        sx, sy, cx, cy, ex, ey = curve
        u = 1.0 - t

        tx = 2.0 * u * (cx - sx) + 2.0 * t * (ex - cx)
        ty = 2.0 * u * (cy - sy) + 2.0 * t * (ey - cy)

        angle = math.degrees(math.atan2(ty, tx))

        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        return angle

    def _edge_label_size_data_units(self, label: str) -> tuple[float, float]:
        clean = (
            label.replace("$", "")
            .replace("{", "")
            .replace("}", "")
            .replace("^", "")
        )

        width = max(0.72, 0.060 * len(clean))
        height = 0.28

        return width, height

    def _label_hits_node(
        self,
        x: float,
        y: float,
        label_width: float,
        label_height: float,
        pos: dict[str, tuple[float, float]],
        node_radii: dict[str, float],
    ) -> bool:
        label_half_diag = 0.5 * math.hypot(label_width, label_height)

        for node, (node_x, node_y) in pos.items():
            safe_distance = (
                node_radii.get(node, 0.45)
                + label_half_diag
                + NetworkControlPanel.edge_label_node_margin
            )

            if math.hypot(x - node_x, y - node_y) < safe_distance:
                return True

        return False

    def _label_hits_existing_label(
        self,
        x: float,
        y: float,
        label_width: float,
        label_height: float,
        occupied_labels: list[tuple[float, float, float, float]],
    ) -> bool:
        margin = NetworkControlPanel.edge_label_label_margin

        left = x - label_width / 2.0 - margin
        right = x + label_width / 2.0 + margin
        bottom = y - label_height / 2.0 - margin
        top = y + label_height / 2.0 + margin

        for other_left, other_right, other_bottom, other_top in occupied_labels:
            separated = (
                right < other_left
                or left > other_right
                or top < other_bottom
                or bottom > other_top
            )

            if not separated:
                return True

        return False

    def _candidate_t_values_near_middle(self) -> list[float]:
        candidates = [0.50]

        step = NetworkControlPanel.edge_label_t_step
        window = NetworkControlPanel.edge_label_t_search_window
        delta = step

        while delta <= window + 1e-9:
            left = 0.50 - delta
            right = 0.50 + delta

            if 0.08 <= left <= 0.92:
                candidates.append(left)

            if 0.08 <= right <= 0.92:
                candidates.append(right)

            delta += step

        return candidates

    def _choose_edge_label_position_on_curve(
        self,
        curve: tuple[float, float, float, float, float, float],
        label: str,
        pos: dict[str, tuple[float, float]],
        node_radii: dict[str, float],
        occupied_labels: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float, float]:
        label_width, label_height = self._edge_label_size_data_units(label)

        best = None
        best_score = float("inf")

        for t in self._candidate_t_values_near_middle():
            x, y = self._bezier_point(curve, t)

            hits_node = self._label_hits_node(
                x=x,
                y=y,
                label_width=label_width,
                label_height=label_height,
                pos=pos,
                node_radii=node_radii,
            )

            hits_label = self._label_hits_existing_label(
                x=x,
                y=y,
                label_width=label_width,
                label_height=label_height,
                occupied_labels=occupied_labels,
            )

            score = abs(t - 0.50) * 500.0

            if hits_node:
                score += 100000.0

            if hits_label:
                score += 1200.0

            if score < best_score:
                best_score = score
                best = (x, y, label_width, label_height, t)

            if not hits_node and not hits_label:
                return x, y, label_width, label_height, t

        assert best is not None
        return best

    def _draw_background_lanes(
        self,
        ax: Any,
        pos: dict[str, tuple[float, float]],
    ) -> None:
        group_titles = {
            "liquid": "Liquid",
            "partial": "Partial / open cages",
            "half": "Half cages",
            "transition": "Transition cages",
            "amorphous": "Amorphous cages",
            "crystal": "Crystal cages",
        }

        group_x = {
            "liquid": -6.4,
            "partial": -4.0,
            "half": -1.55,
            "transition": 1.05,
            "amorphous": 3.65,
            "crystal": 6.20,
        }

        y_min = min(y for _, y in pos.values()) - 0.90
        y_max = max(y for _, y in pos.values()) + 0.90

        for group, x in group_x.items():
            color = self.group_light_colors.get(group, "#F4F5F6")

            ax.axvspan(
                x - 0.72,
                x + 0.72,
                ymin=0.055,
                ymax=0.94,
                color=color,
                alpha=0.58,
                zorder=0,
            )

            ax.text(
                x,
                y_max + 0.30,
                group_titles[group],
                ha="center",
                va="bottom",
                fontsize=NetworkControlPanel.group_label_font_size,
                fontweight="bold",
                color="#39444E",
                zorder=5,
            )

        ax.set_ylim(y_min, y_max + 0.80)

    def _draw_edges_and_labels(
        self,
        ax: Any,
        graph: Any,
        pos: dict[str, tuple[float, float]],
        node_sizes: dict[str, float],
    ) -> None:
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.path import Path
        import matplotlib.patheffects as path_effects

        display_edges = self._iter_display_edges(graph)

        if not display_edges:
            return

        max_weight = max(edge["weight"] for edge in display_edges)
        node_radii = self._node_radii_data_units(graph, node_sizes)
        edge_geometries: list[dict[str, Any]] = []

        for edge in display_edges:
            source = edge["source"]
            target = edge["target"]
            weight = edge["weight"]

            curve = self._edge_curve_geometry(
                source=source,
                target=target,
                pos=pos,
                node_radii=node_radii,
            )

            sx, sy, cx, cy, ex, ey = curve

            path = Path(
                [(sx, sy), (cx, cy), (ex, ey)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )

            normalized = weight / max_weight
            width = (0.45 + 4.25 * normalized) * NetworkControlPanel.edge_scale
            alpha = 0.16 + 0.62 * normalized

            arrowstyle = "<|-|>" if edge["is_reciprocal"] else "-|>"

            patch = FancyArrowPatch(
                path=path,
                arrowstyle=arrowstyle,
                mutation_scale=NetworkControlPanel.arrow_size,
                linewidth=width,
                color="#5F6F7B",
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
                capstyle="round",
                joinstyle="round",
            )

            ax.add_patch(patch)

            edge_geometries.append(
                {
                    "edge": edge,
                    "curve": curve,
                }
            )

        occupied_labels: list[tuple[float, float, float, float]] = []

        if NetworkControlPanel.node_label_outside:
            # Reserve the box of every node label so an edge label can never be
            # dropped on top of the node text.
            for node in graph.nodes():
                node_x, node_y = pos[node]
                node_label = self._format_node_label(node)
                width, height = self._edge_label_size_data_units(node_label)
                centre_y = (
                    node_y
                    - node_radii.get(node, 0.45)
                    - NetworkControlPanel.node_label_offset
                    - height / 2.0
                )
                occupied_labels.append(
                    (
                        node_x - width / 2.0,
                        node_x + width / 2.0,
                        centre_y - height / 2.0,
                        centre_y + height / 2.0,
                    )
                )

        label_geometries = sorted(
            edge_geometries,
            key=lambda item: item["edge"]["weight"],
            reverse=True,
        )

        placed_labels = 0
        label_limit = NetworkControlPanel.max_edge_labels

        for item in label_geometries:
            edge = item["edge"]
            weight = edge["weight"]

            if not NetworkControlPanel.show_edge_labels:
                continue

            if weight < NetworkControlPanel.edge_label_min_weight:
                continue

            if label_limit and placed_labels >= label_limit:
                break

            label = self._format_edge_label(edge)
            curve = item["curve"]

            label_x, label_y, label_width, label_height, t = (
                self._choose_edge_label_position_on_curve(
                    curve=curve,
                    label=label,
                    pos=pos,
                    node_radii=node_radii,
                    occupied_labels=occupied_labels,
                )
            )

            angle = 0.0

            if NetworkControlPanel.rotate_edge_labels_along_edge:
                angle = self._bezier_tangent_angle(curve, t)

            bbox = None

            if NetworkControlPanel.use_edge_label_box:
                bbox = dict(
                    boxstyle="round,pad=0.08,rounding_size=0.03",
                    fc="white",
                    ec="#DDE3EA",
                    lw=0.45,
                    alpha=0.82,
                )

            text = ax.text(
                label_x,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=NetworkControlPanel.edge_label_font_size,
                fontweight="bold",
                color="#1F2A33",
                rotation=angle,
                rotation_mode="anchor",
                transform_rotates_text=True,
                zorder=8,
                bbox=bbox,
            )

            text.set_path_effects(
                [
                    path_effects.withStroke(
                        linewidth=2.6,
                        foreground="white",
                        alpha=0.95,
                    )
                ]
            )

            text.set_clip_on(False)
            placed_labels += 1

            occupied_labels.append(
                (
                    label_x - label_width / 2.0,
                    label_x + label_width / 2.0,
                    label_y - label_height / 2.0,
                    label_y + label_height / 2.0,
                )
            )

    def _draw_nodes(
        self,
        ax: Any,
        graph: Any,
        pos: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        import matplotlib.patheffects as path_effects

        node_list = list(graph.nodes())
        node_sizes = {node: self._node_size_for_label(node) for node in node_list}

        x_values = [pos[node][0] for node in node_list]
        y_values = [pos[node][1] for node in node_list]
        colors = [self._node_color(node) for node in node_list]
        sizes = [node_sizes[node] for node in node_list]

        shadow = ax.scatter(
            [x + 0.035 for x in x_values],
            [y - 0.035 for y in y_values],
            s=[s * 1.10 for s in sizes],
            c="#000000",
            alpha=0.12,
            linewidths=0,
            zorder=3,
        )
        shadow.set_clip_on(False)

        rings = ax.scatter(
            x_values,
            y_values,
            s=[s * 1.16 for s in sizes],
            facecolors="white",
            edgecolors="#FFFFFF",
            linewidths=1.0,
            alpha=0.98,
            zorder=4,
        )
        rings.set_clip_on(False)

        nodes = ax.scatter(
            x_values,
            y_values,
            s=sizes,
            c=colors,
            edgecolors="white",
            linewidths=2.4,
            alpha=0.99,
            zorder=5,
        )
        nodes.set_clip_on(False)

        radii = self._node_radii_data_units(graph, node_sizes)

        for node in node_list:
            x, y = pos[node]
            label = self._format_node_label(node)

            if NetworkControlPanel.node_label_outside:
                # Below the marker with an opaque backdrop: long labels such as
                # "half 5^12 6^2" cannot overflow into a neighbour this way.
                offset = radii.get(node, 0.45) + NetworkControlPanel.node_label_offset

                text = ax.text(
                    x,
                    y - offset,
                    label,
                    ha="center",
                    va="top",
                    fontsize=NetworkControlPanel.node_label_outside_font_size,
                    fontweight="bold",
                    color="#1F2937",
                    linespacing=0.92,
                    zorder=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 0.6,
                    },
                )
                text.set_clip_on(False)
                continue

            text_color = self._text_color_for_node(node)
            font_size = self._node_label_font(node)

            text = ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color=text_color,
                linespacing=0.86,
                zorder=7,
            )

            stroke_color = "#22303A" if text_color == "white" else "white"
            stroke_width = 1.2 if text_color == "white" else 2.0

            text.set_path_effects(
                [
                    path_effects.withStroke(
                        linewidth=stroke_width,
                        foreground=stroke_color,
                        alpha=0.40 if text_color == "white" else 0.95,
                    )
                ]
            )

            text.set_clip_on(False)

        return node_sizes

    def _draw_transition_panel(self, ax: Any, display_edges: list[dict[str, Any]]) -> None:
        """Side list of the strongest transitions, in data coordinates.

        The graph keeps every edge; the numbers for the busiest pathways are
        listed in a reserved strip to the right so nothing has to be squeezed
        onto the arrows.
        """
        rows = NetworkControlPanel.transition_panel_rows

        if not rows:
            return

        ordered = sorted(display_edges, key=lambda item: item["weight"], reverse=True)[:rows]

        if not ordered:
            return

        lines = [r"Strongest transitions (ns$^{-1}$)", ""]

        for edge in ordered:
            source = self._format_node_label(edge["source"]).replace("\n", " ")
            target = self._format_node_label(edge["target"]).replace("\n", " ")
            rate = edge["weight"] / self.total_time_ns
            separator = r"$\leftrightarrow$" if edge.get("is_reciprocal") else "->"
            lines.append(f"{source}  {separator}  {target}    {rate:.1f}")

        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()

        ax.text(
            x_right + 0.35,
            y_top - 0.35,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=NetworkControlPanel.transition_panel_font_size,
            color="#1F2937",
            linespacing=1.45,
            family="DejaVu Sans",
            zorder=12,
        )

        ax.set_xlim(x_left, x_right + NetworkControlPanel.transition_panel_width)

    def _draw_legend(self, ax: Any) -> None:
        from matplotlib.lines import Line2D

        legend_items = [
            ("liquid", "Liquid"),
            ("partial", "Partial"),
            ("half", "Half"),
            ("transition", "Transition"),
            ("amorphous", "Amorphous"),
            ("crystal", "Crystal"),
        ]

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=label,
                markerfacecolor=self.group_colors[group],
                markeredgecolor="white",
                markeredgewidth=1.5,
                markersize=10,
            )
            for group, label in legend_items
        ]

        legend = ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.065),
            ncol=6,
            frameon=True,
            fancybox=True,
            framealpha=0.97,
            borderpad=0.65,
            columnspacing=1.50,
            handletextpad=0.45,
            fontsize=NetworkControlPanel.legend_font_size,
        )

        legend.get_frame().set_edgecolor("#D8DEE4")
        legend.get_frame().set_linewidth(0.8)

    def plot(self, filename: str = NetworkControlPanel.output_figure_name) -> None:
        if not self.transitions:
            return

        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("\nNote: networkx or matplotlib not found. Skipping plot.")
            print("Install them with:")
            print("    pip install networkx matplotlib numpy")
            return

        mpl.rcParams["font.family"] = "DejaVu Sans"
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42

        filtered_transitions = self._filtered_transitions()

        if not filtered_transitions:
            print(
                f"Note: No transition with weight >= "
                f"{NetworkControlPanel.min_edge_weight} found to plot."
            )
            return

        graph = nx.DiGraph()

        for (from_type, to_type), weight in filtered_transitions.items():
            graph.add_edge(from_type, to_type, weight=weight)

        fig, ax = plt.subplots(
            figsize=(NetworkControlPanel.figure_width, NetworkControlPanel.figure_height),
            dpi=NetworkControlPanel.figure_dpi,
        )

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")

        pos = self._get_positions(graph)

        self._draw_background_lanes(ax, pos)

        node_sizes = {
            node: self._node_size_for_label(node)
            for node in graph.nodes()
        }

        self._draw_edges_and_labels(ax, graph, pos, node_sizes)
        self._draw_nodes(ax, graph, pos)
        self._draw_legend(ax)



        ax.set_title(
            NetworkControlPanel.title,
            fontsize=NetworkControlPanel.title_font_size,
            fontweight="bold",
            color="#111827",
            pad=34,
        )

        x_values = [point[0] for point in pos.values()]
        y_values = [point[1] for point in pos.values()]

        if x_values and y_values:
            ax.set_xlim(min(x_values) - 1.20, max(x_values) + 1.20)

            current_ymin, current_ymax = ax.get_ylim()

            ax.set_ylim(
                min(current_ymin, min(y_values) - 1.10),
                max(current_ymax, max(y_values) + 1.35),
            )

        if NetworkControlPanel.show_transition_panel:
            self._draw_transition_panel(ax, self._iter_display_edges(graph))

        ax.axis("off")

        os.makedirs(self.outdir, exist_ok=True)
        save_path = categorised_path(self.outdir, filename, PLOTS)

        plt.subplots_adjust(
            left=0.035,
            right=0.985,
            top=0.90,
            bottom=0.105,
        )

        plt.savefig(
            save_path,
            dpi=NetworkControlPanel.figure_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)

        print(f"\n[Reaction Network] Success! Pathway graph generated: {save_path}")
        print(f"                   Layout: {NetworkControlPanel.layout_type}")
        print(f"                   Filter: weight >= {NetworkControlPanel.min_edge_weight}")
        print(f"                   Edge label min weight: {NetworkControlPanel.edge_label_min_weight}")
        print(f"                   Label : {NetworkControlPanel.label_mode}")
        print(f"                   Time  : {self.total_time_ns:.6f} ns")
        print(f"                   Nodes : {len(graph.nodes())}")
        print(f"                   Edges : {len(graph.edges())}\n")


def generate_network_from_tracker(tracker: Any, outdir: str) -> None:
    """
    Runner entry point from CageTracker.
    """

    network = TransitionNetwork(outdir)
    network.build_from_tracker(tracker)
    network.write_tables()
    network.plot()


def generate_network_from_tables(
    table_dir: str,
    outdir: str | None = None,
    filename: str = NetworkControlPanel.output_figure_name,
) -> None:
    """
    Runner entry point from existing dat tables.
    """

    if outdir is None:
        outdir = table_dir

    edges_file = os.path.join(table_dir, NetworkControlPanel.edges_table_name)
    nodes_file = os.path.join(table_dir, NetworkControlPanel.nodes_table_name)

    network = TransitionNetwork(outdir)
    network.load_from_tables(edges_file=edges_file, nodes_file=nodes_file)
    network.plot(filename=filename)
