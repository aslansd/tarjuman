"""Map a NeuroML morphology onto Jaxley's branch/compartment structure.

NeuroML describes a cell as a tree of *segments* (truncated cones) and gives
optional *segmentGroups*, some of which are marked with the neuroLex id
``sao864921383`` to say "this group is an unbranched cable", i.e. a NEURON
section.  Jaxley describes a cell as a tree of *branches*, each of which is an
unbranched cable divided into a number of equal-length compartments.

The mapping is therefore:

    NeuroML section (or derived unbranched chain)  ->  Jaxley branch
    numberInternalDivisions (or ``ncomp``)         ->  compartments per branch

Everything downstream — where to put a channel, where to attach a synapse,
where to inject current — is expressed in NeuroML as ``(segment id, fraction
along)`` and has to be resolved to a ``(branch, compartment)`` pair.  That is
what :meth:`CellMorphology.locate` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ir
from .errors import MorphologyError
from .report import ConversionReport

__all__ = ["BranchSpec", "CellMorphology", "build_morphology"]

#: neuroLex id marking a segmentGroup as an unbranched cable (a NEURON section).
SECTION_NEUROLEX_ID = "sao864921383"


@dataclass
class BranchSpec:
    """One Jaxley branch: an ordered chain of NeuroML segments."""

    index: int
    segments: list[int]
    lengths: list[float]  # um, per segment
    radii: list[float]  # um, per segment
    parent: int  # branch index, -1 for the root
    ncomp: int = 1
    group: Optional[str] = None
    xyzr: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    @property
    def length(self) -> float:
        return float(sum(self.lengths))

    @property
    def radius(self) -> float:
        """Length-weighted mean radius (falls back on the plain mean)."""
        total = self.length
        if total <= 0:
            return float(np.mean(self.radii)) if self.radii else 1.0
        return float(
            sum(l * r for l, r in zip(self.lengths, self.radii)) / total
        )


@dataclass
class CellMorphology:
    """The result of mapping one NeuroML cell onto Jaxley branches."""

    cell_id: str
    branches: list[BranchSpec]
    segment_to_branch: dict[int, int]
    #: Distance in um from the start of its branch to the start of each segment.
    segment_offset: dict[int, float]
    groups: dict[str, list[int]]  # segment group id -> segment ids

    @property
    def parents(self) -> list[int]:
        return [branch.parent for branch in self.branches]

    @property
    def n_compartments(self) -> int:
        return sum(branch.ncomp for branch in self.branches)

    def locate(self, segment_id: int, fraction_along: float = 0.5) -> tuple[int, int]:
        """Resolve a NeuroML ``(segment, fractionAlong)`` to ``(branch, comp)``.

        ``fraction_along`` is measured from the proximal end of the segment, as
        in NeuroML; the returned compartment index is measured from the
        proximal end of the branch, as in Jaxley.
        """
        if segment_id not in self.segment_to_branch:
            raise MorphologyError(
                f"Segment {segment_id} is not part of cell '{self.cell_id}'."
            )
        branch_index = self.segment_to_branch[segment_id]
        branch = self.branches[branch_index]
        position = self.segment_offset[segment_id] + fraction_along * branch.lengths[
            branch.segments.index(segment_id)
        ]
        if branch.length <= 0:
            return branch_index, 0
        comp = int(np.floor(position / branch.length * branch.ncomp))
        return branch_index, int(np.clip(comp, 0, branch.ncomp - 1))

    def compartments_of_group(self, group_id: str) -> list[tuple[int, int]]:
        """Every ``(branch, comp)`` covered by a segment group."""
        if group_id in ("all", "", None):
            return [
                (branch.index, comp)
                for branch in self.branches
                for comp in range(branch.ncomp)
            ]
        if group_id not in self.groups:
            raise MorphologyError(
                f"Unknown segmentGroup '{group_id}' on cell '{self.cell_id}'."
            )
        located = set()
        for segment_id in self.groups[group_id]:
            branch_index = self.segment_to_branch[segment_id]
            branch = self.branches[branch_index]
            start = self.segment_offset[segment_id]
            end = start + branch.lengths[branch.segments.index(segment_id)]
            if branch.length <= 0:
                located.add((branch_index, 0))
                continue
            comp_length = branch.length / branch.ncomp
            first = int(np.clip(np.floor(start / comp_length), 0, branch.ncomp - 1))
            last = int(
                np.clip(np.ceil(end / comp_length) - 1, first, branch.ncomp - 1)
            )
            for comp in range(first, last + 1):
                located.add((branch_index, comp))
        return sorted(located)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def resolve_groups(morphology: ir.Morphology) -> dict[str, list[int]]:
    """Flatten segment groups, following ``<include>`` and ``<path>`` elements."""
    resolved: dict[str, list[int]] = {}

    def resolve(group_id: str, stack: tuple[str, ...] = ()) -> list[int]:
        if group_id in resolved:
            return resolved[group_id]
        if group_id in stack:
            raise MorphologyError(f"Cyclic segmentGroup include at '{group_id}'.")
        group = morphology.groups.get(group_id)
        if group is None:
            raise MorphologyError(f"Unknown segmentGroup '{group_id}'.")
        segments = list(group.members)
        for included in group.includes:
            segments.extend(resolve(included, stack + (group_id,)))
        for start, end in group.paths:
            segments.extend(_segments_on_path(morphology, start, end))
        ordered = sorted(dict.fromkeys(segments))
        resolved[group_id] = ordered
        return ordered

    for group_id in morphology.groups:
        resolve(group_id)
    return resolved


def _segments_on_path(
    morphology: ir.Morphology, start: Optional[int], end: Optional[int]
) -> list[int]:
    """Segments between ``start`` and ``end``, walking parent links from ``end``."""
    if end is None:
        return [] if start is None else [start]
    path = []
    current: Optional[int] = end
    while current is not None:
        path.append(current)
        if current == start:
            break
        segment = morphology.segments.get(current)
        current = segment.parent if segment is not None else None
    return list(reversed(path))


def build_morphology(
    cell: ir.Cell,
    report: ConversionReport,
    ncomp: int = 1,
    max_comp_length: Optional[float] = None,
    use_neuroml_sections: bool = True,
) -> CellMorphology:
    """Map a NeuroML cell onto Jaxley branches.

    Args:
        cell: The IR cell.
        report: Conversion report for approximations.
        ncomp: Default number of compartments per branch, used where the
            NeuroML file gives no ``numberInternalDivisions``.
        max_comp_length: If given, each branch gets enough compartments that no
            compartment is longer than this (in um).  Overrides ``ncomp`` when
            it asks for more compartments.
        use_neuroml_sections: Honour ``segmentGroup`` elements marked as
            unbranched cables; when False, sections are derived from the
            segment tree alone.

    Returns:
        A :class:`CellMorphology`.
    """
    morphology = cell.morphology
    if not morphology.segments:
        raise MorphologyError(f"Cell '{cell.id}' has no segments.")

    groups = resolve_groups(morphology)
    chains = _section_chains(morphology, groups, use_neuroml_sections, report)

    segment_to_branch: dict[int, int] = {}
    for index, (segment_ids, _) in enumerate(chains):
        for segment_id in segment_ids:
            segment_to_branch[segment_id] = index

    branches: list[BranchSpec] = []
    segment_offset: dict[int, float] = {}
    for index, (segment_ids, group_id) in enumerate(chains):
        lengths, radii, offsets = [], [], []
        offset = 0.0
        for segment_id in segment_ids:
            segment = morphology.segments[segment_id]
            length = segment.length
            if length <= 0.0:
                # NeuroML represents a spherical soma as a segment whose
                # proximal and distal points coincide.  A cylinder of
                # length = diameter has the same surface area as that sphere,
                # which is the convention jnml and NEURON use.
                length = segment.distal[3]
                report.approximation(
                    f"cell '{cell.id}' segment {segment_id}",
                    "zero-length (spherical) segment converted to a cylinder of "
                    f"length = diameter = {length:g} um, preserving surface area",
                )
            offsets.append(offset)
            offset += length
            lengths.append(length)
            radii.append(segment.radius)

        first_segment = morphology.segments[segment_ids[0]]
        parent_branch = -1
        if first_segment.parent is not None:
            parent_branch = segment_to_branch.get(first_segment.parent, -1)
            if first_segment.fraction_along not in (0.0, 1.0):
                report.approximation(
                    f"cell '{cell.id}' segment {segment_ids[0]}",
                    f"attaches to its parent at fractionAlong="
                    f"{first_segment.fraction_along:g}; Jaxley branches connect at "
                    "their ends, so the attachment is moved to the branch end",
                )

        branch_length = sum(lengths)
        n_comp = _n_comp_for(
            morphology, group_id, segment_ids, branch_length, ncomp, max_comp_length
        )
        branch = BranchSpec(
            index=index,
            segments=list(segment_ids),
            lengths=lengths,
            radii=radii,
            parent=parent_branch,
            ncomp=n_comp,
            group=group_id,
            xyzr=_branch_xyzr(morphology, segment_ids),
        )
        branches.append(branch)
        for segment_id, off in zip(segment_ids, offsets):
            segment_offset[segment_id] = off

    _check_parent_order(branches, cell.id)
    return CellMorphology(
        cell_id=cell.id,
        branches=branches,
        segment_to_branch=segment_to_branch,
        segment_offset=segment_offset,
        groups=groups,
    )


def _n_comp_for(
    morphology: ir.Morphology,
    group_id: Optional[str],
    segment_ids: list[int],
    length: float,
    ncomp: int,
    max_comp_length: Optional[float],
) -> int:
    requested = ncomp
    if group_id is not None:
        group = morphology.groups.get(group_id)
        if group is not None and group.n_internal_divisions:
            requested = group.n_internal_divisions
    if max_comp_length is not None and length > 0:
        requested = max(requested, int(np.ceil(length / max_comp_length)))
    return max(1, int(requested))


def _branch_xyzr(morphology: ir.Morphology, segment_ids: list[int]) -> np.ndarray:
    """Traced coordinates of a branch as an ``(n, 4)`` array of x, y, z, radius."""
    points: list[list[float]] = []
    for segment_id in segment_ids:
        segment = morphology.segments[segment_id]
        if segment.proximal is not None:
            point = [*segment.proximal[:3], segment.proximal[3] / 2.0]
            if not points or points[-1] != point:
                points.append(point)
        points.append([*segment.distal[:3], segment.distal[3] / 2.0])
    if len(points) == 1:
        points.append(list(points[0]))
    return np.asarray(points, dtype=float)


def _section_chains(
    morphology: ir.Morphology,
    groups: dict[str, list[int]],
    use_neuroml_sections: bool,
    report: ConversionReport,
) -> list[tuple[list[int], Optional[str]]]:
    """Partition the segments into ordered unbranched chains."""
    sections = {
        group_id: group
        for group_id, group in morphology.groups.items()
        if group.neuro_lex_id == SECTION_NEUROLEX_ID
    }
    covered = {
        segment_id for group_id in sections for segment_id in groups[group_id]
    }
    if use_neuroml_sections and sections and covered == set(morphology.segments):
        chains = [
            (_order_chain(morphology, groups[group_id]), group_id)
            for group_id in sections
        ]
    else:
        if sections and use_neuroml_sections:
            report.info(
                "morphology",
                "segmentGroups marked as unbranched cables do not cover every "
                "segment; sections were derived from the segment tree instead",
            )
        chains = [(chain, None) for chain in _derive_chains(morphology)]

    # Root first, then parents before children, as jx.Cell requires.
    return _sort_chains(morphology, chains)


def _order_chain(morphology: ir.Morphology, segment_ids: list[int]) -> list[int]:
    """Order the segments of a cable from proximal to distal."""
    members = set(segment_ids)
    parents = {
        segment_id: morphology.segments[segment_id].parent for segment_id in members
    }
    roots = [
        segment_id
        for segment_id in members
        if parents[segment_id] is None or parents[segment_id] not in members
    ]
    if len(roots) != 1:
        raise MorphologyError(
            f"Segment group {sorted(members)} is not a single unbranched cable."
        )
    ordered = [roots[0]]
    children = {}
    for segment_id, parent in parents.items():
        if parent in members:
            children.setdefault(parent, []).append(segment_id)
    while True:
        next_segments = children.get(ordered[-1], [])
        if not next_segments:
            break
        if len(next_segments) > 1:
            raise MorphologyError(
                f"Segment {ordered[-1]} branches inside what is declared to be an "
                "unbranched cable."
            )
        ordered.append(next_segments[0])
    return ordered


def _derive_chains(morphology: ir.Morphology) -> list[list[int]]:
    """Split the segment tree into unbranched chains at every branch point."""
    children: dict[Optional[int], list[int]] = {}
    for segment in morphology.segments.values():
        children.setdefault(segment.parent, []).append(segment.id)
    for siblings in children.values():
        siblings.sort()

    roots = children.get(None, [])
    if not roots:
        raise MorphologyError("Morphology has no root segment (none without a parent).")

    chains: list[list[int]] = []
    stack = list(roots)
    while stack:
        start = stack.pop(0)
        chain = [start]
        current = start
        while True:
            next_segments = children.get(current, [])
            if len(next_segments) == 1:
                child = next_segments[0]
                # A child attached part-way along its parent starts a new branch.
                if morphology.segments[child].fraction_along != 1.0:
                    stack.append(child)
                    break
                chain.append(child)
                current = child
            else:
                stack.extend(next_segments)
                break
        chains.append(chain)
    return chains


def _sort_chains(
    morphology: ir.Morphology, chains: list[tuple[list[int], Optional[str]]]
) -> list[tuple[list[int], Optional[str]]]:
    """Order chains so that every branch appears after its parent branch."""
    owner = {
        segment_id: index
        for index, (segment_ids, _) in enumerate(chains)
        for segment_id in segment_ids
    }
    parent_of = {}
    for index, (segment_ids, _) in enumerate(chains):
        parent_segment = morphology.segments[segment_ids[0]].parent
        parent_of[index] = owner.get(parent_segment, -1) if parent_segment is not None else -1

    ordered: list[int] = []
    remaining = set(range(len(chains)))
    while remaining:
        ready = [index for index in sorted(remaining) if parent_of[index] in ordered or parent_of[index] == -1]
        if not ready:
            raise MorphologyError("Segment tree contains a cycle.")
        for index in ready:
            ordered.append(index)
            remaining.discard(index)
    return [chains[index] for index in ordered]


def _check_parent_order(branches: list[BranchSpec], cell_id: str) -> None:
    for branch in branches:
        if branch.parent >= branch.index:
            raise MorphologyError(
                f"Cell '{cell_id}': branch {branch.index} has parent "
                f"{branch.parent}, but Jaxley requires parents to come first."
            )
