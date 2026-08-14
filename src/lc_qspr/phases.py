"""Vocabulary and formatting helpers for liquid-crystal phase sequences."""

from __future__ import annotations

from collections.abc import Mapping

from .constants import CHIRAL_PHASE_TYPES, PHASE_TYPES, TEMPERATURE_TARGETS


def format_phase_sequence(row: Mapping[str, object]) -> str:
    """Build a compact ``Cr ... is`` sequence from one prediction row."""
    melting = float(row["pred_Melting"])
    clearing = float(row["pred_Clearing"])
    transitions: list[tuple[float, str, float]] = []

    for phase in PHASE_TYPES:
        phase_class = int(row.get(f"pred_{phase}type", 0))
        label = phase + "*" if phase_class == 2 else phase
        if phase_class == 0:
            chiral_class = int(row.get(f"pred_{phase}*type", 0))
            if chiral_class > 0:
                phase_class = chiral_class
                label = phase + "*"
        if phase_class > 0:
            lower = float(row[f"pred_{phase}m"])
            upper = float(row[f"pred_{phase}p"])
            transitions.append((lower, label, upper))

    parts = ["Cr", f"{melting:.2f}"]
    for lower, label, upper in sorted(transitions, key=lambda item: item[2]):
        parts.extend((f"{lower:.2f}", label, f"{upper:.2f}"))
    parts.extend((f"{clearing:.2f}", "is"))
    return " ".join(parts)
