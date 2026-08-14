"""Shared scientific vocabulary and column definitions."""

from __future__ import annotations


ABSOLUTE_ZERO_C = -273.15

PHASE_TYPES = ("N", "A", "B", "C", "E", "F", "G", "H", "I", "J", "Dh", "BP")
CHIRAL_PHASE_TYPES = PHASE_TYPES + ("N*", "C*", "E*", "F*", "I*")
PHASE_TARGETS = tuple(f"{phase}type" for phase in CHIRAL_PHASE_TYPES)
TEMPERATURE_TARGETS = ("Clearing", "Melting") + tuple(
    target for phase in PHASE_TYPES for target in (f"{phase}p", f"{phase}m")
)

PERMITTED_ATOMS = (
    "H", "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
    "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
    "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au",
    "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
)

EXCLUDED_PHASE_LABELS = (
    "X", "BS", "So", "cub", "Are", "Nre", "Nre*", "I/F*", "Dsq", "Dob",
    "Bhx", "C~a", "C~g", "CA", "CA*", "Tg", "tr", "Y", "S", "D",
)
MELTING_LABELS = ("Cr", "Cr'", "Tg")
CLEARING_LABELS = ("is", "chg", "dec")

NON_FEATURE_COLUMNS = (
    "ID", "SMILES", "Phases", "rac_en", "Melting", "Meltingtype",
    "Clearingtype", "Clearing", "Stype", "Xtype", "Ytype", "exclude",
    "reentrant", "prohibited",
) + tuple(
    column
    for phase in PHASE_TYPES
    for column in (
        f"{phase}m", f"{phase}p", f"{phase}type",
        f"{phase}*m", f"{phase}*p", f"{phase}*type",
    )
)

