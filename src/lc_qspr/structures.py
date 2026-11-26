"""Read individual SMILES and draw their molecular structures."""

from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D


def molecule_from_smiles(smiles: str) -> Chem.Mol:
    """Read a complete SMILES string for a molecule with defined elements."""
    if not smiles.strip():
        raise ValueError("Enter a SMILES string or draw a molecule.")
    parameters = Chem.SmilesParserParams()
    parameters.parseName = False
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles.strip(), parameters)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Could not read this SMILES. Check ring numbers, brackets, bond orders and atom valences.")
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise ValueError("Replace each wildcard or attachment point with a chemical element.")
    return molecule


def structure_svg(molecule: Chem.Mol, width: int = 480, height: int = 132) -> str:
    """Draw a 2D structure, including specified stereochemistry."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().padding = 0.08
    drawer.drawOptions().fixedBondLength = 30
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
