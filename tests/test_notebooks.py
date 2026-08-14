from pathlib import Path
import re

import nbformat


def test_notebooks_are_executed_and_every_code_cell_has_guidance():
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(paths) == 6

    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is not None, path
            assert index > 0 and notebook.cells[index - 1].cell_type == "markdown", path
            guidance = notebook.cells[index - 1].source
            assert "**Caution.**" in guidance, path
            assert "**Extension.**" in guidance, path
            assert not any(output.output_type == "error" for output in cell.outputs), path


def test_notebook_prose_is_english_and_has_no_next_step_section():
    japanese = re.compile(r"[ぁ-んァ-ヶ一-龠]")
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        prose = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        assert not japanese.search(prose), path
        assert "next steps" not in prose.lower(), path

