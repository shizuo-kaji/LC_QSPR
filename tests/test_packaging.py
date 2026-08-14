import re
import tomllib
from pathlib import Path


def test_descriptor_dependency_does_not_pin_old_numpy_or_networkx():
    configuration = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = configuration["project"]["dependencies"]
    dependency_names = {
        re.split(r"[<>=!~; ]", dependency, maxsplit=1)[0].lower()
        for dependency in dependencies
    }

    assert "mordredcommunity" in dependency_names
    assert "mordred" not in dependency_names
