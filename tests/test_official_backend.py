"""Offline tests for the pinned, author-code reproduction entry points."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_official_scripts_have_valid_shell_syntax() -> None:
    for name in ("official_setup.sh", "official_train.sh", "official_play.sh"):
        subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_official_revision_pins_and_repro_documentation() -> None:
    versions = (SCRIPTS / "official_versions.sh").read_text(encoding="utf-8")
    commits = re.findall(r'(?:SUPERMIMIC|MJLAB)_SHA="([0-9a-f]{40})"', versions)
    assert len(commits) == 2 and len(set(commits)) == 2
    assert "HybridRobotics/supermimic.git" in versions
    assert "HybridRobotics/mjlab_research.git" in versions

    instructions = (ROOT / "docs" / "OFFICIAL_REPRO.md").read_text(encoding="utf-8")
    for sha in commits:
        assert sha in instructions
    assert "--runner.actor_obs proprio" in instructions
    assert "--runner.history_length 10" in instructions
    assert "--runner.max_iterations 320000" in instructions
