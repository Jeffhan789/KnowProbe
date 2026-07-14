from pathlib import Path

import yaml

from knowprobe import __version__
from knowprobe.core.config import AppConfig


def test_default_versions_match_package_metadata() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert AppConfig().version == __version__
    assert config["app"]["version"] == __version__
