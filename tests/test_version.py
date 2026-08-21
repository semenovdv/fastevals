from importlib.metadata import version

from fastevals import __version__


def test_runtime_version_matches_package_metadata():
    """Guards against the hardcoded-string drift seen in 0.1.1."""
    assert __version__ == version("fastevals")
    assert __version__ != "0.0.0.dev0"
