import json

from research_pipeline.cli import main


def test_cli_returns_nonzero_for_errors(tmp_path, capsys):
    code = main(["--registry", str(tmp_path / "registry.sqlite3"), "status", "missing"])
    captured = capsys.readouterr()
    assert code != 0
    assert "error:" in captured.err


def test_cli_init_and_list(tmp_path, capsys):
    registry = tmp_path / "registry.sqlite3"
    assert main(["--registry", str(registry), "init"]) == 0
    capsys.readouterr()
    assert main(["--registry", str(registry), "list-strategies"]) == 0
    assert json.loads(capsys.readouterr().out) == []
