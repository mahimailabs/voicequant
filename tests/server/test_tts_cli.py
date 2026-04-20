from __future__ import annotations

from typer.testing import CliRunner

from voicequant.cli import app



def test_tts_cli_group_exists():
    runner = CliRunner()
    result = runner.invoke(app, ["tts", "--help"])
    assert result.exit_code == 0
    assert "speak" in result.stdout
    assert "voices" in result.stdout
