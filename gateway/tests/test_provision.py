"""Tests for ``stackchan-mcp provision``."""

from __future__ import annotations

import csv
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

from stackchan_mcp import cli, provision
from stackchan_mcp.provision import ProvisionSettings, build_nvs_csv


def _rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def _settings(**kw) -> ProvisionSettings:
    base = dict(
        ssid="Home",
        password="pw",
        websocket_url="ws://192.168.7.207:8765/",
        token="tok",
        ota_url="http://192.168.7.207:8766/ota/",
    )
    base.update(kw)
    return ProvisionSettings(**base)


def test_csv_has_firmware_namespaces_and_keys():
    rows = _rows(build_nvs_csv(_settings()))
    assert rows[0] == ["key", "type", "encoding", "value"]
    assert rows[1] == ["wifi", "namespace", "", ""]
    assert ["ssid", "data", "string", "Home"] in rows
    assert ["password", "data", "string", "pw"] in rows
    assert ["ota_url", "data", "string", "http://192.168.7.207:8766/ota/"] in rows
    ws = rows.index(["websocket", "namespace", "", ""])
    assert rows[ws + 1] == ["url", "data", "string", "ws://192.168.7.207:8765/"]
    assert ["token", "data", "string", "tok"] in rows[ws:]


def test_csv_quotes_commas_and_quotes_in_password():
    rows = _rows(build_nvs_csv(_settings(password='a,b"c')))
    assert ["password", "data", "string", 'a,b"c'] in rows


def test_csv_omits_empty_optional_values():
    rows = _rows(build_nvs_csv(_settings(token="", ota_url="", fallback_url="")))
    keys = [r[0] for r in rows]
    assert "token" not in keys
    assert "ota_url" not in keys
    assert "fallback_url" not in keys


@pytest.mark.parametrize(
    "kw",
    [
        {"ssid": ""},
        {"ssid": "x" * 33},
        {"password": "x" * 65},
        {"websocket_url": "http://host:8765/"},
    ],
)
def test_csv_rejects_invalid_settings(kw):
    with pytest.raises(ValueError):
        build_nvs_csv(_settings(**kw))


def _args(*extra: str):
    parser = cli._build_arg_parser()
    return parser.parse_args(["provision", "--ssid", "Home", *extra])


def test_dry_run_masks_secrets_and_writes_nothing(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))
    monkeypatch.setenv("STACKCHAN_TOKEN", "secret-token")
    args = _args("--password", "secret-pw", "--gateway-host", "10.0.0.5", "--dry-run")
    assert provision.run(args) == 0
    out = capsys.readouterr().out
    assert "secret-pw" not in out
    assert "secret-token" not in out
    assert "ws://10.0.0.5:8765/" in out
    assert "http://10.0.0.5:8766/ota/" in out


def test_run_backs_up_then_writes(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        if "read-flash" in cmd:
            Path(cmd[-1]).write_bytes(b"\xff" * provision.NVS_SIZE)
        if "generate" in cmd:
            # The CSV must exist while the generator runs.
            assert Path(cmd[cmd.index("generate") + 1]).exists()
            Path(cmd[cmd.index("generate") + 2]).write_bytes(b"\0" * provision.NVS_SIZE)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(provision, "_require_tools", lambda: None)
    monkeypatch.setattr(provision, "BACKUP_DIR", tmp_path / "backups")
    args = _args("--password", "pw", "--token", "t", "--port", "/dev/ttyACM9", "--yes")

    assert provision.run(args) == 0

    read, gen, write = calls
    assert "read-flash" in read and "0x9000" in read and "0x4000" in read
    assert read[read.index("-p") + 1] == "/dev/ttyACM9"
    assert "generate" in gen
    assert "write-flash" in write and "0x9000" in write
    # Secrets do not outlive the command.
    assert not Path(gen[gen.index("generate") + 1]).exists()
    assert not Path(write[-1]).exists()
    assert len(list((tmp_path / "backups").glob("nvs-*.bin"))) == 1


def test_run_cancels_without_confirmation(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    args = _args("--password", "pw", "--port", "/dev/ttyACM9")
    assert provision.run(args) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("esp_idf_nvs_partition_gen") is None,
    reason="provision extra not installed",
)
def test_generated_image_contains_settings(tmp_path):
    csv_path = tmp_path / "nvs.csv"
    bin_path = tmp_path / "nvs.bin"
    csv_path.write_text(build_nvs_csv(_settings()))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "esp_idf_nvs_partition_gen",
            "generate",
            str(csv_path),
            str(bin_path),
            hex(provision.NVS_SIZE),
        ],
        check=True,
        capture_output=True,
    )
    data = bin_path.read_bytes()
    assert len(data) == provision.NVS_SIZE
    for needle in (b"wifi", b"websocket", b"Home", b"ws://192.168.7.207:8765/"):
        assert needle in data
