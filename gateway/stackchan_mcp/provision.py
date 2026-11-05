"""``stackchan-mcp provision``: write Wi-Fi and gateway settings over USB.

The firmware's Wi-Fi setup hotspot (captive portal at 192.168.4.1) does not
work with every phone. This command skips it: it builds an NVS partition
image holding the Wi-Fi credentials, the gateway WebSocket URL and token,
and an ``ota_url`` pointing at the gateway's own "no update" endpoint, then
writes it to the robot with esptool.

The keys mirror what the firmware reads:

* ``wifi/ssid``, ``wifi/password`` (components/78__esp-wifi-connect/ssid_manager.cc)
* ``wifi/ota_url`` (main/ota.cc)
* ``websocket/url``, ``websocket/token``, ``websocket/fallback_url``
  (main/protocols/websocket_protocol.cc)

The image replaces the whole NVS partition, so other saved settings
(volume, touch tuning, the xiaozhi board UUID) are reset. The current
partition is always read back to a backup file first.

Needs the ``provision`` extra: ``pip install 'stackchan-mcp[provision]'``.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import glob
import importlib.util
import io
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# firmware/partitions/v2/16m.csv: nvs, data, nvs, 0x9000, 0x4000
NVS_OFFSET = 0x9000
NVS_SIZE = 0x4000

DEFAULT_WS_PORT = 8765
DEFAULT_CAPTURE_PORT = 8766
BACKUP_DIR = Path(os.path.expanduser("~/.stackchan/backups"))


@dataclass(frozen=True)
class ProvisionSettings:
    ssid: str
    password: str
    websocket_url: str
    token: str = ""
    fallback_url: str = ""
    ota_url: str = ""


def build_nvs_csv(settings: ProvisionSettings) -> str:
    """Return the CSV that esp-idf-nvs-partition-gen turns into an image.

    Values are written with the csv module, so commas and quotes in a
    Wi-Fi password are quoted correctly. Empty optional values are left
    out, which the firmware treats the same as "not set".
    """
    if not settings.ssid:
        raise ValueError("ssid must not be empty")
    if len(settings.ssid.encode()) > 32:
        raise ValueError("ssid is longer than 32 bytes")
    if len(settings.password.encode()) > 64:
        raise ValueError("password is longer than 64 bytes")
    if not settings.websocket_url.startswith(("ws://", "wss://")):
        raise ValueError("websocket URL must start with ws:// or wss://")

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["key", "type", "encoding", "value"])
    w.writerow(["wifi", "namespace", "", ""])
    w.writerow(["ssid", "data", "string", settings.ssid])
    w.writerow(["password", "data", "string", settings.password])
    if settings.ota_url:
        w.writerow(["ota_url", "data", "string", settings.ota_url])
    w.writerow(["websocket", "namespace", "", ""])
    w.writerow(["url", "data", "string", settings.websocket_url])
    if settings.token:
        w.writerow(["token", "data", "string", settings.token])
    if settings.fallback_url:
        w.writerow(["fallback_url", "data", "string", settings.fallback_url])
    return buf.getvalue()


def detect_lan_ip() -> str:
    """Best guess at this host's LAN address (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))  # TEST-NET-1; connect() on UDP sends nothing
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def detect_serial_port() -> str | None:
    """Return the only USB serial port, or None if there are zero or several."""
    candidates = sorted(
        glob.glob("/dev/ttyACM*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/cu.usbmodem*")
    )
    return candidates[0] if len(candidates) == 1 else None


def _mask(value: str) -> str:
    return "<set>" if value else "<empty>"


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(cmd[:4])} ..."
        )


def _require_tools() -> None:
    # find_spec only locates the modules; they run as separate processes.
    missing = [
        mod
        for mod in ("esptool", "esp_idf_nvs_partition_gen")
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        raise SystemExit(
            "provision needs the 'provision' extra "
            f"(missing: {', '.join(missing)}). Install it with:\n"
            "  uv tool install --force 'stackchan-mcp[provision]'\n"
            "or: pipx install --force 'stackchan-mcp[provision]'"
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port", help="Serial port (default: the only USB serial port found)."
    )
    parser.add_argument("--ssid", help="Home Wi-Fi network name (2.4 GHz).")
    parser.add_argument(
        "--password",
        help=(
            "Wi-Fi password. Prefer leaving this out: it is then read from "
            "STACKCHAN_WIFI_PASSWORD or prompted for, and stays out of shell history."
        ),
    )
    parser.add_argument(
        "--gateway-host",
        help="Address the robot uses to reach this gateway (default: detected LAN IP).",
    )
    parser.add_argument(
        "--websocket-url",
        help="Full WebSocket URL (default: ws://<gateway-host>:<WS_PORT>/).",
    )
    parser.add_argument(
        "--fallback-url", default="", help="Optional fallback WebSocket URL."
    )
    parser.add_argument(
        "--token",
        help="Gateway token (default: STACKCHAN_TOKEN from the environment / .env).",
    )
    parser.add_argument(
        "--no-ota-url",
        action="store_true",
        help=(
            "Do not point the firmware's version check at the gateway. Firmware "
            "with CONFIG_STACKCHAN_SKIP_CLOUD_VERSION_CHECK skips it anyway; older "
            "firmware will then try the upstream cloud on every boot."
        ),
    )
    parser.add_argument("--baud", type=int, default=460800, help="Serial baud rate.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be written, then exit."
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Do not ask for confirmation."
    )


def run(args: argparse.Namespace) -> int:
    ssid = args.ssid or input("Wi-Fi network name (2.4 GHz): ").strip()
    password = args.password
    if password is None:
        password = os.environ.get("STACKCHAN_WIFI_PASSWORD")
    if password is None:
        password = getpass.getpass(f"Password for {ssid!r}: ")

    host = args.gateway_host or detect_lan_ip()
    ws_port = int(os.environ.get("WS_PORT", DEFAULT_WS_PORT))
    capture_port = int(os.environ.get("CAPTURE_PORT", DEFAULT_CAPTURE_PORT))
    token = args.token
    if token is None:
        token = (
            os.environ.get("STACKCHAN_TOKEN") or os.environ.get("BEARER_TOKEN") or ""
        )

    settings = ProvisionSettings(
        ssid=ssid,
        password=password,
        websocket_url=args.websocket_url or f"ws://{host}:{ws_port}/",
        token=token,
        fallback_url=args.fallback_url,
        ota_url="" if args.no_ota_url else f"http://{host}:{capture_port}/ota/",
    )
    try:
        csv_text = build_nvs_csv(settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    port = args.port or detect_serial_port()
    print("Settings to write to the robot:")
    print(f"  serial port    {port or '<not found>'}")
    print(f"  wifi ssid      {settings.ssid}")
    print(f"  wifi password  {_mask(settings.password)}")
    print(f"  websocket url  {settings.websocket_url}")
    print(f"  token          {_mask(settings.token)}")
    if settings.fallback_url:
        print(f"  fallback url   {settings.fallback_url}")
    print(f"  ota url        {settings.ota_url or '<not set>'}")
    print("This replaces the robot's whole settings partition (volume and touch")
    print("tuning go back to defaults). The current partition is backed up first.")

    if args.dry_run:
        return 0
    if port is None:
        print("error: no single USB serial port found; pass --port", file=sys.stderr)
        return 2
    if not args.yes and input("Write these settings? [y/N] ").strip().lower() not in (
        "y",
        "yes",
    ):
        print("Cancelled.")
        return 1

    _require_tools()
    esptool = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "-p",
        port,
        "-b",
        str(args.baud),
    ]

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"nvs-{time.strftime('%Y%m%d-%H%M%S')}.bin"
    print(f"Backing up current settings to {backup}")
    _run(
        [
            *esptool,
            "--after",
            "no-reset",
            "read-flash",
            hex(NVS_OFFSET),
            hex(NVS_SIZE),
            str(backup),
        ]
    )
    backup.chmod(0o600)

    # The CSV and image hold the Wi-Fi password and token in clear text, so
    # they live in a private temp dir that is removed whatever happens.
    with tempfile.TemporaryDirectory(prefix="stackchan-provision-") as tmp:
        os.chmod(tmp, 0o700)
        csv_path = Path(tmp) / "nvs.csv"
        bin_path = Path(tmp) / "nvs.bin"
        csv_path.write_text(csv_text)
        _run(
            [
                sys.executable,
                "-m",
                "esp_idf_nvs_partition_gen",
                "generate",
                str(csv_path),
                str(bin_path),
                hex(NVS_SIZE),
            ]
        )
        print("Writing settings (the robot restarts when done)")
        _run([*esptool, "write-flash", hex(NVS_OFFSET), str(bin_path)])

    print(
        "Done. The robot should join Wi-Fi and connect to the gateway within a minute."
    )
    print(
        f"To undo: esptool --chip esp32s3 -p {port} write-flash {hex(NVS_OFFSET)} {backup}"
    )
    return 0
