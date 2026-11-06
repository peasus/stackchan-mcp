"""Procedurally drawn avatar sets for ``load_avatar_set``.

Builds a 90-frame "matrix" payload (6 faces x 3 eye states x 5 mouth
shapes, 160x120 RGB565 little-endian, index = face * 15 + eyes * 5 + mouth)
from simple vector shapes, so a new look needs no artwork and no firmware
rebuild. Only numpy is needed (already pulled in by faster-whisper).

    python -m stackchan_mcp.avatar_styles classic --out ~/.stackchan/avatar_sets
    # then call the MCP tool: load_avatar_set(archive_path=..., mode="matrix")

Shapes are rendered as signed distance fields with 4x supersampling for
smooth edges.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

W, H = 160, 120
SS = 4  # supersampling factor
FACES = ["idle", "happy", "thinking", "sad", "surprised", "embarrassed"]
EYES = ["open", "half", "closed"]
MOUTHS = ["closed", "half", "open", "e", "u"]
FRAME_BYTES = W * H * 2


@dataclass(frozen=True)
class Style:
    name: str
    bg: tuple[int, int, int]
    fg: tuple[int, int, int]
    blush: tuple[int, int, int]
    eye_shape: str  # "circle" | "rect" | "big"
    eye_r: float
    highlight: tuple[int, int, int] | None = None
    always_blush: bool = False
    decor: str = ""  # "kitty" | "starry" | ""


STYLES = {
    "classic": Style("classic", (238, 234, 222), (20, 20, 24), (240, 150, 160), "circle", 11.0),
    "retro": Style("retro", (8, 10, 18), (70, 225, 255), (255, 90, 160), "rect", 11.0),
    "cartoon": Style(
        "cartoon",
        (255, 241, 246),
        (40, 30, 50),
        (255, 150, 175),
        "big",
        12.0,
        highlight=(255, 255, 255),
        always_blush=True,
    ),
    "kitty": Style(
        "kitty",
        (255, 236, 210),
        (60, 40, 35),
        (255, 160, 150),
        "big",
        10.0,
        highlight=(255, 255, 255),
        always_blush=True,
        decor="kitty",
    ),
    "starry": Style(
        "starry",
        (18, 22, 58),
        (255, 236, 160),
        (255, 120, 170),
        "circle",
        11.0,
        decor="starry",
    ),
}

# Supersampled pixel-centre grid, in 160x120 units.
_ys, _xs = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
X = (_xs + 0.5) / SS
Y = (_ys + 0.5) / SS


def _circle(cx, cy, r):
    return np.hypot(X - cx, Y - cy) - r


def _ellipse(cx, cy, rx, ry):
    # Approximate SDF, good enough for small shapes.
    k = np.hypot((X - cx) / rx, (Y - cy) / ry)
    return (k - 1.0) * min(rx, ry)


def _rrect(cx, cy, hw, hh, r):
    qx = np.abs(X - cx) - (hw - r)
    qy = np.abs(Y - cy) - (hh - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


def _segment(ax, ay, bx, by, thick):
    px, py = X - ax, Y - ay
    dx, dy = bx - ax, by - ay
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - t * dx, py - t * dy) - thick / 2


def _arc(cx, cy, r, thick, up: bool):
    """Half-circle stroke. up=True draws the top half (a ^ / frown shape)."""
    d = np.abs(np.hypot(X - cx, Y - cy) - r) - thick / 2
    mask = (Y <= cy) if up else (Y >= cy)
    return np.where(mask, d, np.inf)


def _star(cx, cy, r_out, r_in, points=5, rot=-np.pi / 2):
    """Filled star via polar radius interpolation (approximate SDF)."""
    ang = np.arctan2(Y - cy, X - cx) - rot
    seg = 2 * np.pi / points
    t = np.abs(((ang % seg) / seg) * 2 - 1)  # 1 at a point, 0 between points
    r_edge = r_in + (r_out - r_in) * t
    return np.hypot(X - cx, Y - cy) - r_edge


def _triangle(ax, ay, bx, by, cx_, cy_):
    def edge(px, py, qx, qy):
        nx, ny = qy - py, px - qx
        n = np.hypot(nx, ny)
        return ((X - px) * nx + (Y - py) * ny) / n

    e1, e2, e3 = edge(ax, ay, bx, by), edge(bx, by, cx_, cy_), edge(cx_, cy_, ax, ay)
    # Either winding: inside means all edges agree in sign.
    return np.minimum(np.maximum(np.maximum(e1, e2), e3), np.maximum(np.maximum(-e1, -e2), -e3))


_STARS = [(12, 14), (30, 100), (70, 10), (100, 112), (140, 20), (152, 70), (6, 60), (124, 96), (88, 30)]


class _Canvas:
    def __init__(self, bg):
        self.img = np.empty((H * SS, W * SS, 3), np.float32)
        self.img[:] = bg

    def fill(self, sdf, color):
        a = np.clip(0.5 - sdf * SS, 0, 1)[..., None]  # ~1px antialias ramp
        self.img = self.img * (1 - a) + np.asarray(color, np.float32) * a

    def rgb(self) -> np.ndarray:
        small = self.img.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))
        return np.clip(small + 0.5, 0, 255).astype(np.uint8)


EYE_L, EYE_R, EYE_Y = 46.0, 114.0, 54.0
MOUTH_X, MOUTH_Y = 80.0, 90.0
STROKE = 4.0


def _draw_eye(c: _Canvas, s: Style, cx, cy, face, eyes):
    r = s.eye_r * (1.35 if face == "surprised" else 1.0)
    if face == "thinking":
        cx, cy = cx + 4, cy - 4  # glancing up and to the side
    if s.decor == "starry" and face in ("happy", "surprised") and eyes == "open":
        c.fill(_star(cx, cy, r * 1.5, r * 0.6), s.fg)
        return
    if face == "happy" and eyes != "closed":
        # Smiling eyes: a ^ arc, a little flatter when half-closed.
        rr = r if eyes == "open" else r * 0.8
        c.fill(_arc(cx, cy + rr * 0.4, rr, STROKE, up=True), s.fg)
        return
    if eyes == "closed":
        c.fill(_segment(cx - r, cy, cx + r, cy, STROKE), s.fg)
        return
    if s.eye_shape == "rect":
        hh = r * (0.45 if eyes == "half" else 1.0)
        c.fill(_rrect(cx, cy + (r - hh) * 0.6, r * 0.8, hh, 2.0), s.fg)
        return
    if s.eye_shape == "big":
        ry = r * 1.25
        sdf = _ellipse(cx, cy, r, ry)
    else:
        sdf = _circle(cx, cy, r)
    if eyes == "half":
        # Lid comes down: keep only the lower part of the eye.
        sdf = np.maximum(sdf, (cy - r * 0.15) - Y)
    c.fill(sdf, s.fg)
    if s.highlight and eyes == "open":
        c.fill(_circle(cx + r * 0.35, cy - r * 0.45, r * 0.3), s.highlight)


def _draw_brows(c: _Canvas, s: Style, face):
    y = EYE_Y - 18
    if face == "sad":
        c.fill(_segment(EYE_L - 10, y + 2, EYE_L + 8, y - 4, 3.0), s.fg)
        c.fill(_segment(EYE_R + 10, y + 2, EYE_R - 8, y - 4, 3.0), s.fg)
    elif face == "thinking":
        c.fill(_segment(EYE_L - 9, y - 1, EYE_L + 9, y - 1, 3.0), s.fg)
        c.fill(_segment(EYE_R - 9, y - 6, EYE_R + 9, y - 2, 3.0), s.fg)
    elif face == "surprised":
        c.fill(_arc(EYE_L, y - 3, 10, 3.0, up=True), s.fg)
        c.fill(_arc(EYE_R, y - 3, 10, 3.0, up=True), s.fg)


def _draw_mouth(c: _Canvas, s: Style, face, mouth):
    x, y = MOUTH_X, MOUTH_Y
    if mouth == "closed" and s.decor == "kitty" and face in ("idle", "happy", "thinking"):
        c.fill(_arc(x - 6, y - 3, 6, STROKE * 0.8, up=False), s.fg)
        c.fill(_arc(x + 6, y - 3, 6, STROKE * 0.8, up=False), s.fg)
        return
    if mouth == "closed":
        if face == "happy":
            c.fill(_arc(x, y - 6, 12, 3.2, up=False), s.fg)
        elif face == "sad":
            c.fill(_arc(x, y + 6, 10, 3.2, up=True), s.fg)
        elif face == "surprised":
            c.fill(_ellipse(x, y, 6, 7), s.fg)
        elif face == "thinking":
            c.fill(_segment(x - 7, y + 1, x + 7, y - 2, STROKE), s.fg)
        elif face == "embarrassed":
            for i in range(3):  # small wavy mouth
                c.fill(_arc(x - 8 + i * 8, y, 4, 3.0, up=i % 2 == 1), s.fg)
        else:
            c.fill(_segment(x - 13, y, x + 13, y, STROKE), s.fg)
        return
    if s.eye_shape == "rect":
        sizes = {"half": (10, 3), "open": (11, 7), "e": (15, 3), "u": (5, 5)}
        hw, hh = sizes[mouth]
        c.fill(_rrect(x, y, hw, hh, 1.5), s.fg)
        return
    sizes = {"half": (11, 5), "open": (12, 10), "e": (17, 5), "u": (6, 8)}
    rx, ry = sizes[mouth]
    c.fill(_ellipse(x, y, rx, ry), s.fg)


def render(style: Style, face: str, eyes: str, mouth: str) -> np.ndarray:
    c = _Canvas(style.bg)
    if style.decor == "starry":
        for i, (sx, sy) in enumerate(_STARS):
            c.fill(_star(sx, sy, 2.6 if i % 3 else 3.6, 1.1), (190, 200, 255))
    if style.decor == "kitty":
        ear = (240, 200, 170)
        inner = (255, 170, 170)
        c.fill(_triangle(14, 22, 30, -6, 46, 18), ear)
        c.fill(_triangle(146, 22, 130, -6, 114, 18), ear)
        c.fill(_triangle(22, 16, 30, 2, 38, 14), inner)
        c.fill(_triangle(138, 16, 130, 2, 122, 14), inner)
    if face == "embarrassed" or style.always_blush:
        alpha_blush = 1.0 if face == "embarrassed" else 0.55
        col = tuple(
            int(b * alpha_blush + bg * (1 - alpha_blush))
            for b, bg in zip(style.blush, style.bg)
        )
        c.fill(_ellipse(EYE_L - 6, EYE_Y + 20, 11, 6), col)
        c.fill(_ellipse(EYE_R + 6, EYE_Y + 20, 11, 6), col)
    _draw_brows(c, style, face)
    _draw_eye(c, style, EYE_L, EYE_Y, face, eyes)
    _draw_eye(c, style, EYE_R, EYE_Y, face, eyes)
    _draw_mouth(c, style, face, mouth)
    if style.decor == "kitty":
        for dy, tilt in ((-3, -3), (2, 0), (7, 3)):
            c.fill(_segment(EYE_L - 30, MOUTH_Y + dy + tilt, EYE_L - 8, MOUTH_Y + dy, 2.0), style.fg)
            c.fill(_segment(EYE_R + 30, MOUTH_Y + dy + tilt, EYE_R + 8, MOUTH_Y + dy, 2.0), style.fg)
    return c.rgb()


def to_rgb565(rgb: np.ndarray) -> bytes:
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return v.astype("<u2").tobytes()


def build_matrix(style: Style) -> tuple[bytes, list[np.ndarray]]:
    frames = []
    for face in FACES:
        for eyes in EYES:
            for mouth in MOUTHS:
                frames.append(render(style, face, eyes, mouth))
    payload = b"".join(to_rgb565(f) for f in frames)
    assert len(payload) == 90 * FRAME_BYTES
    return payload, frames


def _write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def contact_sheet(frames: list[np.ndarray], scale: int = 2) -> np.ndarray:
    """One row per face: eyes open + mouth closed, half-lid, closed eyes, mouth open."""
    picks = [(0, 0), (1, 0), (2, 0), (0, 2)]
    gap = 4
    rows = []
    for f in range(len(FACES)):
        tiles = [frames[f * 15 + e * 5 + m] for e, m in picks]
        row = np.concatenate(
            [np.pad(t, ((gap, gap), (gap, gap), (0, 0)), constant_values=128) for t in tiles],
            axis=1,
        )
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    return sheet.repeat(scale, axis=0).repeat(scale, axis=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("style", choices=sorted(STYLES))
    ap.add_argument("--out", type=Path, default=Path.home() / ".stackchan" / "avatar_sets")
    ap.add_argument("--preview", action="store_true", help="Also write a contact-sheet PNG.")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    payload, frames = build_matrix(STYLES[args.style])
    bin_path = args.out / f"{args.style}-matrix.rgb565"
    bin_path.write_bytes(payload)
    print(bin_path)
    if args.preview:
        png = args.out / f"{args.style}-preview.png"
        _write_png(png, contact_sheet(frames))
        print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
