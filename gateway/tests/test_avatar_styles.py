"""Tests for the procedural avatar sets."""

from __future__ import annotations

import numpy as np
import pytest

from stackchan_mcp import avatar_styles as av


@pytest.mark.parametrize("style", sorted(av.STYLES))
def test_matrix_payload_matches_firmware_layout(style):
    payload, frames = av.build_matrix(av.STYLES[style])
    # AvatarSet::kMatrixPayloadBytes = 90 * 160 * 120 * 2
    assert len(payload) == 90 * 160 * 120 * 2
    assert len(frames) == 90
    # Every frame draws something on top of the background.
    bg = np.asarray(av.STYLES[style].bg, np.uint8)
    for f in frames:
        assert (f != bg).any()


def test_matrix_index_order_is_face_eyes_mouth():
    s = av.STYLES["classic"]
    _, frames = av.build_matrix(s)
    face, eyes, mouth = av.FACES.index("sad"), av.EYES.index("half"), av.MOUTHS.index("open")
    expected = av.render(s, "sad", "half", "open")
    assert np.array_equal(frames[face * 15 + eyes * 5 + mouth], expected)


def test_rgb565_little_endian():
    px = np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8)
    assert av.to_rgb565(px) == bytes([0x00, 0xF8, 0x1F, 0x00])


def test_cli_writes_payload_and_preview(tmp_path):
    assert av.main(["classic", "--out", str(tmp_path), "--preview"]) == 0
    assert (tmp_path / "classic-matrix.rgb565").stat().st_size == 90 * 38400
    assert (tmp_path / "classic-preview.png").read_bytes().startswith(b"\x89PNG")
