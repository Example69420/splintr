#!/usr/bin/env python3
"""Render an animated demo GIF of the 'locate a tag' flow -- pure standard
library, no Pillow. A self-contained GIF89a + LZW encoder draws the geiger-style
proximity meter accelerating as the target gets closer, then turning green on
'found'. Pair it with media/flow_locate_tag.wav to *hear* the same flow.

Run:  python3 media/generate_gif.py

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import math
import os
import struct
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from soundkey_sim.flows import _approach_curve  # noqa: E402

W, H = 420, 150

# Indexed palette (index -> RGB)
PALETTE = [
    (0xFF, 0xFF, 0xFF),  # 0 paper
    (0x1B, 0x1B, 0x1F),  # 1 ink
    (0x00, 0x72, 0xB2),  # 2 blue (bar)
    (0x00, 0x9E, 0x73),  # 3 green (found / lit click)
    (0xD9, 0xD9, 0xE0),  # 4 grid
    (0xE6, 0x9F, 0x00),  # 5 amber (click blip)
    (0x5C, 0x5C, 0x66),  # 6 muted
]


def blank() -> List[List[int]]:
    return [[0] * W for _ in range(H)]


def fill_rect(px, x, y, w, h, c):
    for yy in range(max(0, y), min(H, y + h)):
        row = px[yy]
        for xx in range(max(0, x), min(W, x + w)):
            row[xx] = c


# A tiny 3x5 bitmap font, enough for digits, '%', and a few words.
FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "%": ["101", "001", "010", "100", "101"],
    " ": ["000", "000", "000", "000", "000"],
    "F": ["111", "100", "111", "100", "100"],
    "O": ["111", "101", "101", "101", "111"],
    "U": ["101", "101", "101", "101", "111"],
    "N": ["101", "111", "111", "111", "101"],
    "D": ["110", "101", "101", "101", "110"],
    "L": ["100", "100", "100", "100", "111"],
    "C": ["111", "100", "100", "100", "111"],
    "A": ["111", "101", "111", "101", "101"],
    "T": ["111", "010", "010", "010", "010"],
    "E": ["111", "100", "111", "100", "111"],
    "G": ["111", "100", "101", "101", "111"],
    ":": ["000", "010", "000", "010", "000"],
}


def draw_text(px, x, y, text, c, scale=2):
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, line in enumerate(glyph):
            for gx, bit in enumerate(line):
                if bit == "1":
                    fill_rect(px, cx + gx * scale, y + gy * scale, scale, scale, c)
        cx += (3 + 1) * scale
    return cx


def frame_for(level: float, found: bool) -> List[List[int]]:
    px = blank()
    # title
    draw_text(px, 16, 12, "LOCATE", 1, 2)
    draw_text(px, 16, 30, "A TAG", 6, 2)
    # proximity bar frame
    bx, by, bw, bh = 16, 60, W - 32, 26
    fill_rect(px, bx, by, bw, bh, 4)
    fill = int(bw * level)
    fill_rect(px, bx, by, fill, bh, 3 if found else 2)
    # click blips: number + brightness scale with level (rate proxy)
    n = 1 + int(level * 14)
    for i in range(n):
        gx = bx + 6 + i * ((bw - 12) // 15)
        fill_rect(px, gx, by + bh + 12, 6, 18, 5 if not found else 3)
    # percent / FOUND caption
    pct = round(level * 100)
    label = "FOUND" if found else f"{pct}%"
    draw_text(px, bx, by + bh + 40, label, 3 if found else 1, 3)
    return px


def build_frames() -> List[List[List[int]]]:
    curve = _approach_curve(12)
    frames = []
    for lvl in curve:
        frames.append(frame_for(lvl, found=False))
    # a few 'found' frames to hold
    for _ in range(4):
        frames.append(frame_for(1.0, found=True))
    return frames


# --- GIF89a + LZW encoder --------------------------------------------------

def _lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    """GIF LZW using the 'store' technique: emit each pixel as a literal code and
    re-send a Clear code before the decoder's dictionary would ever grow enough
    to change the code width. This keeps the code size fixed at
    ``min_code_size + 1`` and sidesteps the notorious growth off-by-one, at the
    cost of a little size. Output is always a valid GIF LZW stream.
    """
    clear = 1 << min_code_size
    end = clear + 1
    code_size = min_code_size + 1

    out_bits = 0
    out_nbits = 0
    out = bytearray()

    def emit(code):
        nonlocal out_bits, out_nbits
        out_bits |= code << out_nbits
        out_nbits += code_size
        while out_nbits >= 8:
            out.append(out_bits & 0xFF)
            out_bits >>= 8
            out_nbits -= 8

    emit(clear)
    count = 0
    reset_after = clear - 2  # re-clear before the decoder's table would widen codes
    for b in indices:
        emit(b)  # b < clear, so it is always a literal code
        count += 1
        if count >= reset_after:
            emit(clear)
            count = 0
    emit(end)
    if out_nbits > 0:
        out.append(out_bits & 0xFF)
    return bytes(out)


def _blockify(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i:i + 255]
        out.append(len(chunk))
        out += chunk
    out.append(0)
    return bytes(out)


def write_gif(path: str, frames: List[List[List[int]]], delay_cs: int = 18):
    ncolors = len(PALETTE)
    # global color table size must be power of two
    gct_size = 1
    while (1 << gct_size) < ncolors:
        gct_size += 1
    table_len = 1 << gct_size

    with open(path, "wb") as f:
        f.write(b"GIF89a")
        f.write(struct.pack("<HH", W, H))
        packed = 0x80 | ((gct_size - 1) << 4) | (gct_size - 1)
        f.write(bytes([packed, 0, 0]))
        for i in range(table_len):
            r, g, b = PALETTE[i] if i < ncolors else (0, 0, 0)
            f.write(bytes([r, g, b]))
        # loop forever (NETSCAPE)
        f.write(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")
        min_code_size = max(2, gct_size)
        for px in frames:
            # graphic control extension (delay)
            f.write(b"\x21\xF9\x04\x00")
            f.write(struct.pack("<H", delay_cs))
            f.write(b"\x00\x00")
            # image descriptor
            f.write(b"\x2C")
            f.write(struct.pack("<HHHH", 0, 0, W, H))
            f.write(b"\x00")
            f.write(bytes([min_code_size]))
            flat = bytes(px[y][x] for y in range(H) for x in range(W))
            f.write(_blockify(_lzw_encode(flat, min_code_size)))
        f.write(b"\x3B")


def main() -> int:
    frames = build_frames()
    out = os.path.join(HERE, "demo_locate_tag.gif")
    write_gif(out, frames)
    print(f"wrote {out} ({os.path.getsize(out)} bytes, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
