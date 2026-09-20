"""A tiny dependency-free SVG builder.

Just enough to draw the architecture diagram, interaction-flow diagrams, and
the evaluation charts as clean, regenerable vector graphics -- no matplotlib,
no external assets. Everything the project renders comes from here.

Colours are defined once so light/dark-friendly, colour-blind-safe choices
stay consistent across every figure.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

import html
from typing import List, Optional, Tuple

# A restrained, colour-blind-safe palette (Okabe-Ito derived).
INK = "#1b1b1f"
MUTED = "#5c5c66"
PAPER = "#ffffff"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRID = "#d9d9e0"


class SVG:
    def __init__(self, width: int, height: int, title: str = ""):
        self.w = width
        self.h = height
        self.title = title
        self.parts: List[str] = []

    def rect(self, x, y, w, h, fill=PAPER, stroke=INK, rx=8, sw=2, opacity=1.0):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=2, dash: Optional[str] = None, marker=True):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if marker else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}{m}/>')

    def polyline(self, points: List[Tuple[float, float]], stroke=BLUE, sw=2, fill="none"):
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, cx, cy, r, fill=BLUE, stroke="none", sw=0):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=14, fill=INK, anchor="start", weight="normal", family="sans-serif"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{family}">{html.escape(str(s))}</text>')

    def render(self) -> str:
        defs = (
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker></defs>')
        title = f"<title>{html.escape(self.title)}</title>" if self.title else ""
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" role="img" '
            f'aria-label="{html.escape(self.title)}">\n{title}{defs}\n'
            f'<rect width="{self.w}" height="{self.h}" fill="{PAPER}"/>\n{body}\n</svg>\n')

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
