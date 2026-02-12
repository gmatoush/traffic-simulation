"""Generate app icon assets for Windows (.ico) and macOS (.icns)."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BRANDING_ICON = ASSETS / "branding" / "traffic_sim.png"
SOURCE_ICON = ASSETS / "app_icon.png"
ICO_PATH = ASSETS / "app_icon.ico"
ICNS_PATH = ASSETS / "app_icon.icns"


def _ensure_source_icon() -> None:
    """Ensure app_icon.png exists, preferring the branded traffic_sim image."""
    if SOURCE_ICON.exists():
        return
    if BRANDING_ICON.exists():
        img = Image.open(BRANDING_ICON).convert("RGBA")
        img.save(SOURCE_ICON)
        return
    size = 512
    img = Image.new("RGBA", (size, size), (20, 24, 32, 255))
    draw = ImageDraw.Draw(img)
    # Draw a simple road + center line motif.
    draw.rectangle([160, 60, 352, 452], fill=(52, 58, 72, 255))
    draw.rectangle([60, 220, 452, 292], fill=(52, 58, 72, 255))
    for y in range(80, 430, 40):
        draw.rectangle([252, y, 260, y + 20], fill=(230, 200, 60, 255))
    for x in range(80, 430, 40):
        draw.rectangle([x, 252, x + 20, 260], fill=(230, 200, 60, 255))
    img.save(SOURCE_ICON)


def _generate_ico() -> None:
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img = Image.open(SOURCE_ICON).convert("RGBA")
    img.save(ICO_PATH, sizes=sizes)


def _generate_icns() -> None:
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    img = Image.open(SOURCE_ICON).convert("RGBA")
    iconset = [(s, s) for s in sizes]
    img.save(ICNS_PATH, sizes=iconset)


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    _ensure_source_icon()
    _generate_ico()
    _generate_icns()
    print(f"Generated {ICO_PATH} and {ICNS_PATH}")


if __name__ == "__main__":
    main()
