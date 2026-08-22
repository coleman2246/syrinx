#!/usr/bin/env python3
"""Generate the app icons.

The desktop has no icon file to copy: the Windows tray draws a feathered disc
at runtime and Linux uses named theme icons. So the phone icon is derived from
those rather than invented -- the tray's listening red on the GUI's canvas,
ringed in the brand colour, so the icon and the tray read as one application.

Only the 60x60 sizes are bundled; iOS is given loose CFBundleIconFiles rather
than an asset catalogue, because actool refuses to run without a simulator
runtime installed. The 1024 is kept here for anywhere that wants it and is
deliberately not in the folder the app bundles.

    python3 ios/icon/make-icon.py
"""

import pathlib
from PIL import Image, ImageDraw

CANVAS = (0x1F, 0x1F, 0x1F)  # palette::CANVAS, the GUI's background
DISC = (0xEB, 0x46, 0x3C)    # the Windows tray's Listening colour
BRAND = (0x5B, 0x5F, 0xC7)   # palette::BRAND

HERE = pathlib.Path(__file__).parent
BUNDLED = HERE.parent / "SyrinxDemo" / "Icons"


def icon(px: int, path: pathlib.Path) -> None:
    # Supersampled, then reduced: the disc edge is the whole design, so it has
    # to be clean at 120 pixels as well as at 1024.
    ss = 8
    size = px * ss
    img = Image.new("RGB", (size, size), CANVAS)
    d = ImageDraw.Draw(img)
    c = size // 2

    ring = int(size * 0.335)
    d.ellipse([c - ring, c - ring, c + ring, c + ring],
              outline=BRAND, width=max(int(size * 0.022), ss // 2))
    disc = int(size * 0.215)
    d.ellipse([c - disc, c - disc, c + disc, c + disc], fill=DISC)

    img.resize((px, px), Image.LANCZOS).save(path)
    print(f"{path} ({px}px)")


if __name__ == "__main__":
    icon(120, BUNDLED / "AppIcon60x60@2x.png")
    icon(180, BUNDLED / "AppIcon60x60@3x.png")
    icon(1024, HERE / "AppIcon1024.png")
