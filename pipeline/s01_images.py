"""Step 1 - get the real H&E tiles and record exactly what was downloaded.

Five breast-tissue H&E tiles published in the StainTools repository (MIT licence) are used
as the real-data example. They were chosen because they were stained to visibly different
strengths, which is the question this project is about. The published file is a figure
strip, so the tiles are cut out of it by finding the non-white panels, and every file is
fingerprinted so a later run can prove it used the same pixels.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request

import numpy as np
from PIL import Image

import config

STRIP = config.RAW / "staintools_strip.png"
MANIFEST = config.RAW / "manifest.json"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    """Fetch the published figure strip once."""
    if STRIP.exists() and not force:
        return
    req = urllib.request.Request(config.TILE_SOURCE_RAW, headers={"User-Agent": "histology-pipeline"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(STRIP, "wb") as out:
        out.write(resp.read())


def panel_bounds(arr: np.ndarray, white: int = 245, min_width: int = 60) -> list[tuple[int, int]]:
    """Column ranges of the image panels inside a figure strip with white margins."""
    non_white = (arr[..., :3].min(axis=2) < white).any(axis=0)
    bounds, start = [], None
    for x, is_ink in enumerate(non_white):
        if is_ink and start is None:
            start = x
        elif not is_ink and start is not None:
            if x - start >= min_width:
                bounds.append((start, x))
            start = None
    if start is not None and len(non_white) - start >= min_width:
        bounds.append((start, len(non_white)))
    return bounds


def row_bounds(arr: np.ndarray, white: int = 245) -> tuple[int, int]:
    """Row range of the coloured panel, dropping the text caption above it."""
    coloured = (arr[..., :3].max(axis=2).astype(int) - arr[..., :3].min(axis=2).astype(int)) > 12
    rows = np.where(coloured.sum(axis=1) > 0.3 * coloured.shape[1])[0]
    if rows.size == 0:
        dark = (arr[..., :3].min(axis=2) < white).any(axis=1)
        rows = np.where(dark)[0]
    return int(rows.min()), int(rows.max()) + 1


def cut_tiles() -> list[dict]:
    """Cut the strip into one PNG per tile and return what was written."""
    arr = np.asarray(Image.open(STRIP).convert("RGB"))
    r0, r1 = row_bounds(arr)
    panels = panel_bounds(arr[r0:r1])
    if len(panels) != len(config.TILE_NAMES):
        raise RuntimeError(f"expected {len(config.TILE_NAMES)} panels in the strip, found {len(panels)}")
    out = []
    for name, (c0, c1) in zip(config.TILE_NAMES, panels):
        tile = arr[r0:r1, c0:c1]
        path = config.RAW / f"{name}.png"
        Image.fromarray(tile).save(path)
        out.append({"name": name, "file": path.name, "height": int(tile.shape[0]),
                    "width": int(tile.shape[1]), "md5": md5(path)})
    return out


def main() -> dict:
    download()
    tiles = cut_tiles()
    manifest = {
        "source_repository": config.TILE_SOURCE_REPO,
        "source_file": config.TILE_SOURCE_FILE,
        "licence": "MIT (StainTools, Peter Byfield)",
        "strip_md5": md5(STRIP),
        "tiles": tiles,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"s01: {len(tiles)} tiles, strip md5 {manifest['strip_md5'][:12]}")
    for t in tiles:
        print(f"     {t['name']}: {t['width']}x{t['height']} px, md5 {t['md5'][:12]}")
    return manifest


if __name__ == "__main__":
    main()
