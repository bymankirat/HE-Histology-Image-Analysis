"""Step 4 - the same measurements on five real H&E sections.

The synthetic work says what should happen when staining varies. These five published
tiles were stained to different strengths by different laboratories, so they are the test
of whether it does. Each tile is measured as it is, and again after being re-stained to a
common appearance, and the spread across tiles is compared.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

import config
from pipeline import nuclei, stains

# The published tiles are 241x180 px crops of a figure, so a nucleus is only a few pixels
# across. Counts are therefore made at a fixed 2x working scale, which puts nuclear
# diameters in the range the detector was validated on. Absolute densities from these
# tiles are not comparable with a full-resolution slide; the comparison between tiles is.
WORK_SCALE = 2


def load_tile(name: str, scale: int = 1) -> np.ndarray:
    img = Image.open(config.RAW / f"{name}.png").convert("RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    return np.asarray(img)


def cohort_target(percentile: float = 99.0) -> tuple[float, float]:
    """Normalisation target: the median stain level of the five tiles.

    Normalising towards the middle of the set rather than an arbitrary fixed value keeps
    the result inside the range the images actually span.
    """
    tops = np.array([stains.percentile_concentrations(load_tile(n), percentile)
                     for n in config.TILE_NAMES])
    return tuple(np.median(tops, axis=0))


def as_uint8(img: np.ndarray) -> np.ndarray:
    return img if img.dtype == np.uint8 else np.clip(img * 255, 0, 255).astype(np.uint8)


def cv(values) -> float:
    """Coefficient of variation in per cent: spread relative to the mean."""
    v = np.asarray(values, dtype=float)
    return float(100.0 * v.std(ddof=1) / abs(v.mean())) if v.mean() else float("nan")


def measure_tiles(normalise: bool = False, target=None) -> list[dict]:
    fixed = json.loads(config.FIXED_THRESHOLD_FILE.read_text())["fixed_threshold"]
    rows = []
    for name in config.TILE_NAMES:
        raw = load_tile(name, WORK_SCALE)
        img = as_uint8(stains.normalise_staining(raw, target_concentrations=target)) if normalise else raw
        mask = stains.tissue_mask(img)
        strength = stains.stain_strength(img, mask=mask)
        seg_adaptive = nuclei.segment(img)
        seg_fixed = nuclei.segment(img, threshold=fixed)
        px = img.shape[0] * img.shape[1]
        areas = np.array([o["area"] for o in seg_adaptive["objects"]], dtype=float)
        rows.append({
            "tile": name,
            "normalised": bool(normalise),
            "haematoxylin": strength["haematoxylin"],
            "eosin": strength["eosin"],
            "he_ratio": strength["he_ratio"],
            "tissue_fraction": strength["tissue_fraction"],
            "n_nuclei_adaptive": seg_adaptive["count"],
            "n_nuclei_fixed": seg_fixed["count"],
            "nuclei_per_megapixel": 1e6 * seg_adaptive["count"] / px,
            "median_radius_px": float(np.median(np.sqrt(areas / np.pi))) if areas.size else None,
            "threshold_adaptive": seg_adaptive["threshold"],
        })
    return rows


def main() -> dict:
    target = cohort_target()
    raw_rows = measure_tiles(normalise=False)
    norm_rows = measure_tiles(normalise=True, target=target)

    def spread(rows):
        return {
            "haematoxylin_cv_pct": cv([r["haematoxylin"] for r in rows]),
            "eosin_cv_pct": cv([r["eosin"] for r in rows]),
            "he_ratio_cv_pct": cv([r["he_ratio"] for r in rows]),
            "n_nuclei_fixed_cv_pct": cv([r["n_nuclei_fixed"] for r in rows]),
            "n_nuclei_adaptive_cv_pct": cv([r["n_nuclei_adaptive"] for r in rows]),
            "eosin_fold_range": float(max(r["eosin"] for r in rows) / max(min(r["eosin"] for r in rows), 1e-6)),
        }

    res = {"raw": raw_rows, "normalised": norm_rows, "work_scale": WORK_SCALE,
           "normalisation_target": {"haematoxylin": float(target[0]), "eosin": float(target[1])},
           "spread_raw": spread(raw_rows), "spread_normalised": spread(norm_rows)}
    (config.DERIVED / "real_tiles.json").write_text(json.dumps(res, indent=2))

    print("s04: five published H&E tiles")
    for r in raw_rows:
        print(f"     {r['tile']}: H {r['haematoxylin']:.3f}  E {r['eosin']:.3f}  "
              f"H/E {r['he_ratio']:5.1f}  nuclei {r['n_nuclei_adaptive']:4d} (adaptive) "
              f"{r['n_nuclei_fixed']:4d} (fixed)")
    print(f"     spread across tiles, eosin: {res['spread_raw']['eosin_fold_range']:.0f}x "
          f"(CV {res['spread_raw']['eosin_cv_pct']:.0f}%) -> after normalisation CV "
          f"{res['spread_normalised']['eosin_cv_pct']:.0f}%")
    norm_counts = [r["n_nuclei_fixed"] for r in norm_rows]
    after = ("no nuclei found at all" if all(c == 0 for c in norm_counts)
             else f"CV {res['spread_normalised']['n_nuclei_fixed_cv_pct']:.0f}%")
    print(f"     nuclei count with the fixed threshold: CV "
          f"{res['spread_raw']['n_nuclei_fixed_cv_pct']:.0f}% -> {after} after normalisation")
    return res


if __name__ == "__main__":
    main()
