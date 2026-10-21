"""Finding and measuring nuclei in the haematoxylin channel.

Haematoxylin binds nucleic acids, so after deconvolution the nuclei are the bright
objects of that channel and everything else is nearly flat. The steps are deliberately
plain -- smooth, threshold, clean, split touching objects by watershed -- because every
one of them is then scored against the synthetic ground truth rather than taken on trust.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

import config
from pipeline import stains


def haematoxylin(img: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    """The haematoxylin concentration map of an RGB image."""
    return stains.separate(img, matrix)[..., 0]


def segment(img: np.ndarray, matrix: np.ndarray | None = None,
            params: dict | None = None, threshold: float | None = None) -> dict:
    """Label the nuclei in one field.

    Returns the label image, the per-object table and the threshold actually used.
    """
    p = dict(config.NUCLEI) | (params or {})
    h = haematoxylin(img, matrix)
    sm = filters.gaussian(h, sigma=p["smooth_sigma"], preserve_range=True)

    if threshold is None:
        thr = float(filters.threshold_otsu(sm))
        if thr < p["min_threshold"]:
            # nothing on the slide: Otsu split noise. Return an empty field rather than debris.
            empty = np.zeros(sm.shape, dtype=int)
            return {"labels": empty, "objects": [], "threshold": thr, "count": 0,
                    "mask_fraction": 0.0}
    else:
        thr = float(threshold)
    mask = sm > thr
    if p["fill_holes"]:
        mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=p["min_area"])
    if p["clear_border"]:
        mask = segmentation.clear_border(mask)

    # split touching nuclei: seed one marker per distance-transform peak
    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=p["peak_min_distance"], labels=mask)
    markers = np.zeros(dist.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labels = segmentation.watershed(-dist, markers, mask=mask)

    table = []
    for reg in measure.regionprops(labels, intensity_image=h):
        if reg.area < p["min_area"] or reg.area > p["max_area"]:
            continue
        table.append({
            "label": int(reg.label),
            "row": float(reg.centroid[0]),
            "col": float(reg.centroid[1]),
            "area": float(reg.area),
            "equivalent_radius": float(np.sqrt(reg.area / np.pi)),
            "eccentricity": float(reg.eccentricity),
            "solidity": float(reg.solidity),
            "mean_haematoxylin": float(reg.intensity_mean),
        })
    keep = {r["label"] for r in table}
    labels = np.where(np.isin(labels, list(keep)), labels, 0)
    return {"labels": labels, "objects": table, "threshold": float(thr),
            "count": len(table), "mask_fraction": float(mask.mean())}


def match_to_truth(detected: list[dict], centres: np.ndarray,
                   max_distance: float = config.MATCH_DISTANCE) -> dict:
    """Greedy nearest-neighbour matching of detections to true nuclei centres.

    Each true nucleus can be claimed once. Pairs further apart than `max_distance` are
    not matched, so a detection on the wrong object counts as a false positive and the
    true nucleus it missed counts as a false negative.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    det = np.array([[d["row"], d["col"]] for d in detected], dtype=float).reshape(-1, 2)
    if len(det) == 0 or len(centres) == 0:
        tp = 0
        pairs = []
    else:
        dist = np.sqrt(((det[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
        pairs = []
        used_det, used_true = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
        for i, j in order:
            if dist[i, j] > max_distance:
                break
            if i in used_det or j in used_true:
                continue
            used_det.add(int(i))
            used_true.add(int(j))
            pairs.append((int(i), int(j), float(dist[i, j])))
        tp = len(pairs)
    fp = len(detected) - tp
    fn = len(centres) - tp
    precision = tp / max(len(detected), 1)
    recall = tp / max(len(centres), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "n_detected": len(detected), "n_true": len(centres),
            "count_error": len(detected) - len(centres),
            "count_error_pct": 100.0 * (len(detected) - len(centres)) / max(len(centres), 1),
            "median_offset": float(np.median([p[2] for p in pairs])) if pairs else None}


def measure_field(img: np.ndarray, matrix: np.ndarray | None = None,
                  params: dict | None = None, mask: np.ndarray | None = None) -> dict:
    """Everything the pipeline reports for one field: counts, sizes and stain strength."""
    seg = segment(img, matrix, params)
    areas = np.array([o["area"] for o in seg["objects"]], dtype=float)
    px = img.shape[0] * img.shape[1]
    out = {
        "n_nuclei": seg["count"],
        "nuclei_per_megapixel": 1e6 * seg["count"] / px,
        "median_area_px": float(np.median(areas)) if areas.size else None,
        "median_radius_px": float(np.median(np.sqrt(areas / np.pi))) if areas.size else None,
        "nuclear_area_fraction": float(areas.sum() / px) if areas.size else 0.0,
        "threshold": seg["threshold"],
    }
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)
    out |= stains.stain_strength(img, matrix, mask=mask)
    return out | {"_segmentation": seg}
