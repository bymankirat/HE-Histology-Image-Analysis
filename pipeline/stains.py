"""Colour deconvolution: separating the two dyes of an H&E section.

A stained section absorbs light, so the useful quantity is optical density (OD), not
brightness. For a pixel of intensity I and a blank-slide intensity I0,

    OD = -log10(I / I0)

and the Beer-Lambert law makes OD additive over stains: each stain contributes its own
unit colour vector times how much dye is there. Written for all three channels at once,

    OD (1x3) = c (1xK) @ M (Kx3)

with M the stain matrix (one unit vector per stain) and c the concentrations. Recovering c
from OD is a linear least-squares problem, which is what Ruifrok & Johnston (2001) do.

Everything here is written from the equations rather than called from a library, so the
result can be checked against scikit-image independently (see tests).
"""
from __future__ import annotations

import numpy as np

import config

EPS = 1e-6


def as_float(img: np.ndarray) -> np.ndarray:
    """uint8 or float image -> float array in [0, 1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    out = img.astype(np.float64)
    return out if out.max() <= 1.0 + EPS else out / 255.0


def rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Transmitted intensity -> optical density. Zeros are clipped so the log stays finite."""
    rgb = np.clip(as_float(img), EPS, 1.0)
    return -np.log10(rgb)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Optical density -> transmitted intensity in [0, 1]."""
    return np.clip(10.0 ** (-np.asarray(od, dtype=np.float64)), 0.0, 1.0)


def normalise_rows(mat: np.ndarray) -> np.ndarray:
    """Scale every row to unit length; a zero row is left alone."""
    mat = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    return mat / norms


def stain_matrix(stains=config.HE_STAINS) -> np.ndarray:
    """Reference stain matrix, one unit OD vector per named stain."""
    return normalise_rows(np.array([config.STAIN_VECTORS[s] for s in stains], dtype=np.float64))


def separate(img: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    """Split an RGB image into per-stain concentration maps.

    Returns an array of shape (H, W, K): one concentration image per stain, in OD units.
    Concentrations are clipped at zero, because a negative amount of dye is not physical.
    """
    matrix = stain_matrix() if matrix is None else normalise_rows(matrix)
    od = rgb_to_od(img)
    flat = od.reshape(-1, 3)
    # least squares: solve flat ~= c @ matrix for c
    conc, *_ = np.linalg.lstsq(matrix.T, flat.T, rcond=None)
    conc = np.clip(conc.T, 0.0, None)
    return conc.reshape(od.shape[0], od.shape[1], matrix.shape[0])


def recombine(conc: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    """Concentration maps -> RGB, the inverse of `separate` when nothing is clipped."""
    matrix = stain_matrix() if matrix is None else normalise_rows(matrix)
    flat = np.asarray(conc, dtype=np.float64).reshape(-1, matrix.shape[0])
    od = (flat @ matrix).reshape(conc.shape[0], conc.shape[1], 3)
    return od_to_rgb(od)


def stain_image(conc: np.ndarray, index: int, matrix: np.ndarray | None = None) -> np.ndarray:
    """One stain shown on its own, in colour: every other stain set to zero."""
    only = np.zeros_like(conc)
    only[..., index] = conc[..., index]
    return recombine(only, matrix)


def tissue_mask(img: np.ndarray, od_threshold: float = 0.15) -> np.ndarray:
    """True where the slide carries tissue rather than empty glass.

    Glass transmits nearly all the light, so its mean OD is close to zero.
    """
    return rgb_to_od(img).mean(axis=2) > od_threshold


# ---------------------------------------------------------------- Macenko estimation
def estimate_stain_matrix(img: np.ndarray, od_threshold: float = 0.15,
                          percentile: float = 1.0) -> np.ndarray:
    """Estimate the two stain vectors from the image itself (Macenko et al., 2009).

    Tissue pixels in OD space lie close to the plane spanned by the two stain vectors.
    Projecting them onto the first two singular vectors of that plane and taking the
    extreme angles gives the two stains, without assuming reference values.

    Rows come back ordered haematoxylin first, eosin second, by their blue content.
    """
    od = rgb_to_od(img).reshape(-1, 3)
    od = od[od.mean(axis=1) > od_threshold]
    if od.shape[0] < 50:
        raise ValueError("too few tissue pixels to estimate a stain matrix")
    # plane of greatest variation
    _, _, vt = np.linalg.svd(od - od.mean(axis=0, keepdims=True), full_matrices=False)
    plane = vt[:2]
    proj = od @ plane.T
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(angles, [percentile, 100 - percentile])
    vecs = np.array([np.cos(lo) * plane[0] + np.sin(lo) * plane[1],
                     np.cos(hi) * plane[0] + np.sin(hi) * plane[1]])
    vecs = np.abs(vecs)  # OD is non-negative; the SVD sign is arbitrary
    vecs = normalise_rows(vecs)
    # haematoxylin is the bluer of the two
    if vecs[0, 2] < vecs[1, 2]:
        vecs = vecs[::-1]
    return vecs


def stain_strength(img: np.ndarray, matrix: np.ndarray | None = None,
                   mask: np.ndarray | None = None) -> dict:
    """Mean concentration of each stain over the tissue, a one-number summary of staining."""
    conc = separate(img, matrix)
    mask = tissue_mask(img) if mask is None else mask
    if mask.sum() == 0:
        return {name: 0.0 for name in config.HE_STAINS} | {"tissue_fraction": 0.0}
    out = {name: float(conc[..., i][mask].mean()) for i, name in enumerate(config.HE_STAINS)}
    out["tissue_fraction"] = float(mask.mean())
    out["he_ratio"] = float(out["haematoxylin"] / max(out["eosin"], EPS))
    return out


def percentile_concentrations(img: np.ndarray, percentile: float = 99.0,
                             matrix: np.ndarray | None = None) -> np.ndarray:
    """The high-percentile concentration of each stain over the tissue.

    Macenko normalisation needs a scale for "how much dye is a lot" in this image; the
    99th percentile is robust to the few very dark pixels at nuclear centres.
    """
    matrix = estimate_stain_matrix(img) if matrix is None else matrix
    conc = separate(img, matrix)
    mask = tissue_mask(img)
    flat = conc[mask] if mask.sum() > 50 else conc.reshape(-1, conc.shape[-1])
    top = np.percentile(flat, percentile, axis=0)
    top[top < EPS] = 1.0
    return top


def normalise_staining(img: np.ndarray, target_matrix: np.ndarray | None = None,
                       target_concentrations: tuple[float, float] | None = None,
                       percentile: float = 99.0) -> np.ndarray:
    """Re-stain an image to a common appearance (Macenko normalisation).

    The image's own stain vectors are estimated, its concentrations rescaled so that their
    high percentile matches the target, and the result rebuilt with the target vectors.
    """
    target_matrix = stain_matrix() if target_matrix is None else normalise_rows(target_matrix)
    source_matrix = estimate_stain_matrix(img)
    conc = separate(img, source_matrix)
    top = percentile_concentrations(img, percentile, source_matrix)
    # Without a target the image is only re-expressed in the reference stain vectors,
    # which leaves its own strength alone; a target rescales it to that strength.
    target = np.array(target_concentrations) if target_concentrations is not None else top
    scaled = conc * (target / top)
    return recombine(scaled, target_matrix)
