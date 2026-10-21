"""A forward model of an H&E field, so that segmentation can be scored against truth.

Real sections come without a list of where the nuclei are, so accuracy cannot be measured
on them. Here the image is built from a known list of nuclei and a known pair of stain
vectors, pushed through the same physics the analysis assumes (Beer-Lambert absorption),
and then degraded the way a microscope degrades it: uneven illumination and read noise.
Anything the pipeline recovers can therefore be compared with what went in.
"""
from __future__ import annotations

import numpy as np

import config
from pipeline import stains


def _poisson_disc_centres(rng, n, size, min_gap, margin=4):
    """Random centres that are at least `min_gap` apart, so nuclei touch but do not merge."""
    centres = []
    tries = 0
    while len(centres) < n and tries < n * 400:
        tries += 1
        p = rng.uniform(margin, size - margin, size=2)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_gap ** 2 for q in centres):
            centres.append(p)
    return np.array(centres)


def _soft_disc(shape, centre, radius, softness=1.5):
    """A disc with a smooth edge: 1 inside, falling to 0 over `softness` pixels."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    d = np.sqrt((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2)
    return np.clip((radius - d) / softness + 0.5, 0.0, 1.0)


def _smooth_field(rng, size, scale=32.0):
    """A smooth random field in [0, 1], used for cytoplasm texture."""
    small = rng.normal(size=(max(4, int(size / scale)), max(4, int(size / scale))))
    yy = np.linspace(0, small.shape[0] - 1, size)
    xx = np.linspace(0, small.shape[1] - 1, size)
    y0 = np.clip(np.floor(yy).astype(int), 0, small.shape[0] - 2)
    x0 = np.clip(np.floor(xx).astype(int), 0, small.shape[1] - 2)
    fy, fx = (yy - y0)[:, None], (xx - x0)[None, :]
    f = (small[y0][:, x0] * (1 - fy) * (1 - fx) + small[y0 + 1][:, x0] * fy * (1 - fx)
         + small[y0][:, x0 + 1] * (1 - fy) * fx + small[y0 + 1][:, x0 + 1] * fy * fx)
    f = (f - f.min()) / max(float(f.max() - f.min()), 1e-9)
    return f


def make_field(seed: int, strength: float = 1.0, params: dict | None = None,
               matrix: np.ndarray | None = None) -> dict:
    """Build one synthetic H&E field.

    `strength` multiplies both stain concentrations, standing in for a longer or shorter
    staining step. Returns the image and the ground truth that produced it.
    """
    p = dict(config.SYNTH) | (params or {})
    rng = np.random.default_rng(seed)
    size = p["size"]
    matrix = stains.stain_matrix() if matrix is None else stains.normalise_rows(matrix)

    centres = _poisson_disc_centres(rng, p["n_nuclei"], size, p["min_centre_gap"])
    radii = rng.uniform(*p["radius_range"], size=len(centres))

    # haematoxylin: dark, round nuclei
    h = np.zeros((size, size))
    for c, r in zip(centres, radii):
        h = np.maximum(h, _soft_disc((size, size), c, r) * p["nucleus_od"]
                       * rng.uniform(0.8, 1.15))
    # eosin: cytoplasm everywhere except inside nuclei
    texture = 0.6 + 0.8 * _smooth_field(rng, size)
    e = p["cytoplasm_od"] * texture * (1.0 - 0.75 * (h > 0.15 * p["nucleus_od"]))

    conc = np.dstack([h, e]) * float(strength)
    rgb = stains.recombine(conc, matrix)

    # uneven illumination across the field, then read noise, then 8-bit quantisation
    yy, xx = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    tilt = 1.0 + p["illumination_tilt"] * (0.5 - 0.5 * (yy + xx))
    counts = np.clip(rgb * 255.0 * tilt[..., None], 0, 255)
    counts = counts + rng.normal(0.0, p["read_noise"], size=counts.shape)
    img = np.clip(counts, 0, 255).astype(np.uint8)

    return {
        "image": img,
        "centres": centres,          # (N, 2) as (row, col)
        "radii": radii,
        "n_true": int(len(centres)),
        "true_conc": conc,
        "matrix": matrix,
        "strength": float(strength),
        "seed": int(seed),
    }


def make_fields(n_fields: int, strength: float = 1.0, seed0: int = config.SEED, **kw):
    """A set of independent fields at one staining strength."""
    return [make_field(seed0 + 1000 * i, strength=strength, **kw) for i in range(n_fields)]


def blank_field(seed: int, size: int | None = None) -> np.ndarray:
    """Empty glass: no tissue at all. Used as a negative control for the detector."""
    p = dict(config.SYNTH)
    size = size or p["size"]
    rng = np.random.default_rng(seed)
    counts = 255.0 - np.abs(rng.normal(0.0, p["read_noise"], size=(size, size, 3)))
    return np.clip(counts, 0, 255).astype(np.uint8)
