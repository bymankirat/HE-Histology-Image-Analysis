"""Settings for the H&E histology image-analysis pipeline.

Everything that can be tuned lives here, so a run is defined by this file plus the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"
FIGURES = ROOT / "docs" / "figures"
for d in (RAW, DERIVED, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 20231101  # the month the DSKC workshop ran; fixed so every run is identical

# ------------------------------------------------------------------ real images
# Five H&E tiles published in the StainTools repository (MIT licence, Peter Byfield),
# used here because they were stained to visibly different strengths.
TILE_SOURCE_REPO = "https://github.com/Peter554/StainTools"
TILE_SOURCE_FILE = "images/normalize_before.png"
TILE_SOURCE_RAW = "https://raw.githubusercontent.com/Peter554/StainTools/master/images/normalize_before.png"
TILE_NAMES = ["tile1", "tile2", "tile3", "tile4", "tile5"]

# ------------------------------------------------------------------ stain vectors
# Ruifrok & Johnston (2001) reference vectors, as unit vectors in optical-density space.
# Rows are stains, columns are R, G, B.
STAIN_VECTORS = {
    "haematoxylin": (0.6500286, 0.704031, 0.2860126),
    "eosin": (0.07, 0.99, 0.11),
    "dab": (0.26814753, 0.57031375, 0.77642715),
}
HE_STAINS = ("haematoxylin", "eosin")

BACKGROUND_INTENSITY = 255.0  # I0 in the Beer-Lambert law, for 8-bit images

# ------------------------------------------------------------------ synthetic images
SYNTH = {
    "size": 512,            # pixels per side
    "n_nuclei": 180,        # ground-truth nuclei per field
    "radius_range": (6, 13),  # pixel radius, drawn uniformly
    "min_centre_gap": 9,    # centres closer than this are rejected, so nuclei can touch but not merge
    "nucleus_od": 0.85,     # peak haematoxylin optical density inside a nucleus
    "cytoplasm_od": 0.35,   # background eosin optical density
    "illumination_tilt": 0.12,  # fractional brightness gradient across the field
    "read_noise": 2.5,      # Gaussian noise in 8-bit counts
    "n_fields": 6,          # independent fields per condition
}

# Staining strengths tested against the ground truth. 1.0 = the reference stain.
# The poster standardised haematoxylin to 35 s and eosin to one dip; these multipliers
# stand in for shorter and longer times.
STAIN_STRENGTHS = (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0)

# ------------------------------------------------------------------ nuclei segmentation
NUCLEI = {
    "smooth_sigma": 0.6,        # Gaussian blur on the haematoxylin channel before thresholding
    "min_area": 60,             # objects smaller than this are debris
    "max_area": 2500,           # objects larger than this are clumps the splitter must divide
    "peak_min_distance": 3,     # minimum distance between watershed seeds
    "fill_holes": True,
    "clear_border": False,      # border nuclei are kept; counts are per field, not per cell type
    # Otsu always returns a threshold, even on an empty slide, so it is floored: below this
    # haematoxylin concentration there is nothing to segment. The blank-slide control set it.
    "min_threshold": 0.08,
}

# Threshold fixed once at the reference staining strength, the computational equivalent of
# a protocol that always stains for the same time. Filled in by the validation step.
FIXED_THRESHOLD_FILE = DERIVED / "fixed_threshold.json"

MATCH_DISTANCE = 12  # a detection within this many pixels of a true centre counts as a hit
BOOTSTRAP = 2000     # resamples for confidence intervals
