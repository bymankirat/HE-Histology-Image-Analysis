"""Checks on the colour deconvolution itself."""
import numpy as np
from skimage import color, data

import config
from pipeline import stains, synthetic


def test_optical_density_round_trip():
    rgb = np.array([[[255, 255, 255], [128, 64, 32], [1, 1, 1]]], dtype=np.uint8)
    back = stains.od_to_rgb(stains.rgb_to_od(rgb))
    assert np.allclose(back, stains.as_float(rgb), atol=1e-6)


def test_white_pixel_has_zero_density():
    od = stains.rgb_to_od(np.full((2, 2, 3), 255, dtype=np.uint8))
    assert np.allclose(od, 0.0, atol=1e-6)


def test_stain_matrix_rows_are_unit_vectors():
    M = stains.stain_matrix()
    assert M.shape == (2, 3)
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0)


def test_known_concentrations_are_recovered_exactly():
    """An image built from known amounts of dye must give those amounts back."""
    rng = np.random.default_rng(0)
    true = rng.uniform(0.0, 1.2, size=(32, 32, 2))
    img = stains.recombine(true)
    got = stains.separate(img)
    assert np.abs(got - true).max() < 1e-10


def test_negative_concentrations_are_clipped():
    """A colour no mix of the two stains can produce must not give a negative amount."""
    green = np.zeros((4, 4, 3), dtype=np.uint8)
    green[..., 1] = 255
    assert (stains.separate(green) >= 0).all()


def test_agrees_with_skimage_on_a_real_section():
    img = data.immunohistochemistry()
    mine = stains.separate(img, stains.stain_matrix(("haematoxylin", "eosin", "dab")))
    theirs = color.rgb2hed(img)
    for i in range(3):
        r = np.corrcoef(mine[..., i].ravel(), theirs[..., i].ravel())[0, 1]
        assert r > 0.99, f"channel {i} correlation {r}"


def test_macenko_recovers_the_vectors_it_was_built_with():
    field = synthetic.make_field(config.SEED + 5)
    est = stains.estimate_stain_matrix(field["image"])
    ref = stains.stain_matrix()
    for i in range(2):
        angle = np.degrees(np.arccos(np.clip(est[i] @ ref[i], -1, 1)))
        assert angle < 5.0, f"stain {i} off by {angle:.2f} degrees"


def test_macenko_refuses_an_empty_slide():
    try:
        stains.estimate_stain_matrix(synthetic.blank_field(1))
    except ValueError:
        return
    raise AssertionError("estimating stain vectors from blank glass should fail")


def test_tissue_mask_is_empty_on_blank_glass():
    assert stains.tissue_mask(synthetic.blank_field(2)).sum() == 0


def test_stain_strength_rises_with_staining():
    weak = synthetic.make_field(7, strength=0.5)["image"]
    strong = synthetic.make_field(7, strength=1.5)["image"]
    whole = np.ones(weak.shape[:2], dtype=bool)
    a = stains.stain_strength(weak, mask=whole)
    b = stains.stain_strength(strong, mask=whole)
    assert b["haematoxylin"] > a["haematoxylin"]
    assert b["eosin"] > a["eosin"]
