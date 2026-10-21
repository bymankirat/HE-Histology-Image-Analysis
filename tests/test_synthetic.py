"""Checks on the forward model, so the ground truth can be trusted."""
import numpy as np

import config
from pipeline import stains, synthetic


def test_same_seed_gives_the_same_field():
    a = synthetic.make_field(42)
    b = synthetic.make_field(42)
    assert np.array_equal(a["image"], b["image"])
    assert np.array_equal(a["centres"], b["centres"])


def test_different_seeds_give_different_fields():
    a = synthetic.make_field(42)
    b = synthetic.make_field(43)
    assert not np.array_equal(a["image"], b["image"])


def test_nuclei_are_placed_without_overlapping_centres():
    f = synthetic.make_field(7)
    c = f["centres"]
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= config.SYNTH["min_centre_gap"] - 1e-9


def test_the_field_is_full_of_nuclei():
    f = synthetic.make_field(7)
    assert f["n_true"] > 0.8 * config.SYNTH["n_nuclei"]
    assert len(f["radii"]) == f["n_true"]


def test_staining_strength_scales_the_concentrations():
    """Doubling the stain must double the optical density it was built from."""
    a = synthetic.make_field(5, strength=1.0)
    b = synthetic.make_field(5, strength=2.0)
    ratio = b["true_conc"].sum() / max(a["true_conc"].sum(), 1e-9)
    assert abs(ratio - 2.0) < 1e-9


def test_stronger_staining_makes_a_darker_image():
    a = synthetic.make_field(5, strength=0.5)["image"].mean()
    b = synthetic.make_field(5, strength=2.0)["image"].mean()
    assert b < a


def test_the_image_is_an_eight_bit_rgb_picture():
    im = synthetic.make_field(1)["image"]
    assert im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 3


def test_a_blank_field_holds_no_dye():
    whole = np.ones((config.SYNTH["size"],) * 2, dtype=bool)
    s = stains.stain_strength(synthetic.blank_field(3), mask=whole)
    assert s["haematoxylin"] < 0.05 and s["eosin"] < 0.05
