"""Checks on nuclei detection and on the scoring that judges it."""
import numpy as np

import config
from pipeline import nuclei, synthetic


def test_blank_slide_gives_no_nuclei():
    """The negative control. Otsu finds a threshold in pure noise, so a floor is needed."""
    for seed in (1, 2, 3):
        assert nuclei.segment(synthetic.blank_field(seed))["count"] == 0


def test_finds_most_nuclei_in_a_synthetic_field():
    f = synthetic.make_field(config.SEED + 900)
    seg = nuclei.segment(f["image"])
    score = nuclei.match_to_truth(seg["objects"], f["centres"])
    assert score["f1"] > 0.90
    assert score["precision"] > 0.95


def test_perfect_detections_score_one():
    centres = np.array([[10.0, 10.0], [50.0, 50.0], [80.0, 20.0]])
    detected = [{"row": r, "col": c} for r, c in centres]
    s = nuclei.match_to_truth(detected, centres)
    assert s["f1"] == 1.0 and s["fp"] == 0 and s["fn"] == 0


def test_a_detection_far_from_any_nucleus_is_a_false_positive():
    centres = np.array([[10.0, 10.0]])
    detected = [{"row": 10.0, "col": 10.0}, {"row": 300.0, "col": 300.0}]
    s = nuclei.match_to_truth(detected, centres)
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0


def test_each_true_nucleus_can_only_be_matched_once():
    """Two detections on one nucleus must not both count as hits."""
    centres = np.array([[20.0, 20.0]])
    detected = [{"row": 20.0, "col": 20.0}, {"row": 21.0, "col": 21.0}]
    s = nuclei.match_to_truth(detected, centres)
    assert s["tp"] == 1 and s["fp"] == 1


def test_empty_inputs_do_not_crash():
    s = nuclei.match_to_truth([], np.zeros((0, 2)))
    assert s["tp"] == 0 and s["f1"] == 0.0


def test_objects_outside_the_size_range_are_dropped():
    f = synthetic.make_field(config.SEED + 901)
    loose = nuclei.segment(f["image"], params={"min_area": 1})
    tight = nuclei.segment(f["image"], params={"min_area": 400})
    assert tight["count"] < loose["count"]
    for o in tight["objects"]:
        assert o["area"] >= 400


def test_adaptive_threshold_tracks_the_staining():
    """A more strongly stained copy of the same field needs a higher threshold."""
    weak = nuclei.segment(synthetic.make_field(11, strength=0.7)["image"])["threshold"]
    strong = nuclei.segment(synthetic.make_field(11, strength=1.5)["image"])["threshold"]
    assert strong > weak


def test_counts_hold_when_only_the_staining_changes():
    """The tissue is identical; only the dye concentration differs."""
    counts = [nuclei.segment(synthetic.make_field(13, strength=s)["image"])["count"]
              for s in (0.6, 1.0, 1.6)]
    assert max(counts) - min(counts) <= 0.05 * max(counts)
