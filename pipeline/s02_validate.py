"""Step 2 - check the two things the rest of the project rests on.

1. Does the deconvolution actually recover the dye concentrations? Tested three ways:
   exactly, on mixtures built from known concentrations; against scikit-image's own
   implementation on a real stained section; and by estimating the stain vectors from the
   image alone (Macenko) and comparing them with the vectors used to build it.
2. Does the nuclei detector find the right nuclei? Tested on synthetic fields whose
   centres are known. Parameters were chosen on development fields and are reported here
   on different fields, so the numbers are not the ones that were tuned on.
"""
from __future__ import annotations

import json

import numpy as np
from skimage import color, data

import config
from pipeline import nuclei, stains, synthetic

DEV_SEEDS = [config.SEED + 100 + 7 * i for i in range(3)]      # used while choosing parameters
TEST_SEEDS = [config.SEED + 900 + 13 * i for i in range(config.SYNTH["n_fields"])]


def bootstrap_ci(values, n=config.BOOTSTRAP, seed=config.SEED, alpha=0.05):
    """Percentile bootstrap confidence interval for a mean."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def angle_between(u, v) -> float:
    """Angle in degrees between two stain vectors."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def check_deconvolution() -> dict:
    """Exactness on known mixtures, agreement with scikit-image, stain-vector recovery."""
    rng = np.random.default_rng(config.SEED)
    M = stains.stain_matrix()

    # (a) build an image from known concentrations and read them back
    true_conc = rng.uniform(0.0, 1.0, size=(64, 64, 2))
    rgb = stains.recombine(true_conc, M)
    got = stains.separate(rgb, M)
    exact_err = float(np.abs(got - true_conc).max())

    # (b) the same image through scikit-image's haematoxylin-eosin-DAB routine
    ihc = data.immunohistochemistry()
    mine = stains.separate(ihc, stains.stain_matrix(("haematoxylin", "eosin", "dab")))
    theirs = color.rgb2hed(ihc)
    corr = [float(np.corrcoef(mine[..., i].ravel(), theirs[..., i].ravel())[0, 1]) for i in range(3)]

    # (c) estimate the stain vectors from a synthetic image and compare with the truth
    field = synthetic.make_field(config.SEED + 5)
    est = stains.estimate_stain_matrix(field["image"])
    angles = [angle_between(est[i], M[i]) for i in range(2)]

    return {
        "max_abs_error_known_mixture": exact_err,
        "pearson_vs_skimage": {"haematoxylin": corr[0], "eosin": corr[1], "dab": corr[2]},
        "stain_vector_angle_deg": {"haematoxylin": angles[0], "eosin": angles[1]},
    }


def score_fields(seeds, strength=1.0, threshold=None, normalise=False,
                 norm_target=None) -> list[dict]:
    """Run the detector over synthetic fields and score each against its ground truth."""
    out = []
    for s in seeds:
        f = synthetic.make_field(s, strength=strength)
        img = (stains.normalise_staining(f["image"], target_concentrations=norm_target)
               if normalise else f["image"])
        img = (np.clip(img * 255, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img)
        seg = nuclei.segment(img, threshold=threshold)
        sc = nuclei.match_to_truth(seg["objects"], f["centres"])
        sc["seed"] = int(s)
        sc["threshold_used"] = seg["threshold"]
        out.append(sc)
    return out


def summarise(scores, keys=("f1", "precision", "recall", "count_error_pct")) -> dict:
    out = {}
    for k in keys:
        v = [s[k] for s in scores if s[k] is not None]
        lo, hi = bootstrap_ci(v)
        out[k] = {"mean": float(np.mean(v)), "ci_low": lo, "ci_high": hi}
    return out


def main() -> dict:
    deconv = check_deconvolution()

    dev = summarise(score_fields(DEV_SEEDS))
    test_scores = score_fields(TEST_SEEDS)
    test = summarise(test_scores)

    # the threshold a fixed protocol would bake in, taken from the reference staining only
    fixed_threshold = float(np.mean([s["threshold_used"] for s in score_fields(DEV_SEEDS)]))
    config.FIXED_THRESHOLD_FILE.write_text(json.dumps({"fixed_threshold": fixed_threshold}))

    # negative control: an empty slide must give no nuclei at all
    blank = [nuclei.segment(synthetic.blank_field(config.SEED + i))["count"] for i in range(3)]

    result = {
        "deconvolution": deconv,
        "development_fields": dev,
        "test_fields": test,
        "n_development_fields": len(DEV_SEEDS),
        "n_test_fields": len(TEST_SEEDS),
        "fixed_threshold": fixed_threshold,
        "blank_slide_counts": blank,
        "per_test_field": test_scores,
    }
    (config.DERIVED / "validation.json").write_text(json.dumps(result, indent=2))

    print(f"s02: deconvolution exact to {deconv['max_abs_error_known_mixture']:.2e} on known mixtures")
    print(f"     scikit-image agreement r = "
          + ", ".join(f"{k} {v:.4f}" for k, v in deconv["pearson_vs_skimage"].items()))
    print(f"     stain vectors recovered to "
          + ", ".join(f"{k} {v:.2f} deg" for k, v in deconv["stain_vector_angle_deg"].items()))
    print(f"     held-out fields: F1 {test['f1']['mean']:.3f} "
          f"({test['f1']['ci_low']:.3f}-{test['f1']['ci_high']:.3f}), "
          f"precision {test['precision']['mean']:.3f}, recall {test['recall']['mean']:.3f}")
    print(f"     blank slides: {blank} nuclei (negative control)")
    return result


if __name__ == "__main__":
    main()
