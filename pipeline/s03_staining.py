"""Step 3 - how much does the staining itself change what you measure?

The workshop protocol standardised haematoxylin to 35 seconds and eosin to a single dip.
That only matters if a different staining time would change the answer, so here the same
fields are re-stained computationally from 0.4x to 2.0x the reference concentration and
counted three ways:

  adaptive   - the threshold is recomputed for every image (Otsu)
  fixed      - one threshold, set once on reference-strength images, applied to all
  normalised - the image is re-stained to a common appearance first, then the fixed
               threshold is applied

Because the fields are synthetic, the true number of nuclei is known at every strength,
so each method's error can be measured rather than argued about.
"""
from __future__ import annotations

import json

import numpy as np

import config
from pipeline import s02_validate as val
from pipeline import stains, synthetic

SEEDS = val.TEST_SEEDS


def reference_target() -> tuple[float, float]:
    """What "normally stained" looks like: the stain levels of reference-strength fields.

    Normalisation needs something to normalise towards. Using the reference staining is the
    computational equivalent of a laboratory's own well-stained control slide.
    """
    tops = np.array([stains.percentile_concentrations(synthetic.make_field(s)["image"])
                     for s in SEEDS[:3]])
    return tuple(np.median(tops, axis=0))


def run() -> dict:
    fixed = json.loads(config.FIXED_THRESHOLD_FILE.read_text())["fixed_threshold"]
    target = reference_target()
    rows = []
    for strength in config.STAIN_STRENGTHS:
        methods = {
            "adaptive": dict(threshold=None, normalise=False),
            "fixed": dict(threshold=fixed, normalise=False),
            "normalised": dict(threshold=fixed, normalise=True, norm_target=target),
        }
        for name, kw in methods.items():
            scores = val.score_fields(SEEDS, strength=strength, **kw)
            rows.append({
                "strength": float(strength),
                "method": name,
                "f1": float(np.mean([s["f1"] for s in scores])),
                "precision": float(np.mean([s["precision"] for s in scores])),
                "recall": float(np.mean([s["recall"] for s in scores])),
                "count_error_pct": float(np.mean([s["count_error_pct"] for s in scores])),
                "abs_count_error_pct": float(np.mean([abs(s["count_error_pct"]) for s in scores])),
                "n_detected": float(np.mean([s["n_detected"] for s in scores])),
                "n_true": float(np.mean([s["n_true"] for s in scores])),
            })
    return {"fixed_threshold": fixed, "rows": rows,
            "normalisation_target": {"haematoxylin": float(target[0]), "eosin": float(target[1])}}


def measured_strength() -> list[dict]:
    """Does the measured stain concentration track the staining actually applied?"""
    out = []
    for strength in config.STAIN_STRENGTHS:
        vals = []
        for s in SEEDS[:3]:
            f = synthetic.make_field(s, strength=strength)
            whole = np.ones(f["image"].shape[:2], dtype=bool)
            vals.append(stains.stain_strength(f["image"], mask=whole))
        out.append({
            "strength": float(strength),
            "haematoxylin": float(np.mean([v["haematoxylin"] for v in vals])),
            "eosin": float(np.mean([v["eosin"] for v in vals])),
        })
    return out


def main() -> dict:
    res = run()
    res["measured_strength"] = measured_strength()

    by = {}
    for r in res["rows"]:
        by.setdefault(r["method"], []).append(r)
    res["summary"] = {
        m: {
            "worst_abs_count_error_pct": max(r["abs_count_error_pct"] for r in rs),
            "mean_abs_count_error_pct": float(np.mean([r["abs_count_error_pct"] for r in rs])),
            "worst_f1": min(r["f1"] for r in rs),
            "mean_f1": float(np.mean([r["f1"] for r in rs])),
        }
        for m, rs in by.items()
    }
    (config.DERIVED / "staining.json").write_text(json.dumps(res, indent=2))

    print(f"s03: staining swept {config.STAIN_STRENGTHS[0]}x to {config.STAIN_STRENGTHS[-1]}x"
          f" on {len(SEEDS)} fields")
    for m, s in res["summary"].items():
        print(f"     {m:<11} worst count error {s['worst_abs_count_error_pct']:6.1f}%"
              f"   worst F1 {s['worst_f1']:.3f}")
    return res


if __name__ == "__main__":
    main()
