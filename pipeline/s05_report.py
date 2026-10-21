"""Step 5 - figures and the interactive report.

Every number printed in the report is read from the JSON written by the earlier steps, so
the page cannot drift away from the run that produced it.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import segmentation as skseg

import config
from pipeline import nuclei, stains, synthetic
from pipeline import s04_real as s4

PLOT = {"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
NAVY, AMBER, RED, GREEN = "#1f3a6e", "#e0a100", "#c2362f", "#1b8a5a"


def _save(fig, name):
    path = config.FIGURES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path.name


def fig_tiles(real: dict) -> str:
    """The five sections with the stain levels measured from each."""
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6))
    for ax, row in zip(axes, real["raw"]):
        ax.imshow(s4.load_tile(row["tile"]))
        ax.set_title(f"{row['tile']}\nH {row['haematoxylin']:.2f}  E {row['eosin']:.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Five published H&E sections: haematoxylin is steady, eosin is not", y=1.04)
    return _save(fig, "tiles.png")


def fig_deconvolution() -> str:
    """One section split into its two dyes and put back together."""
    img = s4.load_tile("tile1", 2)
    conc = stains.separate(img)
    h_only = stains.stain_image(conc, 0)
    e_only = stains.stain_image(conc, 1)
    back = stains.recombine(conc)
    panels = [(img / 255.0, "section"), (h_only, "haematoxylin only"),
              (e_only, "eosin only"), (back, "recombined")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    return _save(fig, "deconvolution.png")


def fig_synthetic() -> str:
    """A synthetic field, its known nuclei and what the detector found."""
    f = synthetic.make_field(config.SEED + 900)
    seg = nuclei.segment(f["image"])
    score = nuclei.match_to_truth(seg["objects"], f["centres"])
    overlay = skseg.mark_boundaries(f["image"] / 255.0, seg["labels"], color=(1, 1, 0))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].imshow(f["image"])
    axes[0].set_title("synthetic field", fontsize=9)
    axes[1].imshow(f["image"])
    axes[1].scatter(f["centres"][:, 1], f["centres"][:, 0], s=6, c="white", edgecolors="k",
                    linewidths=0.3)
    axes[1].set_title(f"ground truth, {f['n_true']} nuclei", fontsize=9)
    axes[2].imshow(overlay)
    axes[2].set_title(f"detected {score['n_detected']}, F1 {score['f1']:.3f}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    return _save(fig, "synthetic.png")


def fig_sweep(staining: dict) -> str:
    """Counting error against staining strength, for each thresholding rule."""
    rows = staining["rows"]
    styles = {"adaptive": (NAVY, "o-", "adaptive threshold (per image)"),
              "fixed": (RED, "s--", "fixed threshold"),
              "normalised": (AMBER, "^:", "normalised, then fixed threshold")}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method, (colour, style, label) in styles.items():
        sub = [r for r in rows if r["method"] == method]
        ax.plot([r["strength"] for r in sub], [r["count_error_pct"] for r in sub],
                style, color=colour, label=label, markersize=4)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.axvline(1.0, color="#ccc", linewidth=0.8)
    ax.set_xlabel("staining strength (1.0 = the reference stain)")
    ax.set_ylabel("error in nuclei count (%)")
    ax.set_title("The same tissue, stained more or less strongly")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "sweep.png")


def fig_normalisation(real: dict) -> str:
    """The five sections before and after being re-stained to a common appearance."""
    target = (real["normalisation_target"]["haematoxylin"], real["normalisation_target"]["eosin"])
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
    for col, row in enumerate(real["raw"]):
        raw = s4.load_tile(row["tile"])
        norm = s4.as_uint8(stains.normalise_staining(raw, target_concentrations=target))
        axes[0, col].imshow(raw)
        axes[0, col].set_title(f"{row['tile']}  E {row['eosin']:.3f}", fontsize=8)
        axes[1, col].imshow(norm)
        axes[1, col].set_title(f"E {real['normalised'][col]['eosin']:.3f}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].set_ylabel("as published")
    fig.suptitle("Top: as published.  Bottom: normalised to the median of the set", y=1.02)
    return _save(fig, "normalisation.png")


def main() -> dict:
    plt.rcParams.update(PLOT)
    val = json.loads((config.DERIVED / "validation.json").read_text())
    staining = json.loads((config.DERIVED / "staining.json").read_text())
    real = json.loads((config.DERIVED / "real_tiles.json").read_text())
    manifest = json.loads((config.RAW / "manifest.json").read_text())

    figures = {
        "tiles": fig_tiles(real),
        "deconvolution": fig_deconvolution(),
        "synthetic": fig_synthetic(),
        "sweep": fig_sweep(staining),
        "normalisation": fig_normalisation(real),
    }

    fixed_rows = {r["strength"]: r for r in staining["rows"] if r["method"] == "fixed"}
    adaptive_rows = {r["strength"]: r for r in staining["rows"] if r["method"] == "adaptive"}
    worst_strength = min(fixed_rows, key=lambda s: fixed_rows[s]["count_error_pct"])

    results = {
        "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
        "seed": config.SEED,
        "figures": figures,
        "manifest": manifest,
        "validation": val,
        "staining": staining,
        "real": real,
        "headline": {
            "deconvolution_error": val["deconvolution"]["max_abs_error_known_mixture"],
            "skimage_r_haematoxylin": val["deconvolution"]["pearson_vs_skimage"]["haematoxylin"],
            "skimage_r_eosin": val["deconvolution"]["pearson_vs_skimage"]["eosin"],
            "stain_angle_h": val["deconvolution"]["stain_vector_angle_deg"]["haematoxylin"],
            "stain_angle_e": val["deconvolution"]["stain_vector_angle_deg"]["eosin"],
            "f1": val["test_fields"]["f1"]["mean"],
            "f1_lo": val["test_fields"]["f1"]["ci_low"],
            "f1_hi": val["test_fields"]["f1"]["ci_high"],
            "precision": val["test_fields"]["precision"]["mean"],
            "recall": val["test_fields"]["recall"]["mean"],
            "blank_counts": val["blank_slide_counts"],
            "adaptive_worst": staining["summary"]["adaptive"]["worst_abs_count_error_pct"],
            "fixed_worst": staining["summary"]["fixed"]["worst_abs_count_error_pct"],
            "fixed_worst_strength": worst_strength,
            "fixed_worst_f1": fixed_rows[worst_strength]["f1"],
            "adaptive_at_worst": abs(adaptive_rows[worst_strength]["count_error_pct"]),
            "eosin_fold": real["spread_raw"]["eosin_fold_range"],
            "eosin_cv": real["spread_raw"]["eosin_cv_pct"],
            "haem_cv": real["spread_raw"]["haematoxylin_cv_pct"],
            "eosin_cv_norm": real["spread_normalised"]["eosin_cv_pct"],
            "eosin_fold_norm": real["spread_normalised"]["eosin_fold_range"],
        },
    }
    docs = config.ROOT / "docs"
    (docs / "results.json").write_text(json.dumps(results, indent=2))
    # a .js copy as well, so the report opens from a file:// path without a web server
    (docs / "results.js").write_text("window.HE = " + json.dumps(results, indent=2) + ";\n")
    print("s05: wrote docs/results.json, docs/results.js and " + ", ".join(figures.values()))
    return results


if __name__ == "__main__":
    main()
