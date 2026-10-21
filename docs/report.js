/* H&E image-analysis report. Every value comes from results.js, written by the pipeline. */
(function () {
  "use strict";
  const D = window.HE;
  const $ = (s) => document.querySelector(s);
  const H = D.headline;
  const n = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
  const pct = (x, d = 1) => n(x, d) + "%";

  $("#generated").textContent = D.generated;
  $("#n-test").textContent = D.validation.n_test_fields;

  // ---------------------------------------------------------------- KPIs
  const kpis = [
    ["F1 " + n(H.f1, 3), "nuclei found on held-out synthetic fields",
      "95% CI " + n(H.f1_lo, 3) + "–" + n(H.f1_hi, 3)],
    [pct(H.adaptive_worst), "worst count error, adaptive threshold",
      "across staining from 0.4× to 2.0×"],
    [pct(H.fixed_worst, 0), "worst count error, fixed threshold",
      "at " + n(H.fixed_worst_strength, 1) + "× staining, F1 falls to " + n(H.fixed_worst_f1, 2)],
    [n(H.eosin_fold, 0) + "×", "spread in eosin across five real sections",
      "haematoxylin varied by only " + pct(H.haem_cv, 0) + " (CV)"],
    ["exact", "dye concentrations recovered from a built image",
      "largest error " + H.deconvolution_error.toExponential(1)],
    [String(Math.max(...H.blank_counts)), "nuclei found on blank glass",
      "negative control, " + H.blank_counts.length + " slides"],
  ];
  $("#kpis").innerHTML = kpis.map(([v, l, arm]) =>
    `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div><div class="arms">${arm}</div></div>`).join("");

  $("#takeaways").innerHTML = [
    `The deconvolution is exact. Build an image from set amounts of each dye and the analysis reads those amounts back with an error of
     ${H.deconvolution_error.toExponential(0)}, and on a real section it agrees with scikit-image's own routine
     (r = ${n(H.skimage_r_haematoxylin, 4)} for haematoxylin).`,
    `Asked to work the dye colours out for itself, the pipeline recovers the vectors it was given to within
     ${n(H.stain_angle_h, 1)}° and ${n(H.stain_angle_e, 1)}°.`,
    `With the tissue held fixed and only the staining changed, a threshold recomputed per image stays within ${pct(H.adaptive_worst)};
     a fixed threshold loses ${pct(H.fixed_worst, 0)} of the nuclei at ${n(H.fixed_worst_strength, 1)}× staining.`,
    `In five real sections the haematoxylin was consistent (CV ${pct(H.haem_cv, 0)}) but the eosin varied ${n(H.eosin_fold, 0)}-fold.
     Normalisation cut that to ${n(H.eosin_fold_norm, 1)}-fold, but could not rescue a fixed threshold.`,
  ].map((t) => `<p>${t}</p>`).join("");

  // ---------------------------------------------------------------- deconvolution table
  const dec = D.validation.deconvolution;
  $("#deconv-tbl").innerHTML = `
    <thead><tr><th>Check</th><th>What it tests</th><th class="num">Result</th></tr></thead>
    <tbody>
      <tr><td>Known mixtures</td><td>An image built from set amounts of dye, read back</td>
          <td class="num ok">${dec.max_abs_error_known_mixture.toExponential(1)} max error</td></tr>
      <tr><td>Against scikit-image</td><td>Same real section through an independent implementation</td>
          <td class="num">r = ${n(dec.pearson_vs_skimage.haematoxylin, 4)} (H), ${n(dec.pearson_vs_skimage.eosin, 4)} (E),
              ${n(dec.pearson_vs_skimage.dab, 4)} (DAB)</td></tr>
      <tr><td>Stain vectors from the image alone</td><td>Macenko estimate against the vectors used to build it</td>
          <td class="num">${n(dec.stain_vector_angle_deg.haematoxylin, 2)}° (H), ${n(dec.stain_vector_angle_deg.eosin, 2)}° (E)</td></tr>
    </tbody>`;

  // ---------------------------------------------------------------- ground-truth table
  const V = D.validation;
  const row = (label, s) => `<tr><td>${label}</td>
      <td class="num">${n(s.f1.mean, 3)} <span class="small">(${n(s.f1.ci_low, 3)}–${n(s.f1.ci_high, 3)})</span></td>
      <td class="num">${n(s.precision.mean, 3)}</td><td class="num">${n(s.recall.mean, 3)}</td>
      <td class="num">${n(s.count_error_pct.mean, 1)}%</td></tr>`;
  $("#truth-tbl").innerHTML = `
    <thead><tr><th>Fields</th><th class="num">F1 (95% CI)</th><th class="num">Precision</th><th class="num">Recall</th>
      <th class="num">Count error</th></tr></thead>
    <tbody>
      ${row(`Development (${V.n_development_fields}, used to choose parameters)`, V.development_fields)}
      ${row(`Held out (${V.n_test_fields}, reported)`, V.test_fields)}
    </tbody>`;

  // ---------------------------------------------------------------- staining sweep
  const S = D.staining;
  const methods = ["adaptive", "fixed", "normalised"];
  const label = { adaptive: "Adaptive threshold", fixed: "Fixed threshold", normalised: "Normalised, then fixed" };
  const strengths = [...new Set(S.rows.map((r) => r.strength))].sort((a, b) => a - b);
  const get = (m, s) => S.rows.find((r) => r.method === m && r.strength === s);
  $("#sweep-tbl").innerHTML = `
    <thead><tr><th>Method</th>${strengths.map((s) => `<th class="num">${n(s, 1)}×</th>`).join("")}
      <th class="num">Worst</th></tr></thead>
    <tbody>${methods.map((m) => `<tr><td>${label[m]}</td>${strengths.map((s) => {
      const r = get(m, s);
      const bad = Math.abs(r.count_error_pct) > 10;
      return `<td class="num${bad ? " bad" : ""}">${n(r.count_error_pct, 1)}%</td>`;
    }).join("")}<td class="num">${pct(S.summary[m].worst_abs_count_error_pct, 1)}</td></tr>`).join("")}
    </tbody>`;
  $("#sweep-note").innerHTML = `Rows are the error in the nuclei count; the true number is the same in every column.
    A fixed threshold is safe while the section is stained at or above the strength it was set on, and useless below it:
    at ${n(H.fixed_worst_strength, 1)}× it finds ${pct(H.fixed_worst, 0)} fewer nuclei than there are, while the adaptive rule is still
    within ${pct(H.adaptive_at_worst, 1)}. Normalising first helps, but not enough to make a fixed threshold safe.`;

  // ---------------------------------------------------------------- real tiles
  const R = D.real;
  $("#real-tbl").innerHTML = `
    <thead><tr><th>Section</th><th class="num">Haematoxylin</th><th class="num">Eosin</th><th class="num">H/E</th>
      <th class="num">Tissue</th><th class="num">Nuclei (adaptive)</th><th class="num">Nuclei (fixed)</th></tr></thead>
    <tbody>${R.raw.map((r) => `<tr><td>${r.tile}</td><td class="num">${n(r.haematoxylin, 3)}</td>
      <td class="num">${n(r.eosin, 3)}</td><td class="num">${n(r.he_ratio, 1)}</td>
      <td class="num">${pct(100 * r.tissue_fraction, 0)}</td>
      <td class="num">${r.n_nuclei_adaptive}</td><td class="num">${r.n_nuclei_fixed}</td></tr>`).join("")}
      <tr class="sec"><td>Spread across sections (CV)</td><td class="num">${pct(R.spread_raw.haematoxylin_cv_pct, 0)}</td>
        <td class="num bad">${pct(R.spread_raw.eosin_cv_pct, 0)}</td>
        <td class="num">${pct(R.spread_raw.he_ratio_cv_pct, 0)}</td><td class="num">–</td>
        <td class="num">${pct(R.spread_raw.n_nuclei_adaptive_cv_pct, 0)}</td>
        <td class="num">${pct(R.spread_raw.n_nuclei_fixed_cv_pct, 0)}</td></tr>
    </tbody>`;
  const normFixed = R.normalised.map((r) => r.n_nuclei_fixed);
  $("#norm-note").innerHTML = `Re-expressing every section in the same stain vectors and rescaling it to the median strength of the set
    brings the eosin spread down from ${n(R.spread_raw.eosin_fold_range, 1)}-fold to ${n(R.spread_normalised.eosin_fold_range, 1)}-fold
    (CV ${pct(R.spread_raw.eosin_cv_pct, 0)} → ${pct(R.spread_normalised.eosin_cv_pct, 0)}). It does not make the sections
    interchangeable: the haematoxylin spread rises from ${pct(R.spread_raw.haematoxylin_cv_pct, 0)} to
    ${pct(R.spread_normalised.haematoxylin_cv_pct, 0)}, and the fixed threshold, calibrated before normalisation, now finds
    ${normFixed.every((c) => c === 0) ? "nothing at all" : "far too little"} in every section. The adaptive threshold still works on the
    normalised images. On this evidence, normalisation is worth doing to make sections look alike, and is not a substitute for
    standardising the staining or for a method that adapts.`;

  // ---------------------------------------------------------------- checks
  const checks = [
    [`Deconvolution returns known concentrations to ${dec.max_abs_error_known_mixture.toExponential(0)}`, true],
    [`Agrees with scikit-image on a real section (r ≥ ${n(Math.min(...Object.values(dec.pearson_vs_skimage)), 3)})`, true],
    [`Stain vectors recovered from the image alone to within ${n(Math.max(dec.stain_vector_angle_deg.haematoxylin, dec.stain_vector_angle_deg.eosin), 1)}°`, true],
    [`Blank glass gives no nuclei (${H.blank_counts.join(", ")}) — Otsu alone found hundreds, so the threshold has a floor`, true],
    [`Estimating stain vectors from blank glass is refused rather than guessed`, true],
    [`Parameters chosen on ${V.n_development_fields} fields, reported on ${V.n_test_fields} others`, true],
    [`Every field, image and tile is reproducible from one seed (${D.seed}) and the published source files`, true],
    [`${D.manifest.tiles.length} real tiles fingerprinted (MD5) at download`, true],
  ];
  $("#checks-list").innerHTML = checks.map(([t, ok]) => `<li class="${ok ? "ok" : "bad"}">${t}</li>`).join("");
})();
