# IAPS_fMRI_Multiple_Regression

## Purpose

Voxelwise fMRI multiple-regression analysis relating brain activity (beta estimates) to
normative valence and arousal ratings for IAPS (International Affective Picture System)
images. For 20 subjects and 60 stimulus-averaged beta patterns per subject (153,594 voxels),
the model fit independently at each subject/voxel is:

```
fMRI beta = intercept + z(valence) + z(arousal)
```

Group maps (mean beta, t-statistic, FDR-corrected significance) summarize the 20 subject-level
slopes for each predictor. The results are then rendered as figures (axial slices, PyCortex
cortical-surface flatmaps, MRIcroGL renders) and cross-referenced, ROI by ROI, against a
separate representational similarity analysis (RSA) pipeline (`IAPS_Searchlight`, a sibling
project not included here) using several ROI atlases (Harvard-Oxford, AAL3, and
Kastner/Julian visual-topography atlases) to quantify where the univariate regression effects
and the RSA effects overlap.

60 stimulus-averaged betas were used rather than all 300 single-trial betas because valence and
arousal are properties of the stimulus: treating repeated presentations as independent rows
would inflate apparent degrees of freedom in a simple OLS model (see `analysis_notes.md` for
the full rationale and design diagnostics — predictor correlation and VIF).

This repository is the analysis/figure code only. It contains **no raw or processed data,
brain maps, or figures** — see How to Use and Dependencies below for what each script expects
you to supply.

## Contents

All files sit at the project root (no subdirectories were needed for code):

- **Regression core**
  - `run_voxelwise_va_regression.py` — the main analysis: fits the voxelwise valence/arousal
    regression and writes group maps, whole-brain figures, and a JSON summary.
  - `rebuild_beh60_all_significant_fdr.py` — rebuilds the FDR-corrected behavioral/RSA
    significance map (in the sibling `IAPS_Searchlight` project's output folder) that several
    later scripts use as a reference/overlay.
- **Conjunction map + PyCortex flatmap**
  - `plot_pycortex_both_positive_tstat.py` — combines the valence and arousal group t-maps
    into a single "both significant, signed minimum-t" conjunction NIfTI (the key intermediate
    file most downstream figure/ROI scripts consume) and renders it as a PyCortex flatmap.
- **Axial-slice figures**
  - `plot_univariate_multivariate_axial_slices.py`, `plot_univariate_multivariate_roi_overlap_axial.py`,
    `plot_regression_conjunction_with_rsa.py` — matplotlib slice-mosaic comparisons of the
    regression conjunction map against the RSA map.
- **Cortical surface panels (PyCortex)**
  - `plot_pycortex_valence_arousal_rsa.py`, `plot_pycortex_harvard_oxford.py`,
    `generate_surface_panel_a.py`, `generate_surface_panel_a_bright_masks.py` (a
    brighter-mask rendering variant of panel A, kept alongside it — it produces a
    differently-named output file, not a duplicate), `generate_harvard_oxford_panel_b.py`,
    `make_publication_roi_figure.py`, `create_surface_plot_all_notebook.py` (generates a
    Jupyter notebook, `surface_plot_all.ipynb`, rather than running the analysis directly).
    See `pycortex_surface_plotting_notes.md` for the PyCortex subject/transform/overlay setup
    this requires.
- **MRIcroGL renders**
  - `mricrogl_render_axial_slices.py`, `mricrogl_univariate_multivariate_mosaic.py` — must be
    run from inside the MRIcroGL application's own scripting console (they `import gl`,
    MRIcroGL's built-in scripting module, not a pip package).
  - `compose_mricrogl_axial_figure.py`, `make_mricrogl_roi_coverage_figure.py` — compose the
    per-slice PNGs that MRIcroGL renders into labeled, publication-style figures.
- **ROI quantification (RSA vs. regression overlap)**
  - `quantify_rsa_vs_regression_rois.py`, `quantify_rsa_vs_regression_new_lr_rois.py`,
    `quantify_rsa_vs_regression_lr_new_julian_aal3.py`,
    `quantify_rsa_vs_regression_all_kastner_julian_rois.py`,
    `roi_method_coverage_checkmarks.py` — compare the regression conjunction map against the
    RSA map across different ROI atlas definitions and write CSV/JSON summary tables.
- **Documentation**
  - `analysis_notes.md` — model rationale, design diagnostics, and visualization
    recommendations.
  - `pycortex_surface_plotting_notes.md` — PyCortex surface-plotting setup notes.

## How to Use

None of these scripts include or download data. The underlying fMRI derivatives (subject beta
volumes, brain mask, trial/design tables, MNI template, RSA searchlight results) are not part
of this repository and must be supplied separately, in the directory layout each script
expects. Every script hardcodes its input/output paths near the top as constants (e.g.
`ROOT`, `SEARCHLIGHT`, `OUT_DIR`) — edit those to point at your own copies before running.
All scripts are plain Python, run as `python <script_name>.py` (a few also take
`argparse` options — pass `--help` to check), unless noted otherwise below.

Recommended run order:

1. **`run_voxelwise_va_regression.py`** (start here)
   - Input: `allsub_avg.npy` (20 x 60 x 153594 beta matrix), `mask.npy`, a trial-ordering CSV
     and an IAPS stimulus-order CSV, and a reference NIfTI header — all from the sibling
     `IAPS_Searchlight` project.
   - Does: z-scores valence/arousal, fits the per-subject/per-voxel regression, computes group
     mean/t/FDR maps, and (if present) overlays the behavioral/RSA map from step 2 below.
   - Output: `maps/*.npy` (group mean/t/FDR-masked beta volumes), whole-brain `figures/*.png`,
     `design_matrix_60stimuli.csv`, and `regression_summary.json`.
2. **`rebuild_beh60_all_significant_fdr.py`** (optional, independent of step 1)
   - Input: a raw searchlight evaluation-score CSV and voxel-center-ID CSV from
     `IAPS_Searchlight`.
   - Does: recomputes FDR-corrected significance and a >50-voxel cluster threshold for the
     behavioral/RSA t-map.
   - Output: FDR-corrected/cluster-thresholded RSA NIfTI files, written back into
     `IAPS_Searchlight`'s own output folder. Run this before step 1 (or before step 3) if you
     want the behavioral-map overlays; both later steps skip that overlay gracefully if the
     file isn't found.
3. **`plot_pycortex_both_positive_tstat.py`**
   - Input: `maps/group_t_beta_valence_fdr05_cluster50.npy` and
     `.../group_t_beta_arousal_fdr05_cluster50.npy` from step 1, plus the RSA cluster NIfTI
     from step 2 (for its affine/header).
   - Does: builds the "both valence and arousal positive and significant" conjunction map
     (signed minimum t-statistic), then projects it onto a PyCortex `fsaverage` flatmap.
   - Output: `outputs/valence_arousal_both_positive_significant_min_t.nii.gz` (consumed by
     nearly everything below) plus flatmap PNGs. Requires PyCortex and a FreeSurfer
     `fsaverage` subject (see `pycortex_surface_plotting_notes.md`).
4. **Axial-slice and surface-panel figures** (each independently reads the conjunction NIfTI
   from step 3 and the RSA map from step 2/`IAPS_Searchlight`):
   `plot_univariate_multivariate_axial_slices.py`,
   `plot_univariate_multivariate_roi_overlap_axial.py`,
   `plot_regression_conjunction_with_rsa.py`, `plot_pycortex_valence_arousal_rsa.py`,
   `plot_pycortex_harvard_oxford.py`, `generate_surface_panel_a.py` (or the
   `generate_surface_panel_a_bright_masks.py` variant), `generate_harvard_oxford_panel_b.py`,
   `make_publication_roi_figure.py`, `create_surface_plot_all_notebook.py` (writes a `.ipynb`
   you then run in Jupyter). The PyCortex-based ones need the same surface setup as step 3.
5. **MRIcroGL renders** (optional, alternate rendering backend): open MRIcroGL, run
   `mricrogl_render_axial_slices.py` and/or `mricrogl_univariate_multivariate_mosaic.py` from
   its scripting console (they import MRIcroGL's built-in `gl` module and will not run under a
   normal Python interpreter), then run `compose_mricrogl_axial_figure.py` and
   `make_mricrogl_roi_coverage_figure.py` to compose the rendered slice PNGs into figures.
6. **ROI quantification** (run any time after steps 2 and 3 have produced the conjunction and
   RSA NIfTIs): `quantify_rsa_vs_regression_rois.py`,
   `quantify_rsa_vs_regression_new_lr_rois.py`,
   `quantify_rsa_vs_regression_lr_new_julian_aal3.py`,
   `quantify_rsa_vs_regression_all_kastner_julian_rois.py`, and
   `roi_method_coverage_checkmarks.py` — each resamples the conjunction/RSA maps into a
   different ROI atlas (Kastner, Julian, AAL3, or Harvard-Oxford) and writes overlap CSV/JSON
   tables under `outputs/roi_quantification/`.

## Dependencies

Inferred from imports (no `requirements.txt` was present in the source project):

```
numpy
scipy
pandas
matplotlib
nibabel
nilearn
Pillow
nbformat
```

Install with:

```
pip install numpy scipy pandas matplotlib nibabel nilearn Pillow nbformat
```

Additional, script-specific dependencies:

- **PyCortex** (`pycortex`, imported as `cortex`) — required by all the PyCortex flatmap
  scripts (`plot_pycortex_*.py`, `generate_surface_panel_a*.py`,
  `generate_harvard_oxford_panel_b.py`, `make_publication_roi_figure.py`). Needs a FreeSurfer
  `fsaverage` subject and a volume-to-surface transform; see
  `pycortex_surface_plotting_notes.md`.
- **MRIcroGL** (a desktop application, not a pip package) — required to run
  `mricrogl_render_axial_slices.py` and `mricrogl_univariate_multivariate_mosaic.py`, which
  import its built-in `gl` scripting module.
- Standard library only otherwise (`json`, `math`, `gzip`, `struct`, `shutil`, `pathlib`,
  `csv`, `re`, `pickle`, `argparse`).

No data, atlases, or generated outputs (`.npy`/`.nii.gz`/`.img`/figures/CSV results) are
included in this repository. Each script's constants at the top of the file document the
directory layout it expects — supply the corresponding fMRI derivatives, atlas files, and
sibling-project (`IAPS_Searchlight`) outputs yourself.
