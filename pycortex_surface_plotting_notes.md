# PyCortex Surface Plotting Notes

Goal: plot these signed, thresholded maps on a cortical flatmap with ROI borders/labels:

- `maps/group_t_beta_valence_fdr05_cluster50.img`
- `maps/group_t_beta_arousal_fdr05_cluster50.img`
- `N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight\outputs\tBrainmap\final_results\tmap_beh60_all_significant_fdr05_fisherz_cluster_gt50.nii.gz`

The helper script is:

```powershell
python plot_pycortex_valence_arousal_rsa.py --subject SUBJECT --xfm XFM
```

It writes separate PNGs to:

```text
N:\Experimental_Data\yujunchen\projects\IAPS_fMRI_multiple_reg\figures\pycortex
```

## What Is Still Needed

PyCortex cannot draw the paper-style ROI names by itself. The ROI borders and labels must already exist in a PyCortex overlay SVG for the target subject.

Needed assets:

1. A PyCortex subject imported from FreeSurfer.
2. A PyCortex transform from these group map volumes to the surface subject.
3. An overlay SVG containing ROI names/borders, for example:
   `V1`, `V2`, `V3`, `V3A`, `V3B`, `V4`, `LO`, `STS`, `pSTS`, `FFA`, `OFA`, `EBA`, `PPA`, `OPA`, `IPS`, `RSC`, `ITS`, `CoS`, `ATFP`.

The paper used subject-specific functional/retinotopic ROIs, so the exact ROI overlay is not a standard PyCortex atlas. If we do not have those localizers, the closest practical alternative is to use an atlas-derived fsaverage overlay, but that would be an approximation rather than the same ROI definition.

## Why `identity` Is Usually Not Enough

The maps here are volumetric group maps. PyCortex needs to know how each voxel maps to a cortical surface. That mapping is stored as a PyCortex transform. `identity` only works if the PyCortex subject/reference volume is already in the exact same space and grid as the NIfTI maps.

## Example Setup

After installing PyCortex and nibabel in the desired Python environment:

```powershell
pip install pycortex nibabel
```

Import a FreeSurfer subject:

```python
import cortex
cortex.freesurfer.import_subj(
    "SUBJECT",
    freesurfer_subject_dir=r"PATH_TO_FREESURFER_SUBJECTS_DIR",
)
```

Then create or import the correct transform between the map volume space and that subject. Once the transform and overlay exist, run:

```powershell
python plot_pycortex_valence_arousal_rsa.py --subject SUBJECT --xfm XFM --overlay-file PATH_TO_OVERLAY.svg
```
