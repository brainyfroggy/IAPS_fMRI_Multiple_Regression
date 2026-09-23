from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn import datasets


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SEARCHLIGHT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")
MASK_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/data/masks")

RSA_NIFTI = SEARCHLIGHT / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
MULTIPLE_REGRESSION_NIFTI = ROOT / "outputs/valence_arousal_both_positive_significant_min_t.nii.gz"

KASTNER_DIR = MASK_ROOT / "new"
KASTNER_LABEL_FILE = MASK_ROOT / "ROIfiles_Labeling.txt"
JULIAN_DIR = MASK_ROOT / "Julian2012"
AAL3_ATLAS = MASK_ROOT / "AAL3/AAL3v1.nii.gz"
AAL3_LABELS = MASK_ROOT / "AAL3/roi_labels.csv"
HARVARD_OXFORD_ATLAS = "cortl-maxprob-thr25-2mm"
NILEARN_DATA_DIR = Path("/home/yujun/nilearn_data")

OUT_DIR = ROOT / "outputs/roi_quantification"
THRESHOLD_PERCENT = 25.0
CHECK = "✓"


def load_img(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def positive_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data) & (data > 0)


def parse_kastner_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"%\s*(\d+)\s*-\s*([A-Za-z0-9]+)", line)
        if match:
            labels[int(match.group(1))] = match.group(2)
    if not labels:
        raise ValueError(f"No labels parsed from {path}")
    return labels


def resample_binary_roi(roi_path: Path, reference_img: nib.spatialimages.SpatialImage) -> np.ndarray:
    roi_img = load_img(roi_path)
    same_grid = roi_img.shape == reference_img.shape and np.allclose(roi_img.affine, reference_img.affine)
    if same_grid:
        roi_data = roi_img.get_fdata(dtype=np.float32)
    else:
        roi_resampled = resample_to_img(
            roi_img,
            reference_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        roi_data = roi_resampled.get_fdata(dtype=np.float32)
    return np.isfinite(roi_data) & (roi_data > 0)


def resample_label_atlas(atlas_path: Path, reference_img: nib.spatialimages.SpatialImage) -> np.ndarray:
    atlas_img = load_img(atlas_path)
    same_grid = atlas_img.shape == reference_img.shape and np.allclose(atlas_img.affine, reference_img.affine)
    if same_grid:
        atlas_data = atlas_img.get_fdata(dtype=np.float32)
    else:
        atlas_resampled = resample_to_img(
            atlas_img,
            reference_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_data = atlas_resampled.get_fdata(dtype=np.float32)
    return np.rint(atlas_data).astype(np.int16)


def julian_family(path: Path) -> str:
    rel = path.relative_to(JULIAN_DIR).as_posix()
    if "face_parcels" in rel:
        return "Julian2012 face parcels"
    if "scene_parcels" in rel:
        return "Julian2012 scene parcels"
    if "body_parcels" in rel:
        return "Julian2012 body parcels"
    if "object_parcels" in rel:
        return "Julian2012 object parcels"
    return "Julian2012"


def julian_name_and_hemi(path: Path) -> tuple[str, str]:
    name = path.stem
    if re.match(r"^[lr][A-Za-z]", name):
        hemi = "left" if name[0] == "l" else "right"
        return name[1:], hemi
    return name, "unknown"


def aal3_name_and_hemi(roi_name: str) -> tuple[str, str]:
    if roi_name.endswith("_L"):
        return roi_name[:-2], "left"
    if roi_name.endswith("_R"):
        return roi_name[:-2], "right"
    return roi_name, "unknown"


def add_roi_row(
    rows: list[dict[str, object]],
    atlas: str,
    hemisphere: str,
    roi: str,
    roi_id: str | int,
    roi_file_or_source: str,
    roi_mask: np.ndarray,
    rsa_mask: np.ndarray,
    multiple_regression_mask: np.ndarray,
) -> None:
    roi_voxels = int(roi_mask.sum())
    rsa_voxels = int((roi_mask & rsa_mask).sum())
    mr_voxels = int((roi_mask & multiple_regression_mask).sum())

    rsa_pct = 100 * rsa_voxels / roi_voxels if roi_voxels else np.nan
    mr_pct = 100 * mr_voxels / roi_voxels if roi_voxels else np.nan

    rsa_check = CHECK if np.isfinite(rsa_pct) and rsa_pct >= THRESHOLD_PERCENT else ""
    mr_check = CHECK if np.isfinite(mr_pct) and mr_pct >= THRESHOLD_PERCENT else ""

    rows.append(
        {
            "atlas": atlas,
            "hemisphere": hemisphere,
            "roi_id": roi_id,
            "roi": roi,
            "roi_file_or_source": roi_file_or_source,
            "roi_voxels": roi_voxels,
            "RSA_voxels_in_roi": rsa_voxels,
            "RSA_overlap_pct": rsa_pct,
            "RSA": rsa_check,
            "Multiple_Regression_voxels_in_roi": mr_voxels,
            "Multiple_Regression_overlap_pct": mr_pct,
            "Multiple_Regression": mr_check,
        }
    )


def collect_kastner_rows(
    rows: list[dict[str, object]],
    reference_img: nib.spatialimages.SpatialImage,
    rsa_mask: np.ndarray,
    multiple_regression_mask: np.ndarray,
) -> None:
    labels = parse_kastner_labels(KASTNER_LABEL_FILE)
    for hemisphere, prefix in [("left", "lh"), ("right", "rh")]:
        for roi_index, roi_label in sorted(labels.items()):
            roi_path = KASTNER_DIR / f"maxprob_vol_{prefix}_{roi_index}.nii"
            if not roi_path.exists():
                continue
            roi_mask = resample_binary_roi(roi_path, reference_img)
            add_roi_row(
                rows,
                "Kastner",
                hemisphere,
                roi_label,
                roi_index,
                str(roi_path),
                roi_mask,
                rsa_mask,
                multiple_regression_mask,
            )


def collect_julian_rows(
    rows: list[dict[str, object]],
    reference_img: nib.spatialimages.SpatialImage,
    rsa_mask: np.ndarray,
    multiple_regression_mask: np.ndarray,
) -> None:
    paths = sorted(
        path
        for path in JULIAN_DIR.rglob("*.img")
        if "__MACOSX" not in path.as_posix()
    )
    for path in paths:
        roi, hemi = julian_name_and_hemi(path)
        roi_mask = resample_binary_roi(path, reference_img)
        add_roi_row(
            rows,
            julian_family(path),
            hemi,
            roi,
            path.stem,
            str(path),
            roi_mask,
            rsa_mask,
            multiple_regression_mask,
        )


def collect_aal3_rows(
    rows: list[dict[str, object]],
    reference_img: nib.spatialimages.SpatialImage,
    rsa_mask: np.ndarray,
    multiple_regression_mask: np.ndarray,
) -> None:
    aal3_labels = pd.read_csv(AAL3_LABELS)
    atlas_data = resample_label_atlas(AAL3_ATLAS, reference_img)
    for _, label_row in aal3_labels.iterrows():
        roi_id = int(label_row["id"])
        roi_name = str(label_row["roi_name"])
        roi, hemi = aal3_name_and_hemi(roi_name)
        if hemi == "unknown":
            continue
        roi_mask = atlas_data == roi_id
        add_roi_row(
            rows,
            "AAL3",
            hemi,
            roi,
            roi_id,
            f"{AAL3_ATLAS} label_id={roi_id}",
            roi_mask,
            rsa_mask,
            multiple_regression_mask,
        )


def collect_harvard_oxford_rows(
    rows: list[dict[str, object]],
    reference_img: nib.spatialimages.SpatialImage,
    rsa_mask: np.ndarray,
    multiple_regression_mask: np.ndarray,
) -> None:
    atlas = datasets.fetch_atlas_harvard_oxford(
        HARVARD_OXFORD_ATLAS,
        data_dir=str(NILEARN_DATA_DIR),
    )
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.spatialimages.SpatialImage) else load_img(Path(atlas.maps))
    same_grid = atlas_img.shape == reference_img.shape and np.allclose(atlas_img.affine, reference_img.affine)
    if same_grid:
        atlas_data = atlas_img.get_fdata(dtype=np.float32)
    else:
        atlas_resampled = resample_to_img(
            atlas_img,
            reference_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_data = atlas_resampled.get_fdata(dtype=np.float32)
    atlas_data = np.rint(atlas_data).astype(np.int16)

    for roi_id, full_name in enumerate(atlas.labels):
        if roi_id == 0:
            continue
        full_name = str(full_name)
        if full_name.startswith("Left "):
            roi, hemi = full_name[5:], "left"
        elif full_name.startswith("Right "):
            roi, hemi = full_name[6:], "right"
        else:
            continue
        roi_mask = atlas_data == roi_id
        add_roi_row(
            rows,
            "Harvard-Oxford",
            hemi,
            roi,
            roi_id,
            f"{HARVARD_OXFORD_ATLAS} label_id={roi_id}",
            roi_mask,
            rsa_mask,
            multiple_regression_mask,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rsa_img = load_img(RSA_NIFTI)
    multiple_regression_img = load_img(MULTIPLE_REGRESSION_NIFTI)
    if rsa_img.shape != multiple_regression_img.shape or not np.allclose(
        rsa_img.affine,
        multiple_regression_img.affine,
    ):
        raise ValueError("RSA and Multiple_Regression maps are not on the same grid.")

    rsa_data = rsa_img.get_fdata(dtype=np.float32)
    multiple_regression_data = multiple_regression_img.get_fdata(dtype=np.float32)
    rsa_mask = positive_mask(rsa_data)
    multiple_regression_mask = positive_mask(multiple_regression_data)

    rows: list[dict[str, object]] = []
    collect_kastner_rows(rows, rsa_img, rsa_mask, multiple_regression_mask)
    collect_julian_rows(rows, rsa_img, rsa_mask, multiple_regression_mask)
    collect_aal3_rows(rows, rsa_img, rsa_mask, multiple_regression_mask)
    collect_harvard_oxford_rows(rows, rsa_img, rsa_mask, multiple_regression_mask)

    all_df = pd.DataFrame(rows)
    kept_df = all_df[(all_df["RSA"] == CHECK) | (all_df["Multiple_Regression"] == CHECK)].copy()
    kept_df = kept_df.sort_values(
        ["RSA", "Multiple_Regression", "RSA_overlap_pct", "Multiple_Regression_overlap_pct"],
        ascending=[False, False, False, False],
    )

    all_csv = OUT_DIR / "rsa_and_multiple_regression_roi_coverage_all_rows_with_harvard_oxford.csv"
    kept_csv = OUT_DIR / "rsa_and_multiple_regression_roi_coverage_ge25_checkmarks_with_harvard_oxford.csv"
    summary_json = OUT_DIR / "rsa_and_multiple_regression_roi_coverage_ge25_summary_with_harvard_oxford.json"

    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    kept_df.to_csv(kept_csv, index=False, encoding="utf-8-sig")

    summary = {
        "rsa_file": str(RSA_NIFTI),
        "Multiple_Regression_file": str(MULTIPLE_REGRESSION_NIFTI),
        "threshold_percent": THRESHOLD_PERCENT,
        "coverage_definition": "100 * method_positive_voxels_inside_roi / roi_voxels",
        "rsa_positive_voxels": int(rsa_mask.sum()),
        "Multiple_Regression_positive_voxels": int(multiple_regression_mask.sum()),
        "n_all_roi_rows": int(len(all_df)),
        "n_kept_roi_rows": int(len(kept_df)),
        "n_rsa_checked_rows": int((kept_df["RSA"] == CHECK).sum()),
        "n_Multiple_Regression_checked_rows": int((kept_df["Multiple_Regression"] == CHECK).sum()),
        "output_all_rows_csv": str(all_csv),
        "output_kept_csv": str(kept_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nKept ROIs (>={THRESHOLD_PERCENT:g}% coverage by RSA or Multiple_Regression):")
    columns = [
        "atlas",
        "hemisphere",
        "roi",
        "roi_voxels",
        "RSA_overlap_pct",
        "RSA",
        "Multiple_Regression_overlap_pct",
        "Multiple_Regression",
    ]
    print(kept_df[columns].to_string(index=False))
    print(f"\nWrote kept table: {kept_csv}")
    print(f"Wrote all rows: {all_csv}")
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
