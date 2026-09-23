from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SEARCHLIGHT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")
MASK_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/data/masks")

RSA_NIFTI = SEARCHLIGHT / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
REG_NIFTI = ROOT / "outputs/valence_arousal_both_positive_significant_min_t.nii.gz"

NEW_ROI_DIR = MASK_ROOT / "new"
NEW_LABEL_FILE = MASK_ROOT / "ROIfiles_Labeling.txt"
JULIAN_DIR = MASK_ROOT / "Julian2012"
AAL3_ATLAS = MASK_ROOT / "AAL3/AAL3v1.nii.gz"
AAL3_LABELS = MASK_ROOT / "AAL3/roi_labels.csv"

OUT_DIR = ROOT / "outputs/roi_quantification"


def load_img(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def positive_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data) & (data > 0)


def parse_new_roi_labels(path: Path) -> dict[int, str]:
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


def summarize_values(data: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    values = data[mask]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def summarize_roi_mask(
    atlas: str,
    roi: str,
    hemisphere: str,
    roi_id: str | int,
    roi_file_or_source: str,
    roi_mask: np.ndarray,
    rsa_data: np.ndarray,
    rsa_mask: np.ndarray,
    reg_data: np.ndarray,
    reg_mask: np.ndarray,
    n_rsa: int,
    n_reg: int,
) -> dict[str, object]:
    roi_rsa = roi_mask & rsa_mask
    roi_reg = roi_mask & reg_mask
    roi_overlap = roi_rsa & roi_reg
    roi_union = roi_rsa | roi_reg

    n_roi = int(roi_mask.sum())
    n_roi_rsa = int(roi_rsa.sum())
    n_roi_reg = int(roi_reg.sum())
    n_roi_overlap = int(roi_overlap.sum())
    n_roi_union = int(roi_union.sum())
    overlap_rate_pct = 100 * n_roi_overlap / n_roi_union if n_roi_union else np.nan

    row: dict[str, object] = {
        "atlas": atlas,
        "hemisphere": hemisphere,
        "roi_id": roi_id,
        "roi": roi,
        "roi_file_or_source": roi_file_or_source,
        "roi_voxels": n_roi,
        "rsa_voxels_in_roi": n_roi_rsa,
        "Multiple_Regression": n_roi_reg,
        "overlap_voxels_in_roi": n_roi_overlap,
        "union_voxels_in_roi": n_roi_union,
        "overlap_rate_pct": overlap_rate_pct,
        "dice_pct": 100 * (2 * n_roi_overlap / (n_roi_rsa + n_roi_reg)) if (n_roi_rsa + n_roi_reg) else np.nan,
        "pct_all_rsa_voxels_in_roi": 100 * n_roi_rsa / n_rsa if n_rsa else np.nan,
        "pct_all_Multiple_Regression_voxels_in_roi": 100 * n_roi_reg / n_reg if n_reg else np.nan,
    }
    row.update(summarize_values(rsa_data, roi_rsa, "rsa_t_in_roi"))
    row.update(summarize_values(reg_data, roi_reg, "reg_min_t_in_roi"))
    row.update(summarize_values(rsa_data, roi_overlap, "rsa_t_overlap_in_roi"))
    row.update(summarize_values(reg_data, roi_overlap, "reg_min_t_overlap_in_roi"))
    return row


def collect_new_rows(
    reference_img: nib.spatialimages.SpatialImage,
    rsa_data: np.ndarray,
    rsa_mask: np.ndarray,
    reg_data: np.ndarray,
    reg_mask: np.ndarray,
    n_rsa: int,
    n_reg: int,
) -> list[dict[str, object]]:
    labels = parse_new_roi_labels(NEW_LABEL_FILE)
    rows = []
    for hemisphere, prefix in [("left", "lh"), ("right", "rh")]:
        for roi_index, roi_label in sorted(labels.items()):
            roi_path = NEW_ROI_DIR / f"maxprob_vol_{prefix}_{roi_index}.nii"
            if not roi_path.exists():
                continue
            roi_mask = resample_binary_roi(roi_path, reference_img)
            rows.append(
                summarize_roi_mask(
                    "Kastner",
                    roi_label,
                    hemisphere,
                    roi_index,
                    str(roi_path),
                    roi_mask,
                    rsa_data,
                    rsa_mask,
                    reg_data,
                    reg_mask,
                    n_rsa,
                    n_reg,
                )
            )
    return rows


def collect_julian_rows(
    reference_img: nib.spatialimages.SpatialImage,
    rsa_data: np.ndarray,
    rsa_mask: np.ndarray,
    reg_data: np.ndarray,
    reg_mask: np.ndarray,
    n_rsa: int,
    n_reg: int,
) -> list[dict[str, object]]:
    rows = []
    paths = sorted(
        path
        for path in JULIAN_DIR.rglob("*.img")
        if "__MACOSX" not in path.as_posix()
    )
    for path in paths:
        roi, hemi = julian_name_and_hemi(path)
        roi_mask = resample_binary_roi(path, reference_img)
        rows.append(
            summarize_roi_mask(
                julian_family(path),
                roi,
                hemi,
                path.stem,
                str(path),
                roi_mask,
                rsa_data,
                rsa_mask,
                reg_data,
                reg_mask,
                n_rsa,
                n_reg,
            )
        )
    return rows


def collect_aal3_rows(
    reference_img: nib.spatialimages.SpatialImage,
    rsa_data: np.ndarray,
    rsa_mask: np.ndarray,
    reg_data: np.ndarray,
    reg_mask: np.ndarray,
    n_rsa: int,
    n_reg: int,
) -> list[dict[str, object]]:
    aal3_labels = pd.read_csv(AAL3_LABELS)
    atlas_data = resample_label_atlas(AAL3_ATLAS, reference_img)
    rows = []
    for _, label_row in aal3_labels.iterrows():
        roi_id = int(label_row["id"])
        roi_name = str(label_row["roi_name"])
        roi, hemi = aal3_name_and_hemi(roi_name)
        if hemi == "unknown":
            continue
        roi_mask = atlas_data == roi_id
        rows.append(
            summarize_roi_mask(
                "AAL3",
                roi,
                hemi,
                roi_id,
                f"{AAL3_ATLAS} label_id={roi_id}",
                roi_mask,
                rsa_data,
                rsa_mask,
                reg_data,
                reg_mask,
                n_rsa,
                n_reg,
            )
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rsa_img = load_img(RSA_NIFTI)
    reg_img = load_img(REG_NIFTI)
    if rsa_img.shape != reg_img.shape or not np.allclose(rsa_img.affine, reg_img.affine):
        raise ValueError("RSA and regression maps are not on the same grid.")

    rsa_data = rsa_img.get_fdata(dtype=np.float32)
    reg_data = reg_img.get_fdata(dtype=np.float32)
    rsa_mask = positive_mask(rsa_data)
    reg_mask = positive_mask(reg_data)
    overlap_mask = rsa_mask & reg_mask
    union_mask = rsa_mask | reg_mask

    n_rsa = int(rsa_mask.sum())
    n_reg = int(reg_mask.sum())
    n_overlap = int(overlap_mask.sum())
    n_union = int(union_mask.sum())

    rows = []
    rows.extend(collect_new_rows(rsa_img, rsa_data, rsa_mask, reg_data, reg_mask, n_rsa, n_reg))
    rows.extend(collect_julian_rows(rsa_img, rsa_data, rsa_mask, reg_data, reg_mask, n_rsa, n_reg))
    rows.extend(collect_aal3_rows(rsa_img, rsa_data, rsa_mask, reg_data, reg_mask, n_rsa, n_reg))

    all_df = pd.DataFrame(rows)
    nonzero_df = all_df[np.isfinite(all_df["overlap_rate_pct"]) & (all_df["overlap_rate_pct"] > 0)].copy()
    nonzero_df = nonzero_df.sort_values(
        ["overlap_rate_pct", "overlap_voxels_in_roi", "Multiple_Regression"],
        ascending=False,
    )

    summary = {
        "rsa_file": str(RSA_NIFTI),
        "regression_file": str(REG_NIFTI),
        "kastner_left_right_roi_directory": str(NEW_ROI_DIR),
        "new_label_file": str(NEW_LABEL_FILE),
        "julian_directory": str(JULIAN_DIR),
        "aal3_atlas": str(AAL3_ATLAS),
        "aal3_labels": str(AAL3_LABELS),
        "map_shape": list(rsa_img.shape),
        "rsa_positive_voxels": n_rsa,
        "Multiple_Regression_positive_voxels": n_reg,
        "overlap_voxels": n_overlap,
        "union_voxels": n_union,
        "whole_map_overlap_rate_pct": 100 * n_overlap / n_union if n_union else np.nan,
        "whole_map_dice_pct": 100 * (2 * n_overlap / (n_rsa + n_reg)) if (n_rsa + n_reg) else np.nan,
        "overlap_rate_definition": "100 * overlap_voxels_in_roi / union_voxels_in_roi; Jaccard-style percent",
        "n_all_left_right_roi_rows": int(len(all_df)),
        "n_nonzero_overlap_rows": int(len(nonzero_df)),
    }

    all_csv = OUT_DIR / "rsa_vs_valence_arousal_both_positive_new_julian_aal3_lr_all_rows.csv"
    nonzero_csv = OUT_DIR / "rsa_vs_valence_arousal_both_positive_new_julian_aal3_lr_nonzero_overlap.csv"
    summary_json = OUT_DIR / "rsa_vs_valence_arousal_both_positive_new_julian_aal3_lr_summary.json"
    all_df.to_csv(all_csv, index=False)
    nonzero_df.to_csv(nonzero_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nNonzero Jaccard-style overlap rows:")
    cols = [
        "atlas",
        "hemisphere",
        "roi",
        "roi_voxels",
        "rsa_voxels_in_roi",
        "Multiple_Regression",
        "overlap_voxels_in_roi",
        "union_voxels_in_roi",
        "overlap_rate_pct",
    ]
    print(nonzero_df[cols].to_string(index=False))
    print(f"\nWrote nonzero rows: {nonzero_csv}")
    print(f"Wrote all rows: {all_csv}")
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
