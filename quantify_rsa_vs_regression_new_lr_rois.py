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
LABEL_FILE = MASK_ROOT / "ROIfiles_Labeling.txt"
OUT_DIR = ROOT / "outputs/roi_quantification"


def load_img(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def positive_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data) & (data > 0)


def parse_roi_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"%\s*(\d+)\s*-\s*([A-Za-z0-9]+)", line)
        if match:
            labels[int(match.group(1))] = match.group(2)
    if not labels:
        raise ValueError(f"No labels parsed from {path}")
    return labels


def resample_roi_to_reference(roi_path: Path, reference_img: nib.spatialimages.SpatialImage) -> np.ndarray:
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


def summarize_roi(
    hemisphere: str,
    roi_index: int,
    roi_label: str,
    roi_path: Path,
    rsa_img: nib.spatialimages.SpatialImage,
    rsa_data: np.ndarray,
    rsa_mask: np.ndarray,
    reg_data: np.ndarray,
    reg_mask: np.ndarray,
    n_rsa: int,
    n_reg: int,
) -> dict[str, object]:
    roi_mask = resample_roi_to_reference(roi_path, rsa_img)
    roi_rsa = roi_mask & rsa_mask
    roi_reg = roi_mask & reg_mask
    roi_overlap = roi_rsa & roi_reg
    roi_union = roi_rsa | roi_reg

    n_roi = int(roi_mask.sum())
    n_roi_rsa = int(roi_rsa.sum())
    n_roi_reg = int(roi_reg.sum())
    n_roi_overlap = int(roi_overlap.sum())
    n_roi_union = int(roi_union.sum())

    # Single symmetric overlap percentage requested by user.
    overlap_rate_pct = 100 * n_roi_overlap / n_roi_union if n_roi_union else np.nan

    row: dict[str, object] = {
        "atlas": "new maxprob left/right ROI files",
        "hemisphere": hemisphere,
        "roi_index": roi_index,
        "roi": roi_label,
        "roi_file": str(roi_path),
        "roi_voxels": n_roi,
        "rsa_voxels_in_roi": n_roi_rsa,
        "regression_voxels_in_roi": n_roi_reg,
        "overlap_voxels_in_roi": n_roi_overlap,
        "union_voxels_in_roi": n_roi_union,
        "overlap_rate_pct": overlap_rate_pct,
        "pct_roi_occupied_by_rsa": 100 * n_roi_rsa / n_roi if n_roi else np.nan,
        "pct_roi_occupied_by_regression": 100 * n_roi_reg / n_roi if n_roi else np.nan,
        "pct_roi_occupied_by_overlap": 100 * n_roi_overlap / n_roi if n_roi else np.nan,
        "pct_all_rsa_voxels_in_roi": 100 * n_roi_rsa / n_rsa if n_rsa else np.nan,
        "pct_all_regression_voxels_in_roi": 100 * n_roi_reg / n_reg if n_reg else np.nan,
        "dice_pct": 100 * (2 * n_roi_overlap / (n_roi_rsa + n_roi_reg)) if (n_roi_rsa + n_roi_reg) else np.nan,
    }
    row.update(summarize_values(rsa_data, roi_rsa, "rsa_t_in_roi"))
    row.update(summarize_values(reg_data, roi_reg, "reg_min_t_in_roi"))
    row.update(summarize_values(rsa_data, roi_overlap, "rsa_t_overlap_in_roi"))
    row.update(summarize_values(reg_data, roi_overlap, "reg_min_t_overlap_in_roi"))
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = parse_roi_labels(LABEL_FILE)

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
    missing = []
    for hemisphere, prefix in [("left", "lh"), ("right", "rh")]:
        for roi_index, roi_label in sorted(labels.items()):
            roi_path = NEW_ROI_DIR / f"maxprob_vol_{prefix}_{roi_index}.nii"
            if not roi_path.exists():
                missing.append(str(roi_path))
                continue
            rows.append(
                summarize_roi(
                    hemisphere,
                    roi_index,
                    roi_label,
                    roi_path,
                    rsa_img,
                    rsa_data,
                    rsa_mask,
                    reg_data,
                    reg_mask,
                    n_rsa,
                    n_reg,
                )
            )

    df = pd.DataFrame(rows).sort_values(
        ["overlap_voxels_in_roi", "regression_voxels_in_roi", "rsa_voxels_in_roi"],
        ascending=False,
    )

    summary = {
        "rsa_file": str(RSA_NIFTI),
        "regression_file": str(REG_NIFTI),
        "roi_directory": str(NEW_ROI_DIR),
        "label_file": str(LABEL_FILE),
        "roi_indices_used": sorted(labels.keys()),
        "excluded_unlabeled_index_0": True,
        "map_shape": list(rsa_img.shape),
        "rsa_positive_voxels": n_rsa,
        "regression_positive_voxels": n_reg,
        "overlap_voxels": n_overlap,
        "union_voxels": n_union,
        "whole_map_overlap_rate_pct": 100 * n_overlap / n_union if n_union else np.nan,
        "whole_map_dice_pct": 100 * (2 * n_overlap / (n_rsa + n_reg)) if (n_rsa + n_reg) else np.nan,
        "n_roi_rows": int(len(df)),
        "missing_roi_files": missing,
        "overlap_rate_definition": "100 * overlap_voxels_in_roi / union_voxels_in_roi; a single symmetric Jaccard-style percentage",
    }

    csv_path = OUT_DIR / "rsa_vs_valence_arousal_both_positive_new_lr_rois.csv"
    json_path = OUT_DIR / "rsa_vs_valence_arousal_both_positive_new_lr_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nTop left/right ROIs by overlap:")
    columns = [
        "hemisphere",
        "roi_index",
        "roi",
        "roi_voxels",
        "rsa_voxels_in_roi",
        "regression_voxels_in_roi",
        "overlap_voxels_in_roi",
        "union_voxels_in_roi",
        "overlap_rate_pct",
        "dice_pct",
    ]
    print(df.head(30)[columns].to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
