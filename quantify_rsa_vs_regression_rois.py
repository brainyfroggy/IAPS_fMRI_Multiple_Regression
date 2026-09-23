from __future__ import annotations

import json
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

OUT_DIR = ROOT / "outputs/roi_quantification"


ROI_DEFS = {
    "V1": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/V1.img"],
    },
    "V2": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/V2.img"],
    },
    "V3": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/V3.img"],
    },
    "V3A": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/V3a.img"],
    },
    "V3B": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/V3b.img"],
    },
    "V4": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/V4.img"],
    },
    "LO": {
        "source": "Kastner",
        "files": [
            MASK_ROOT / "Kastner/LO1.img",
            MASK_ROOT / "Kastner/LO2.img",
        ],
    },
    "IPS": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/IPS.img"],
    },
    "MT": {
        "source": "Kastner",
        "files": [MASK_ROOT / "Kastner/MT.img"],
    },
    "STS": {
        "source": "Julian2012 face parcels",
        "files": [
            MASK_ROOT / "Julian2012/face_parcels/face_parcels/lSTS.img",
            MASK_ROOT / "Julian2012/face_parcels/face_parcels/rSTS.img",
        ],
    },
}


def load_img(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def finite_positive_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data) & (data > 0)


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


def summarize_values(data: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float | int]:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rsa_img = load_img(RSA_NIFTI)
    reg_img = load_img(REG_NIFTI)
    if rsa_img.shape != reg_img.shape or not np.allclose(rsa_img.affine, reg_img.affine):
        raise ValueError("RSA and regression maps are not on the same grid.")

    rsa_data = rsa_img.get_fdata(dtype=np.float32)
    reg_data = reg_img.get_fdata(dtype=np.float32)
    rsa_mask = finite_positive_mask(rsa_data)
    reg_mask = finite_positive_mask(reg_data)
    overlap_mask = rsa_mask & reg_mask
    union_mask = rsa_mask | reg_mask

    n_rsa = int(rsa_mask.sum())
    n_reg = int(reg_mask.sum())
    n_overlap = int(overlap_mask.sum())
    n_union = int(union_mask.sum())

    whole_summary = {
        "rsa_file": str(RSA_NIFTI),
        "regression_file": str(REG_NIFTI),
        "map_shape": list(rsa_img.shape),
        "rsa_positive_voxels": n_rsa,
        "regression_positive_voxels": n_reg,
        "overlap_voxels": n_overlap,
        "union_voxels": n_union,
        "jaccard_overlap_over_union": n_overlap / n_union if n_union else np.nan,
        "dice_coefficient": (2 * n_overlap) / (n_rsa + n_reg) if (n_rsa + n_reg) else np.nan,
        "percent_rsa_overlapping_regression": 100 * n_overlap / n_rsa if n_rsa else np.nan,
        "percent_regression_overlapping_rsa": 100 * n_overlap / n_reg if n_reg else np.nan,
    }
    whole_summary.update(summarize_values(rsa_data, rsa_mask, "rsa_t_all"))
    whole_summary.update(summarize_values(reg_data, reg_mask, "reg_min_t_all"))
    whole_summary.update(summarize_values(rsa_data, overlap_mask, "rsa_t_overlap"))
    whole_summary.update(summarize_values(reg_data, overlap_mask, "reg_min_t_overlap"))

    rows = []
    for roi_name, roi_def in ROI_DEFS.items():
        roi_mask = np.zeros(rsa_img.shape, dtype=bool)
        roi_files = []
        for roi_file in roi_def["files"]:
            roi_files.append(str(roi_file))
            roi_mask |= resample_roi_to_reference(roi_file, rsa_img)

        roi_rsa = roi_mask & rsa_mask
        roi_reg = roi_mask & reg_mask
        roi_overlap = roi_mask & overlap_mask
        roi_union = roi_mask & union_mask

        n_roi = int(roi_mask.sum())
        n_roi_rsa = int(roi_rsa.sum())
        n_roi_reg = int(roi_reg.sum())
        n_roi_overlap = int(roi_overlap.sum())
        n_roi_union = int(roi_union.sum())

        row = {
            "roi": roi_name,
            "source": roi_def["source"],
            "roi_files": ";".join(roi_files),
            "roi_voxels": n_roi,
            "rsa_voxels_in_roi": n_roi_rsa,
            "regression_voxels_in_roi": n_roi_reg,
            "overlap_voxels_in_roi": n_roi_overlap,
            "union_voxels_in_roi": n_roi_union,
            "pct_roi_occupied_by_rsa": 100 * n_roi_rsa / n_roi if n_roi else np.nan,
            "pct_roi_occupied_by_regression": 100 * n_roi_reg / n_roi if n_roi else np.nan,
            "pct_roi_occupied_by_overlap": 100 * n_roi_overlap / n_roi if n_roi else np.nan,
            "pct_all_rsa_voxels_in_roi": 100 * n_roi_rsa / n_rsa if n_rsa else np.nan,
            "pct_all_regression_voxels_in_roi": 100 * n_roi_reg / n_reg if n_reg else np.nan,
            "pct_roi_rsa_overlapping_regression": 100 * n_roi_overlap / n_roi_rsa if n_roi_rsa else np.nan,
            "pct_roi_regression_overlapping_rsa": 100 * n_roi_overlap / n_roi_reg if n_roi_reg else np.nan,
            "roi_jaccard_rsa_regression": n_roi_overlap / n_roi_union if n_roi_union else np.nan,
            "roi_dice_rsa_regression": (
                2 * n_roi_overlap / (n_roi_rsa + n_roi_reg) if (n_roi_rsa + n_roi_reg) else np.nan
            ),
        }
        row.update(summarize_values(rsa_data, roi_rsa, "rsa_t_in_roi"))
        row.update(summarize_values(reg_data, roi_reg, "reg_min_t_in_roi"))
        row.update(summarize_values(rsa_data, roi_overlap, "rsa_t_overlap_in_roi"))
        row.update(summarize_values(reg_data, roi_overlap, "reg_min_t_overlap_in_roi"))
        rows.append(row)

    roi_df = pd.DataFrame(rows).sort_values(
        ["overlap_voxels_in_roi", "regression_voxels_in_roi", "rsa_voxels_in_roi"],
        ascending=False,
    )

    roi_csv = OUT_DIR / "rsa_vs_valence_arousal_both_positive_roi_overlap.csv"
    summary_json = OUT_DIR / "rsa_vs_valence_arousal_both_positive_summary.json"
    roi_df.to_csv(roi_csv, index=False)
    summary_json.write_text(json.dumps(whole_summary, indent=2), encoding="utf-8")

    print(json.dumps(whole_summary, indent=2))
    print("\nROI summary:")
    cols = [
        "roi",
        "source",
        "roi_voxels",
        "rsa_voxels_in_roi",
        "regression_voxels_in_roi",
        "overlap_voxels_in_roi",
        "pct_roi_regression_overlapping_rsa",
        "pct_roi_rsa_overlapping_regression",
        "roi_dice_rsa_regression",
    ]
    print(roi_df[cols].to_string(index=False))
    print(f"\nWrote {roi_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
