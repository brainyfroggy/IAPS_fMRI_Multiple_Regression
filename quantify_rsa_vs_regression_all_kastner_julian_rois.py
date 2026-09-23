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

KASTNER_DIR = MASK_ROOT / "Kastner"
JULIAN_DIR = MASK_ROOT / "Julian2012"
OUT_DIR = ROOT / "outputs/roi_quantification"


def load_img(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def positive_mask(data: np.ndarray) -> np.ndarray:
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


def clean_julian_label(path: Path) -> tuple[str, str | None, str]:
    name = path.stem
    hemi = None
    bilateral_label = name
    if re.match(r"^[lr][A-Za-z]", name):
        hemi = "left" if name[0] == "l" else "right"
        bilateral_label = name[1:]

    rel = path.relative_to(JULIAN_DIR).as_posix()
    if "face_parcels" in rel:
        family = "Julian2012 face parcels"
    elif "scene_parcels" in rel:
        family = "Julian2012 scene parcels"
    elif "body_parcels" in rel:
        family = "Julian2012 body parcels"
    elif "object_parcels" in rel:
        family = "Julian2012 object parcels"
    else:
        family = "Julian2012"
    return bilateral_label, hemi, family


def discover_rois() -> list[dict[str, object]]:
    rois: list[dict[str, object]] = []

    for path in sorted(KASTNER_DIR.glob("*.img")):
        rois.append(
            {
                "atlas": "Kastner",
                "roi": path.stem,
                "hemisphere": "bilateral_or_combined",
                "level": "individual",
                "files": [path],
            }
        )

    julian_paths = sorted(
        path
        for path in JULIAN_DIR.rglob("*.img")
        if "__MACOSX" not in path.as_posix()
    )
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in julian_paths:
        label, hemi, family = clean_julian_label(path)
        rois.append(
            {
                "atlas": family,
                "roi": path.stem,
                "hemisphere": hemi or "unknown",
                "level": "individual",
                "files": [path],
            }
        )
        grouped.setdefault((family, label), []).append(path)

    for (family, label), paths in sorted(grouped.items()):
        if len(paths) > 1:
            rois.append(
                {
                    "atlas": family,
                    "roi": label,
                    "hemisphere": "bilateral_combined",
                    "level": "bilateral_combined",
                    "files": sorted(paths),
                }
            )

    return rois


def summarize_roi(
    roi: dict[str, object],
    rsa_img: nib.spatialimages.SpatialImage,
    rsa_data: np.ndarray,
    rsa_mask: np.ndarray,
    reg_data: np.ndarray,
    reg_mask: np.ndarray,
    n_rsa: int,
    n_reg: int,
) -> dict[str, object]:
    roi_mask = np.zeros(rsa_img.shape, dtype=bool)
    roi_files = [Path(path) for path in roi["files"]]
    for roi_file in roi_files:
        roi_mask |= resample_roi_to_reference(roi_file, rsa_img)

    roi_rsa = roi_mask & rsa_mask
    roi_reg = roi_mask & reg_mask
    roi_overlap = roi_rsa & roi_reg
    roi_union = roi_rsa | roi_reg

    n_roi = int(roi_mask.sum())
    n_roi_rsa = int(roi_rsa.sum())
    n_roi_reg = int(roi_reg.sum())
    n_roi_overlap = int(roi_overlap.sum())
    n_roi_union = int(roi_union.sum())

    row = {
        "atlas": roi["atlas"],
        "roi": roi["roi"],
        "hemisphere": roi["hemisphere"],
        "level": roi["level"],
        "roi_files": ";".join(str(path) for path in roi_files),
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
    return row


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

    rows = [
        summarize_roi(roi, rsa_img, rsa_data, rsa_mask, reg_data, reg_mask, n_rsa, n_reg)
        for roi in discover_rois()
    ]
    df = pd.DataFrame(rows).sort_values(
        ["level", "overlap_voxels_in_roi", "regression_voxels_in_roi", "rsa_voxels_in_roi"],
        ascending=[True, False, False, False],
    )

    summary = {
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
        "n_roi_rows": int(len(df)),
        "n_kastner_individual_rois": int(((df["atlas"] == "Kastner") & (df["level"] == "individual")).sum()),
        "n_julian_individual_rois": int((df["atlas"].str.startswith("Julian2012") & (df["level"] == "individual")).sum()),
        "n_julian_bilateral_combined_rois": int((df["atlas"].str.startswith("Julian2012") & (df["level"] == "bilateral_combined")).sum()),
    }

    csv_path = OUT_DIR / "rsa_vs_valence_arousal_both_positive_all_kastner_julian_rois.csv"
    json_path = OUT_DIR / "rsa_vs_valence_arousal_both_positive_all_kastner_julian_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nTop individual ROIs by overlap:")
    top_cols = [
        "atlas",
        "roi",
        "hemisphere",
        "roi_voxels",
        "rsa_voxels_in_roi",
        "regression_voxels_in_roi",
        "overlap_voxels_in_roi",
        "pct_roi_regression_overlapping_rsa",
        "roi_dice_rsa_regression",
    ]
    individual = df[df["level"] == "individual"].copy()
    print(
        individual.sort_values(
            ["overlap_voxels_in_roi", "regression_voxels_in_roi", "rsa_voxels_in_roi"],
            ascending=False,
        )
        .head(20)[top_cols]
        .to_string(index=False)
    )

    combined = df[df["level"] == "bilateral_combined"].copy()
    if not combined.empty:
        print("\nBilateral Julian combined ROIs:")
        print(
            combined.sort_values(
                ["overlap_voxels_in_roi", "regression_voxels_in_roi", "rsa_voxels_in_roi"],
                ascending=False,
            )[top_cols]
            .to_string(index=False)
        )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
