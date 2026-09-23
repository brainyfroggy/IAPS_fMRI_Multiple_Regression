from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.table import Table
from nilearn import datasets
from nilearn.image import resample_to_img
from PIL import Image, ImageChops


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SEARCHLIGHT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")
MASK_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/data/masks")

RSA_NIFTI = SEARCHLIGHT / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
UNIVARIATE_NIFTI = ROOT / "outputs/valence_arousal_both_positive_significant_min_t.nii.gz"
PANEL_A = ROOT / "figures/mricrogl_harvard_oxford_panel_a_tvalues.png"

AAL3_ATLAS = MASK_ROOT / "AAL3/AAL3v1.nii.gz"
AAL3_LABELS = MASK_ROOT / "AAL3/roi_labels.csv"
HARVARD_OXFORD_ATLAS = "cortl-maxprob-thr25-2mm"
NILEARN_DATA_DIR = Path("/home/yujun/nilearn_data")

OUTPUT_DIR = ROOT / "outputs/roi_quantification"
FIGURE_DIR = ROOT / "figures"
THRESHOLD_PERCENT = 5.0

DISPLAY_NAMES = {
    "Occipital_Inf": "Inferior occipital gyrus",
    "Occipital_Mid": "Middle occipital gyrus",
    "Temporal_Inf": "Inferior temporal gyrus",
    "Temporal_Mid": "Middle temporal gyrus",
    "ACC_pre": "Pregenual anterior cingulate cortex",
    "ACC_sub": "Subgenual anterior cingulate cortex",
    "Frontal_Med_Orb": "Medial orbital frontal gyrus",
    "Lateral Occipital Cortex, inferior division": "Inferior lateral occipital cortex",
    "Middle Temporal Gyrus, temporooccipital part": "Temporo-occipital middle temporal gyrus",
    "Frontal Medial Cortex": "Frontal medial cortex",
    "Paracingulate Gyrus": "Paracingulate gyrus",
    "Superior Parietal Lobule": "Superior parietal lobule",
    "Angular Gyrus": "Angular gyrus",
}

STATUS_ORDER = {
    "Shared": 0,
    "Multivariate RSA only": 1,
    "Univariate only": 2,
}

STATUS_COLORS = {
    "Shared": "#e7f3e8",
    "Multivariate RSA only": "#fff2df",
    "Univariate only": "#e8f1fb",
}


def positive_mask(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = image.get_fdata(dtype=np.float32)
    return np.isfinite(data) & (data > 0)


def resample_labels(
    atlas_image: nib.spatialimages.SpatialImage,
    reference_image: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    if atlas_image.shape == reference_image.shape and np.allclose(atlas_image.affine, reference_image.affine):
        data = atlas_image.get_fdata(dtype=np.float32)
    else:
        data = resample_to_img(
            atlas_image,
            reference_image,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        ).get_fdata(dtype=np.float32)
    return np.rint(data).astype(np.int16)


def roi_row(
    atlas: str,
    hemisphere: str,
    roi: str,
    roi_id: int,
    roi_mask: np.ndarray,
    rsa_mask: np.ndarray,
    univariate_mask: np.ndarray,
) -> dict[str, object] | None:
    roi_voxels = int(roi_mask.sum())
    if roi_voxels == 0:
        return None
    rsa_overlap_voxels = int((roi_mask & rsa_mask).sum())
    univariate_overlap_voxels = int((roi_mask & univariate_mask).sum())
    rsa_percent = 100.0 * rsa_overlap_voxels / roi_voxels
    univariate_percent = 100.0 * univariate_overlap_voxels / roi_voxels
    rsa_pass = rsa_percent >= THRESHOLD_PERCENT
    univariate_pass = univariate_percent >= THRESHOLD_PERCENT
    if not (rsa_pass or univariate_pass):
        return None
    if rsa_pass and univariate_pass:
        status = "Shared"
    elif rsa_pass:
        status = "Multivariate RSA only"
    else:
        status = "Univariate only"
    return {
        "atlas": atlas,
        "hemisphere": hemisphere,
        "roi_id": roi_id,
        "roi": roi,
        "display_roi": DISPLAY_NAMES.get(roi, roi.replace("_", " ")),
        "atlas_roi_voxels": roi_voxels,
        "multivariate_rsa_overlap_voxels": rsa_overlap_voxels,
        "multivariate_rsa_overlap_percent": rsa_percent,
        "multivariate_rsa_pass_5pct": rsa_pass,
        "univariate_overlap_voxels": univariate_overlap_voxels,
        "univariate_overlap_percent": univariate_percent,
        "univariate_pass_5pct": univariate_pass,
        "classification": status,
    }


def collect_harvard_oxford(
    reference_image: nib.spatialimages.SpatialImage,
    rsa_mask: np.ndarray,
    univariate_mask: np.ndarray,
) -> list[dict[str, object]]:
    atlas = datasets.fetch_atlas_harvard_oxford(
        HARVARD_OXFORD_ATLAS,
        data_dir=str(NILEARN_DATA_DIR),
    )
    atlas_image = atlas.maps if isinstance(atlas.maps, nib.spatialimages.SpatialImage) else nib.load(str(atlas.maps))
    atlas_data = resample_labels(atlas_image, reference_image)
    rows: list[dict[str, object]] = []
    for roi_id, label in enumerate(atlas.labels):
        if roi_id == 0:
            continue
        full_name = str(label)
        if full_name.startswith("Left "):
            hemisphere, roi = "L", full_name[5:]
        elif full_name.startswith("Right "):
            hemisphere, roi = "R", full_name[6:]
        else:
            continue
        row = roi_row(
            "Harvard-Oxford",
            hemisphere,
            roi,
            roi_id,
            atlas_data == roi_id,
            rsa_mask,
            univariate_mask,
        )
        if row is not None:
            rows.append(row)
    return rows


def collect_aal3(
    reference_image: nib.spatialimages.SpatialImage,
    rsa_mask: np.ndarray,
    univariate_mask: np.ndarray,
) -> list[dict[str, object]]:
    atlas_image = nib.load(str(AAL3_ATLAS))
    atlas_data = resample_labels(atlas_image, reference_image)
    labels = pd.read_csv(AAL3_LABELS)
    rows: list[dict[str, object]] = []
    for _, label_row in labels.iterrows():
        roi_id = int(label_row["id"])
        full_name = str(label_row["roi_name"])
        if full_name.endswith("_L"):
            hemisphere, roi = "L", full_name[:-2]
        elif full_name.endswith("_R"):
            hemisphere, roi = "R", full_name[:-2]
        else:
            continue
        row = roi_row(
            "AAL3",
            hemisphere,
            roi,
            roi_id,
            atlas_data == roi_id,
            rsa_mask,
            univariate_mask,
        )
        if row is not None:
            rows.append(row)
    return rows


def percent_label(value: float, passed: bool) -> str:
    return f"{value:.1f}%" + ("  \u2713" if passed else "")


def crop_white_margin(image: Image.Image, padding: int = 18) -> Image.Image:
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    bbox = difference.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def add_table_panel(axis: plt.Axes, rows: pd.DataFrame) -> None:
    axis.axis("off")
    axis.text(0.0, 0.995, "B", transform=axis.transAxes, fontsize=30, fontweight="normal", va="top")
    axis.text(
        0.065,
        0.988,
        "Harvard-Oxford ROI coverage",
        transform=axis.transAxes,
        fontsize=25,
        fontweight="bold",
        va="top",
    )

    headers = [
        "Hemisphere",
        "ROI",
        "Univariate",
        "Multivariate",
    ]
    column_widths = [0.18, 0.48, 0.17, 0.17]
    table = Table(axis, bbox=[0.03, 0.20, 0.94, 0.68])
    row_height = 1.0 / (len(rows) + 1)

    for column, (header, width) in enumerate(zip(headers, column_widths)):
        cell = table.add_cell(0, column, width, row_height, text=header, loc="center", facecolor="#333333")
        cell.get_text().set_color("white")
        cell.get_text().set_fontsize(24)
        cell.get_text().set_fontweight("bold")
        cell.set_edgecolor("white")
        cell.set_linewidth(1.0)

    for row_index, (_, row) in enumerate(rows.iterrows(), start=1):
        values = [
            row["hemisphere"],
            row["display_roi"],
            "\u2713" if row["univariate_pass_5pct"] else "",
            "\u2713" if row["multivariate_rsa_pass_5pct"] else "",
        ]
        background = STATUS_COLORS[row["classification"]]
        for column, (value, width) in enumerate(zip(values, column_widths)):
            location = "left" if column == 1 else "center"
            cell = table.add_cell(
                row_index,
                column,
                width,
                row_height,
                text=str(value),
                loc=location,
                facecolor=background,
            )
            cell.PAD = 0.025
            cell.set_edgecolor("#c9c9c9")
            cell.set_linewidth(0.55)
            cell.get_text().set_fontsize(23)
            if column in (2, 3):
                cell.get_text().set_fontsize(38)
                cell.get_text().set_fontweight("bold")

    axis.add_table(table)
    axis.text(
        0.0,
        0.115,
        "\u2713 indicates that positive significant voxels cover at least 5% of the atlas ROI.",
        transform=axis.transAxes,
        fontsize=16,
        color="#333333",
        va="bottom",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rsa_image = nib.load(str(RSA_NIFTI))
    univariate_image = nib.load(str(UNIVARIATE_NIFTI))
    if rsa_image.shape != univariate_image.shape or not np.allclose(rsa_image.affine, univariate_image.affine):
        raise ValueError("The multivariate RSA and univariate maps do not share a voxel grid.")

    rsa_mask = positive_mask(rsa_image)
    univariate_mask = positive_mask(univariate_image)
    rows = collect_harvard_oxford(rsa_image, rsa_mask, univariate_mask)
    table_data = pd.DataFrame(rows)
    table_data["status_order"] = table_data["classification"].map(STATUS_ORDER)
    table_data = table_data.sort_values(
        ["status_order", "hemisphere", "display_roi"],
        kind="stable",
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "harvard_oxford_roi_coverage_ge5_balanced_figure_data.csv"
    export_columns = [column for column in table_data.columns if not column.endswith("_order")]
    table_data[export_columns].to_csv(csv_path, index=False, encoding="utf-8-sig")

    panel_a_image = np.asarray(crop_white_margin(Image.open(PANEL_A)))
    figure = plt.figure(figsize=(18.5, 7.5), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.02, 1.18],
        left=0.02,
        right=0.985,
        bottom=0.045,
        top=0.97,
        wspace=0.035,
    )

    panel_a_axis = figure.add_subplot(grid[0, 0])
    panel_a_axis.imshow(panel_a_image, interpolation="lanczos")
    panel_a_axis.axis("off")
    panel_a_axis.set_anchor("N")
    panel_a_axis.text(
        0.003,
        0.993,
        "A",
        transform=panel_a_axis.transAxes,
        fontsize=24,
        fontweight="normal",
        va="top",
        ha="left",
    )

    panel_b_axis = figure.add_subplot(grid[0, 1])
    add_table_panel(panel_b_axis, table_data)

    png_path = FIGURE_DIR / "mricrogl_harvard_oxford_roi_figure_balanced.png"
    pdf_path = FIGURE_DIR / "mricrogl_harvard_oxford_roi_figure_balanced.pdf"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)

    counts = table_data["classification"].value_counts().to_dict()
    print(f"Qualifying ROI rows: {len(table_data)}")
    print(f"Classification counts: {counts}")
    print(f"Saved table: {csv_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
