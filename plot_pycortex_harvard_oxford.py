from __future__ import annotations

import csv
from pathlib import Path

import cortex
import cortex.database
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import font_manager
from nilearn import datasets
from nilearn.image import resample_to_img
from PIL import Image, ImageChops, ImageDraw, ImageFont

import plot_pycortex_both_positive_tstat as base


ATLAS_NAME = "cortl-maxprob-thr25-2mm"
ATLAS_DATA_DIR = Path("/home/yujun/nilearn_data")

# These settings soften voxel-to-surface stair steps while preserving parcel shape.
base.ROI_BORDER_LINEWIDTH = 2.6
base.ROI_SMOOTH_STEPS = 16
base.ROI_SMOOTH_SELF_WEIGHT = 0.52
base.ROI_CONTOUR_LEVEL = 0.40
FOCUS_XLIM = (-170, 170)
FOCUS_YLIM = (-95, 85)
base.CROP_XLIM = FOCUS_XLIM
base.CROP_YLIM = FOCUS_YLIM

SHORT_LABELS = {
    "Frontal Pole": "FP",
    "Frontal Medial Cortex": "FMC",
    "Subcallosal Cortex": "SubCal",
    "Paracingulate Gyrus": "ParaCG",
    "Cingulate Gyrus, anterior division": "aCG",
    "Postcentral Gyrus": "PostCG",
    "Superior Parietal Lobule": "SPL",
    "Supramarginal Gyrus, anterior division": "aSMG",
    "Angular Gyrus": "AG",
    "Lateral Occipital Cortex, superior division": "LOCs",
    "Lateral Occipital Cortex, inferior division": "LOCi",
    "Middle Temporal Gyrus, temporooccipital part": "MTGto",
    "Inferior Temporal Gyrus, temporooccipital part": "ITGto",
    "Temporal Occipital Fusiform Cortex": "TOF",
    "Occipital Fusiform Gyrus": "OFG",
}

ROI_PALETTE = (
    "#00a9c7",
    "#8e5bb7",
    "#2ca25f",
    "#e07a1f",
    "#d64f74",
    "#4778c8",
    "#b59b00",
    "#00a087",
    "#b05a9d",
    "#6b8e23",
)


def split_lateralized_label(label: str) -> tuple[str, str]:
    for prefix, hemisphere in (("Left ", "left"), ("Right ", "right")):
        if label.startswith(prefix):
            return hemisphere, label[len(prefix) :]
    return "unknown", label


def related_harvard_oxford_rois(
    fsaverage_ni: object,
    n_lh: int,
    flat_pts: np.ndarray,
    functional_surface_union: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    atlas = datasets.fetch_atlas_harvard_oxford(
        ATLAS_NAME,
        data_dir=str(ATLAS_DATA_DIR),
    )
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.spatialimages.SpatialImage) else nib.load(str(atlas.maps))
    atlas_native = np.rint(atlas_img.get_fdata(dtype=np.float32)).astype(np.int16)

    rsa_img = nib.load(str(base.RSA_PATH))
    mr_img = nib.load(str(base.CONJ_TSTAT_NIFTI))
    atlas_on_stats = resample_to_img(
        atlas_img,
        rsa_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_stats = np.rint(atlas_on_stats.get_fdata(dtype=np.float32)).astype(np.int16)
    rsa_mask = np.isfinite(rsa_img.get_fdata(dtype=np.float32)) & (rsa_img.get_fdata(dtype=np.float32) > 0)
    mr_mask = np.isfinite(mr_img.get_fdata(dtype=np.float32)) & (mr_img.get_fdata(dtype=np.float32) > 0)

    roi_specs: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    visible_vertices = (
        (flat_pts[:, 0] >= FOCUS_XLIM[0])
        & (flat_pts[:, 0] <= FOCUS_XLIM[1])
        & (flat_pts[:, 1] >= FOCUS_YLIM[0])
        & (flat_pts[:, 1] <= FOCUS_YLIM[1])
    )
    for roi_id, full_label in enumerate(atlas.labels):
        if roi_id == 0:
            continue
        hemisphere, anatomical_label = split_lateralized_label(str(full_label))
        roi_stats = atlas_stats == roi_id
        roi_voxels = int(roi_stats.sum())
        if roi_voxels == 0:
            continue

        rsa_voxels = int((roi_stats & rsa_mask).sum())
        mr_voxels = int((roi_stats & mr_mask).sum())
        rsa_coverage = 100.0 * rsa_voxels / roi_voxels
        mr_coverage = 100.0 * mr_voxels / roi_voxels
        if rsa_voxels == 0 and mr_voxels == 0:
            continue

        roi_native = atlas_native == roi_id
        roi_img = nib.Nifti1Image(roi_native.astype(np.uint8), atlas_img.affine, atlas_img.header)
        surface_mask = base.project_mask_to_fsaverage(roi_img, fsaverage_ni)
        if hemisphere == "left":
            surface_mask[n_lh:] = False
        elif hemisphere == "right":
            surface_mask[:n_lh] = False
        visible_functional_overlap = surface_mask & functional_surface_union & visible_vertices
        if not visible_functional_overlap.any():
            continue

        side = "L" if hemisphere == "left" else "R" if hemisphere == "right" else ""
        short_name = SHORT_LABELS.get(anatomical_label, anatomical_label)
        plot_label = f"{short_name}-{side}".strip("-")
        bilateral_key = list(SHORT_LABELS).index(anatomical_label) if anatomical_label in SHORT_LABELS else roi_id
        color = ROI_PALETTE[bilateral_key % len(ROI_PALETTE)]
        roi_specs.append(
            {
                "label": plot_label,
                "mask": surface_mask,
                "label_mask": surface_mask & visible_vertices,
                "color": color,
                "source": "Harvard-Oxford",
            }
        )
        rows.append(
            {
                "atlas": "Harvard-Oxford cortical lateralized maxprob 25%",
                "hemisphere": hemisphere,
                "roi_id": roi_id,
                "roi": anatomical_label,
                "roi_voxels": roi_voxels,
                "RSA_voxels_in_roi": rsa_voxels,
                "RSA_roi_coverage_percent": rsa_coverage,
                "Multiple_Regression_voxels_in_roi": mr_voxels,
                "Multiple_Regression_roi_coverage_percent": mr_coverage,
                "surface_vertices_overlapping_visible_functional_maps": int(visible_functional_overlap.sum()),
            }
        )

    return roi_specs, rows


def save_selection_table(rows: list[dict[str, object]]) -> Path:
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = base.OUT_DIR / "harvard_oxford_rois_overlapping_either_analysis.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def make_harvard_oxford_flatmap(
    title: str,
    texture: np.ndarray,
    vmax: float,
    colorbar_label: str,
    stem: str,
    roi_specs: list[dict[str, object]],
    flat_pts: np.ndarray,
    flat_polys: np.ndarray,
    n_lh: int,
    mesh_adjacency: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    return base.make_flatmap(
        title,
        texture,
        "hot",
        0.0,
        vmax,
        colorbar_label,
        stem,
        roi_specs,
        flat_pts,
        flat_polys,
        n_lh,
        mesh_adjacency,
        label_fontsize=10,
        plot_rect=(0.01, 0.01, 0.98, 0.80),
        colorbar_rect=(0.39, 0.865, 0.22, 0.022),
        title_y=0.965,
        title_fontsize=24,
    )


def crop_white_margin(image: Image.Image, padding: int = 8) -> Image.Image:
    """Remove export whitespace while retaining a small publication margin."""
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


def functional_overlap_metrics() -> dict[str, float | int]:
    rsa_img = nib.load(str(base.RSA_PATH))
    mr_img = nib.load(str(base.CONJ_TSTAT_NIFTI))
    if rsa_img.shape != mr_img.shape or not np.allclose(rsa_img.affine, mr_img.affine):
        raise ValueError("RSA and multiple-regression maps must share a voxel grid.")
    rsa = np.isfinite(rsa_img.get_fdata(dtype=np.float32)) & (rsa_img.get_fdata(dtype=np.float32) > 0)
    mr = np.isfinite(mr_img.get_fdata(dtype=np.float32)) & (mr_img.get_fdata(dtype=np.float32) > 0)
    shared = rsa & mr
    union = rsa | mr
    return {
        "RSA voxels": int(rsa.sum()),
        "Regression voxels": int(mr.sum()),
        "Shared voxels": int(shared.sum()),
        "Regression inside RSA (%)": 100.0 * shared.sum() / mr.sum(),
        "RSA inside regression (%)": 100.0 * shared.sum() / rsa.sum(),
        "Jaccard index (%)": 100.0 * shared.sum() / union.sum(),
    }


def save_overlap_metrics(metrics: dict[str, float | int]) -> Path:
    out_path = base.OUT_DIR / "functional_map_directional_overlap_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    return out_path


def make_overlap_table_panel(metrics: dict[str, float | int], height: int) -> Image.Image:
    width = max(390, int(height * 0.55))
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    font_path = font_manager.findfont("DejaVu Sans")
    bold_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    title_font = ImageFont.truetype(bold_path, max(24, int(height * 0.040)))
    subtitle_font = ImageFont.truetype(font_path, max(15, int(height * 0.022)))
    row_font = ImageFont.truetype(font_path, max(17, int(height * 0.025)))
    value_font = ImageFont.truetype(bold_path, max(18, int(height * 0.026)))
    note_font = ImageFont.truetype(bold_path, max(16, int(height * 0.023)))

    margin = max(20, int(width * 0.07))
    draw.text((margin, int(height * 0.06)), "Functional overlap", fill="#111111", font=title_font)
    draw.text(
        (margin, int(height * 0.115)),
        "Significant positive voxels",
        fill="#555555",
        font=subtitle_font,
    )

    rows = list(metrics.items())
    table_top = int(height * 0.19)
    row_height = int(height * 0.082)
    key_row = "Regression inside RSA (%)"
    for index, (label, value) in enumerate(rows):
        top = table_top + index * row_height
        bottom = top + row_height
        if label == key_row:
            draw.rounded_rectangle(
                (margin - 8, top + 4, width - margin + 8, bottom - 4),
                radius=8,
                fill="#e7f4f7",
            )
            draw.rectangle((margin - 8, top + 4, margin - 2, bottom - 4), fill="#00a9c7")
        draw.line((margin, bottom, width - margin, bottom), fill="#d4d4d4", width=1)
        display_label = label.replace(" (%)", "")
        display_value = f"{value:,.0f}" if isinstance(value, int) else f"{value:.1f}%"
        draw.text((margin, top + row_height * 0.28), display_label, fill="#222222", font=row_font)
        value_box = draw.textbbox((0, 0), display_value, font=value_font)
        draw.text(
            (width - margin - (value_box[2] - value_box[0]), top + row_height * 0.27),
            display_value,
            fill="#111111",
            font=value_font,
        )

    note_y = table_top + len(rows) * row_height + int(height * 0.055)
    note_lines = ["55.8% of regression voxels", "are contained in the RSA result."]
    for line_index, line in enumerate(note_lines):
        draw.text(
            (margin, note_y + line_index * int(height * 0.040)),
            line,
            fill="#006b7d",
            font=note_font,
        )
    return panel


def main() -> None:
    plt.close("all")
    cortex.database.db = cortex.database.Database()
    base.save_conjunction_tstat_nifti()
    fsaverage_ni = base.ensure_flat_surfaces()
    flat_pts, flat_polys = cortex.db.get_surf(base.SUBJECT, "flat", merge=True, nudge=True)
    n_lh = cortex.db.get_surf(base.SUBJECT, "flat", "lh")[0].shape[0]
    mesh_adjacency = base.build_mesh_adjacency(flat_polys, flat_pts.shape[0])

    rsa_texture = base.project_positive_volume(base.RSA_PATH, fsaverage_ni)
    rsa_values = rsa_texture[np.isfinite(rsa_texture)]
    rsa_vmax = max(float(np.nanpercentile(rsa_values, 99)), 1e-6)

    mr_texture = base.project_positive_volume(base.CONJ_TSTAT_NIFTI, fsaverage_ni)
    mr_values = mr_texture[np.isfinite(mr_texture)]
    mr_vmax = max(float(np.nanpercentile(mr_values, 99)), 1e-6)
    functional_surface_union = np.isfinite(rsa_texture) | np.isfinite(mr_texture)

    roi_specs, rows = related_harvard_oxford_rois(
        fsaverage_ni,
        n_lh,
        flat_pts,
        functional_surface_union,
    )
    if not roi_specs:
        raise RuntimeError("No Harvard-Oxford ROIs overlap the visible functional maps.")
    table_path = save_selection_table(rows)
    overlap_metrics = functional_overlap_metrics()
    overlap_metrics_path = save_overlap_metrics(overlap_metrics)

    pngs = [
        make_harvard_oxford_flatmap(
            "Valence & Arousal RDM RSA",
            rsa_texture,
            rsa_vmax,
            "t-value",
            "valence_arousal_rdm_rsa_pycortex_harvard_oxford_focused",
            roi_specs,
            flat_pts,
            flat_polys,
            n_lh,
            mesh_adjacency,
        ),
        make_harvard_oxford_flatmap(
            "Valence & Arousal Multiple Regression",
            mr_texture,
            mr_vmax,
            "t-value",
            "valence_arousal_multiple_regression_pycortex_harvard_oxford_focused",
            roi_specs,
            flat_pts,
            flat_polys,
            n_lh,
            mesh_adjacency,
        ),
    ]

    images = [crop_white_margin(Image.open(path)) for path in pngs]
    map_height = max(image.height for image in images)
    overlap_panel = make_overlap_table_panel(overlap_metrics, map_height)
    gap = 10
    width = sum(image.width for image in images) + overlap_panel.width + 2 * gap
    height = map_height
    combined = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        y = (height - image.height) // 2
        combined.paste(image, (x, y))
        x += image.width + gap
    combined.paste(overlap_panel, (x, 0))

    combined_path = base.OUT_DIR / "valence_arousal_rsa_and_multiple_regression_harvard_oxford_focused_with_overlap_table.png"
    combined.save(combined_path)
    print(f"Selected {len(rows)} Harvard-Oxford ROI/hemisphere parcels overlapping either analysis")
    for row in rows:
        print(
            f"{row['hemisphere']:>5s} {row['roi']}: "
            f"RSA={row['RSA_roi_coverage_percent']:.1f}%, "
            f"Multiple Regression={row['Multiple_Regression_roi_coverage_percent']:.1f}%"
        )
    print(f"Saved selection table: {table_path}")
    print(f"Saved overlap metrics: {overlap_metrics_path}")
    print(f"Saved combined figure: {combined_path}")


if __name__ == "__main__":
    main()
