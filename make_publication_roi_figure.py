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


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SURFACE_OUT = Path(
    "/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight/"
    "visualization/pycortex/surface_plot_selected_outputs"
)
ROI_TABLE_ALL = ROOT / "outputs/roi_quantification/rsa_and_multiple_regression_roi_coverage_all_rows_with_harvard_oxford.csv"
POSTERIOR_RSA = SURFACE_OUT / "valence_arousal_rdm_rsa_pycortex_harvard_oxford_focused.png"
POSTERIOR_MR = SURFACE_OUT / "valence_arousal_multiple_regression_pycortex_harvard_oxford_focused.png"
ATLAS_NAME = "cortl-maxprob-thr25-2mm"
ATLAS_DATA_DIR = Path("/home/yujun/nilearn_data")
THRESHOLD_PERCENT = 25.0

LEFT_MPFC_XLIM = (-318, -245)
RIGHT_MPFC_XLIM = (245, 318)
MPFC_YLIM = (-24, 55)

MPFC_COMPONENTS = {"Frontal Medial Cortex", "Paracingulate Gyrus"}

PUBLICATION_ROWS = [
    ("Kastner", "left", "hMT", "hMT"),
    ("Kastner", "right", "hMT", "hMT"),
    ("Julian2012 body parcels", "left", "EBA", "Extrastriate body area"),
    ("Julian2012 body parcels", "right", "EBA", "Extrastriate body area"),
    (
        "Harvard-Oxford",
        "left",
        "Lateral Occipital Cortex, inferior division",
        "Inferior lateral occipital cortex",
    ),
    (
        "Harvard-Oxford",
        "right",
        "Lateral Occipital Cortex, inferior division",
        "Inferior lateral occipital cortex",
    ),
    (
        "Harvard-Oxford",
        "right",
        "Middle Temporal Gyrus, temporooccipital part",
        "Temporo-occipital middle temporal gyrus",
    ),
    ("Julian2012 face parcels", "right", "OFA", "Occipital face area"),
    ("AAL3", "left", "Frontal_Med_Orb", "Medial orbital prefrontal cortex"),
]


def crop_white_margin(image: Image.Image, padding: int = 4) -> Image.Image:
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


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    properties = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(properties), size)


def positive_mask_image(path: Path) -> nib.Nifti1Image:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return nib.Nifti1Image((np.isfinite(data) & (data > 0)).astype(np.uint8), image.affine, image.header)


def build_mpfc_roi_specs(
    fsaverage: object,
    flat_pts: np.ndarray,
    n_lh: int,
    functional_surface_union: np.ndarray,
) -> list[dict[str, object]]:
    atlas = datasets.fetch_atlas_harvard_oxford(ATLAS_NAME, data_dir=str(ATLAS_DATA_DIR))
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.spatialimages.SpatialImage) else nib.load(str(atlas.maps))
    atlas_data = np.rint(atlas_img.get_fdata(dtype=np.float32)).astype(np.int16)
    visible = (
        (
            ((flat_pts[:, 0] >= LEFT_MPFC_XLIM[0]) & (flat_pts[:, 0] <= LEFT_MPFC_XLIM[1]))
            | ((flat_pts[:, 0] >= RIGHT_MPFC_XLIM[0]) & (flat_pts[:, 0] <= RIGHT_MPFC_XLIM[1]))
        )
        & (flat_pts[:, 1] >= MPFC_YLIM[0])
        & (flat_pts[:, 1] <= MPFC_YLIM[1])
    )

    hemisphere_masks = {
        "left": np.zeros(flat_pts.shape[0], dtype=bool),
        "right": np.zeros(flat_pts.shape[0], dtype=bool),
    }
    for roi_id, full_name in enumerate(atlas.labels):
        if roi_id == 0:
            continue
        full_name = str(full_name)
        if full_name.startswith("Left "):
            hemisphere, anatomical_name = "left", full_name[5:]
        elif full_name.startswith("Right "):
            hemisphere, anatomical_name = "right", full_name[6:]
        else:
            continue
        if anatomical_name not in MPFC_COMPONENTS:
            continue

        roi_img = nib.Nifti1Image((atlas_data == roi_id).astype(np.uint8), atlas_img.affine, atlas_img.header)
        surface_mask = base.project_mask_to_fsaverage(roi_img, fsaverage)
        if hemisphere == "left":
            surface_mask[n_lh:] = False
        else:
            surface_mask[:n_lh] = False
        hemisphere_masks[hemisphere] |= surface_mask

    specs: list[dict[str, object]] = []
    for hemisphere, surface_mask in hemisphere_masks.items():
        functional_overlap = surface_mask & functional_surface_union & visible
        if not functional_overlap.any():
            continue
        side = "L" if hemisphere == "left" else "R"
        specs.append(
            {
                "label": f"mPFC-{side}",
                "mask": surface_mask,
                "label_mask": surface_mask & visible,
                "color": "#00a9c7",
                "source": "Harvard-Oxford",
            }
        )
    return specs


def render_surface_detail(
    texture: np.ndarray,
    roi_specs: list[dict[str, object]],
    flat_pts: np.ndarray,
    flat_polys: np.ndarray,
    n_lh: int,
    mesh_adjacency: tuple[np.ndarray, np.ndarray, np.ndarray],
    xlim: tuple[float, float],
    stem: str,
    vmax: float,
) -> Path:
    vertex = cortex.Vertex(texture, base.SUBJECT, cmap="hot", vmin=0.0, vmax=vmax)
    fig = cortex.quickflat.make_figure(
        vertex,
        with_curvature=True,
        with_colorbar=False,
        with_rois=False,
        with_sulci=False,
        with_labels=False,
        curvature_brightness=0.5,
        curvature_contrast=0.25,
        height=520,
    )
    ax = fig.axes[0]
    base.add_roi_boundaries_and_labels(
        ax,
        texture,
        roi_specs,
        flat_pts,
        flat_polys,
        n_lh,
        mesh_adjacency,
        label_fontsize=13,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*MPFC_YLIM)
    ax.set_aspect("equal")
    ax.set_position((0.0, 0.0, 1.0, 1.0))
    output = SURFACE_OUT / f"{stem}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output


def make_symmetric_mpfc_strip(left_path: Path, right_path: Path, target_width: int) -> Image.Image:
    left = crop_white_margin(Image.open(left_path), padding=1)
    right = crop_white_margin(Image.open(right_path), padding=1)
    tile_width = (target_width - 8) // 2
    tile_height = max(120, int(tile_width * 0.72))
    left = left.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    right = right.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    strip = Image.new("RGB", (2 * tile_width + 8, tile_height), "white")
    strip.paste(left, (0, 0))
    strip.paste(right, (tile_width + 8, 0))
    draw = ImageDraw.Draw(strip)
    draw.rectangle((0, 0, tile_width - 1, tile_height - 1), outline="#b8b8b8", width=1)
    draw.rectangle((tile_width + 8, 0, strip.width - 1, tile_height - 1), outline="#b8b8b8", width=1)
    hemi_font = load_font(max(12, int(tile_height * 0.09)), bold=True)
    draw.text((8, 6), "L", fill="#333333", font=hemi_font)
    draw.text((tile_width + 16, 6), "R", fill="#333333", font=hemi_font)
    return strip


def make_method_column(posterior_path: Path, mpfc_strip: Image.Image) -> Image.Image:
    posterior = crop_white_margin(Image.open(posterior_path), padding=3)
    inset_width = int(posterior.width * 0.65)
    inset_height = int(mpfc_strip.height * inset_width / mpfc_strip.width)
    mpfc_strip = mpfc_strip.resize((inset_width, inset_height), Image.Resampling.LANCZOS)
    label_font = load_font(max(15, int(posterior.height * 0.025)), bold=True)
    label_height = max(28, int(posterior.height * 0.045))
    column = Image.new("RGB", (posterior.width, posterior.height + label_height + inset_height + 8), "white")
    column.paste(posterior, (0, 0))
    draw = ImageDraw.Draw(column)
    label = "Medial prefrontal cortex"
    bbox = draw.textbbox((0, 0), label, font=label_font)
    draw.text(((posterior.width - (bbox[2] - bbox[0])) // 2, posterior.height + 2), label, fill="#222222", font=label_font)
    column.paste(mpfc_strip, ((posterior.width - inset_width) // 2, posterior.height + label_height))
    return column


def read_publication_rows() -> list[dict[str, object]]:
    with ROI_TABLE_ALL.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    selected: list[dict[str, object]] = []
    for atlas, hemisphere, roi, display_name in PUBLICATION_ROWS:
        matches = [
            row
            for row in all_rows
            if row["atlas"] == atlas and row["hemisphere"] == hemisphere and row["roi"] == roi
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one ROI row for {(atlas, hemisphere, roi)}, found {len(matches)}")
        row = matches[0]
        selected.append(
            {
                "Atlas": atlas.replace("Julian2012 body parcels", "Julian body").replace(
                    "Julian2012 face parcels", "Julian face"
                ),
                "Hemi": "L" if hemisphere == "left" else "R",
                "ROI": display_name,
                "RSA_percent": float(row["RSA_overlap_pct"]),
                "Regression_percent": float(row["Multiple_Regression_overlap_pct"]),
            }
        )
    return selected


def save_publication_rows(rows: list[dict[str, object]]) -> Path:
    output = ROOT / "outputs/roi_quantification/publication_roi_check_table_ge25.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Atlas", "Hemi", "ROI", "RSA_percent", "RSA_check", "Regression_percent", "Regression_check"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "RSA_check": row["RSA_percent"] >= THRESHOLD_PERCENT,
                    "Regression_check": row["Regression_percent"] >= THRESHOLD_PERCENT,
                }
            )
    return output


def make_roi_table_panel(rows: list[dict[str, object]], height: int) -> Image.Image:
    width = max(900, int(height * 1.18))
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    panel_font = load_font(max(32, int(height * 0.042)), bold=True)
    title_font = load_font(max(25, int(height * 0.034)), bold=True)
    header_font = load_font(max(17, int(height * 0.022)), bold=True)
    row_font = load_font(max(16, int(height * 0.021)))
    row_bold = load_font(max(16, int(height * 0.021)), bold=True)
    note_font = load_font(max(14, int(height * 0.018)))

    title_x = int(width * 0.11)
    draw.text((int(width * 0.025), int(height * 0.012)), "b", fill="#111111", font=panel_font)
    draw.text((title_x, int(height * 0.018)), "ROI coverage by method", fill="#111111", font=title_font)
    draw.text(
        (title_x, int(height * 0.072)),
        "Representative nonredundant parcels",
        fill="#555555",
        font=note_font,
    )

    margin = int(width * 0.035)
    table_top = int(height * 0.145)
    table_bottom = int(height * 0.855)
    row_height = (table_bottom - table_top) // (len(rows) + 1)
    columns = {
        "Atlas": margin,
        "Hemi": int(width * 0.20),
        "ROI": int(width * 0.29),
        "RSA": int(width * 0.76),
        "Regression": int(width * 0.88),
    }
    headers = ["Atlas", "Hemi", "ROI", "RSA, %", "Regression, %"]
    keys = ["Atlas", "Hemi", "ROI", "RSA", "Regression"]
    for key, header in zip(keys, headers):
        draw.text((columns[key], table_top + int(row_height * 0.25)), header, fill="#222222", font=header_font)
    draw.line((margin, table_top + row_height - 2, width - margin, table_top + row_height - 2), fill="#333333", width=2)

    for index, row in enumerate(rows):
        top = table_top + (index + 1) * row_height
        is_mpfc = row["ROI"] == "Medial orbital prefrontal cortex"
        if is_mpfc:
            draw.rectangle((margin - 8, top, width - margin + 8, top + row_height), fill="#fff2df")
        draw.line((margin, top + row_height - 1, width - margin, top + row_height - 1), fill="#d7d7d7", width=1)
        row_y = top + int(row_height * 0.27)
        draw.text((columns["Atlas"], row_y), str(row["Atlas"]), fill="#222222", font=row_font)
        draw.text((columns["Hemi"], row_y), str(row["Hemi"]), fill="#222222", font=row_font)
        draw.text((columns["ROI"], row_y), str(row["ROI"]), fill="#222222", font=row_bold if is_mpfc else row_font)
        rsa = float(row["RSA_percent"])
        regression = float(row["Regression_percent"])
        rsa_text = f"{rsa:.1f}" + ("  ✓" if rsa >= THRESHOLD_PERCENT else "")
        regression_text = f"{regression:.1f}" + ("  ✓" if regression >= THRESHOLD_PERCENT else "")
        draw.text((columns["RSA"], row_y), rsa_text, fill="#111111", font=row_bold)
        draw.text((columns["Regression"], row_y), regression_text, fill="#111111", font=row_bold)

    note_y = int(height * 0.895)
    draw.text(
        (margin, note_y),
        "✓  Significant positive voxels occupy at least 25% of the atlas ROI.",
        fill="#333333",
        font=note_font,
    )
    draw.text(
        (margin, note_y + int(height * 0.035)),
        "The medial prefrontal parcel is retained for anatomical interpretation; it does not meet 25% coverage.",
        fill="#7a4a00",
        font=note_font,
    )
    return panel


def main() -> None:
    SURFACE_OUT.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    cortex.database.db = cortex.database.Database()
    fsaverage = base.ensure_flat_surfaces()
    flat_pts, flat_polys = cortex.db.get_surf(base.SUBJECT, "flat", merge=True, nudge=True)
    n_lh = cortex.db.get_surf(base.SUBJECT, "flat", "lh")[0].shape[0]
    mesh_adjacency = base.build_mesh_adjacency(flat_polys, flat_pts.shape[0])

    rsa_texture = base.project_positive_volume(base.RSA_PATH, fsaverage)
    mr_texture = base.project_positive_volume(base.CONJ_TSTAT_NIFTI, fsaverage)
    functional_union = np.isfinite(rsa_texture) | np.isfinite(mr_texture)
    roi_specs = build_mpfc_roi_specs(fsaverage, flat_pts, n_lh, functional_union)
    rsa_vmax = float(np.nanpercentile(rsa_texture[np.isfinite(rsa_texture)], 99))
    mr_vmax = float(np.nanpercentile(mr_texture[np.isfinite(mr_texture)], 99))

    details: dict[str, Path] = {}
    for method, texture, vmax in (("rsa", rsa_texture, rsa_vmax), ("regression", mr_texture, mr_vmax)):
        details[f"{method}_left"] = render_surface_detail(
            texture,
            roi_specs,
            flat_pts,
            flat_polys,
            n_lh,
            mesh_adjacency,
            LEFT_MPFC_XLIM,
            f"{method}_mpfc_left_harvard_oxford_detail",
            vmax,
        )
        details[f"{method}_right"] = render_surface_detail(
            texture,
            roi_specs,
            flat_pts,
            flat_polys,
            n_lh,
            mesh_adjacency,
            RIGHT_MPFC_XLIM,
            f"{method}_mpfc_right_harvard_oxford_detail",
            vmax,
        )

    posterior_rsa = crop_white_margin(Image.open(POSTERIOR_RSA))
    inset_target = int(posterior_rsa.width * 0.55)
    rsa_strip = make_symmetric_mpfc_strip(details["rsa_left"], details["rsa_right"], inset_target)
    mr_strip = make_symmetric_mpfc_strip(details["regression_left"], details["regression_right"], inset_target)
    method_columns = [make_method_column(POSTERIOR_RSA, rsa_strip), make_method_column(POSTERIOR_MR, mr_strip)]

    panel_gap = 10
    panel_a_width = sum(column.width for column in method_columns) + panel_gap
    panel_a_height = max(column.height for column in method_columns)
    panel_a = Image.new("RGB", (panel_a_width, panel_a_height), "white")
    x = 0
    for column in method_columns:
        panel_a.paste(column, (x, 0))
        x += column.width + panel_gap
    draw_a = ImageDraw.Draw(panel_a)
    draw_a.text((10, 5), "a", fill="#111111", font=load_font(max(34, int(panel_a_height * 0.045)), bold=True))

    publication_rows = read_publication_rows()
    publication_csv = save_publication_rows(publication_rows)
    panel_b = make_roi_table_panel(publication_rows, panel_a_height)

    outer_gap = 18
    final = Image.new("RGB", (panel_a.width + outer_gap + panel_b.width, panel_a_height), "white")
    final.paste(panel_a, (0, 0))
    final.paste(panel_b, (panel_a.width + outer_gap, 0))
    separator_x = panel_a.width + outer_gap // 2
    ImageDraw.Draw(final).line((separator_x, 12, separator_x, panel_a_height - 12), fill="#bdbdbd", width=2)

    output = SURFACE_OUT / "publication_valence_arousal_surface_and_roi_coverage_ge25.png"
    final.save(output, dpi=(300, 300))
    print(f"mPFC atlas parcels displayed: {[spec['label'] for spec in roi_specs]}")
    print(f"Saved publication table: {publication_csv}")
    print(f"Saved publication figure: {output}")


if __name__ == "__main__":
    main()
