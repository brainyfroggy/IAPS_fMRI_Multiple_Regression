"""Render a bright categorical-mask version of the univariate/RSA flatmap."""

from __future__ import annotations

from pathlib import Path

import cortex
import cortex.database
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import surface

import generate_surface_panel_a as base


OUT_PNG = base.ROOT / "figures/mricrogl_harvard_oxford_panel_a_bright_masks.png"
OUT_SVG = base.ROOT / "figures/mricrogl_harvard_oxford_panel_a_bright_masks.svg"
OUT_CSV = base.ROOT / "outputs/roi_quantification/bright_surface_panel_a_included_aal3_parcels.csv"
MASK_ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/data/masks")
AAL3_ATLAS = MASK_ROOT / "AAL3/AAL3v1.nii.gz"
AAL3_LABELS = MASK_ROOT / "AAL3/roi_labels.csv"
COLORS = {
    "Univariate multiple regression": "#007BFF",
    "Multivariate RSA": "#FF3030",
}
OPACITY = 0.95
ROI_COLORS = ["#00D9FF", "#F5C542", "#9AFF66", "#C68CFF", "#FFB05B", "#69E6C2", "#F7A8D9"]
ROI_SMOOTH_STEPS = 8
ROI_SMOOTH_SELF_WEIGHT = 0.62
ROI_CONTOUR_LEVEL = 0.42
ROI_BORDER_LINEWIDTH = 2.4


def concise_aal3_name(roi_name: str) -> str:
    """Use compact, readable AAL3 labels while retaining each parcel's identity."""
    base_name = roi_name.rsplit("_", 1)[0]
    replacements = {
        "ACC_pre": "preACC",
        "ACC_sub": "subACC",
        "ACC_sup": "supACC",
        "Cingulate_Ant": "ACC",
        "Cingulate_Mid": "MCC",
        "Cingulate_Post": "PCC",
        "Frontal_Med_Orb": "mOFC",
        "Frontal_Med": "mPFC",
        "Frontal_Sup_Medial": "dmPFC",
        "Occipital_Inf": "Inf Occ",
        "Occipital_Mid": "Mid Occ",
        "Occipital_Sup": "Sup Occ",
        "Temporal_Inf": "ITG",
        "Temporal_Mid": "MTG",
        "Temporal_Pole_Mid": "Mid Temp Pole",
        "Temporal_Pole_Sup": "Sup Temp Pole",
        "Parietal_Inf": "IPL",
        "Parietal_Sup": "SPL",
        "Postcentral": "Postcentral",
        "Precentral": "Precentral",
        "Supp_Motor_Area": "SMA",
        "Rolandic_Oper": "Rolandic Op",
        "Heschl": "Heschl",
        "Parahippocampal": "PHG",
        "Insula": "Insula",
        "Calcarine": "Calcarine",
        "Lingual": "Lingual",
        "Fusiform": "Fusiform",
        "Angular": "Angular",
        "Precuneus": "Precuneus",
    }
    return replacements.get(base_name, base_name.replace("_", " "))


def hex_to_rgb01(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[index : index + 2], 16) for index in (0, 2, 4)]) / 255.0


def build_mesh_adjacency(flat_polys: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse vertex-neighbor representation for gentle contour smoothing."""
    edges = np.vstack(
        [
            flat_polys[:, [0, 1]],
            flat_polys[:, [1, 2]],
            flat_polys[:, [2, 0]],
        ]
    )
    edges = np.vstack([edges, edges[:, ::-1]])
    edges = np.unique(edges, axis=0)
    src, dst = edges[:, 0], edges[:, 1]
    degree = np.bincount(src, minlength=n_vertices).astype(np.float32)
    return src, dst, degree


def smooth_mask_on_mesh(mask: np.ndarray, mesh_adjacency: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Lightly smooth binary parcel membership solely for clean plotted contours."""
    src, dst, degree = mesh_adjacency
    values = mask.astype(np.float32)
    for _ in range(ROI_SMOOTH_STEPS):
        neighbor_sum = np.bincount(src, weights=values[dst], minlength=values.size)
        neighbor_mean = neighbor_sum / np.maximum(degree, 1.0)
        values = ROI_SMOOTH_SELF_WEIGHT * values + (1.0 - ROI_SMOOTH_SELF_WEIGHT) * neighbor_mean
    return values


def project_aal3_parcels(fsaverage, n_lh: int, functional_masks: dict[str, np.ndarray]) -> list[dict]:
    """Return only AAL3 parcels with at least one projected functional overlap vertex."""
    atlas_image = nib.load(str(AAL3_ATLAS))
    labels = pd.read_csv(AAL3_LABELS)
    parcels: list[dict] = []

    # Project integer parcel IDs once per hemisphere.  Nearest-most-frequent
    # sampling keeps the discrete atlas identity at each cortical vertex.
    left_ids = np.rint(
        surface.vol_to_surf(
            atlas_image,
            fsaverage.pial_left,
            inner_mesh=fsaverage.white_left,
            interpolation="nearest_most_frequent",
            n_samples=7,
        )
    ).astype(int)
    right_ids = np.rint(
        surface.vol_to_surf(
            atlas_image,
            fsaverage.pial_right,
            inner_mesh=fsaverage.white_right,
            interpolation="nearest_most_frequent",
            n_samples=7,
        )
    ).astype(int)
    atlas_ids = np.hstack([left_ids, right_ids])

    for row in labels.itertuples(index=False):
        surface_mask = atlas_ids == row.id
        hemisphere = row.roi_name.rsplit("_", 1)[-1]
        if hemisphere == "L":
            surface_mask[n_lh:] = False
        elif hemisphere == "R":
            surface_mask[:n_lh] = False

        blue_count = int(np.count_nonzero(surface_mask & functional_masks["Univariate multiple regression"]))
        red_count = int(np.count_nonzero(surface_mask & functional_masks["Multivariate RSA"]))
        if blue_count == 0 and red_count == 0:
            continue
        parcels.append(
            {
                "aal3_id": int(row.id),
                "roi_name": row.roi_name,
                "label": f"{concise_aal3_name(row.roi_name)}-{hemisphere}",
                "hemisphere": hemisphere,
                "mask": surface_mask,
                "univariate_overlap_vertices": blue_count,
                "multivariate_overlap_vertices": red_count,
                "either_overlap_vertices": int(np.count_nonzero(surface_mask & (functional_masks["Univariate multiple regression"] | functional_masks["Multivariate RSA"]))),
            }
        )
    return parcels


def add_aal3_boundaries_and_labels(
    axis,
    figure,
    parcels: list[dict],
    flat_pts: np.ndarray,
    flat_polys: np.ndarray,
    mesh_adjacency: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Draw smoothed colored parcel contours and collision-aware white labels."""
    triangulation = mtri.Triangulation(flat_pts[:, 0], flat_pts[:, 1], flat_polys)
    label_specs: list[dict] = []
    for index, parcel in enumerate(parcels):
        smoothed = smooth_mask_on_mesh(parcel["mask"], mesh_adjacency)
        axis.tricontour(
            triangulation,
            smoothed,
            levels=[ROI_CONTOUR_LEVEL],
            colors=[ROI_COLORS[index % len(ROI_COLORS)]],
            linewidths=ROI_BORDER_LINEWIDTH,
            alpha=1.0,
            zorder=50,
        )
        vertices = np.flatnonzero(parcel["mask"])
        flat_xy = flat_pts[:, :2]
        centroid = flat_xy[vertices].mean(axis=0)
        anchor_index = vertices[np.argmin(np.sum((flat_xy[vertices] - centroid) ** 2, axis=1))]
        label_specs.append(
            {
                "text": parcel["label"],
                "anchor": flat_xy[anchor_index],
                "color": ROI_COLORS[index % len(ROI_COLORS)],
            }
        )

    # Try a compact series of positions, then connect moved labels back to their parcels.
    x_span = np.ptp(flat_pts[:, 0])
    y_span = np.ptp(flat_pts[:, 1])
    offsets = [(0, 0), (0.035, 0), (-0.035, 0), (0, 0.045), (0, -0.045), (0.055, 0.04), (-0.055, 0.04), (0.055, -0.04), (-0.055, -0.04), (0.085, 0), (-0.085, 0), (0, 0.075), (0, -0.075)]
    placed_boxes = []
    renderer = figure.canvas.get_renderer()
    for spec in sorted(label_specs, key=lambda item: (-item["anchor"][1], item["anchor"][0])):
        chosen = None
        chosen_text = None
        for x_factor, y_factor in offsets:
            position = spec["anchor"] + np.array([x_factor * x_span, y_factor * y_span])
            text = axis.text(
                *position,
                spec["text"],
                color="white",
                fontsize=15,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=60,
                path_effects=[pe.withStroke(linewidth=4.0, foreground="black")],
            )
            figure.canvas.draw()
            bounding_box = text.get_window_extent(renderer).expanded(1.22, 1.55)
            if not any(bounding_box.overlaps(existing) for existing in placed_boxes):
                chosen = position
                chosen_text = text
                placed_boxes.append(bounding_box)
                break
            text.remove()
        if chosen_text is None:
            chosen = spec["anchor"] + np.array([0.11 * x_span, 0.06 * y_span])
            chosen_text = axis.text(
                *chosen,
                spec["text"],
                color="white",
                fontsize=15,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=60,
                path_effects=[pe.withStroke(linewidth=4.0, foreground="black")],
            )
            figure.canvas.draw()
            placed_boxes.append(chosen_text.get_window_extent(renderer).expanded(1.22, 1.55))
        if not np.allclose(chosen, spec["anchor"]):
            axis.annotate(
                "",
                xy=spec["anchor"],
                xytext=chosen,
                arrowprops={"arrowstyle": "-", "color": spec["color"], "lw": 1.6},
                zorder=55,
            )


def main() -> None:
    plt.close("all")
    cortex.database.db = cortex.database.Database()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fsaverage = base.ensure_flat_surfaces()
    textures = {
        "Univariate multiple regression": base.project_positive_tvalues(base.REGRESSION_NIFTI, fsaverage),
        "Multivariate RSA": base.project_positive_tvalues(base.RSA_NIFTI, fsaverage),
    }
    masks = {name: np.isfinite(texture) for name, texture in textures.items()}
    names = list(masks)
    mask_stack = np.vstack([masks[name] for name in names]).astype(np.float32)
    color_stack = np.vstack([hex_to_rgb01(COLORS[name]) for name in names])
    active_count = mask_stack.sum(axis=0)
    rgb = np.zeros((active_count.size, 3), dtype=np.float32)
    active = active_count > 0
    rgb[active] = (mask_stack.T @ color_stack)[active] / active_count[active, None]
    alpha = 1.0 - np.power(1.0 - OPACITY, active_count)

    vertex_rgb = cortex.VertexRGB(
        cortex.Vertex(rgb[:, 0], "fsaverage", vmin=0, vmax=1),
        cortex.Vertex(rgb[:, 1], "fsaverage", vmin=0, vmax=1),
        cortex.Vertex(rgb[:, 2], "fsaverage", vmin=0, vmax=1),
        alpha=alpha,
    )
    figure = cortex.quickflat.make_figure(
        vertex_rgb,
        with_curvature=True,
        with_colorbar=False,
        with_rois=False,
        with_sulci=False,
        with_labels=False,
        curvature_brightness=0.56,
        curvature_contrast=0.24,
        height=1000,
    )
    axis = figure.axes[0]
    flat_lh, flat_rh = cortex.database.db.get_surf("fsaverage", "flat", merge=False)
    flat_pts = np.vstack([flat_lh[0], flat_rh[0]])
    flat_polys = np.vstack([flat_lh[1], flat_rh[1] + len(flat_lh[0])])
    mesh_adjacency = build_mesh_adjacency(flat_polys, len(flat_pts))
    parcels = project_aal3_parcels(fsaverage, len(flat_lh[0]), masks)
    add_aal3_boundaries_and_labels(axis, figure, parcels, flat_pts, flat_polys, mesh_adjacency)

    axis.set_title(
        "Univariate and Multivariate Valence-Arousal Effects",
        fontsize=38,
        fontweight="bold",
        pad=10,
    )
    legend = axis.legend(
        handles=[
            mpatches.Patch(facecolor=COLORS[name], edgecolor="none", alpha=OPACITY, label=name)
            for name in names
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        fontsize=28,
        handlelength=1.6,
        labelspacing=1.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "aal3_id": parcel["aal3_id"],
                "roi_name": parcel["roi_name"],
                "label": parcel["label"],
                "hemisphere": parcel["hemisphere"],
                "univariate_overlap_vertices": parcel["univariate_overlap_vertices"],
                "multivariate_overlap_vertices": parcel["multivariate_overlap_vertices"],
                "either_overlap_vertices": parcel["either_overlap_vertices"],
            }
            for parcel in parcels
        ]
    ).sort_values(["hemisphere", "roi_name"]).to_csv(OUT_CSV, index=False)
    figure.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_SVG}")
    print(f"Saved {OUT_CSV}")
    print(f"Included {len(parcels)} AAL3 parcels")


if __name__ == "__main__":
    main()
