from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import cortex
import cortex.database
import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
from nilearn import datasets, surface
from PIL import Image


SUBJECT = "fsaverage"
HEIGHT = 1024
CROP_XLIM = (-185, 185)
CROP_YLIM = (-138, 138)
MASK_INTERPOLATION = "nearest_most_frequent"
VALUE_INTERPOLATION = "linear"
SAVE_DPI = 300
ROI_BORDER_LINEWIDTH = 2.2
ROI_SMOOTH_STEPS = 8
ROI_SMOOTH_SELF_WEIGHT = 0.62
ROI_CONTOUR_LEVEL = 0.42

SEARCHLIGHT_DIR = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")
REG_DIR = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
PYCO_DIR = SEARCHLIGHT_DIR / "visualization/pycortex"
OUT_DIR = PYCO_DIR / "surface_plot_selected_outputs"
REG_OUT_DIR = REG_DIR / "outputs"
MASK_DIR = Path("/mnt/n/Experimental_Data/yujunchen/projects/data/masks")
FS_SUBJ = Path("/usr/local/freesurfer/7.4.1/subjects/fsaverage")
PYCO_FSAVG_SURF_DIR = Path("/home/yujun/miniconda3/envs/pycortex/share/pycortex/db/fsaverage/surfaces")

RSA_PATH = SEARCHLIGHT_DIR / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
VAL_T_PATH = REG_DIR / "maps/group_t_beta_valence_fdr05_cluster50.npy"
ARO_T_PATH = REG_DIR / "maps/group_t_beta_arousal_fdr05_cluster50.npy"
REF_NIFTI = SEARCHLIGHT_DIR / "outputs/tBrainmap/final_results/tmap_beh60_all_significant_fdr05_fisherz_cluster_gt50.nii.gz"
CONJ_TSTAT_NIFTI = REG_OUT_DIR / "valence_arousal_both_positive_significant_min_t.nii.gz"


def ensure_flat_surfaces() -> object:
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    for hemi, side in [("lh", "left"), ("rh", "right")]:
        out_path = PYCO_FSAVG_SURF_DIR / f"flat_{hemi}.gii"
        if out_path.exists():
            continue
        gii_path = Path(getattr(fsaverage, f"flat_{side}"))
        if gii_path.suffix == ".gz":
            with gzip.open(gii_path, "rb") as f:
                gii = nib.GiftiImage.from_bytes(f.read())
        else:
            gii = nib.load(str(gii_path))
        nib.save(gii, str(out_path))
        print(f"Saved missing flat surface: {out_path}")
    return datasets.fetch_surf_fsaverage("fsaverage")


def save_conjunction_tstat_nifti() -> None:
    val_t = np.load(VAL_T_PATH)
    aro_t = np.load(ARO_T_PATH)
    ref = nib.load(str(REF_NIFTI))
    if ref.shape != val_t.shape:
        raise ValueError(f"Reference shape {ref.shape} does not match regression maps {val_t.shape}")

    both_positive = (val_t > 0) & (aro_t > 0)
    min_t = np.where(both_positive, np.minimum(val_t, aro_t), 0.0).astype(np.float32)
    REG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(min_t, ref.affine, ref.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, str(CONJ_TSTAT_NIFTI))
    print(
        f"Saved {CONJ_TSTAT_NIFTI}: voxels={int(np.count_nonzero(min_t))}, "
        f"min={float(min_t[min_t > 0].min()):.3f}, max={float(min_t.max()):.3f}"
    )


def project_positive_volume(path: Path, fsaverage_ni: object) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    nonzero_img = nib.Nifti1Image((data > 0).astype(np.float32), img.affine, img.header)
    tex_l = surface.vol_to_surf(str(path), fsaverage_ni.pial_left, interpolation=VALUE_INTERPOLATION)
    tex_r = surface.vol_to_surf(str(path), fsaverage_ni.pial_right, interpolation=VALUE_INTERPOLATION)
    texture = np.hstack([tex_l, tex_r]).astype(float)
    mask_l = surface.vol_to_surf(nonzero_img, fsaverage_ni.pial_left, interpolation=MASK_INTERPOLATION)
    mask_r = surface.vol_to_surf(nonzero_img, fsaverage_ni.pial_right, interpolation=MASK_INTERPOLATION)
    mask = np.hstack([mask_l, mask_r]) > 0
    texture[~mask] = np.nan
    texture[texture <= 0] = np.nan
    return texture


def project_mask_to_fsaverage(mask_img_or_path: object, fsaverage_ni: object) -> np.ndarray:
    try:
        tex_l = surface.vol_to_surf(mask_img_or_path, fsaverage_ni.pial_left, interpolation=MASK_INTERPOLATION)
        tex_r = surface.vol_to_surf(mask_img_or_path, fsaverage_ni.pial_right, interpolation=MASK_INTERPOLATION)
    except TypeError:
        tex_l = surface.vol_to_surf(str(mask_img_or_path), fsaverage_ni.pial_left, interpolation=MASK_INTERPOLATION)
        tex_r = surface.vol_to_surf(str(mask_img_or_path), fsaverage_ni.pial_right, interpolation=MASK_INTERPOLATION)
    return np.hstack([tex_l, tex_r]) > 0


def build_roi_specs(fsaverage_ni: object, flat_pts: np.ndarray, n_lh: int) -> list[dict[str, object]]:
    n_vertices = flat_pts.shape[0]
    roi_specs: list[dict[str, object]] = []

    kastner_path = MASK_DIR / "kastner_dict.pkl"
    if kastner_path.exists():
        with open(kastner_path, "rb") as f:
            kastner = pickle.load(f)
        groups = {
            "V1": ["V1v", "V1d"],
            "V2": ["V2v", "V2d"],
            "V3": ["V3v", "V3d"],
            "V3A": ["V3a"],
            "V3B": ["V3b"],
            "V4": ["hV4"],
            "LO": ["LO1", "LO2"],
            "IPS": ["IPS"],
        }
        colors = {
            "V1": "#e41a1c",
            "V2": "#ff7f00",
            "V3": "#f1c40f",
            "V3A": "#00a6d6",
            "V3B": "#4daf4a",
            "V4": "#a65628",
            "LO": "#8e44ad",
            "IPS": "#00bcd4",
        }
        for label, keys in groups.items():
            mask = np.zeros(n_vertices, dtype=bool)
            for key in keys:
                if key in kastner:
                    mask |= project_mask_to_fsaverage(kastner[key], fsaverage_ni)
            if mask.any():
                roi_specs.append({"label": label, "mask": mask, "color": colors[label], "source": "Kastner"})

    mt_v5_path = MASK_DIR / "Kastner/MT.img"
    if mt_v5_path.exists():
        mt_v5_mask = project_mask_to_fsaverage(str(mt_v5_path), fsaverage_ni)
        if mt_v5_mask.any():
            roi_specs.append({"label": "MT", "mask": mt_v5_mask, "color": "#2ca02c", "source": "Kastner"})

    julian_base = MASK_DIR / "Julian2012"
    julian_specs = {
        "pSTS": ["face_parcels/face_parcels/lSTS.img", "face_parcels/face_parcels/rSTS.img"],
    }
    julian_colors = {
        "pSTS": "#1f77b4",
    }
    for label, rel_paths in julian_specs.items():
        mask = np.zeros(n_vertices, dtype=bool)
        for rel_path in rel_paths:
            parcel_path = julian_base / rel_path
            if parcel_path.exists():
                mask |= project_mask_to_fsaverage(str(parcel_path), fsaverage_ni)
        if mask.any():
            roi_specs.append({"label": label, "mask": mask, "color": julian_colors[label], "source": "Julian2012"})

    aparc_map = {}
    labels_lh, _, names_lh = nib.freesurfer.read_annot(str(FS_SUBJ / "label/lh.aparc.annot"))
    labels_rh, _, names_rh = nib.freesurfer.read_annot(str(FS_SUBJ / "label/rh.aparc.annot"))
    names_lh = [n.decode() if isinstance(n, bytes) else n for n in names_lh]
    names_rh = [n.decode() if isinstance(n, bytes) else n for n in names_rh]
    colors = plt.get_cmap("tab20", len(aparc_map))
    for idx, (aparc_name, label) in enumerate(aparc_map.items()):
        mask = np.zeros(n_vertices, dtype=bool)
        if aparc_name in names_lh:
            mask[:n_lh] = labels_lh == names_lh.index(aparc_name)
        if aparc_name in names_rh:
            mask[n_lh:] = labels_rh == names_rh.index(aparc_name)
        if mask.any():
            roi_specs.append({"label": label, "mask": mask, "color": colors(idx), "source": "aparc"})
    return roi_specs


def build_mesh_adjacency(flat_polys: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build vertex-neighbor arrays for lightweight smoothing on the surface mesh."""
    edges = np.vstack(
        [
            flat_polys[:, [0, 1]],
            flat_polys[:, [1, 2]],
            flat_polys[:, [2, 0]],
        ]
    )
    bidirectional = np.vstack([edges, edges[:, ::-1]]).astype(np.int64, copy=False)
    src = bidirectional[:, 0]
    dst = bidirectional[:, 1]
    degree = np.bincount(src, minlength=n_vertices).astype(np.float32)
    degree[degree == 0] = 1.0
    return src, dst, degree


def smooth_mask_on_mesh(
    mask: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    """Smooth a binary ROI mask over neighboring surface vertices before contouring."""
    values = mask.astype(np.float32)
    for _ in range(ROI_SMOOTH_STEPS):
        neighbor_sum = np.zeros_like(values, dtype=np.float32)
        np.add.at(neighbor_sum, src, values[dst])
        neighbor_mean = neighbor_sum / degree
        values = ROI_SMOOTH_SELF_WEIGHT * values + (1.0 - ROI_SMOOTH_SELF_WEIGHT) * neighbor_mean
    return values


def add_roi_boundaries_and_labels(
    ax: plt.Axes,
    texture: np.ndarray,
    roi_specs: list[dict[str, object]],
    flat_pts: np.ndarray,
    flat_polys: np.ndarray,
    n_lh: int,
    mesh_adjacency: tuple[np.ndarray, np.ndarray, np.ndarray],
    label_fontsize: int = 17,
) -> None:
    src, dst, degree = mesh_adjacency
    triangulation = mtri.Triangulation(flat_pts[:, 0], flat_pts[:, 1], flat_polys)

    for spec in roi_specs:
        mask = spec["mask"]

        smooth_values = smooth_mask_on_mesh(mask, src, dst, degree)
        if np.nanmax(smooth_values) > ROI_CONTOUR_LEVEL:
            ax.tricontour(
                triangulation,
                smooth_values,
                levels=[ROI_CONTOUR_LEVEL],
                colors=[spec["color"]],
                linewidths=ROI_BORDER_LINEWIDTH,
                alpha=0.95 if spec["source"] != "aparc" else 0.75,
                zorder=50,
            )

        label_mask = np.asarray(spec.get("label_mask", mask), dtype=bool)
        hemi_masks = [label_mask.copy(), label_mask.copy()]
        hemi_masks[0][n_lh:] = False
        hemi_masks[1][:n_lh] = False
        for hemi_mask in hemi_masks:
            if hemi_mask.sum() == 0:
                continue
            pts = flat_pts[hemi_mask]
            if len(pts) == 0:
                continue
            planar_pts = pts[:, :2]
            centroid = planar_pts.mean(axis=0)
            center_vertex = planar_pts[np.argmin(np.sum((planar_pts - centroid) ** 2, axis=1))]
            cx, cy = center_vertex
            ax.text(
                cx,
                cy,
                spec["label"],
                color="white",
                fontsize=label_fontsize if spec["source"] != "aparc" else label_fontsize - 2,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=60,
                clip_on=True,
                path_effects=[pe.withStroke(linewidth=4.5, foreground="black")],
            )


def make_flatmap(
    name: str,
    texture: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    stem: str,
    roi_specs: list[dict[str, object]],
    flat_pts: np.ndarray,
    flat_polys: np.ndarray,
    n_lh: int,
    mesh_adjacency: tuple[np.ndarray, np.ndarray, np.ndarray],
    label_fontsize: int = 17,
    plot_rect: tuple[float, float, float, float] | None = None,
    colorbar_rect: tuple[float, float, float, float] = (0.28, 0.90, 0.44, 0.035),
    title_y: float = 0.985,
    title_fontsize: int = 28,
) -> Path:
    vertex_data = cortex.Vertex(texture, SUBJECT, cmap=cmap_name, vmin=vmin, vmax=vmax)
    fig = cortex.quickflat.make_figure(
        vertex_data,
        with_curvature=True,
        with_colorbar=False,
        with_rois=False,
        with_sulci=False,
        with_labels=False,
        curvature_brightness=0.5,
        curvature_contrast=0.25,
        height=HEIGHT,
    )
    ax = fig.axes[0]
    add_roi_boundaries_and_labels(
        ax,
        texture,
        roi_specs,
        flat_pts,
        flat_polys,
        n_lh,
        mesh_adjacency,
        label_fontsize=label_fontsize,
    )
    ax.set_xlim(*CROP_XLIM)
    ax.set_ylim(*CROP_YLIM)
    ax.set_aspect("equal")
    if plot_rect is not None:
        ax.set_position(plot_rect)
    fig.text(0.5, title_y, name, ha="center", va="top", fontsize=title_fontsize, fontweight="bold")

    cbar_ax = fig.add_axes(colorbar_rect)
    sm = cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"{stem}.png"
    out_svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def main() -> None:
    plt.close("all")
    cortex.database.db = cortex.database.Database()
    save_conjunction_tstat_nifti()
    fsaverage_ni = ensure_flat_surfaces()
    flat_pts, flat_polys = cortex.db.get_surf(SUBJECT, "flat", merge=True, nudge=True)
    n_lh = cortex.db.get_surf(SUBJECT, "flat", "lh")[0].shape[0]
    mesh_adjacency = build_mesh_adjacency(flat_polys, flat_pts.shape[0])
    roi_specs = build_roi_specs(fsaverage_ni, flat_pts, n_lh)
    print(f"Prepared {len(roi_specs)} ROI masks")

    rsa_texture = project_positive_volume(RSA_PATH, fsaverage_ni)
    rsa_vals = rsa_texture[np.isfinite(rsa_texture)]
    rsa_vmax = max(float(np.nanpercentile(rsa_vals, 99)), 1e-6)

    conj_texture = project_positive_volume(CONJ_TSTAT_NIFTI, fsaverage_ni)
    conj_vals = conj_texture[np.isfinite(conj_texture)]
    conj_vmax = max(float(np.nanpercentile(conj_vals, 99)), 1e-6)

    pngs = [
        make_flatmap(
            "Behavioral RDM RSA: tmap_beh60_p05_v2",
            rsa_texture,
            "hot",
            0.0,
            rsa_vmax,
            "t-value",
            "tmap_beh60_p05_v2_pycortex_flatmap",
            roi_specs,
            flat_pts,
            flat_polys,
            n_lh,
            mesh_adjacency,
        ),
        make_flatmap(
            "Valence & Arousal Both Positive Significant",
            conj_texture,
            "hot",
            0.0,
            conj_vmax,
            "min(valence t, arousal t)",
            "valence_arousal_both_positive_significant_min_t_pycortex_flatmap",
            roi_specs,
            flat_pts,
            flat_polys,
            n_lh,
            mesh_adjacency,
        ),
    ]

    imgs = [Image.open(p).convert("RGB") for p in pngs]
    width = max(im.width for im in imgs)
    height = sum(im.height for im in imgs)
    combined = Image.new("RGB", (width, height), "white")
    y = 0
    for im in imgs:
        x = (width - im.width) // 2
        combined.paste(im, (x, y))
        y += im.height
    combined_path = OUT_DIR / "tmap_beh60_and_valence_arousal_both_positive_min_t_pycortex_combined.png"
    combined.save(combined_path)
    print(f"Saved {combined_path}")
    print(f"RSA surface vertices: {len(rsa_vals)}, min/max: {float(np.nanmin(rsa_vals)):.3f}/{float(np.nanmax(rsa_vals)):.3f}")
    print(
        f"Both-positive conjunction t surface vertices: {len(conj_vals)}, "
        f"min/max: {float(np.nanmin(conj_vals)):.3f}/{float(np.nanmax(conj_vals)):.3f}"
    )


if __name__ == "__main__":
    main()
