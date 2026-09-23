"""Render panel A as overlapping positive univariate and multivariate flatmaps."""

from __future__ import annotations

import gzip
from pathlib import Path

import cortex
import cortex.database
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import nibabel as nib
import numpy as np
from nilearn import datasets, surface


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SEARCHLIGHT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")
PYCORTEX_SURFACE_DIR = Path(
    "/home/yujun/miniconda3/envs/pycortex/share/pycortex/db/fsaverage/surfaces"
)

RSA_NIFTI = SEARCHLIGHT / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
REGRESSION_NIFTI = ROOT / "outputs/valence_arousal_both_positive_significant_min_t.nii.gz"
OUT_PNG = ROOT / "figures/mricrogl_harvard_oxford_panel_a_tvalues.png"
OUT_SVG = ROOT / "figures/mricrogl_harvard_oxford_panel_a_tvalues.svg"

SURFACE_THRESHOLD = 0.01
OPACITY = 0.90
T_MIN = 2.0
T_MAX = 7.0
CMAPS = {
    "Univariate multiple regression": "Blues",
    "Multivariate RSA": "Reds",
}
BAR_LABELS = {
    "Univariate multiple regression": "Univariate multiple regression t-value",
    "Multivariate RSA": "Multivariate RSA t-value",
}


def ensure_flat_surfaces() -> object:
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    for hemisphere, side in (("lh", "left"), ("rh", "right")):
        output = PYCORTEX_SURFACE_DIR / f"flat_{hemisphere}.gii"
        if output.exists():
            continue
        source = Path(getattr(fsaverage, f"flat_{side}"))
        if source.suffix == ".gz":
            with gzip.open(source, "rb") as handle:
                gifti = nib.GiftiImage.from_bytes(handle.read())
        else:
            gifti = nib.load(str(source))
        nib.save(gifti, str(output))
    return fsaverage


def project_positive_tvalues(path: Path, fsaverage: object) -> np.ndarray:
    """Project positive t values with the manuscript's cortical-ribbon method."""
    left = surface.vol_to_surf(
        path,
        fsaverage.pial_left,
        inner_mesh=fsaverage.white_left,
        interpolation="linear",
        n_samples=7,
    )
    right = surface.vol_to_surf(
        path,
        fsaverage.pial_right,
        inner_mesh=fsaverage.white_right,
        interpolation="linear",
        n_samples=7,
    )
    texture = np.hstack([left, right])
    texture = np.asarray(texture, dtype=np.float32)
    texture[~np.isfinite(texture) | (texture <= SURFACE_THRESHOLD)] = np.nan
    return texture


def colorize_tvalues(name: str, texture: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB, opacity weights, and active vertices for one projected t map."""
    active = np.isfinite(texture)
    normalized = np.clip((texture - T_MIN) / (T_MAX - T_MIN), 0.0, 1.0)
    # Avoid nearly white low t values so blue and red effects remain legible.
    cmap_values = 0.35 + 0.65 * normalized
    rgb = cm.get_cmap(CMAPS[name])(cmap_values)[..., :3]
    strength = 0.45 + 0.55 * normalized
    return rgb.astype(np.float32), strength.astype(np.float32), active


def composite_tvalues(textures: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Blend projected maps by their displayed t-value strength at overlaps."""
    n_vertices = next(iter(textures.values())).size
    rgb = np.zeros((n_vertices, 3), dtype=np.float32)
    weighted_rgb = np.zeros_like(rgb)
    total_weight = np.zeros(n_vertices, dtype=np.float32)
    active_count = np.zeros(n_vertices, dtype=np.float32)

    for name, texture in textures.items():
        layer_rgb, weight, active = colorize_tvalues(name, texture)
        weighted_rgb[active] += layer_rgb[active] * weight[active, None]
        total_weight[active] += weight[active]
        active_count[active] += 1

    active = total_weight > 0
    rgb[active] = weighted_rgb[active] / total_weight[active, None]
    alpha = 1.0 - np.power(1.0 - OPACITY, active_count)
    return rgb, alpha


def add_colorbar(figure: plt.Figure, rect: tuple[float, float, float, float], name: str) -> None:
    axis = figure.add_axes(rect)
    mapper = cm.ScalarMappable(cmap=CMAPS[name], norm=colors.Normalize(vmin=T_MIN, vmax=T_MAX))
    mapper.set_array([])
    colorbar = figure.colorbar(mapper, cax=axis, orientation="horizontal", ticks=np.arange(T_MIN, T_MAX + 1))
    colorbar.set_label(BAR_LABELS[name], fontsize=24, fontweight="bold", labelpad=7)
    colorbar.ax.tick_params(labelsize=20, length=5, width=1.2)


def main() -> None:
    plt.close("all")
    cortex.database.db = cortex.database.Database()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fsaverage = ensure_flat_surfaces()
    textures = {
        "Univariate multiple regression": project_positive_tvalues(REGRESSION_NIFTI, fsaverage),
        "Multivariate RSA": project_positive_tvalues(RSA_NIFTI, fsaverage),
    }
    rgb, alpha = composite_tvalues(textures)
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
    axis.set_title(
        "Univariate and Multivariate Valence-Arousal Effects",
        fontsize=38,
        fontweight="bold",
        pad=10,
    )
    add_colorbar(figure, (0.16, 0.04, 0.29, 0.026), "Univariate multiple regression")
    add_colorbar(figure, (0.55, 0.04, 0.29, 0.026), "Multivariate RSA")

    figure.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    univariate_active = np.isfinite(textures["Univariate multiple regression"])
    multivariate_active = np.isfinite(textures["Multivariate RSA"])
    print(f"Univariate surface vertices: {int(univariate_active.sum())}")
    print(f"Multivariate surface vertices: {int(multivariate_active.sum())}")
    print(f"Shared surface vertices: {int((univariate_active & multivariate_active).sum())}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_SVG}")


if __name__ == "__main__":
    main()
