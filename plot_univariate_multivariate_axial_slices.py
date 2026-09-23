from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn.image import resample_to_img


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SEARCHLIGHT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")

UNIVARIATE_MAP = ROOT / "outputs/valence_arousal_both_positive_significant_min_t.nii.gz"
MULTIVARIATE_MAP = SEARCHLIGHT / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
MNI_TEMPLATE = SEARCHLIGHT / "visualization/pycortex/mni152.nii.gz"
OUTPUT_DIR = ROOT / "figures"

WORLD_Z_COORDINATES = (-11, -5, 1, 7, 13, 19, 31, 64)

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "significant_blue",
    ["#19b5fe", "#006bd6", "#001b8e"],
)
RED_CMAP = LinearSegmentedColormap.from_list(
    "significant_red",
    ["#ff5a47", "#e0002b", "#850000"],
)


def robust_normalize(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data) & (data > 0)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    normalized = (data - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def world_z_to_index(image: nib.spatialimages.SpatialImage, world_z: float) -> int:
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), [0.0, 0.0, world_z])
    return int(np.clip(np.rint(voxel[2]), 0, image.shape[2] - 1))


def positive_stat_data(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = image.get_fdata(dtype=np.float32)
    return np.where(np.isfinite(data) & (data > 0), data, np.nan).astype(np.float32)


def draw_slice_row(
    axes: list[plt.Axes],
    template: np.ndarray,
    stat_data: np.ndarray,
    stat_image: nib.spatialimages.SpatialImage,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
) -> matplotlib.image.AxesImage:
    last_image = None
    for axis, world_z in zip(axes, WORLD_Z_COORDINATES):
        z_index = world_z_to_index(stat_image, world_z)
        background = np.rot90(template[:, :, z_index])
        overlay = np.ma.masked_invalid(np.rot90(stat_data[:, :, z_index]))

        axis.set_facecolor("#050505")
        background_masked = np.ma.masked_where(background <= 0.015, background)
        axis.imshow(
            background_masked,
            cmap="gray",
            vmin=0.02,
            vmax=0.92,
            interpolation="bilinear",
        )
        last_image = axis.imshow(
            overlay,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            alpha=0.96,
        )
        axis.set_title(f"z = {world_z:+d} mm", fontsize=16, fontweight="semibold", pad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    if last_image is None:
        raise RuntimeError("No slices were rendered.")
    return last_image


def add_compact_colorbar(
    figure: plt.Figure,
    grid_spec: matplotlib.gridspec.SubplotSpec,
    image: matplotlib.image.AxesImage,
    label: str,
) -> None:
    holder = figure.add_subplot(grid_spec)
    holder.axis("off")
    colorbar_axis = holder.inset_axes([0.365, 0.48, 0.27, 0.25])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(label, fontsize=16, fontweight="semibold", labelpad=5)
    colorbar.ax.tick_params(labelsize=13, length=4, width=1)
    colorbar.outline.set_linewidth(0.8)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    univariate_img = nib.load(str(UNIVARIATE_MAP))
    multivariate_img = nib.load(str(MULTIVARIATE_MAP))
    if univariate_img.shape != multivariate_img.shape or not np.allclose(
        univariate_img.affine,
        multivariate_img.affine,
    ):
        raise ValueError("The univariate and multivariate maps do not share a voxel grid.")

    template_img = nib.load(str(MNI_TEMPLATE))
    template_resampled = resample_to_img(
        template_img,
        univariate_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    template = robust_normalize(template_resampled.get_fdata(dtype=np.float32))

    univariate = positive_stat_data(univariate_img)
    multivariate = positive_stat_data(multivariate_img)
    univariate_values = univariate[np.isfinite(univariate)]
    multivariate_values = multivariate[np.isfinite(multivariate)]
    univariate_vmin = float(np.min(univariate_values))
    multivariate_vmin = float(np.min(multivariate_values))
    univariate_vmax = float(np.percentile(univariate_values, 99.5))
    multivariate_vmax = float(np.percentile(multivariate_values, 99.5))

    figure = plt.figure(figsize=(22, 12.0), facecolor="white")
    grid = figure.add_gridspec(
        4,
        len(WORLD_Z_COORDINATES),
        height_ratios=(1.0, 0.16, 1.0, 0.16),
        left=0.025,
        right=0.985,
        bottom=0.055,
        top=0.79,
        wspace=0.035,
        hspace=0.72,
    )

    univariate_axes = [figure.add_subplot(grid[0, column]) for column in range(len(WORLD_Z_COORDINATES))]
    multivariate_axes = [figure.add_subplot(grid[2, column]) for column in range(len(WORLD_Z_COORDINATES))]

    univariate_image = draw_slice_row(
        univariate_axes,
        template,
        univariate,
        univariate_img,
        BLUE_CMAP,
        univariate_vmin,
        univariate_vmax,
    )
    multivariate_image = draw_slice_row(
        multivariate_axes,
        template,
        multivariate,
        multivariate_img,
        RED_CMAP,
        multivariate_vmin,
        multivariate_vmax,
    )

    add_compact_colorbar(
        figure,
        grid[1, :],
        univariate_image,
        "Minimum t-value across valence and arousal coefficients",
    )
    add_compact_colorbar(
        figure,
        grid[3, :],
        multivariate_image,
        "tRSA t-value",
    )

    figure.suptitle(
        "Valence and Arousal Effects Across Methods",
        fontsize=29,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        "Positive significant voxels on the MNI152 template",
        ha="center",
        va="center",
        fontsize=18,
        color="#3f3f3f",
    )

    univariate_top = univariate_axes[0].get_position().y1
    multivariate_top = multivariate_axes[0].get_position().y1
    figure.text(
        0.026,
        univariate_top + 0.070,
        "Univariate multiple regression",
        fontsize=22,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.026,
        univariate_top + 0.043,
        "Positive conjunction of valence and arousal coefficients",
        fontsize=15,
        color="#444444",
        ha="left",
    )
    figure.text(
        0.026,
        multivariate_top + 0.070,
        "Multivariate tRSA",
        fontsize=22,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.026,
        multivariate_top + 0.043,
        "Valence-arousal representational similarity analysis",
        fontsize=15,
        color="#444444",
        ha="left",
    )

    png_path = OUTPUT_DIR / "combined_univariate_multivariate_positive_axial_slices_vibrant.png"
    svg_path = OUTPUT_DIR / "combined_univariate_multivariate_positive_axial_slices_vibrant.svg"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)

    print(f"Univariate range: {univariate_vmin:.3f} to {univariate_vmax:.3f}")
    print(f"Multivariate range: {multivariate_vmin:.3f} to {multivariate_vmax:.3f}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


if __name__ == "__main__":
    main()
