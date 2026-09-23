from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter
from nilearn.image import resample_to_img


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SEARCHLIGHT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight")

UNIVARIATE_MAP = ROOT / "outputs/valence_arousal_both_positive_significant_min_t.nii.gz"
MULTIVARIATE_MAP = SEARCHLIGHT / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
MNI_TEMPLATE = SEARCHLIGHT / "visualization/pycortex/mni152.nii.gz"
OUTPUT_DIR = ROOT / "figures"

WORLD_Z_COORDINATES = (-11, -5, 1, 7, 13, 19, 31, 64)

# Deep teal-green and bright lime distinguish the two requested green categories.
UNIVARIATE_COLOR = "#007A5E"
SHARED_COLOR = "#62B44A"
MULTIVARIATE_COLOR = "#F5C400"
ROI_CMAP = ListedColormap(
    [UNIVARIATE_COLOR, SHARED_COLOR, MULTIVARIATE_COLOR],
    name="roi_comparison",
)
ROI_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], ROI_CMAP.N)
SMOOTHING_SIGMA_MM = 0.9
MASK_CONTOUR_LEVEL = 0.5


def robust_normalize(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data) & (data > 0)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32)


def world_z_to_index(image: nib.spatialimages.SpatialImage, world_z: float) -> int:
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), [0.0, 0.0, world_z])
    return int(np.clip(np.rint(voxel[2]), 0, image.shape[2] - 1))


def positive_mask(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = image.get_fdata(dtype=np.float32)
    return np.isfinite(data) & (data > 0)


def smooth_mask_on_template(
    mask: np.ndarray,
    source_image: nib.spatialimages.SpatialImage,
    template_image: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    mask_image = nib.Nifti1Image(mask.astype(np.float32), source_image.affine, source_image.header)
    high_resolution = resample_to_img(
        mask_image,
        template_image,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    ).get_fdata(dtype=np.float32)
    sigma_voxels = tuple(
        SMOOTHING_SIGMA_MM / float(zoom)
        for zoom in template_image.header.get_zooms()[:3]
    )
    return gaussian_filter(high_resolution, sigma=sigma_voxels, mode="constant")


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
    template = robust_normalize(template_img.get_fdata(dtype=np.float32))

    univariate = positive_mask(univariate_img)
    multivariate = positive_mask(multivariate_img)
    univariate_only = smooth_mask_on_template(univariate & ~multivariate, univariate_img, template_img)
    shared_mask = smooth_mask_on_template(univariate & multivariate, univariate_img, template_img)
    multivariate_only = smooth_mask_on_template(multivariate & ~univariate, univariate_img, template_img)

    figure, axes = plt.subplots(2, 4, figsize=(13.8, 9.4), facecolor="white")
    figure.subplots_adjust(
        left=0.012,
        right=0.988,
        top=0.835,
        bottom=0.145,
        wspace=0.01,
        hspace=0.16,
    )

    for axis, world_z in zip(axes.flat, WORLD_Z_COORDINATES):
        z_index = world_z_to_index(template_img, world_z)
        background = np.rot90(template[:, :, z_index])
        background = np.ma.masked_where(background <= 0.015, background)

        axis.set_facecolor("#050505")
        axis.imshow(background, cmap="gray", vmin=0.02, vmax=0.92, interpolation="bilinear")
        for mask, color in (
            (univariate_only, UNIVARIATE_COLOR),
            (multivariate_only, MULTIVARIATE_COLOR),
            (shared_mask, SHARED_COLOR),
        ):
            mask_slice = np.rot90(mask[:, :, z_index])
            if np.max(mask_slice) >= MASK_CONTOUR_LEVEL:
                axis.contourf(
                    mask_slice,
                    levels=[MASK_CONTOUR_LEVEL, 1.1],
                    colors=[color],
                    alpha=0.97,
                    antialiased=True,
                    corner_mask=True,
                )
        axis.set_title(f"z = {world_z:+d} mm", fontsize=21, fontweight="bold", pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        "Spatial Comparison of Univariate and Multivariate ROIs",
        fontsize=28,
        fontweight="bold",
        y=0.972,
    )
    figure.text(
        0.5,
        0.921,
        "Positive significant voxels on the MNI152 template",
        ha="center",
        va="center",
        fontsize=18,
        color="#3c3c3c",
    )

    colorbar_axis = figure.add_axes([0.255, 0.067, 0.49, 0.034])
    colorbar = matplotlib.colorbar.ColorbarBase(
        colorbar_axis,
        cmap=ROI_CMAP,
        norm=ROI_NORM,
        orientation="horizontal",
        ticks=[1, 2, 3],
        boundaries=[0.5, 1.5, 2.5, 3.5],
        spacing="uniform",
    )
    colorbar.ax.set_xticklabels(
        ["Univariate only", "Shared", "Multivariate only"],
        fontsize=15,
        fontweight="semibold",
    )
    colorbar.ax.tick_params(length=0, pad=7)
    colorbar.outline.set_linewidth(0.9)

    png_path = OUTPUT_DIR / "univariate_multivariate_roi_overlap_axial_smooth.png"
    svg_path = OUTPUT_DIR / "univariate_multivariate_roi_overlap_axial_smooth.svg"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)

    shared = univariate & multivariate
    print(f"Univariate voxels: {int(univariate.sum())}")
    print(f"Multivariate voxels: {int(multivariate.sum())}")
    print(f"Shared voxels: {int(shared.sum())}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


if __name__ == "__main__":
    main()
