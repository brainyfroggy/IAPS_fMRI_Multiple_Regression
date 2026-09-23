from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path("/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_multiple_reg")
SLICE_DIR = ROOT / "figures/mricrogl_axial_slices"
OUTPUT_DIR = ROOT / "figures"

SLICES = (
    (-8, "z_neg8.png"),
    (-4, "z_neg4.png"),
    (0, "z_pos0.png"),
    (4, "z_pos4.png"),
    (8, "z_pos8.png"),
    (12, "z_pos12.png"),
    (16, "z_pos16.png"),
    (66, "z_pos66.png"),
)

# These nodes reproduce MRIcroGL's bundled 5winter and 4hot color tables.
WINTER_CMAP = LinearSegmentedColormap.from_list(
    "mricrogl_5winter",
    [(0.0, "#0000ff"), (128 / 255, "#0080c4"), (1.0, "#00ff80")],
)
HOT_CMAP = LinearSegmentedColormap.from_list(
    "mricrogl_4hot",
    [
        (0.0, "#030000"),
        (95 / 255, "#ff0000"),
        (191 / 255, "#ffff00"),
        (1.0, "#ffffff"),
    ],
)


def add_colorbar(
    figure: plt.Figure,
    bounds: tuple[float, float, float, float],
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    title: str,
    label: str,
) -> None:
    axis = figure.add_axes(bounds)
    colorbar = ColorbarBase(
        axis,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        orientation="horizontal",
    )
    axis.set_title(title, fontsize=21, fontweight="bold", pad=10)
    colorbar.set_label(label, fontsize=18, fontweight="semibold", labelpad=7)
    colorbar.set_ticks([2, 3, 4, 5, 6, 7])
    colorbar.ax.tick_params(labelsize=16, length=5, width=1.0)
    colorbar.outline.set_linewidth(0.9)


def main() -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.8, 10.1), facecolor="white")
    figure.subplots_adjust(
        left=0.018,
        right=0.982,
        top=0.83,
        bottom=0.205,
        wspace=0.012,
        hspace=0.13,
    )

    for axis, (z_coordinate, filename) in zip(axes.flat, SLICES):
        image = mpimg.imread(SLICE_DIR / filename)
        axis.imshow(image, interpolation="lanczos")
        z_label = "0" if z_coordinate == 0 else f"{z_coordinate:+d}"
        axis.set_title(
            f"z = {z_label} mm",
            fontsize=25,
            fontweight="bold",
            pad=6,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        "Univariate and Multivariate Valence-Arousal Effects",
        fontsize=28,
        fontweight="bold",
        y=0.973,
    )
    add_colorbar(
        figure,
        (0.13, 0.075, 0.31, 0.026),
        WINTER_CMAP,
        2.0,
        7.0,
        "Univariate multiple regression",
        "minimum t across valence and arousal",
    )
    add_colorbar(
        figure,
        (0.56, 0.075, 0.31, 0.026),
        HOT_CMAP,
        2.0,
        7.0,
        "Multivariate RSA",
        "t-value",
    )

    png_path = OUTPUT_DIR / "mricrogl_mni152_univariate_multivariate_axial.png"
    svg_path = OUTPUT_DIR / "mricrogl_mni152_univariate_multivariate_axial.svg"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


if __name__ == "__main__":
    main()
