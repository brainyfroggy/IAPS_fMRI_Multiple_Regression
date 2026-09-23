"""Export the Harvard-Oxford table as a standalone panel B figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

import make_mricrogl_roi_coverage_figure as base


def main() -> None:
    rsa_image = nib.load(str(base.RSA_NIFTI))
    regression_image = nib.load(str(base.UNIVARIATE_NIFTI))
    if rsa_image.shape != regression_image.shape or not np.allclose(rsa_image.affine, regression_image.affine):
        raise ValueError("The multivariate RSA and univariate maps do not share a voxel grid.")

    rows = pd.DataFrame(
        base.collect_harvard_oxford(
            rsa_image,
            base.positive_mask(rsa_image),
            base.positive_mask(regression_image),
        )
    )
    rows["status_order"] = rows["classification"].map(base.STATUS_ORDER)
    rows = rows.sort_values(["status_order", "hemisphere", "display_roi"], kind="stable").reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(12.8, 8.2), facecolor="white")
    base.add_table_panel(axis, rows)
    figure.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.97)

    png = base.FIGURE_DIR / "mricrogl_harvard_oxford_panel_b.png"
    pdf = base.FIGURE_DIR / "mricrogl_harvard_oxford_panel_b.pdf"
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
