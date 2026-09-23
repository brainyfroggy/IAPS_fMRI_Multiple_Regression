from __future__ import annotations

import gzip
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_fMRI_multiple_reg")
SEARCHLIGHT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight")
MAPS_DIR = ROOT / "maps"
FIGS_DIR = ROOT / "figures"

MNI_TEMPLATE = SEARCHLIGHT_DIR / "visualization" / "pycortex" / "mni152.nii.gz"
RSA_SIGNED_CLUSTER_TMAP = (
    SEARCHLIGHT_DIR
    / "outputs"
    / "tBrainmap"
    / "final_results"
    / "tmap_beh60_all_significant_fdr05_fisherz_cluster_gt50.nii.gz"
)

NIFTI_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def load_nifti_volume(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(348)
        sizeof_hdr = struct.unpack("<i", header[:4])[0]
        if sizeof_hdr != 348:
            raise ValueError(f"{path} does not look like a little-endian NIfTI file.")
        dim = struct.unpack("<8h", header[40:56])
        datatype = struct.unpack("<h", header[70:72])[0]
        vox_offset = int(round(struct.unpack("<f", header[108:112])[0]))
        dtype = NIFTI_DTYPES.get(datatype)
        if dtype is None:
            raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
        shape = tuple(int(x) for x in dim[1 : 1 + dim[0]])
        f.seek(vox_offset)
        data = np.frombuffer(f.read(), dtype=dtype)
    return data[: int(np.prod(shape))].reshape(shape, order="F").astype(np.float32)


def resample_template_to_shape(template: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    indices = [
        np.clip(
            np.rint(np.linspace(0, template.shape[axis] - 1, target_shape[axis])).astype(int),
            0,
            template.shape[axis] - 1,
        )
        for axis in range(3)
    ]
    return template[np.ix_(indices[0], indices[1], indices[2])]


def robust_normalize(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)


def display_volume(volume: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(volume) & (volume != 0), volume, np.nan)


def choose_z_slices(volumes: list[np.ndarray], n_slices: int = 8) -> np.ndarray:
    active = np.zeros(volumes[0].shape, dtype=bool)
    for volume in volumes:
        active |= np.isfinite(volume) & (volume != 0)
    counts = active.sum(axis=(0, 1))
    nonzero_z = np.flatnonzero(counts > 0)
    if nonzero_z.size == 0:
        return np.linspace(6, volumes[0].shape[2] - 7, n_slices).astype(int)

    if nonzero_z.size <= n_slices:
        return nonzero_z.astype(int)

    quantile_positions = np.linspace(0, nonzero_z.size - 1, n_slices)
    return np.unique(np.rint(quantile_positions).astype(int)).astype(int)


def add_row_colorbar(fig: plt.Figure, image, ax, label: str) -> None:
    pos = ax.get_position()
    cax = fig.add_axes([0.925, pos.y0, 0.015, pos.height])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def plot_conjunction_and_rsa(
    coefficient_volumes: dict[str, np.ndarray],
    rsa_volume: np.ndarray,
    template: np.ndarray,
    out_path: Path,
) -> None:
    all_volumes = list(coefficient_volumes.values()) + [rsa_volume]
    z_slices = choose_z_slices(all_volumes, n_slices=8)

    coef_values = np.concatenate(
        [v[np.isfinite(v) & (v != 0)] for v in coefficient_volumes.values()]
    )
    coef_vmax = np.nanpercentile(np.abs(coef_values), 99) if coef_values.size else 1.0
    coef_vmax = max(float(coef_vmax), 1e-6)

    rsa_values = rsa_volume[np.isfinite(rsa_volume) & (rsa_volume != 0)]
    rsa_vmax = np.nanpercentile(np.abs(rsa_values), 99) if rsa_values.size else 1.0
    rsa_vmax = max(float(rsa_vmax), 1e-6)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad((1, 1, 1, 0))

    rows = list(coefficient_volumes.items()) + [("Behavioral RSA", rsa_volume)]
    fig, axes = plt.subplots(len(rows), len(z_slices), figsize=(18, 4.1 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)

    row_images = []
    for row_idx, (row_name, volume) in enumerate(rows):
        is_rsa = row_name == "Behavioral RSA"
        vmax = rsa_vmax if is_rsa else coef_vmax
        vmin = -vmax
        last_img = None
        for col_idx, z in enumerate(z_slices):
            ax = axes[row_idx, col_idx]
            bg_slice = np.rot90(template[:, :, z])
            stat_slice = np.ma.masked_invalid(np.rot90(display_volume(volume[:, :, z])))
            ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            last_img = ax.imshow(
                stat_slice,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.88,
                interpolation="nearest",
            )
            ax.set_title(f"z={z}", fontsize=10)
            ax.axis("off")

        axes[row_idx, 0].text(
            0.0,
            1.17,
            row_name,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
        row_images.append(last_img)

    fig.suptitle(
        "Regression coefficients where valence and arousal are both significant, with RSA below",
        fontsize=16,
    )
    fig.subplots_adjust(right=0.90, top=0.91, hspace=0.42, wspace=0.03)
    add_row_colorbar(fig, row_images[0], axes[0, -1], "regression coefficient")
    add_row_colorbar(fig, row_images[-1], axes[-1, -1], "RSA t-value")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_conjunction_and_rsa(
    conjunction_categories: np.ndarray,
    rsa_volume: np.ndarray,
    template: np.ndarray,
    out_path: Path,
) -> None:
    z_slices = choose_z_slices([conjunction_categories, rsa_volume], n_slices=8)

    category_cmap = ListedColormap(
        [
            "#2166ac",  # both negative
            "#67a9cf",  # valence negative, arousal positive
            "#ef8a62",  # valence positive, arousal negative
            "#b2182b",  # both positive
        ]
    )
    category_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], category_cmap.N)
    category_labels = [
        ("Both negative", "#2166ac"),
        ("Valence -, Arousal +", "#67a9cf"),
        ("Valence +, Arousal -", "#ef8a62"),
        ("Both positive", "#b2182b"),
    ]

    rsa_values = rsa_volume[np.isfinite(rsa_volume) & (rsa_volume != 0)]
    rsa_vmax = np.nanpercentile(np.abs(rsa_values), 99) if rsa_values.size else 1.0
    rsa_vmax = max(float(rsa_vmax), 1e-6)
    rsa_cmap = plt.get_cmap("coolwarm").copy()
    rsa_cmap.set_bad((1, 1, 1, 0))

    fig, axes = plt.subplots(2, len(z_slices), figsize=(18, 8.2))

    for col_idx, z in enumerate(z_slices):
        ax = axes[0, col_idx]
        bg_slice = np.rot90(template[:, :, z])
        cat_slice = np.ma.masked_invalid(np.rot90(display_volume(conjunction_categories[:, :, z])))
        ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.imshow(
            cat_slice,
            cmap=category_cmap,
            norm=category_norm,
            alpha=0.92,
            interpolation="nearest",
        )
        ax.set_title(f"z={z}", fontsize=10)
        ax.axis("off")

    axes[0, 0].text(
        0.0,
        1.17,
        "Valence & arousal both significant",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    rsa_last_img = None
    for col_idx, z in enumerate(z_slices):
        ax = axes[1, col_idx]
        bg_slice = np.rot90(template[:, :, z])
        rsa_slice = np.ma.masked_invalid(np.rot90(display_volume(rsa_volume[:, :, z])))
        ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        rsa_last_img = ax.imshow(
            rsa_slice,
            cmap=rsa_cmap,
            vmin=-rsa_vmax,
            vmax=rsa_vmax,
            alpha=0.88,
            interpolation="nearest",
        )
        ax.set_title(f"z={z}", fontsize=10)
        ax.axis("off")

    axes[1, 0].text(
        0.0,
        1.17,
        "Behavioral RSA",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    fig.suptitle("Conjunction of significant valence and arousal coefficients, with RSA below", fontsize=16)
    fig.subplots_adjust(right=0.90, top=0.89, hspace=0.48, wspace=0.03)

    legend_handles = [Patch(facecolor=color, edgecolor="black", label=label) for label, color in category_labels]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    add_row_colorbar(fig, rsa_last_img, axes[1, -1], "RSA t-value")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    val_t = np.load(MAPS_DIR / "group_t_beta_valence_fdr05_cluster50.npy")
    aro_t = np.load(MAPS_DIR / "group_t_beta_arousal_fdr05_cluster50.npy")
    val_coef = np.load(MAPS_DIR / "group_mean_beta_valence.npy")
    aro_coef = np.load(MAPS_DIR / "group_mean_beta_arousal.npy")

    if val_t.shape != aro_t.shape or val_t.shape != val_coef.shape or val_t.shape != aro_coef.shape:
        raise ValueError("Regression map shapes do not match.")

    conjunction = (
        np.isfinite(val_t)
        & np.isfinite(aro_t)
        & (val_t != 0)
        & (aro_t != 0)
    )
    print(f"Valence significant voxels: {int(np.count_nonzero(val_t))}")
    print(f"Arousal significant voxels: {int(np.count_nonzero(aro_t))}")
    print(f"Conjunction voxels, valence AND arousal significant: {int(conjunction.sum())}")

    val_coef_conj = np.where(conjunction, val_coef, 0.0).astype(np.float32)
    aro_coef_conj = np.where(conjunction, aro_coef, 0.0).astype(np.float32)
    conjunction_categories = np.zeros(val_t.shape, dtype=np.float32)
    conjunction_categories[conjunction & (val_t < 0) & (aro_t < 0)] = 1
    conjunction_categories[conjunction & (val_t < 0) & (aro_t > 0)] = 2
    conjunction_categories[conjunction & (val_t > 0) & (aro_t < 0)] = 3
    conjunction_categories[conjunction & (val_t > 0) & (aro_t > 0)] = 4
    conjunction_categories[conjunction_categories == 0] = np.nan

    print("Conjunction sign-pattern voxels:")
    print(f"  both negative: {int(np.count_nonzero(conjunction & (val_t < 0) & (aro_t < 0)))}")
    print(f"  valence -, arousal +: {int(np.count_nonzero(conjunction & (val_t < 0) & (aro_t > 0)))}")
    print(f"  valence +, arousal -: {int(np.count_nonzero(conjunction & (val_t > 0) & (aro_t < 0)))}")
    print(f"  both positive: {int(np.count_nonzero(conjunction & (val_t > 0) & (aro_t > 0)))}")

    rsa = load_nifti_volume(RSA_SIGNED_CLUSTER_TMAP)
    if rsa.shape != val_t.shape:
        raise ValueError(f"RSA map shape {rsa.shape} does not match regression maps {val_t.shape}")

    template = robust_normalize(resample_template_to_shape(load_nifti_volume(MNI_TEMPLATE), val_t.shape))

    np.save(MAPS_DIR / "valence_arousal_both_significant_mask.npy", conjunction.astype(np.uint8))
    np.save(MAPS_DIR / "valence_arousal_both_significant_sign_categories.npy", conjunction_categories)
    np.save(MAPS_DIR / "group_mean_beta_valence_both_sig.npy", val_coef_conj)
    np.save(MAPS_DIR / "group_mean_beta_arousal_both_sig.npy", aro_coef_conj)

    single_out_path = FIGS_DIR / "valence_arousal_both_significant_conjunction_with_rsa_below.png"
    plot_single_conjunction_and_rsa(
        conjunction_categories,
        np.where(rsa != 0, rsa, 0.0).astype(np.float32),
        template,
        single_out_path,
    )
    print(f"Wrote {single_out_path}")

    out_path = FIGS_DIR / "valence_arousal_coefficients_both_significant_with_rsa_below.png"
    plot_conjunction_and_rsa(
        {
            "Valence coefficient": val_coef_conj,
            "Arousal coefficient": aro_coef_conj,
        },
        np.where(rsa != 0, rsa, 0.0).astype(np.float32),
        template,
        out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
