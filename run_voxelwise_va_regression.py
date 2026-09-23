from __future__ import annotations

import json
import math
import gzip
import struct
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEARCHLIGHT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight")
OUT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_fMRI_multiple_reg")

ALLSUB_PATH = SEARCHLIGHT_DIR / "allsub_avg.npy"
MASK_PATH = SEARCHLIGHT_DIR / "mask.npy"
TRIAL_CSV = SEARCHLIGHT_DIR / "fmri_conditions_trial_ordered.csv"
IAPS_ORDER_CSV = SEARCHLIGHT_DIR / "IAPS_60_pnu.csv"
REFERENCE_HDR = SEARCHLIGHT_DIR / "tmp.hdr"
MNI_TEMPLATE = SEARCHLIGHT_DIR / "visualization" / "pycortex" / "mni152.nii.gz"
BEH60_TMAP = SEARCHLIGHT_DIR / "outputs" / "tBrainmap" / "final_results" / "tmap_beh60_p05_v2.nii.gz"
BEH60_SIGNED_CLUSTER_TMAP = (
    SEARCHLIGHT_DIR
    / "outputs"
    / "tBrainmap"
    / "final_results"
    / "tmap_beh60_all_significant_fdr05_fisherz_cluster_gt50.nii.gz"
)

PREDICTORS = ("valence", "arousal")
FDR_ALPHA = 0.05
CLUSTER_THRESHOLD_VOXELS = 50

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


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std(ddof=1)


def load_design() -> tuple[pd.DataFrame, np.ndarray]:
    trials = pd.read_csv(TRIAL_CSV)
    unique_stimuli = (
        trials[["iaps_id", "type", "category", "stim_order", "Mean_Valence", "Mean_Arousal"]]
        .value_counts()
        .reset_index(name="n_repetitions")
    )
    unique_stimuli["type"] = pd.Categorical(unique_stimuli["type"], ["Pl", "Nt", "Up"])
    unique_stimuli = unique_stimuli.sort_values(["type", "stim_order"]).reset_index(drop=True)

    if len(unique_stimuli) != 60:
        raise ValueError(f"Expected 60 unique stimuli, found {len(unique_stimuli)}")
    if not np.all(unique_stimuli["n_repetitions"].to_numpy() == 5):
        raise ValueError("Not every stimulus has 5 repetitions in the trial file.")

    iaps_order = pd.read_csv(IAPS_ORDER_CSV)
    if not unique_stimuli["iaps_id"].astype(int).equals(iaps_order["iaps_id"].astype(int)):
        raise ValueError("Stimulus order from trial CSV does not match IAPS_60_pnu.csv.")

    design = unique_stimuli.copy()
    design.insert(0, "stimulus_index_1based", np.arange(1, len(design) + 1))
    design["emotion_type"] = iaps_order["emotion_type"].to_numpy()
    design["valence_z"] = zscore(design["Mean_Valence"].to_numpy())
    design["arousal_z"] = zscore(design["Mean_Arousal"].to_numpy())

    x = np.column_stack(
        [
            np.ones(len(design), dtype=np.float64),
            design["valence_z"].to_numpy(dtype=np.float64),
            design["arousal_z"].to_numpy(dtype=np.float64),
        ]
    )
    return design, x


def regression_by_subject(data: np.ndarray, x: np.ndarray, mask_flat: np.ndarray) -> np.ndarray:
    """Return subject slopes with shape subjects x 2 predictors x mask_voxels."""
    n_subjects, n_stimuli, n_voxels = data.shape
    if n_stimuli != x.shape[0]:
        raise ValueError(f"Data has {n_stimuli} stimuli but design has {x.shape[0]} rows.")
    if n_voxels != mask_flat.size:
        raise ValueError(f"Data has {n_voxels} voxels but mask has {mask_flat.size}.")

    x_pinv = np.linalg.pinv(x)
    slopes = np.empty((n_subjects, 2, int(mask_flat.sum())), dtype=np.float32)

    for subj_idx in range(n_subjects):
        y = np.asarray(data[subj_idx][:, mask_flat], dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        beta = x_pinv @ y
        slopes[subj_idx] = beta[1:3].astype(np.float32)
        print(f"Finished subject {subj_idx + 1:02d}/{n_subjects}")

    return slopes


def group_stats(subject_slopes: np.ndarray) -> dict[str, np.ndarray]:
    n_subjects = subject_slopes.shape[0]
    mean_beta = subject_slopes.mean(axis=0)
    sd_beta = subject_slopes.std(axis=0, ddof=1)
    sem_beta = sd_beta / np.sqrt(n_subjects)
    t_beta = np.divide(mean_beta, sem_beta, out=np.zeros_like(mean_beta), where=sem_beta > 0)
    return {
        "mean_beta": mean_beta.astype(np.float32),
        "sd_beta": sd_beta.astype(np.float32),
        "sem_beta": sem_beta.astype(np.float32),
        "t_beta": t_beta.astype(np.float32),
    }


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter = 200
    eps = 3.0e-12
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_to_two_sided_p(t_values: np.ndarray, df: int) -> np.ndarray:
    """Two-sided Student t p-values without requiring scipy."""
    t_flat = np.abs(np.asarray(t_values, dtype=np.float64)).ravel()
    a = df / 2.0
    b = 0.5
    p_flat = np.empty_like(t_flat)
    for idx, t_val in enumerate(t_flat):
        if not np.isfinite(t_val):
            p_flat[idx] = np.nan
            continue
        x = df / (df + t_val * t_val)
        p_flat[idx] = regularized_incomplete_beta(a, b, x)
    return p_flat.reshape(np.asarray(t_values).shape).astype(np.float32)


def fdr_bh(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR adjusted p-values and rejection mask."""
    p = np.asarray(p_values, dtype=np.float64)
    p_flat = p.ravel()
    q_flat = np.full_like(p_flat, np.nan)
    finite = np.isfinite(p_flat)
    finite_p = p_flat[finite]

    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    finite_indices = np.flatnonzero(finite)
    q_flat[finite_indices[order]] = adjusted
    q = q_flat.reshape(p.shape).astype(np.float32)
    return q, q < alpha


def to_volume(mask: np.ndarray, masked_values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    volume = np.full(mask.size, fill_value, dtype=np.float32)
    volume[mask.ravel().astype(bool)] = masked_values.astype(np.float32)
    return volume.reshape(mask.shape)


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
    # Display-only nearest-neighbor resampling into the functional-map grid.
    indices = [
        np.clip(np.rint(np.linspace(0, template.shape[axis] - 1, target_shape[axis])).astype(int), 0, template.shape[axis] - 1)
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


def cluster_threshold_volume(volume: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, int, int]:
    """Keep 26-connected nonzero finite clusters with at least min_cluster_size voxels."""
    active = np.isfinite(volume) & (volume != 0)
    visited = np.zeros(active.shape, dtype=bool)
    kept = np.zeros(active.shape, dtype=bool)
    total_clusters = 0
    kept_clusters = 0

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    active_coords = np.argwhere(active)
    shape = active.shape
    for start in active_coords:
        start_tuple = tuple(int(v) for v in start)
        if visited[start_tuple]:
            continue

        total_clusters += 1
        stack = [start_tuple]
        visited[start_tuple] = True
        cluster = []

        while stack:
            voxel = stack.pop()
            cluster.append(voxel)
            x, y, z = voxel
            for dx, dy, dz in neighbor_offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= shape[0] or ny >= shape[1] or nz >= shape[2]:
                    continue
                neighbor = (nx, ny, nz)
                if active[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        if len(cluster) >= min_cluster_size:
            kept_clusters += 1
            for voxel in cluster:
                kept[voxel] = True

    thresholded = np.where(kept, volume, 0.0).astype(np.float32)
    return thresholded, total_clusters, kept_clusters


def save_analyze_pair(volume: np.ndarray, out_stem: Path) -> None:
    """Save an Analyze .hdr/.img pair by reusing the existing header geometry."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if REFERENCE_HDR.exists():
        shutil.copyfile(REFERENCE_HDR, out_stem.with_suffix(".hdr"))
    volume.astype("<f4", copy=False).tofile(out_stem.with_suffix(".img"))


def signed_max_projection(volume: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(volume)
    abs_for_index = np.where(finite, np.abs(volume), -1.0)
    abs_idx = np.argmax(abs_for_index, axis=axis)
    projection = np.take_along_axis(volume, np.expand_dims(abs_idx, axis=axis), axis=axis).squeeze(axis)
    has_data = np.any(finite, axis=axis)
    return np.where(has_data, projection, np.nan)


def plot_slice_mosaic(
    volumes: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    cmap: str = "coolwarm",
    positive_only: bool = False,
) -> None:
    z_slices = np.linspace(6, volumes[next(iter(volumes))].shape[2] - 7, 8).astype(int)
    finite_values = np.concatenate([v[np.isfinite(v)] for v in volumes.values()])
    if positive_only:
        finite_values = finite_values[finite_values > 0]
        vmax = np.nanpercentile(finite_values, 99) if finite_values.size else 1.0
        vmin = 0.0
    else:
        vmax = np.nanpercentile(np.abs(finite_values), 99) if finite_values.size else 1.0
        vmin = -vmax
    vmax = max(float(vmax), 1e-6)

    fig, axes = plt.subplots(len(volumes), len(z_slices), figsize=(18, 4.4 * len(volumes)))
    if len(volumes) == 1:
        axes = np.expand_dims(axes, 0)

    last_img = None
    for row, (name, volume) in enumerate(volumes.items()):
        for col, z in enumerate(z_slices):
            ax = axes[row, col]
            slice_2d = np.rot90(volume[:, :, z])
            last_img = ax.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{name}, z={z}", fontsize=10)
            ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(right=0.90, top=0.90, hspace=0.05, wspace=0.03)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(last_img, cax=cax)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_slice_mosaic_on_template(
    stat_volumes: dict[str, np.ndarray],
    template: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "coolwarm",
    positive_only: bool = False,
) -> None:
    z_slices = np.linspace(6, template.shape[2] - 7, 8).astype(int)
    finite_values = np.concatenate([v[np.isfinite(v)] for v in stat_volumes.values()])
    if positive_only:
        finite_values = finite_values[finite_values > 0]
        vmax = np.nanpercentile(finite_values, 99) if finite_values.size else 1.0
        vmin = 0.0
    else:
        vmax = np.nanpercentile(np.abs(finite_values), 99) if finite_values.size else 1.0
        vmin = -vmax
    vmax = max(float(vmax), 1e-6)

    overlay_cmap = plt.get_cmap(cmap).copy()
    overlay_cmap.set_bad((1, 1, 1, 0))

    fig, axes = plt.subplots(len(stat_volumes), len(z_slices), figsize=(18, 4.4 * len(stat_volumes)))
    if len(stat_volumes) == 1:
        axes = np.expand_dims(axes, 0)

    last_img = None
    for row, (name, volume) in enumerate(stat_volumes.items()):
        for col, z in enumerate(z_slices):
            ax = axes[row, col]
            bg_slice = np.rot90(template[:, :, z])
            stat_slice = np.ma.masked_invalid(np.rot90(volume[:, :, z]))
            ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            last_img = ax.imshow(
                stat_slice,
                cmap=overlay_cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.88,
                interpolation="nearest",
            )
            ax.set_title(f"{name}, z={z}", fontsize=10)
            ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(right=0.90, top=0.90, hspace=0.05, wspace=0.03)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(last_img, cax=cax)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_slice_mosaic_on_template_row_labels(
    stat_volumes: dict[str, np.ndarray],
    template: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "coolwarm",
) -> None:
    z_slices = np.linspace(6, template.shape[2] - 7, 8).astype(int)
    finite_values = np.concatenate([v[np.isfinite(v)] for v in stat_volumes.values()])
    vmax = np.nanpercentile(np.abs(finite_values), 99) if finite_values.size else 1.0
    vmax = max(float(vmax), 1e-6)

    overlay_cmap = plt.get_cmap(cmap).copy()
    overlay_cmap.set_bad((1, 1, 1, 0))

    fig, axes = plt.subplots(len(stat_volumes), len(z_slices), figsize=(18, 4.4 * len(stat_volumes)))
    if len(stat_volumes) == 1:
        axes = np.expand_dims(axes, 0)

    last_img = None
    for row, (name, volume) in enumerate(stat_volumes.items()):
        for col, z in enumerate(z_slices):
            ax = axes[row, col]
            bg_slice = np.rot90(template[:, :, z])
            stat_slice = np.ma.masked_invalid(np.rot90(volume[:, :, z]))
            ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            last_img = ax.imshow(
                stat_slice,
                cmap=overlay_cmap,
                vmin=-vmax,
                vmax=vmax,
                alpha=0.88,
                interpolation="nearest",
            )
            ax.set_title(f"z={z}", fontsize=10)
            ax.axis("off")

        axes[row, 0].text(
            0.0,
            1.18,
            name,
            transform=axes[row, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(right=0.90, top=0.90, hspace=0.42, wspace=0.03)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(last_img, cax=cax)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_projection_summary(
    volumes: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    cmap: str = "coolwarm",
    positive_only: bool = False,
) -> None:
    finite_values = np.concatenate([v[np.isfinite(v)] for v in volumes.values()])
    if positive_only:
        finite_values = finite_values[finite_values > 0]
        vmax = np.nanpercentile(finite_values, 99) if finite_values.size else 1.0
        vmin = 0.0
    else:
        vmax = np.nanpercentile(np.abs(finite_values), 99) if finite_values.size else 1.0
        vmin = -vmax
    vmax = max(float(vmax), 1e-6)

    axes_names = [("sagittal", 0), ("coronal", 1), ("axial", 2)]
    fig, axes = plt.subplots(len(volumes), 3, figsize=(11, 3.8 * len(volumes)))
    if len(volumes) == 1:
        axes = np.expand_dims(axes, 0)

    last_img = None
    for row, (name, volume) in enumerate(volumes.items()):
        for col, (axis_name, axis) in enumerate(axes_names):
            ax = axes[row, col]
            projection = np.rot90(signed_max_projection(volume, axis=axis))
            last_img = ax.imshow(projection, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{name}: {axis_name}", fontsize=10)
            ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(right=0.88, top=0.88, hspace=0.10, wspace=0.05)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    fig.colorbar(last_img, cax=cax)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    design: pd.DataFrame,
    x: np.ndarray,
    subject_slopes: np.ndarray,
    stats: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    corr = float(np.corrcoef(design["valence_z"], design["arousal_z"])[0, 1])
    vif = 1.0 / (1.0 - corr**2)

    summary = {
        "analysis": "voxelwise multiple regression",
        "dependent_variable": "fMRI beta, stimulus-averaged, per subject and voxel",
        "model": "beta_voxel ~ intercept + z(valence) + z(arousal)",
        "n_subjects": int(subject_slopes.shape[0]),
        "n_stimuli": int(x.shape[0]),
        "n_repetitions_per_stimulus": 5,
        "n_mask_voxels": int(subject_slopes.shape[2]),
        "predictor_correlation_valence_arousal": corr,
        "predictor_vif_each_predictor": vif,
        "mean_beta_range": {
            predictor: [
                float(np.nanmin(stats["mean_beta"][idx])),
                float(np.nanmax(stats["mean_beta"][idx])),
            ]
            for idx, predictor in enumerate(PREDICTORS)
        },
        "t_beta_range": {
            predictor: [
                float(np.nanmin(stats["t_beta"][idx])),
                float(np.nanmax(stats["t_beta"][idx])),
            ]
            for idx, predictor in enumerate(PREDICTORS)
        },
    }

    (out_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    notes = f"""# Voxelwise valence/arousal regression

## Model

The analysis uses the available `allsub_avg.npy` beta matrix, which has shape `20 x 60 x 153594`: 20 subjects, 60 stimulus-averaged beta patterns, and 153594 voxels. The model fit at each subject/voxel is:

`fMRI beta = intercept + z(valence) + z(arousal)`

The output maps are group summaries across the 20 subject-level slopes.

## Why 60 averaged stimuli instead of 300 trials?

The current reusable data in `IAPS_Searchlight` are already averaged to 60 stimuli. This is also the better simple model for these predictors: valence and arousal are properties of the stimulus, so repeating the same rating 5 times and treating the 300 rows as independent would inflate the apparent degrees of freedom unless the model explicitly handled stimulus/repetition/subject dependence. For a simple voxelwise regression, the 60 averaged stimulus betas are the cleaner choice.

If you want to use all 300 single-trial betas later, the better version would be a mixed-effects or two-stage model that accounts for repeated presentations rather than a plain OLS on 300 rows.

## Design diagnostics

- Valence/arousal predictor correlation: `{corr:.3f}`
- VIF for each predictor in the two-predictor model: `{vif:.3f}`

## Visualization recommendation

Use unthresholded group mean beta maps to show effect direction and scale, and group t maps to show reliability across subjects. For a paper figure, the most interpretable static view is usually a small set of anatomically meaningful slices from the t maps, with matching unthresholded beta maps in the supplement. The projection PNGs here are useful as a fast whole-brain overview, but slice maps are better for localization.

For final reporting, the strongest next visualization would be thresholded group t maps on an anatomical template with a diverging colormap centered at zero, plus ROI summaries or scatterplots from peak/ROI voxels to show how beta changes with valence and arousal.
"""
    (out_dir / "analysis_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    maps_dir = OUT_DIR / "maps"
    figs_dir = OUT_DIR / "figures"
    maps_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    design, x = load_design()
    design.to_csv(OUT_DIR / "design_matrix_60stimuli.csv", index=False)

    mask = np.load(MASK_PATH)
    mask_flat = mask.ravel().astype(bool)
    np.save(OUT_DIR / "brain_mask.npy", mask.astype(np.int8))
    np.save(OUT_DIR / "mask_flat_indices.npy", np.flatnonzero(mask_flat).astype(np.int32))

    data = np.load(ALLSUB_PATH, mmap_mode="r")
    print(f"Loaded fMRI data: {data.shape}")
    print(f"Mask voxels: {mask_flat.sum()} / {mask_flat.size}")

    subject_slopes = regression_by_subject(data, x, mask_flat)
    np.save(OUT_DIR / "subject_betas_valence_arousal_masked.npy", subject_slopes)

    stats = group_stats(subject_slopes)
    df = subject_slopes.shape[0] - 1
    p_two_sided = t_to_two_sided_p(stats["t_beta"], df=df)
    q_fdr = np.empty_like(p_two_sided)
    fdr_rejected = np.empty(p_two_sided.shape, dtype=bool)
    for predictor_idx, predictor in enumerate(PREDICTORS):
        q_fdr[predictor_idx], fdr_rejected[predictor_idx] = fdr_bh(
            p_two_sided[predictor_idx],
            alpha=FDR_ALPHA,
        )
        print(
            f"{predictor}: {int(fdr_rejected[predictor_idx].sum())} / "
            f"{fdr_rejected.shape[1]} voxels survive FDR q < {FDR_ALPHA}"
        )

    for stat_name, arr in stats.items():
        np.save(OUT_DIR / f"group_{stat_name}_valence_arousal_masked.npy", arr)
    np.save(OUT_DIR / "group_p_two_sided_valence_arousal_masked.npy", p_two_sided)
    np.save(OUT_DIR / "group_p_fdr_valence_arousal_masked.npy", q_fdr)
    np.save(OUT_DIR / "group_fdr05_mask_valence_arousal_masked.npy", fdr_rejected)

    mean_volumes: dict[str, np.ndarray] = {}
    t_volumes: dict[str, np.ndarray] = {}
    t_fdr05_volumes: dict[str, np.ndarray] = {}
    t_fdr05_positive_volumes: dict[str, np.ndarray] = {}
    mean_positive_volumes: dict[str, np.ndarray] = {}
    t_positive_volumes: dict[str, np.ndarray] = {}

    for predictor_idx, predictor in enumerate(PREDICTORS):
        mean_volume = to_volume(mask, stats["mean_beta"][predictor_idx], fill_value=0.0)
        t_volume = to_volume(mask, stats["t_beta"][predictor_idx], fill_value=0.0)
        t_fdr05_volume_raw = to_volume(
            mask,
            np.where(fdr_rejected[predictor_idx], stats["t_beta"][predictor_idx], 0.0),
            fill_value=0.0,
        )
        t_fdr05_pos_clustered, pos_clusters_total, pos_clusters_kept = cluster_threshold_volume(
            np.where(t_fdr05_volume_raw > 0, t_fdr05_volume_raw, 0.0),
            CLUSTER_THRESHOLD_VOXELS,
        )
        t_fdr05_neg_clustered, neg_clusters_total, neg_clusters_kept = cluster_threshold_volume(
            np.where(t_fdr05_volume_raw < 0, t_fdr05_volume_raw, 0.0),
            CLUSTER_THRESHOLD_VOXELS,
        )
        t_fdr05_volume = t_fdr05_pos_clustered + t_fdr05_neg_clustered
        print(
            f"{predictor}: cluster threshold >= {CLUSTER_THRESHOLD_VOXELS} voxels kept "
            f"{pos_clusters_kept}/{pos_clusters_total} positive clusters and "
            f"{neg_clusters_kept}/{neg_clusters_total} negative clusters"
        )
        mean_volumes[predictor.capitalize()] = np.where(mask.astype(bool), mean_volume, np.nan)
        t_volumes[predictor.capitalize()] = np.where(mask.astype(bool), t_volume, np.nan)
        t_fdr05_volumes[predictor.capitalize()] = np.where(
            (mask.astype(bool)) & (t_fdr05_volume != 0),
            t_fdr05_volume,
            np.nan,
        )
        t_fdr05_positive_volumes[predictor.capitalize()] = np.where(
            (mask.astype(bool)) & (t_fdr05_volume > 0),
            t_fdr05_volume,
            np.nan,
        )
        mean_positive_volumes[predictor.capitalize()] = np.where(
            (mask.astype(bool)) & (mean_volume > 0),
            mean_volume,
            np.nan,
        )
        t_positive_volumes[predictor.capitalize()] = np.where(
            (mask.astype(bool)) & (t_volume > 0),
            t_volume,
            np.nan,
        )

        np.save(maps_dir / f"group_mean_beta_{predictor}.npy", mean_volume)
        np.save(maps_dir / f"group_t_beta_{predictor}.npy", t_volume)
        np.save(maps_dir / f"group_t_beta_{predictor}_fdr05_uncorrected_clusters.npy", t_fdr05_volume_raw)
        np.save(maps_dir / f"group_t_beta_{predictor}_fdr05_cluster50.npy", t_fdr05_volume)
        save_analyze_pair(mean_volume, maps_dir / f"group_mean_beta_{predictor}")
        save_analyze_pair(t_volume, maps_dir / f"group_t_beta_{predictor}")
        save_analyze_pair(t_fdr05_volume, maps_dir / f"group_t_beta_{predictor}_fdr05_cluster50")

    plot_slice_mosaic(
        mean_volumes,
        figs_dir / "whole_brain_group_mean_betas_axial.png",
        "Group mean regression slopes: z(valence) and z(arousal)",
    )
    plot_slice_mosaic(
        t_volumes,
        figs_dir / "whole_brain_group_t_betas_axial.png",
        "Group t maps for regression slopes across subjects",
    )
    plot_projection_summary(
        mean_volumes,
        figs_dir / "whole_brain_group_mean_betas_projection.png",
        "Signed max projection of group mean slopes",
    )
    plot_projection_summary(
        t_volumes,
        figs_dir / "whole_brain_group_t_betas_projection.png",
        "Signed max projection of group t maps",
    )
    plot_slice_mosaic(
        t_fdr05_volumes,
        figs_dir / "whole_brain_group_t_betas_axial_fdr05.png",
        f"Group t maps, FDR q < 0.05, cluster >= {CLUSTER_THRESHOLD_VOXELS} voxels",
    )
    if MNI_TEMPLATE.exists():
        template = robust_normalize(resample_template_to_shape(load_nifti_volume(MNI_TEMPLATE), mask.shape))
        np.save(OUT_DIR / "mni152_resampled_to_regression_grid.npy", template)
        plot_slice_mosaic_on_template(
            t_fdr05_volumes,
            template,
            figs_dir / "whole_brain_group_t_betas_axial_fdr05_on_template.png",
            f"Group t maps on MNI template, FDR q < 0.05, cluster >= {CLUSTER_THRESHOLD_VOXELS}",
        )
        plot_slice_mosaic_on_template(
            t_fdr05_positive_volumes,
            template,
            figs_dir / "whole_brain_group_t_betas_axial_fdr05_positive_only_on_template.png",
            f"Positive group t maps on MNI template, FDR q < 0.05, cluster >= {CLUSTER_THRESHOLD_VOXELS}",
            cmap="Reds",
            positive_only=True,
        )
        if BEH60_TMAP.exists():
            beh60_tmap = load_nifti_volume(BEH60_TMAP)
            if beh60_tmap.shape != mask.shape:
                raise ValueError(f"Expected {BEH60_TMAP} to have shape {mask.shape}, found {beh60_tmap.shape}")
            beh60_positive = {
                "Behavioral RDM": np.where(beh60_tmap > 0, beh60_tmap, np.nan),
            }
            plot_slice_mosaic_on_template(
                beh60_positive,
                template,
                figs_dir / "tmap_beh60_p05_v2_positive_only_axial_on_template.png",
                "Positive behavioral RDM t map on MNI template",
                cmap="Reds",
                positive_only=True,
            )
            plot_slice_mosaic_on_template(
                beh60_positive,
                template,
                BEH60_TMAP.parent / "tmap_beh60_p05_v2_positive_only_axial_on_template_matched_style.png",
                "Positive behavioral RDM t map on MNI template",
                cmap="Reds",
                positive_only=True,
            )
            combined_positive = {
                "Valence": t_fdr05_positive_volumes["Valence"],
                "Arousal": t_fdr05_positive_volumes["Arousal"],
                "Behavioral RDM": beh60_positive["Behavioral RDM"],
            }
            plot_slice_mosaic_on_template(
                combined_positive,
                template,
                figs_dir / "combined_positive_regression_and_beh60_tmaps_on_template_same_scale.png",
                f"Positive t maps on MNI template, shared scale; regression clusters >= {CLUSTER_THRESHOLD_VOXELS}",
                cmap="Reds",
                positive_only=True,
            )
            plot_slice_mosaic_on_template(
                combined_positive,
                template,
                BEH60_TMAP.parent / "combined_positive_regression_and_beh60_tmaps_on_template_same_scale.png",
                f"Positive t maps on MNI template, shared scale; regression clusters >= {CLUSTER_THRESHOLD_VOXELS}",
                cmap="Reds",
                positive_only=True,
            )
            if BEH60_SIGNED_CLUSTER_TMAP.exists():
                beh60_signed_cluster_tmap = load_nifti_volume(BEH60_SIGNED_CLUSTER_TMAP)
                if beh60_signed_cluster_tmap.shape != mask.shape:
                    raise ValueError(
                        f"Expected {BEH60_SIGNED_CLUSTER_TMAP} to have shape {mask.shape}, "
                        f"found {beh60_signed_cluster_tmap.shape}"
                    )
                combined_signed = {
                    "Valence": t_fdr05_volumes["Valence"],
                    "Arousal": t_fdr05_volumes["Arousal"],
                    "Behavioral RSA": np.where(beh60_signed_cluster_tmap != 0, beh60_signed_cluster_tmap, np.nan),
                }
                plot_slice_mosaic_on_template_row_labels(
                    combined_signed,
                    template,
                    figs_dir / "combined_signed_regression_and_beh60_tmaps_on_template_same_scale_cluster50.png",
                    "Signed t maps on MNI template, shared scale",
                    cmap="coolwarm",
                )
                plot_slice_mosaic_on_template_row_labels(
                    combined_signed,
                    template,
                    BEH60_TMAP.parent / "combined_signed_regression_and_beh60_tmaps_on_template_same_scale_cluster50.png",
                    "Signed t maps on MNI template, shared scale",
                    cmap="coolwarm",
                )
            else:
                print(f"Signed behavioral t map not found, skipping: {BEH60_SIGNED_CLUSTER_TMAP}")
        else:
            print(f"Behavioral t map not found, skipping: {BEH60_TMAP}")
    else:
        print(f"Template not found, skipping template overlay: {MNI_TEMPLATE}")
    plot_projection_summary(
        t_fdr05_volumes,
        figs_dir / "whole_brain_group_t_betas_projection_fdr05.png",
        f"Signed max projection of group t maps, FDR q < 0.05, cluster >= {CLUSTER_THRESHOLD_VOXELS}",
    )
    plot_slice_mosaic(
        mean_positive_volumes,
        figs_dir / "whole_brain_group_mean_betas_positive_only_axial.png",
        "Positive group mean regression slopes: z(valence) and z(arousal)",
        cmap="Reds",
        positive_only=True,
    )
    plot_slice_mosaic(
        t_positive_volumes,
        figs_dir / "whole_brain_group_t_betas_positive_only_axial.png",
        "Positive group t maps for regression slopes across subjects",
        cmap="Reds",
        positive_only=True,
    )
    plot_projection_summary(
        mean_positive_volumes,
        figs_dir / "whole_brain_group_mean_betas_positive_only_projection.png",
        "Positive signed max projection of group mean slopes",
        cmap="Reds",
        positive_only=True,
    )
    plot_projection_summary(
        t_positive_volumes,
        figs_dir / "whole_brain_group_t_betas_positive_only_projection.png",
        "Positive signed max projection of group t maps",
        cmap="Reds",
        positive_only=True,
    )

    write_summary(design, x, subject_slopes, stats, OUT_DIR)
    print("Done.")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
