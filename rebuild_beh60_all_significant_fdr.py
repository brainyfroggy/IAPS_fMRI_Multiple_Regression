from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight")
SCORE_CSV = ROOT / "outputs" / "20sub_SL_eval_score_norm_042726.csv"
CENTER_CSV = ROOT / "outputs" / "voxel_center_id.csv"
REFERENCE_NIFTI = ROOT / "outputs" / "tBrainmap" / "final_results" / "tmap_beh60_p05_v2.nii.gz"
OUT_DIR = ROOT / "outputs" / "tBrainmap" / "final_results"
OUT_NIFTI = OUT_DIR / "tmap_beh60_all_significant_fdr05_fisherz.nii.gz"
OUT_CLUSTER_NIFTI = OUT_DIR / "tmap_beh60_all_significant_fdr05_fisherz_cluster_gt50.nii.gz"
OUT_SUMMARY = OUT_DIR / "tmap_beh60_all_significant_fdr05_fisherz_summary.json"
CLUSTER_THRESHOLD_VOXELS = 50


def betacf(a: float, b: float, x: float) -> float:
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
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_to_two_sided_p(t_values: np.ndarray, df: int) -> np.ndarray:
    t_flat = np.abs(np.asarray(t_values, dtype=np.float64)).ravel()
    p_flat = np.empty_like(t_flat)
    a = df / 2.0
    b = 0.5
    for idx, t_val in enumerate(t_flat):
        x = df / (df + t_val * t_val)
        p_flat[idx] = regularized_incomplete_beta(a, b, x)
    return p_flat.reshape(np.asarray(t_values).shape)


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p_flat = np.asarray(p_values, dtype=np.float64).ravel()
    order = np.argsort(p_flat)
    ranked = p_flat[order]
    m = ranked.size
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_flat = np.empty_like(p_flat)
    q_flat[order] = q_ranked
    q = q_flat.reshape(np.asarray(p_values).shape)
    return q, q < alpha


def read_nifti_header(path: Path) -> tuple[bytes, tuple[int, int, int], int]:
    with gzip.open(path, "rb") as f:
        header = bytearray(f.read(348))
    if struct.unpack("<i", header[:4])[0] != 348:
        raise ValueError(f"{path} is not a little-endian NIfTI file.")
    dim = struct.unpack("<8h", header[40:56])
    shape = tuple(int(x) for x in dim[1 : 1 + dim[0]])
    vox_offset = int(round(struct.unpack("<f", header[108:112])[0]))
    return bytes(header), shape, vox_offset


def write_nifti_like(reference_path: Path, data: np.ndarray, out_path: Path) -> None:
    header, shape, vox_offset = read_nifti_header(reference_path)
    if data.shape != shape:
        raise ValueError(f"Data shape {data.shape} does not match reference shape {shape}")
    with gzip.open(out_path, "wb") as f:
        f.write(header)
        if vox_offset > 348:
            f.write(b"\x00" * (vox_offset - 348))
        f.write(np.asarray(data, dtype="<f8").ravel(order="F").tobytes())


def cluster_threshold_volume(volume: np.ndarray, threshold_voxels: int) -> tuple[np.ndarray, dict[str, int]]:
    """Keep 26-connected same-sign clusters with more than threshold_voxels voxels."""
    thresholded = np.zeros_like(volume, dtype=np.float64)
    summary: dict[str, int] = {}

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for label, active in {
        "positive": np.isfinite(volume) & (volume > 0),
        "negative": np.isfinite(volume) & (volume < 0),
    }.items():
        visited = np.zeros(active.shape, dtype=bool)
        total_clusters = 0
        kept_clusters = 0
        kept_voxels = 0

        for start in np.argwhere(active):
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
                    if (
                        nx < 0
                        or ny < 0
                        or nz < 0
                        or nx >= active.shape[0]
                        or ny >= active.shape[1]
                        or nz >= active.shape[2]
                    ):
                        continue
                    neighbor = (nx, ny, nz)
                    if active[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            if len(cluster) > threshold_voxels:
                kept_clusters += 1
                kept_voxels += len(cluster)
                for voxel in cluster:
                    thresholded[voxel] = volume[voxel]

        summary[f"{label}_clusters_total"] = int(total_clusters)
        summary[f"{label}_clusters_kept_gt{threshold_voxels}"] = int(kept_clusters)
        summary[f"{label}_voxels_kept_gt{threshold_voxels}"] = int(kept_voxels)

    return thresholded, summary


def main() -> None:
    scores = pd.read_csv(SCORE_CSV, index_col=0).to_numpy(dtype=np.float64)
    centers = pd.read_csv(CENTER_CSV).iloc[:, 1].to_numpy(dtype=np.int64)
    _, shape, _ = read_nifti_header(REFERENCE_NIFTI)

    if scores.shape[1] != centers.size:
        raise ValueError(f"Score columns {scores.shape[1]} do not match centers {centers.size}")

    fisher_z = np.arctanh(np.clip(scores, -1 + 1e-10, 1 - 1e-10))
    t_stats = fisher_z.mean(axis=0) / (fisher_z.std(axis=0, ddof=1) / math.sqrt(fisher_z.shape[0]))
    p_values = t_to_two_sided_p(t_stats, df=fisher_z.shape[0] - 1)
    q_values, rejected = fdr_bh(p_values, alpha=0.05)

    flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
    flat[centers] = np.where(rejected, t_stats, 0.0)
    volume = flat.reshape(shape, order="C")

    write_nifti_like(REFERENCE_NIFTI, volume, OUT_NIFTI)
    clustered_volume, cluster_summary = cluster_threshold_volume(volume, CLUSTER_THRESHOLD_VOXELS)
    write_nifti_like(REFERENCE_NIFTI, clustered_volume, OUT_CLUSTER_NIFTI)

    summary = {
        "source_scores": str(SCORE_CSV),
        "source_centers": str(CENTER_CSV),
        "reference_header": str(REFERENCE_NIFTI),
        "output_nifti": str(OUT_NIFTI),
        "output_cluster_thresholded_nifti": str(OUT_CLUSTER_NIFTI),
        "model": "behavioral RSA, Fisher-z transformed subject searchlight scores",
        "group_test": "two-sided one-sample t-test against 0 across 20 subjects",
        "multiple_comparisons": "Benjamini-Hochberg FDR q < 0.05 across searchlight centers",
        "cluster_threshold": "26-connected same-sign clusters, keep clusters with > 50 voxels",
        "n_subjects": int(scores.shape[0]),
        "n_centers": int(scores.shape[1]),
        "n_fdr_significant_total": int(rejected.sum()),
        "n_positive": int(np.sum(rejected & (t_stats > 0))),
        "n_negative": int(np.sum(rejected & (t_stats < 0))),
        "t_min_significant": float(np.min(t_stats[rejected])),
        "t_max_significant": float(np.max(t_stats[rejected])),
        "n_cluster_thresholded_total": int(np.sum(clustered_volume != 0)),
        "n_cluster_thresholded_positive": int(np.sum(clustered_volume > 0)),
        "n_cluster_thresholded_negative": int(np.sum(clustered_volume < 0)),
        **cluster_summary,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
