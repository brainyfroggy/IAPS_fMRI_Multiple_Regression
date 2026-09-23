"""Plot the regression and RSA maps on a PyCortex flatmap.

This script expects an existing PyCortex subject database with:

1. A cortical surface subject, usually imported from FreeSurfer.
2. A transform from the NIfTI map space to that subject.
3. An overlay SVG containing ROI labels/borders if ROI labels are desired.

The paper-style labels such as V1, V2, V3, LO, STS, FFA, EBA, PPA, OPA,
IPS, and CoS are not generated automatically by PyCortex; they must exist
in the subject's PyCortex overlay file.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
SEARCHLIGHT_FINAL = Path(
    r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight"
    r"\outputs\tBrainmap\final_results"
)

DEFAULT_MAPS = {
    "valence_regression": ROOT / "maps" / "group_t_beta_valence_fdr05_cluster50.img",
    "arousal_regression": ROOT / "maps" / "group_t_beta_arousal_fdr05_cluster50.img",
    "behavioral_rsa": SEARCHLIGHT_FINAL
    / "tmap_beh60_all_significant_fdr05_fisherz_cluster_gt50.nii.gz",
}


def _import_or_explain() -> tuple[object, object, object]:
    try:
        import cortex
        import cortex.quickflat
    except Exception as exc:  # pragma: no cover - environment check
        raise SystemExit(
            "PyCortex is not available in this Python environment.\n"
            "Install it in the environment you want to use, for example:\n"
            "  pip install pycortex nibabel\n\n"
            f"Original import error: {exc!r}"
        ) from exc

    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - environment check
        raise SystemExit(
            "nibabel is required to read the NIfTI/Analyze maps.\n"
            "Install it in the same environment, for example:\n"
            "  pip install nibabel\n\n"
            f"Original import error: {exc!r}"
        ) from exc

    return cortex, cortex.quickflat, nib


def _finite_nonzero_values(images: Iterable[object]) -> np.ndarray:
    values = []
    for img in images:
        data = np.asarray(img.get_fdata(dtype=np.float32))
        keep = np.isfinite(data) & (data != 0)
        if np.any(keep):
            values.append(data[keep])
    if not values:
        raise SystemExit("No nonzero finite values were found in the input maps.")
    return np.concatenate(values)


def _load_maps(nib: object, map_paths: dict[str, Path]) -> dict[str, object]:
    images = {}
    for name, path in map_paths.items():
        if not path.exists():
            raise SystemExit(f"Missing input map for {name}: {path}")
        images[name] = nib.load(str(path))
    return images


def _check_pycortex_target(cortex: object, subject: str, xfm: str) -> None:
    subjects = list(cortex.db.subjects)
    if subject not in subjects:
        joined = ", ".join(subjects) if subjects else "(none)"
        raise SystemExit(
            f"PyCortex subject '{subject}' was not found.\n"
            f"Available subjects: {joined}\n\n"
            "Import or create the subject first. For a FreeSurfer subject:\n"
            "  import cortex\n"
            "  cortex.freesurfer.import_subj('SUBJECT', freesurfer_subject_dir='PATH_TO_SUBJECTS_DIR')"
        )

    try:
        cortex.db.get_xfm(subject, xfm)
    except Exception as exc:
        raise SystemExit(
            f"PyCortex transform '{xfm}' was not found for subject '{subject}'.\n"
            "Create a transform from the NIfTI map space to the subject surface first.\n"
            "For group MNI maps, this usually means using an fsaverage/MNI surface setup "
            "or a subject-specific registration.\n\n"
            f"Original transform error: {exc!r}"
        ) from exc


def _robust_symmetric_limits(values: np.ndarray, percentile: float) -> tuple[float, float]:
    vmax = float(np.percentile(np.abs(values), percentile))
    if not math.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(values)))
    return -vmax, vmax


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make PyCortex flatmaps for valence, arousal, and behavioral RSA t maps."
    )
    parser.add_argument(
        "--subject",
        default="fsaverage",
        help="PyCortex subject name. Default: fsaverage.",
    )
    parser.add_argument(
        "--xfm",
        default="identity",
        help="PyCortex transform name from NIfTI space to subject. Default: identity.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "pycortex",
        help="Output directory for PNG flatmaps.",
    )
    parser.add_argument(
        "--overlay-file",
        type=Path,
        default=None,
        help="Optional PyCortex overlay SVG. If omitted, PyCortex uses the subject default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1600,
        help="Flatmap image height in pixels. Default: 1600.",
    )
    parser.add_argument(
        "--cmap",
        default="RdBu_r",
        help="Matplotlib colormap name. Default: RdBu_r.",
    )
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=99.0,
        help="Shared symmetric color scale percentile over nonzero voxels. Default: 99.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed absolute color limit. Overrides --scale-percentile.",
    )
    parser.add_argument(
        "--recache",
        action="store_true",
        help="Force PyCortex to recache flatmap geometry.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cortex, quickflat, nib = _import_or_explain()

    _check_pycortex_target(cortex, args.subject, args.xfm)
    if args.overlay_file is not None and not args.overlay_file.exists():
        raise SystemExit(f"Overlay SVG not found: {args.overlay_file}")

    images = _load_maps(nib, DEFAULT_MAPS)
    all_values = _finite_nonzero_values(images.values())
    if args.vmax is None:
        vmin, vmax = _robust_symmetric_limits(all_values, args.scale_percentile)
    else:
        vmax = abs(float(args.vmax))
        vmin = -vmax

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using shared color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    overlay_file = str(args.overlay_file) if args.overlay_file is not None else None
    for name, img in images.items():
        data = np.asarray(img.get_fdata(dtype=np.float32))
        data[~np.isfinite(data)] = 0

        vol = cortex.Volume(
            data,
            args.subject,
            args.xfm,
            vmin=vmin,
            vmax=vmax,
            cmap=args.cmap,
        )

        out_png = args.out_dir / f"{name}_pycortex_flatmap.png"
        quickflat.make_png(
            str(out_png),
            vol,
            height=args.height,
            recache=args.recache,
            with_rois=True,
            with_sulci=True,
            with_labels=True,
            with_curvature=True,
            with_colorbar=True,
            overlay_file=overlay_file,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
