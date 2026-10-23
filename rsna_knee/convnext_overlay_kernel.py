"""TEST overlay: mill V10 plus ConvNeXt Effusion+Synovitis at weight 0.15.

Importing this module does not start inference and does not submit.
The Kaggle script unpacks this file and calls ``main`` only under ``__main__``.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

try:
    from rsna_knee.fusion import fail_closed_rank_blend, overlay_weights_from_findings
    from rsna_knee.scoring_time import (
        CONVNEXT_HOLDOUT_WALL_S,
        MILL_HOLDOUT_WALL_S,
        N_HOLDOUT,
        convnext_overlay_clock_allowed,
        convnext_overlay_estimate_s,
        scale_wall_seconds,
    )
    from rsna_knee.study_count import discovered_study_ids
except ImportError:  # unpacked next to this file on Kaggle
    from fusion import fail_closed_rank_blend, overlay_weights_from_findings
    from scoring_time import (
        CONVNEXT_HOLDOUT_WALL_S,
        MILL_HOLDOUT_WALL_S,
        N_HOLDOUT,
        convnext_overlay_clock_allowed,
        convnext_overlay_estimate_s,
        scale_wall_seconds,
    )
    from study_count import discovered_study_ids

OVERLAY_FINDINGS = ("Effusion", "Synovitis")
OVERLAY_WEIGHT = 0.15
HEADER = (
    "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,"
    "Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
)
TAG = "[cnxt-ov]"


def overlay_weights() -> dict[str, float]:
    return overlay_weights_from_findings(OVERLAY_FINDINGS, OVERLAY_WEIGHT)


def blend_tables(
    mill: pd.DataFrame,
    overlay: pd.DataFrame | None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Fail-closed rank blend. A bad ConvNeXt frame returns mill unchanged."""
    return fail_closed_rank_blend(mill, overlay, weights if weights is not None else overlay_weights())


def clock_allows(n_studies: int) -> bool:
    return convnext_overlay_clock_allowed(n_studies)


def find_comp_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "test_series").is_dir() or (path / "test.csv").is_file():
            print(f"{TAG} competition root {path}", flush=True)
            return path
    base = Path("/kaggle/input")
    if base.is_dir():
        for path in base.rglob("test_series.csv"):
            root = path.parent
            if (root / "test_series").is_dir() or (root / "test.csv").is_file():
                print(f"{TAG} competition root (scan) {root}", flush=True)
                return root
    raise FileNotFoundError("competition test_series not mounted")


def _find_file(name: str) -> Path:
    skip = {"train_series", "test_series", "test_images", "train_images"}
    base = Path("/kaggle/input")
    hits: list[Path] = []
    if base.is_dir():
        stack = [base]
        depth = 0
        while stack and depth < 120:
            cur = stack.pop(0)
            depth += 1
            try:
                kids = list(cur.iterdir())
            except OSError:
                continue
            if name in {kid.name for kid in kids}:
                hits.append(cur / name)
            for kid in kids:
                if kid.is_dir() and kid.name not in skip and not kid.name.startswith("1.2."):
                    stack.append(kid)
    if not hits:
        raise FileNotFoundError(name)
    print(f"{TAG} found {name} at {hits[0]}", flush=True)
    return hits[0]


def _gpu_probe() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    probe = subprocess.run(
        ["nvidia-smi", "-L"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(f"{TAG} nvidia-smi rc={probe.returncode} {probe.stdout.strip()}", flush=True)


def _run_mill(mill_py: Path, work: Path, timeout_s: float) -> Path:
    dest = work / "submission.csv"
    if dest.is_file():
        dest.unlink()
    print(f"{TAG} START mill_v10 timeout={timeout_s:.0f}s", flush=True)
    proc = subprocess.run(
        [sys.executable, str(mill_py)],
        cwd=str(work),
        timeout=timeout_s,
        check=False,
    )
    print(f"{TAG} END mill_v10 rc={proc.returncode}", flush=True)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(f"mill V10 did not write submission.csv rc={proc.returncode}")
    kept = work / "mill_v10_base.csv"
    shutil.copy2(dest, kept)
    return kept


def interpreter_wheel_tag(major: int | None = None, minor: int | None = None) -> str:
    """CPython wheel tag for the running interpreter, for example cp312."""
    major = sys.version_info.major if major is None else int(major)
    minor = sys.version_info.minor if minor is None else int(minor)
    return f"cp{major}{minor}"


def compatible_wheels(wheels: list[Path], py_tag: str) -> list[Path]:
    """Keep pure-python wheels and the wheels built for this interpreter."""
    tag = py_tag.lower()
    kept: list[Path] = []
    for wheel in wheels:
        name = wheel.name.lower()
        if "py3-none-any" in name or "py2.py3-none-any" in name or tag in name:
            kept.append(wheel)
    return kept


def wheel_install_argv(python: str, wheels: list[Path], target: Path) -> list[str]:
    """Install by absolute path so pip does not look for the wheel in the working directory."""
    return [
        python,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-index",
        "--no-deps",
        "--target",
        str(target),
        *[str(Path(wheel).resolve()) for wheel in wheels],
    ]


def convnext_child_pythonpath(own: Path, wheel_target: Path | None, existing: str = "") -> str:
    """Child path: reader sources, then installed DICOM wheels, then the parent path."""
    parts = [str(own)]
    if wheel_target is not None:
        parts.append(str(wheel_target))
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _run_convnext(comp: Path, work: Path, study_ids: list[str], timeout_s: float) -> Path:
    ckpt0 = _find_file("cnxt_v0_fold0.pt")
    assets = ckpt0.parent
    ckpts = sorted(glob.glob(str(assets / "cnxt_v0_fold*.pt")))
    if len(ckpts) < 3:
        raise RuntimeError(f"need 3 ConvNeXt fold checkpoints, found {ckpts}")
    own = work / "own_src"
    own.mkdir(parents=True, exist_ok=True)
    for name in ("infer.py", "knee.py", "preprocess.py"):
        src = assets / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, own / name)
    wheels = compatible_wheels(list(assets.glob("*.whl")), interpreter_wheel_tag())
    target = work / "_own_env"
    if wheels:
        target.mkdir(exist_ok=True)
        cmd_pip = wheel_install_argv(sys.executable, wheels, target)
        print(f"{TAG} pip wheels={len(wheels)}", flush=True)
        subprocess.run(cmd_pip, check=False)
    out_csv = work / "_convnext_test.csv"
    driver = work / "_convnext_driver.py"
    driver.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from infer import run\n"
        "ids = Path(sys.argv[4]).read_text(encoding='utf-8').splitlines()\n"
        "frame = run(sys.argv[2], 'test', sys.argv[5:], studies=ids)\n"
        "frame.to_csv(sys.argv[3])\n"
        "print('wrote', sys.argv[3], frame.shape)\n",
        encoding="utf-8",
    )
    id_file = work / "_study_ids.txt"
    id_file.write_text("\n".join(study_ids) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = convnext_child_pythonpath(
        own, target if wheels else None, env.get("PYTHONPATH", "")
    )
    cmd = [sys.executable, str(driver), str(own), str(comp), str(out_csv), str(id_file), *ckpts]
    print(f"{TAG} START convnext timeout={timeout_s:.0f}s studies={len(study_ids)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(work), env=env, timeout=timeout_s, check=False)
    print(f"{TAG} END convnext rc={proc.returncode}", flush=True)
    if proc.returncode != 0 or not out_csv.is_file():
        raise RuntimeError(f"ConvNeXt reader failed rc={proc.returncode}")
    return out_csv


def refuse_over_clock(n_studies: int, submission: Path | None = None) -> None:
    """Exit 2 and delete submission.csv when the discovered count misses 8.75 h."""
    if clock_allows(n_studies):
        return
    if submission is not None and submission.is_file():
        submission.unlink()
    estimate = convnext_overlay_estimate_s(n_studies) if n_studies > 0 else float("inf")
    print(
        f"{TAG} KILL discovered n={n_studies} estimate_s={estimate:.1f} misses the 8.75h line. "
        "No submission.csv. keep 55779066.",
        flush=True,
    )
    raise SystemExit(2)


def run_overlay_map(weights: dict[str, float], *, tag: str = TAG) -> None:
    """Score hidden test with a per-finding weight map. Exit 2 over the 8.75 h line."""
    t0 = time.time()
    if any(float(value) >= 0.50 for value in weights.values()):
        raise RuntimeError("overlay weight must stay under 0.50")
    findings = tuple(
        name for name in HEADER.split(",")[1:] if float(weights.get(name, 0.0)) > 0.0
    )
    print(
        f"{tag} TEST overlay map {dict((name, weights[name]) for name in findings)} "
        f"header {HEADER}. keep 55779066 until a COMPLETE publicScore is strictly greater. "
        "This file is not a train-holdout submit.",
        flush=True,
    )
    _gpu_probe()
    work = Path("/kaggle/working")
    work.mkdir(parents=True, exist_ok=True)
    comp = find_comp_root()
    ids = discovered_study_ids(comp)
    n = len(ids)
    estimate = convnext_overlay_estimate_s(n) if n > 0 else float("inf")
    print(f"{tag} discovered n={n} estimate_s={estimate:.1f}", flush=True)
    sub = work / "submission.csv"
    if not clock_allows(n):
        refuse_over_clock(n, sub)

    mill_py = work / "_mill_v10.py"
    if not mill_py.is_file():
        raise FileNotFoundError(mill_py)
    mill_timeout = scale_wall_seconds(MILL_HOLDOUT_WALL_S, N_HOLDOUT, n) + 180.0
    cnxt_timeout = scale_wall_seconds(CONVNEXT_HOLDOUT_WALL_S, N_HOLDOUT, n) + 180.0
    mill_csv = _run_mill(mill_py, work, timeout_s=mill_timeout)
    mill = pd.read_csv(mill_csv, dtype={"StudyInstanceUID": str})
    try:
        cnxt_csv = _run_convnext(
            comp,
            work,
            mill["StudyInstanceUID"].astype(str).tolist(),
            timeout_s=cnxt_timeout,
        )
        overlay = pd.read_csv(cnxt_csv, dtype={"StudyInstanceUID": str})
        if "StudyInstanceUID" not in overlay.columns and overlay.index.name == "StudyInstanceUID":
            overlay = overlay.reset_index()
        out = blend_tables(mill, overlay, weights)
    except Exception as exc:
        print(f"{tag} convnext FAILED; fail-closed to mill: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        out = mill
    missing = [name for name in HEADER.split(",") if name not in out.columns]
    if missing or list(out.columns) != HEADER.split(","):
        raise RuntimeError(f"submission header {list(out.columns)} != {HEADER}")
    out.to_csv(sub, index=False)
    changed = [name for name in findings if not out[name].equals(mill[name])]
    print(
        f"{tag} wrote submission.csv shape={out.shape} overlay_columns={changed} "
        f"elapsed={time.time() - t0:.0f}s "
        "not 1.0 replace. keep 55779066 until publicScore > 0.936.",
        flush=True,
    )


def run_overlay(
    findings: tuple[str, ...],
    weight: float,
    *,
    tag: str = TAG,
) -> None:
    """Score hidden test. Exit 2 when the discovered-n clock misses 8.75 h."""
    run_overlay_map(overlay_weights_from_findings(findings, weight), tag=tag)


def main() -> None:
    """Two-finding Effusion+Synovitis path. The k7 kernel calls run_overlay directly."""
    run_overlay(OVERLAY_FINDINGS, OVERLAY_WEIGHT, tag=TAG)
