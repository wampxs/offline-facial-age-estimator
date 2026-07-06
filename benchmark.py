#!/usr/bin/env python3
"""
benchmark.py - Core-algorithm performance benchmark
===================================================
Measures the resource cost of the two inference stages — face detection and
age estimation — in isolation, excluding image file I/O and result drawing.

Run from the same folder as age_estimator.py:

    python benchmark.py INPUT_DIR
    python benchmark.py INPUT_DIR --runs 300 --warmup 20

INPUT_DIR should hold a sample of representative face images (a few hundred
UTKFace images is ideal). The script:
  1. records the hardware/software context,
  2. performs warmup iterations (excluded from timing) to let caches and the
     ONNX session settle,
  3. times detection per frame and age inference per face separately,
  4. measures peak memory, and reports model file sizes on disk.

All numbers are specific to the machine that runs this script, so the
reported hardware context must accompany them in any writeup.
"""

import os
import sys
import time
import argparse
import platform
from pathlib import Path

import numpy as np
import cv2

import age_estimator as ae

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Environment / hardware context
# ---------------------------------------------------------------------------

def hardware_context():
    info = {}
    info['platform'] = platform.platform()
    info['processor'] = platform.processor() or platform.machine()
    info['python'] = platform.python_version()
    try:
        info['logical_cores'] = os.cpu_count()
    except Exception:
        info['logical_cores'] = 'unknown'

    # Physical cores + RAM if psutil is available (optional)
    try:
        import psutil
        info['physical_cores'] = psutil.cpu_count(logical=False)
        info['total_ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
        try:
            freq = psutil.cpu_freq()
            info['cpu_freq_mhz'] = round(freq.max) if freq and freq.max else 'unknown'
        except Exception:
            info['cpu_freq_mhz'] = 'unknown'
    except ImportError:
        info['physical_cores'] = 'n/a (install psutil for this)'
        info['total_ram_gb'] = 'n/a (install psutil for this)'
        info['cpu_freq_mhz'] = 'n/a'

    # ONNX Runtime version + provider actually used
    try:
        import onnxruntime as ort
        info['onnxruntime'] = ort.__version__
    except Exception:
        info['onnxruntime'] = 'unknown'
    info['opencv'] = cv2.__version__
    return info


def peak_rss_mb():
    """Peak resident memory in MB, best-effort across platforms."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KB, macOS reports bytes
            if sys.platform == 'darwin':
                return round(ru / 1e6, 1)
            return round(ru / 1e3, 1)
        except Exception:
            return None


def model_sizes_mb(est):
    """Best-effort sizes of the loaded model files on disk."""
    sizes = {}
    # Age model
    try:
        p = est._resolve_age_model()
        sizes['age_model_mb'] = round(os.path.getsize(p) / 1e6, 1)
    except Exception:
        sizes['age_model_mb'] = 'unknown'
    # Detector pack (InsightFace caches under ~/.insightface/models/<pack>)
    try:
        pack_dir = Path.home() / '.insightface' / 'models' / ae.DETECTOR_MODEL
        if pack_dir.exists():
            total = sum(f.stat().st_size for f in pack_dir.rglob('*') if f.is_file())
            sizes['detector_pack_mb'] = round(total / 1e6, 1)
        else:
            sizes['detector_pack_mb'] = 'unknown'
    except Exception:
        sizes['detector_pack_mb'] = 'unknown'
    return sizes


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def summarize(times_s, label):
    """Return a dict of ms statistics from a list of second-valued samples."""
    if not times_s:
        return None
    a = np.array(times_s) * 1000.0  # -> milliseconds
    return {
        'label':  label,
        'n':      len(a),
        'mean':   round(float(a.mean()), 2),
        'median': round(float(np.median(a)), 2),
        'p95':    round(float(np.percentile(a, 95)), 2),
        'min':    round(float(a.min()), 2),
        'max':    round(float(a.max()), 2),
    }


def print_stat_row(s):
    if s is None:
        return
    print(f"  {s['label']:<22} mean {s['mean']:>7.2f}  median {s['median']:>7.2f}  "
          f"p95 {s['p95']:>7.2f}  min {s['min']:>7.2f}  max {s['max']:>7.2f}   (n={s['n']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Core-algorithm performance benchmark.")
    ap.add_argument('input_dir', help="Folder of sample face images")
    ap.add_argument('--runs', type=int, default=200,
                    help="Number of timed images (default 200)")
    ap.add_argument('--warmup', type=int, default=15,
                    help="Warmup images, excluded from timing (default 15)")
    args = ap.parse_args()

    in_path = Path(args.input_dir)
    images = sorted(p for p in in_path.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if not images:
        print(f"[ERROR] No images in {in_path}")
        return

    print("=" * 68)
    print("CORE-ALGORITHM PERFORMANCE BENCHMARK")
    print("=" * 68)

    # --- hardware context ---
    ctx = hardware_context()
    print("\nHARDWARE / SOFTWARE CONTEXT")
    print("-" * 68)
    for k, v in ctx.items():
        print(f"  {k:<16}: {v}")

    # --- load models (not timed) ---
    print("\nLoading models...")
    est = ae.AgeEstimator()
    mem_after_load = peak_rss_mb()
    sizes = model_sizes_mb(est)

    # --- warmup ---
    print(f"\nWarmup ({args.warmup} images, not timed)...")
    est.timing = False
    wcount = 0
    for img_path in images:
        if wcount >= args.warmup:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        face = est.detect(img)
        if face is not None:
            est.analyse_face(img, face)
        wcount += 1

    # --- timed run ---
    print(f"Timing (up to {args.runs} images)...")
    est.timing = True
    est.reset_timing()
    done = 0
    for img_path in images:
        if done >= args.runs:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        face = est.detect(img)          # detection timed inside
        if face is not None:
            est.analyse_face(img, face)  # age inference timed inside
        done += 1

    mem_peak = peak_rss_mb()

    # --- report ---
    det = summarize(est.t_detect, "Face detection")
    age = summarize(est.t_age,    "Age inference")

    print("\nLATENCY PER STAGE (milliseconds, CPU)")
    print("-" * 68)
    print_stat_row(det)
    print_stat_row(age)

    if det and age:
        combined_mean = det['mean'] + age['mean']
        print("-" * 68)
        print(f"  {'Combined / face':<22} mean {combined_mean:>7.2f} ms")
        if combined_mean > 0:
            print(f"  {'Throughput':<22} {1000.0/combined_mean:>7.2f} faces/second")

    print("\nMEMORY")
    print("-" * 68)
    print(f"  RSS after model load  : {mem_after_load} MB")
    print(f"  Peak RSS during run   : {mem_peak} MB")

    print("\nMODEL SIZE ON DISK")
    print("-" * 68)
    print(f"  Age model (ONNX)      : {sizes['age_model_mb']} MB")
    print(f"  Detector pack         : {sizes['detector_pack_mb']} MB")

    print("\n" + "=" * 68)
    print("Report these numbers WITH the hardware context above.")
    print("=" * 68)


if __name__ == '__main__':
    main()
