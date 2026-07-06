#!/usr/bin/env python3
"""
run_sweep.py - Batch directory age classification
==================================================
Keep this file in the same folder as age_estimator.py and run:

    python run_sweep.py INPUT_DIR OUTPUT_DIR
    python run_sweep.py INPUT_DIR OUTPUT_DIR --quiet

Writes annotated images + results.csv. UTKFace filenames
([age]_[gender]_[race]_[date].jpg) are auto-parsed; the run prints binary
accuracy and age MAE against ground truth.
"""

import os
import csv
import argparse
from pathlib import Path

import cv2

import age_estimator as ae

_IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
_RACE_STR   = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
_GENDER_STR = {0: 'Male', 1: 'Female'}

_CSV_FIELDS = ['filename', 'face_idx', 'face_count', 'classification',
               'certainty', 'pred_age', 'pred_gender',
               'gt_age', 'gt_gender', 'gt_race', 'gt_label',
               'age_error', 'is_correct']


def parse_utk(stem):
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
        if not (0 <= gender <= 1) or not (0 <= race <= 4):
            return None
        return {'gt_age': age, 'gt_gender_str': _GENDER_STR[gender],
                'gt_race_str': _RACE_STR[race],
                'gt_label': 'UNDERAGE' if age < 18 else 'ADULT'}
    except (ValueError, IndexError):
        return None


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def write_row(w, path, idx, count, v, gt):
    row = {'filename': path.name, 'face_idx': idx, 'face_count': count}
    if v is None:
        row.update(classification='NO_FACE', certainty='', pred_age='', pred_gender='')
    elif v.classification == 'TOO_SMALL':
        row.update(classification='TOO_SMALL', certainty='', pred_age='', pred_gender='')
    else:
        row.update(classification=v.classification, certainty=v.certainty,
                   pred_age=v.age, pred_gender=v.gender)
    if gt:
        row.update(gt_age=gt['gt_age'], gt_gender=gt['gt_gender_str'],
                   gt_race=gt['gt_race_str'], gt_label=gt['gt_label'])
        if v is not None and v.classification in ('ADULT', 'UNDERAGE'):
            row['is_correct'] = (v.classification == gt['gt_label'])
            row['age_error']  = round(abs(v.age - gt['gt_age']), 1)
        elif v is None:
            row['is_correct'] = False
            row['age_error']  = ''
        else:
            row['is_correct'] = ''
            row['age_error']  = ''
    else:
        row.update(gt_age='', gt_gender='', gt_race='', gt_label='',
                   age_error='', is_correct='')
    w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Batch age classification.")
    p.add_argument('input_dir')
    p.add_argument('output_dir')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--timing', action='store_true',
                   help="Also measure per-stage inference latency (CPU)")
    args = p.parse_args()
    verbose = not args.quiet

    in_path = Path(args.input_dir).resolve()
    out_path = Path(args.output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in in_path.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if not images:
        print(f"[WARN] No images found in {in_path}")
        return

    total = len(images)
    print(f"[INFO] Found {total} image(s) -> loading models...")
    est = ae.AgeEstimator()
    if args.timing:
        est.timing = True
        est.reset_timing()
    print(f"[INFO] Writing to {out_path}\n")

    csv_path = out_path / 'results.csv'
    n_no_face = n_faces = n_correct = n_scored = 0
    age_err_sum = 0.0

    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for i, img_path in enumerate(images, 1):
            if verbose:
                print(f"  [{i:>{len(str(total))}}/{total}] {img_path.name:<36}", end='', flush=True)
            img = load_image(img_path)
            if img is None:
                if verbose:
                    print("  SKIP - unreadable")
                continue
            faces = est.detect_all(img)
            verdicts = sorted((est.analyse_face(img, f) for f in faces),
                              key=lambda v: v.det_score, reverse=True)
            cv2.imwrite(str(out_path / img_path.name),
                        ae.make_polaroid(img, verdicts, img_path.name))
            gt = parse_utk(img_path.stem)
            if not verdicts:
                n_no_face += 1
                write_row(w, img_path, 0, 0, None, gt)
            else:
                n_faces += len(verdicts)
                for j, v in enumerate(verdicts, 1):
                    write_row(w, img_path, j, len(verdicts), v, gt)
                    if gt and v.classification in ('ADULT', 'UNDERAGE'):
                        n_scored += 1
                        if v.classification == gt['gt_label']:
                            n_correct += 1
                        age_err_sum += abs(v.age - gt['gt_age'])
            if verbose:
                if not verdicts:
                    print("  -> no face")
                else:
                    print("  -> " + ', '.join(
                        f"F{j}:{v.classification}" + (f"(~{v.age:.0f})" if v.age is not None else "")
                        for j, v in enumerate(verdicts, 1)))

    print("\n[INFO] Done.")
    print(f"         Images processed : {total - n_no_face}")
    print(f"         No face detected : {n_no_face}")
    print(f"         Total faces      : {n_faces}")
    if n_scored:
        print(f"         Benchmark (UTKFace):")
        print(f"           Binary accuracy : {100.0*n_correct/n_scored:.1f}%  ({n_correct}/{n_scored})")
        print(f"           Age MAE         : {age_err_sum/n_scored:.2f} years")
    print(f"         CSV              : {csv_path}")

    if args.timing and est.t_age:
        import numpy as _np
        det_ms = _np.array(est.t_detect) * 1000.0 if est.t_detect else _np.array([0.0])
        age_ms = _np.array(est.t_age) * 1000.0
        print(f"         Performance (CPU, per-stage latency):")
        print(f"           Detection : mean {det_ms.mean():.2f} ms  "
              f"median {_np.median(det_ms):.2f} ms  (n={len(det_ms)})")
        print(f"           Age infer : mean {age_ms.mean():.2f} ms  "
              f"median {_np.median(age_ms):.2f} ms  (n={len(age_ms)})")
        combined = det_ms.mean() + age_ms.mean()
        if combined > 0:
            print(f"           Combined  : {combined:.2f} ms/face  "
                  f"({1000.0/combined:.1f} faces/s)")


if __name__ == '__main__':
    main()
