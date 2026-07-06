#!/usr/bin/env python3
"""
run_live.py - Live webcam age classification
=============================================
Keep this file in the same folder as age_estimator.py and run:

    python run_live.py                  (continuous, default)
    python run_live.py --mode stable    (hold still, capture once)
    python run_live.py --camera 1

Controls:  D = debug panel   SPACE = re-capture (stable)   Q / ESC = quit
"""

import argparse
import cv2

import age_estimator as ae


def run_continuous(est, cap):
    n = 0
    last = None
    debug = False
    print("[INFO] Continuous mode.  D = debug   Q/ESC = quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if n % ae.LIVE_INFER_EVERY == 0:
            face = est.detect(frame)
            last = est.analyse_face(frame, face) if face is not None else None
        if last is not None:
            ae.draw_verdict(frame, last)
            if debug:
                ae.draw_debug(frame, last)
        else:
            ae.draw_status(frame, "No face detected")
        ae.draw_privacy_badge(frame)
        cv2.imshow("Age Estimator (continuous)  |  D=debug  Q=quit", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break
        if k in (ord('d'), ord('D')):
            debug = not debug
            print(f"[INFO] Debug: {'ON' if debug else 'OFF'}")


def run_stable(est, cap):
    captured = None
    history = []
    debug = False
    print("[INFO] Stable mode.  Hold still to capture.")
    print("[INFO] SPACE = re-capture   D = debug   Q/ESC = quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if captured is None:
            face = est.detect(frame)
            if face is None:
                history.clear()
                ae.draw_status(frame, "No face - step into view")
            else:
                v = est.analyse_face(frame, face)
                x, y, w, h = v.bbox
                history.append((x + w/2, y + h/2))
                if len(history) > ae.STABLE_FRAMES:
                    history.pop(0)
                stable = False
                if len(history) >= ae.STABLE_FRAMES:
                    mx = sum(p[0] for p in history)/len(history)
                    my = sum(p[1] for p in history)/len(history)
                    spread = max(((px-mx)**2+(py-my)**2)**0.5 for px,py in history)
                    stable = spread <= ae.STABLE_TOLERANCE_PX
                ae.draw_verdict(frame, v)
                pct = int(100*len(history)/ae.STABLE_FRAMES)
                ae.draw_status(frame, f"Hold still... {min(pct,100)}%" if not stable else "Capturing...")
                if stable:
                    captured = v
                    history.clear()
                    print(f"[INFO] Captured: {v.classification} (age ~{v.age:.0f}, certainty {v.certainty})")
        else:
            ae.draw_verdict(frame, captured)
            ae.draw_status(frame, "Captured - SPACE to redo")
            if debug:
                ae.draw_debug(frame, captured)
        ae.draw_privacy_badge(frame)
        cv2.imshow("Age Estimator (stable)  |  SPACE=redo  D=debug  Q=quit", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break
        if k == ord(' '):
            captured = None
            history.clear()
            print("[INFO] Re-capturing...")
        if k in (ord('d'), ord('D')):
            debug = not debug


def main():
    p = argparse.ArgumentParser(description="Live webcam age classification.")
    p.add_argument('--mode', choices=['continuous', 'stable'], default='continuous')
    p.add_argument('--camera', type=int, default=0)
    args = p.parse_args()

    est = ae.AgeEstimator()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        (run_stable if args.mode == 'stable' else run_continuous)(est, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Session ended. No data was stored or transmitted.")


if __name__ == '__main__':
    main()
