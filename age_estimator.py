"""
age_estimator.py
================
Everything for the age classifier in one file: configuration, face
detection, the pre-trained CNN age model, classification, and drawing.

This is imported by run_live.py and run_sweep.py. Keep all three files in
the same folder and run them from there — no installation needed.

Approach:
    detect face (InsightFace) -> crop -> CNN age model (UTKFace ViT, ONNX)
    -> age >= 18 means ADULT, else UNDERAGE.

The CNN is pre-trained; nothing is trained here. It downloads once on first
run (~345 MB) and then runs fully offline. No data is transmitted or stored.
"""

from __future__ import annotations
import os
import time
import warnings
from typing import Optional, NamedTuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import cv2


# ===========================================================================
# CONFIGURATION  (edit these values to tune behaviour)
# ===========================================================================

# -- Detection --------------------------------------------------------------
DETECTOR_MODEL     = 'buffalo_s'        # InsightFace pack (detection only)
DETECTOR_DET_SIZE  = (640, 640)
DETECTOR_PROVIDERS = ['CPUExecutionProvider']
MIN_FACE_HEIGHT    = 40                  # px; smaller faces are skipped

# -- Age model --------------------------------------------------------------
# Pre-trained Vision Transformer (UTKFace) run locally via ONNX Runtime.
# Auto-downloads on first run. To use a local file instead, set the path:
AGE_MODEL_PATH   = None                  # e.g. r"C:\models\age_model.onnx"
_HF_REPO         = "onnx-community/age-gender-prediction-ONNX"
_HF_FILE         = "onnx/model.onnx"

# Binary decision: predicted age >= this -> ADULT
AGE_ADULT_CUTOFF   = 18.0
# Ages within this band of the cutoff are flagged "borderline" (model error
# is ~4.5 yrs, so this band is inherently uncertain).
AGE_UNCERTAIN_BAND = 5.0

# -- Live mode --------------------------------------------------------------
LIVE_INFER_EVERY    = 5                  # run pipeline every Nth frame
STABLE_TOLERANCE_PX = 40                 # stable-mode: max bbox jitter
STABLE_FRAMES       = 8                  # stable-mode: frames to hold still

# -- Colours (BGR) ----------------------------------------------------------
COLOR_UNDERAGE = ( 50,  50, 210)         # red
COLOR_ADULT    = ( 70, 200,  70)         # green
COLOR_UNKNOWN  = (140, 140, 140)         # grey
STRIP_BG       = (245, 245, 245)
STRIP_TEXT     = ( 35,  35,  35)
STRIP_DIM      = (120, 120, 120)
STRIP_BORDER   = (190, 190, 190)


# ===========================================================================
# RESULT CONTAINER
# ===========================================================================

class FaceVerdict(NamedTuple):
    bbox:           tuple              # (x, y, w, h)
    classification: str                # ADULT | UNDERAGE | TOO_SMALL
    certainty:      Optional[float]    # 1.0 adult .. 0.0 underage
    color:          tuple
    age:            Optional[float]
    gender:         Optional[str]
    gender_conf:    Optional[float]
    det_score:      float
    uncertain:      bool


# ===========================================================================
# CLASSIFICATION RULES
# ===========================================================================

def classify(age: float) -> tuple:
    if age >= AGE_ADULT_CUTOFF:
        return ('ADULT', COLOR_ADULT)
    return ('UNDERAGE', COLOR_UNDERAGE)


def certainty(age: float) -> float:
    """1.0 = definitely adult, 0.0 = definitely underage."""
    score = 0.5 + (age - AGE_ADULT_CUTOFF) / (2.0 * AGE_UNCERTAIN_BAND)
    return round(float(np.clip(score, 0.0, 1.0)), 2)


# ===========================================================================
# THE ENGINE: detector + age model + full pipeline
# ===========================================================================

class AgeEstimator:
    """
    Loads the face detector and the CNN age model once, then analyses faces.
    Construct one of these and reuse it for every frame / image.
    """

    _AGE_INPUT = 224
    _MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __init__(self):
        self._load_detector()
        self._load_age_model()
        # --- optional performance instrumentation (off by default) ---
        # When self.timing is True, per-stage wall-clock times (seconds) are
        # appended to these lists. Used by the benchmark and the sweep's
        # --timing flag; has zero effect on normal operation.
        self.timing = False
        self.t_detect = []   # detection time per frame
        self.t_age    = []   # age-inference time per face

    def reset_timing(self):
        self.t_detect.clear()
        self.t_age.clear()

    # ---- model loading ----------------------------------------------------

    def _load_detector(self):
        from insightface.app import FaceAnalysis
        print("[INFO] Loading face detector (detection only)...")
        self._det = FaceAnalysis(
            name=DETECTOR_MODEL,
            allowed_modules=['detection'],
            providers=DETECTOR_PROVIDERS,
        )
        self._det.prepare(ctx_id=0, det_size=DETECTOR_DET_SIZE)
        print("[INFO] Face detector ready.")

    def _load_age_model(self):
        import onnxruntime as ort

        path = self._resolve_age_model()
        print("[INFO] Loading age model into ONNX Runtime (CPU)...")
        self._age = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self._age_in   = self._age.get_inputs()[0].name
        self._age_outs = [o.name for o in self._age.get_outputs()]
        print(f"[INFO] Age model ready. input='{self._age_in}' outputs={self._age_outs}")

    def _resolve_age_model(self) -> str:
        if AGE_MODEL_PATH and os.path.isfile(AGE_MODEL_PATH):
            return AGE_MODEL_PATH
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("[ERROR] huggingface_hub is needed to download the age model.")
            print("        Run:  pip install huggingface_hub")
            print("        Or download the .onnx file and set AGE_MODEL_PATH.")
            raise
        print(f"[INFO] Resolving age model (first run downloads ~345 MB)...")
        p = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        print(f"[INFO] Age model file: {p}")
        return p

    # ---- detection --------------------------------------------------------

    def detect(self, frame):
        """Highest-confidence face above the size threshold, or None."""
        if self.timing:
            _t = time.perf_counter()
            faces = self._det.get(frame)
            self.t_detect.append(time.perf_counter() - _t)
        else:
            faces = self._det.get(frame)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        if (best.bbox[3] - best.bbox[1]) < MIN_FACE_HEIGHT:
            return None
        return best

    def detect_all(self, frame):
        """All faces above the size threshold."""
        if self.timing:
            _t = time.perf_counter()
            faces = self._det.get(frame)
            self.t_detect.append(time.perf_counter() - _t)
        else:
            faces = self._det.get(frame)
        return [f for f in faces
                if (f.bbox[3] - f.bbox[1]) >= MIN_FACE_HEIGHT]

    # ---- age inference ----------------------------------------------------

    def _predict_age(self, face_bgr):
        img = cv2.resize(face_bgr, (self._AGE_INPUT, self._AGE_INPUT),
                         interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

        if self.timing:
            _t = time.perf_counter()
            outputs = self._age.run(self._age_outs, {self._age_in: img})
            self.t_age.append(time.perf_counter() - _t)
        else:
            outputs = self._age.run(self._age_outs, {self._age_in: img})

        flat = None
        for o in outputs:
            a = np.asarray(o).flatten()
            if a.size >= 2:
                flat = a
                break
        if flat is None:
            age = float(np.asarray(outputs[0]).flatten()[0])
            return float(np.clip(age, 0, 100)), 'Unknown', 0.5

        age = float(np.clip(flat[0], 0, 100))
        g_logit = float(flat[1])
        female_p = g_logit if 0.0 <= g_logit <= 1.0 else 1.0/(1.0+np.exp(-g_logit))
        if female_p >= 0.5:
            gender, conf = 'Female', female_p
        else:
            gender, conf = 'Male', 1.0 - female_p
        return age, gender, round(float(conf), 3)

    # ---- full analysis ----------------------------------------------------

    def analyse_face(self, frame, face) -> FaceVerdict:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bbox = (x1, y1, x2 - x1, y2 - y1)
        det  = round(float(face.det_score), 3)

        mx, my = int((x2-x1)*0.10), int((y2-y1)*0.10)
        crop = frame[max(0,y1-my):min(h_img,y2+my),
                     max(0,x1-mx):min(w_img,x2+mx)]

        if crop.size == 0 or (y2 - y1) < MIN_FACE_HEIGHT:
            return FaceVerdict(bbox, 'TOO_SMALL', None, COLOR_UNKNOWN,
                               None, None, None, det, False)

        age, gender, gconf = self._predict_age(crop)
        label, color = classify(age)
        return FaceVerdict(
            bbox, label, certainty(age), color,
            round(age, 1), gender, gconf, det,
            abs(age - AGE_ADULT_CUTOFF) <= AGE_UNCERTAIN_BAND,
        )


# ===========================================================================
# DRAWING
# ===========================================================================

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MONO = cv2.FONT_HERSHEY_PLAIN
_LINE_H, _PAD_V, _PAD_H = 28, 13, 16


def _rounded_rect(img, x1, y1, x2, y2, color, t, r=5):
    cv2.line(img, (x1+r,y1),(x2-r,y1), color, t); cv2.line(img,(x1+r,y2),(x2-r,y2),color,t)
    cv2.line(img, (x1,y1+r),(x1,y2-r), color, t); cv2.line(img,(x2,y1+r),(x2,y2-r),color,t)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,t); cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,t)
    cv2.ellipse(img,(x1+r,y2-r),(r,r),90,0,90,color,t);  cv2.ellipse(img,(x2-r,y2-r),(r,r),0,0,90,color,t)


def _tag(img, text, x, y, bg, above, scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1); p = 4
    if above: bx1,by1,bx2,by2 = x, max(0,y-th-p*2), x+tw+p*2, y
    else:     bx1,by1,bx2,by2 = x, y, x+tw+p*2, y+th+p*2
    cv2.rectangle(img,(bx1,by1),(bx2,by2),bg,cv2.FILLED)
    cv2.putText(img,text,(bx1+p,by2-p),_FONT,scale,(255,255,255),1,cv2.LINE_AA)


def draw_verdict(frame, v, index=None):
    x, y, w, h = v.bbox
    _rounded_rect(frame, x, y, x+w, y+h, v.color, 2)
    prefix = f"F{index} " if index is not None else ""
    if v.classification == 'TOO_SMALL':
        _tag(frame, f"{prefix}TOO SMALL", x, y, v.color, True); return
    _tag(frame, f"{prefix}age ~{v.age:.0f}" + ("?" if v.uncertain else ""),
         x, y, v.color, True)
    _tag(frame, v.classification, x, y+h, v.color, False)


def draw_status(frame, msg):
    fh, fw = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(msg, _FONT, 0.75, 2)
    x, y = (fw-tw)//2, fh-30
    cv2.putText(frame, msg, (x+1,y+1), _FONT, 0.75, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, msg, (x,y),     _FONT, 0.75, (180,180,180), 2, cv2.LINE_AA)


def draw_privacy_badge(frame):
    text = "LOCAL  |  NO DATA STORED"
    fh, fw = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.42, 1); pad = 8
    x1, y1 = fw-tw-pad*2-10, 10; x2, y2 = fw-10, y1+th+pad*2
    ov = frame.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),(20,20,20),cv2.FILLED)
    cv2.addWeighted(ov,0.70,frame,0.30,0,frame)
    cv2.putText(frame,text,(x1+pad,y2-pad),_FONT,0.42,(230,230,230),1,cv2.LINE_AA)


def draw_debug(frame, v):
    if v.classification == 'TOO_SMALL':
        return
    lines = [
        f"Predicted age: {v.age:.1f} yrs",
        f"Cutoff:        {AGE_ADULT_CUTOFF:.0f} yrs",
        f"Class:         {v.classification}",
        f"Certainty:     {v.certainty:.2f}",
        f"Borderline:    {'YES' if v.uncertain else 'no'}",
        f"Gender:        {v.gender} ({v.gender_conf:.0%})",
        f"Det score:     {v.det_score:.3f}",
    ]
    lh, pad, pw = 20, 9, 290
    ph = len(lines)*lh + pad*2
    fh, fw = frame.shape[:2]
    px, py = 10, fh-ph-10
    ov = frame.copy(); cv2.rectangle(ov,(px,py),(px+pw,py+ph),(12,12,12),cv2.FILLED)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (px+pad, py+pad+i*lh+lh-3),
                    _FONT_MONO, 1.0, (200,200,200), 1, cv2.LINE_AA)


def make_polaroid(img, verdicts, filename):
    """Original image + white description strip below (for batch mode)."""
    out = img.copy()
    for i, v in enumerate(verdicts, 1):
        x, y, w, h = v.bbox
        _rounded_rect(out, x, y, x+w, y+h, v.color, 2)
        if v.classification == 'TOO_SMALL':
            _tag(out, f"F{i} TOO SMALL", x, y, v.color, True, 0.52)
        else:
            _tag(out, f"F{i} age ~{v.age:.0f}", x, y, v.color, True, 0.52)
            _tag(out, v.classification, x, y+h, v.color, False, 0.52)
    if not verdicts:
        fh, fw = out.shape[:2]
        ov = out.copy(); cv2.rectangle(ov,(0,fh-44),(fw,fh),(30,30,30),cv2.FILLED)
        cv2.addWeighted(ov,0.60,out,0.40,0,out)
        cv2.putText(out,"NO FACE DETECTED",(fw//2-115,fh-14),_FONT,0.70,(210,210,210),2,cv2.LINE_AA)

    img_w   = img.shape[1]
    n_info  = max(1, len(verdicts))
    strip_h = _PAD_V + _LINE_H + 6 + _LINE_H*n_info + _LINE_H + _PAD_V
    strip   = np.full((strip_h, img_w, 3), STRIP_BG, np.uint8)
    strip[0:2,:] = STRIP_BORDER

    n = len(verdicts)
    fs = "no face detected" if n==0 else ("1 face detected" if n==1 else f"{n} faces detected")
    cv2.putText(strip, f"  {filename}   |   {fs}", (_PAD_H,_PAD_V+_LINE_H-6),
                _FONT, 0.44, STRIP_TEXT, 1, cv2.LINE_AA)
    sep = _PAD_V+_LINE_H+3
    strip[sep:sep+1,_PAD_H:img_w-_PAD_H] = STRIP_BORDER

    top = sep+7
    if not verdicts:
        cv2.putText(strip,"  No face detected - no facial region found.",
                    (_PAD_H+8, top+_LINE_H-5),_FONT_MONO,0.95,STRIP_DIM,1,cv2.LINE_AA)
    else:
        for i, v in enumerate(verdicts, 1):
            ty = top + (i-1)*_LINE_H + _LINE_H - 6
            cv2.circle(strip,(_PAD_H+7,ty-5),6,v.color,cv2.FILLED)
            cv2.circle(strip,(_PAD_H+7,ty-5),6,STRIP_BORDER,1)
            if v.classification == 'TOO_SMALL':
                line = f"  F{i}  TOO SMALL (below measurement threshold)"
                cv2.putText(strip,line,(_PAD_H+18,ty),_FONT_MONO,0.96,STRIP_DIM,1,cv2.LINE_AA)
            else:
                unc = "  [borderline]" if v.uncertain else ""
                line = (f"  F{i}  {v.classification:<9}  certainty: {v.certainty:.2f}"
                        f"  age ~{v.age:.0f}  {v.gender}{unc}")
                cv2.putText(strip,line,(_PAD_H+18,ty),_FONT_MONO,0.96,STRIP_TEXT,1,cv2.LINE_AA)

    fy = strip_h - _PAD_V + 2
    strip[fy-_LINE_H-2:fy-_LINE_H-1,_PAD_H:img_w-_PAD_H] = STRIP_BORDER
    cv2.putText(strip,"  LOCAL ANALYSIS  |  NO DATA STORED",
                (_PAD_H,fy),_FONT_MONO,0.85,STRIP_DIM,1,cv2.LINE_AA)

    return np.vstack([out, strip])
