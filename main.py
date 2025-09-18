import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pyvirtualcam

# =========================
# Config
# =========================
MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")
PREFERRED_RES = [(1280, 720)]   # 固定解像度（Zoom用に安定化）
TARGET_FPS = 30
FLIP_HORIZONTAL_DEFAULT = True

# Debug / UI
SHOW_LANDMARKS = False   # 仮想カメラ出力ではデバッグ描画を切る
BOX_SMOOTH_ALPHA = 0.35
HIGHLIGHT_WARN_RATIO = 0.01

# ===== Beauty parameters =====
BEAUTY_DEFAULT_ON = True
ROI_DOWNSCALE = 0.6
BILATERAL_ITER   = 2
BILATERAL_SIG_C  = 38
BILATERAL_SIG_S  = 7
CONTRAST_ALPHA = 1.18
BRIGHT_BETA    = 18
CLAHE_CLIP = 1.6
BLEND_ALPHA = 0.52
ELLIPSE_MARGIN = 0.10
FEATHER_PX = 25
EDGE_PROTECT_WEIGHT = 0.35

# =========================
# Utils
# =========================
def enumerate_devices(max_index: int = 5):
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found

def try_open(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def highlight_warnings(bgr, ratio_threshold=0.01):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pct = float((gray >= 250).sum()) / float(gray.size)
    return (pct >= ratio_threshold, pct * 100.0)

# =========================
# Beauty core
# =========================
def make_ellipse_mask(h, w, rect, margin=0.10, feather=25):
    x1, y1, x2, y2 = rect
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    mask = np.zeros((h, w), dtype=np.uint8)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = int((x2 - x1) * (1.0 + margin) / 2.0)
    ry = int((y2 - y1) * (1.0 + margin) / 2.0)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask

def edge_protect_mask(bgr, rect, feather=7):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, v) for v in map(int, rect)]
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.ones((h, w), dtype=np.float32)

    gray = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    edges = cv2.GaussianBlur(edges, (0, 0), feather)
    edges = edges.astype(np.float32) / 255.0

    protect = 1.0 - EDGE_PROTECT_WEIGHT * edges
    protect = np.clip(protect, 0.0, 1.0)

    full = np.ones((h, w), dtype=np.float32)
    full[y1:y2, x1:x2] = protect
    return full

def apply_edge_preserving_beauty(bgr, rect):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, v) for v in map(int, rect)]
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return bgr

    roi = bgr[y1:y2, x1:x2].copy()
    if ROI_DOWNSCALE < 1.0:
        small_size = (max(1, int(roi.shape[1] * ROI_DOWNSCALE)),
                      max(1, int(roi.shape[0] * ROI_DOWNSCALE)))
        proc = cv2.resize(roi, small_size, interpolation=cv2.INTER_AREA)
    else:
        proc = roi

    for _ in range(max(1, BILATERAL_ITER)):
        proc = cv2.bilateralFilter(proc, d=0, sigmaColor=BILATERAL_SIG_C, sigmaSpace=BILATERAL_SIG_S)

    if proc.shape[:2] != roi.shape[:2]:
        proc = cv2.resize(proc, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=CONTRAST_ALPHA, beta=BRIGHT_BETA)

    blended = cv2.addWeighted(enhanced, BLEND_ALPHA, roi, 1.0 - BLEND_ALPHA, 0)

    canvas = bgr.copy()
    canvas[y1:y2, x1:x2] = blended

    mask = make_ellipse_mask(h, w, (x1, y1, x2, y2), margin=ELLIPSE_MARGIN, feather=FEATHER_PX).astype(np.float32) / 255.0
    protect = edge_protect_mask(bgr, (x1, y1, x2, y2))
    mask *= protect
    mask3 = cv2.merge([mask, mask, mask])

    out = (bgr.astype(np.float32) * (1.0 - mask3) +
           canvas.astype(np.float32) * mask3)
    return out.astype(np.uint8)

# =========================
# Main
# =========================
def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH.resolve()}")
        return

    detector = cv2.FaceDetectorYN_create(
        model=str(MODEL_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    cands = enumerate_devices(5)
    if not cands:
        print("No camera found.")
        return
    device_index = cands[0]
    print(f"[INFO] Candidates: {cands} -> use: {device_index}")

    width, height = PREFERRED_RES[0]
    flip = FLIP_HORIZONTAL_DEFAULT
    beauty_on = BEAUTY_DEFAULT_ON

    cap = try_open(device_index, width, height, TARGET_FPS)
    if cap is None:
        print("Failed to initialize camera.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((actual_w, actual_h))
    print(f"[INFO] Resolution: {actual_w}x{actual_h} / Target FPS: {TARGET_FPS}")

    smoothed_box = None

    # 🎥 Virtual Camera Start
    with pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=TARGET_FPS) as cam:
        print(f"[INFO] Virtual camera started: {cam.device}")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            retval, faces = detector.detect(frame)
            best = None
            if faces is not None and len(faces) > 0:
                faces = faces[np.argsort(faces[:, 4])[::-1]]
                best = faces[0] if faces[0, 4] >= 0.8 else None

            if best is not None:
                x, y, w, h = best[:4]
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                if smoothed_box is None:
                    smoothed_box = box
                else:
                    smoothed_box = BOX_SMOOTH_ALPHA * smoothed_box + (1.0 - BOX_SMOOTH_ALPHA) * box
            else:
                smoothed_box = None

            output = frame
            if beauty_on and smoothed_box is not None:
                output = apply_edge_preserving_beauty(output, smoothed_box)

            # 仮想カメラに送信 (RGB形式に変換)
            rgb_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cam.send(rgb_frame)
            cam.sleep_until_next_frame()

            # ESCで終了（仮想カメラでもキー入力を拾えるようにする）
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
