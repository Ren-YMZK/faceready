import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np

# =========================
# Config
# =========================
MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")
PREFERRED_RES = [(1280, 720), (960, 540), (640, 360)]
TARGET_FPS = 30
FLIP_HORIZONTAL_DEFAULT = True

# Debug / UI
SHOW_LANDMARKS = True
BOX_SMOOTH_ALPHA = 0.35
HIGHLIGHT_WARN_RATIO = 0.01

# ===== Beauty parameters (STRONG but natural) =====
BEAUTY_DEFAULT_ON = True
# 1) 処理解像度を下げて高速化（0.6〜0.8が目安、0.5だとさらに高速・画質やや低下）
ROI_DOWNSCALE = 0.6
# 2) エッジ保存スムージング（バイラテラル）
BILATERAL_ITER   = 2         # 繰り返し回数（1〜3）
BILATERAL_SIG_C  = 38        # sigmaColor（色の平滑）
BILATERAL_SIG_S  = 7         # sigmaSpace（空間）
# 3) トーン調整（やや強め）
CONTRAST_ALPHA = 1.18        # 1.10〜1.25
BRIGHT_BETA    = 18          # 10〜30
# 4) CLAHE（明部の伸びすぎ抑制・軽め）
CLAHE_CLIP = 1.6             # 1.2〜2.0
# 5) 合成
BLEND_ALPHA = 0.52           # 0.45〜0.60（高いほど強い）
ELLIPSE_MARGIN = 0.10
FEATHER_PX = 25
EDGE_PROTECT_WEIGHT = 0.35   # 輪郭部分は処理を弱める（0.0で無効、0.2〜0.5）

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

def overlay_text(img, lines, pos=(10, 25)):
    x, y = pos
    for line in lines:
        cv2.putText(img, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 20

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
    """輪郭で処理を弱めるための0..1マスク（1=処理強、エッジは小さく）"""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, v) for v in map(int, rect)]
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.ones((h, w), dtype=np.float32)

    # 顔周辺のエッジを検出（Canny）
    gray = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    edges = cv2.GaussianBlur(edges, (0, 0), feather)
    edges = edges.astype(np.float32) / 255.0  # 0..1（1=強いエッジ）

    # 1 - w*edges でエッジ周辺は処理弱め
    protect = 1.0 - EDGE_PROTECT_WEIGHT * edges
    protect = np.clip(protect, 0.0, 1.0)

    full = np.ones((h, w), dtype=np.float32)
    full[y1:y2, x1:x2] = protect
    return full

def apply_edge_preserving_beauty(bgr, rect):
    """
    顔矩形ベースのエッジ保存・高速美顔
    - ROIを縮小してバイラテラル(反復) → 拡大
    - CLAHE(軽) + コントラスト/明るさ
    - 輪郭部は処理を弱める（edge_protect）
    - 楕円マスクで合成
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, v) for v in map(int, rect)]
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return bgr

    roi = bgr[y1:y2, x1:x2].copy()

    # ダウンサンプリング（高速化＆モザイク感の抑制のバランス）
    if ROI_DOWNSCALE < 1.0:
        small_size = (max(1, int(roi.shape[1] * ROI_DOWNSCALE)),
                      max(1, int(roi.shape[0] * ROI_DOWNSCALE)))
        proc = cv2.resize(roi, small_size, interpolation=cv2.INTER_AREA)
    else:
        proc = roi

    # エッジ保持スムージング（バイラテラル複数回）
    for _ in range(max(1, BILATERAL_ITER)):
        proc = cv2.bilateralFilter(proc, d=0, sigmaColor=BILATERAL_SIG_C, sigmaSpace=BILATERAL_SIG_S)

    # 元サイズへ戻す
    if proc.shape[:2] != roi.shape[:2]:
        proc = cv2.resize(proc, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

    # CLAHE（軽め）＋ コントラスト/明るさ
    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=CONTRAST_ALPHA, beta=BRIGHT_BETA)

    # ROIとブレンド（処理の“強さ”）
    blended = cv2.addWeighted(enhanced, BLEND_ALPHA, roi, 1.0 - BLEND_ALPHA, 0)

    # フルキャンバスへ
    canvas = bgr.copy()
    canvas[y1:y2, x1:x2] = blended

    # 楕円マスク
    mask = make_ellipse_mask(h, w, (x1, y1, x2, y2), margin=ELLIPSE_MARGIN, feather=FEATHER_PX).astype(np.float32) / 255.0

    # 輪郭保護マスク（エッジ付近の処理を弱める）
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
        score_threshold=0.9,   # 誤検出を減らす
        nms_threshold=0.3,
        top_k=5000
    )

    if len(sys.argv) >= 2:
        try:
            device_index = int(sys.argv[1])
        except ValueError:
            print("device_index must be int.")
            return
    else:
        cands = enumerate_devices(5)
        if not cands:
            print("No camera found.")
            return
        device_index = cands[0]
        print(f"[INFO] Candidates: {cands} -> use: {device_index}")

    res_idx = 0
    width, height = PREFERRED_RES[res_idx]
    flip = FLIP_HORIZONTAL_DEFAULT
    beauty_on = BEAUTY_DEFAULT_ON

    cap = try_open(device_index, width, height, TARGET_FPS)
    if cap is None:
        for (w, h) in PREFERRED_RES[1:]:
            cap = try_open(device_index, w, h, TARGET_FPS)
            if cap is not None:
                width, height = w, h
                break
    if cap is None:
        print("Failed to initialize camera.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((actual_w, actual_h))
    print(f"[INFO] Resolution: {actual_w}x{actual_h} / Target FPS: {TARGET_FPS}")

    ts_hist = deque(maxlen=60)
    smoothed_box = None

    help_lines = ["Keys: ESC=Exit  m=Mirror  r=Res cycle  b=Beauty ON/OFF"]

    while True:
        t0 = time.time()
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

        disp = output.copy()
        if smoothed_box is not None:
            x1, y1, x2, y2 = [int(v) for v in smoothed_box]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 255), 2)
        if SHOW_LANDMARKS and best is not None:
            pts = [
                tuple(best[5:7].astype(int)),
                tuple(best[7:9].astype(int)),
                tuple(best[9:11].astype(int)),
                tuple(best[11:13].astype(int)),
                tuple(best[13:15].astype(int)),
            ]
            for p in pts:
                cv2.circle(disp, p, 2, (0, 255, 0), -1, cv2.LINE_AA)

        t1 = time.time()
        ts_hist.append(t1 - t0)
        avg_ms = (sum(ts_hist) / max(1, len(ts_hist))) * 1000.0
        warn, pct = highlight_warnings(disp, HIGHLIGHT_WARN_RATIO)

        info_lines = [
            f"Device: {device_index}",
            f"Res: {actual_w}x{actual_h}",
            f"Avg Proc: {avg_ms:.1f} ms",
            f"Flip: {'ON' if flip else 'OFF'}",
            f"Beauty: {'ON' if beauty_on else 'OFF'}",
        ]
        if warn:
            info_lines.append(f"Highlight warn: {pct:.2f}% near 255")

        overlay_text(disp, help_lines + info_lines, pos=(10, 25))
        cv2.imshow("Face Ready - Phase 3 (Edge-preserving beauty)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('m'), ord('M')):
            flip = not flip
        elif key in (ord('r'), ord('R')):
            res_idx = (res_idx + 1) % len(PREFERRED_RES)
            w, h = PREFERRED_RES[res_idx]
            print(f"[INFO] Switch resolution -> {w}x{h}")
            cap.release()
            cap = try_open(device_index, w, h, TARGET_FPS)
            if cap is None:
                print("[WARN] Unsupported. Reverting.")
                res_idx = (res_idx - 1) % len(PREFERRED_RES)
                w, h = PREFERRED_RES[res_idx]
                cap = try_open(device_index, w, h, TARGET_FPS)
                if cap is None:
                    print("[ERROR] Re-init failed."); break
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detector.setInputSize((actual_w, actual_h))
            smoothed_box = None
        elif key in (ord('b'), ord('B')):
            beauty_on = not beauty_on

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
