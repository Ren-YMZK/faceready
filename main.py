import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# =========================
# Config
# =========================
# ダウンロードした YuNet モデルのパス
MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")

# 解像度プリセット（rキーで切替）
PREFERRED_RES = [(1280, 720), (960, 540), (640, 360)]
TARGET_FPS = 30
FLIP_HORIZONTAL_DEFAULT = True

# 描画や平滑化のパラメータ
BOX_SMOOTH_ALPHA = 0.35      # 枠の指数移動平均（0=追従速い, 1=全く更新しない わけではない点に注意）
SHOW_LANDMARKS = True        # YuNetは目・鼻・口角の5点を返す
HIGHLIGHT_WARN_RATIO = 0.01  # 飽和ピクセル（255）の割合が1%超で警告

# =========================
# Utils
# =========================
def enumerate_devices(max_index: int = 5):
    """MSMFで0..max_indexを走査して利用可能なデバイス番号を返す"""
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
    """指定デバイスをMSMFでオープン（1フレーム読めたらcapを返す）"""
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
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
    """白飛び簡易検出（グレイ255近傍の画素割合で判定）"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pct = float((gray >= 250).sum()) / float(gray.size)
    return (pct >= ratio_threshold, pct * 100.0)

# =========================
# Main
# =========================
def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH.resolve()}")
        print("→ models フォルダに face_detection_yunet_2023mar.onnx を配置してください。")
        return

    # YuNet のフェイスディテクタ作成
    # 注意: 入力サイズは毎回 setInputSize で現在のフレーム解像度を渡す必要がある
    detector = cv2.FaceDetectorYN_create(
        model=str(MODEL_PATH),
        config="",                # 使用しない
        input_size=(320, 320),    # 仮（後で実際のフレームサイズに更新）
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )

    # カメラ選択
    if len(sys.argv) >= 2:
        try:
            device_index = int(sys.argv[1])
        except ValueError:
            print("device_index must be integer. e.g., python main.py 0")
            return
    else:
        cands = enumerate_devices(5)
        if not cands:
            print("No camera found. Close other apps using the camera and try again.")
            return
        device_index = cands[0]
        print(f"[INFO] Candidates: {cands} -> use: {device_index}")

    # カメラ開始
    res_idx = 0
    width, height = PREFERRED_RES[res_idx]
    flip = FLIP_HORIZONTAL_DEFAULT

    cap = try_open(device_index, width, height, TARGET_FPS)
    if cap is None:
        for (w, h) in PREFERRED_RES[1:]:
            cap = try_open(device_index, w, h, TARGET_FPS)
            if cap is not None:
                width, height = w, h
                break
    if cap is None:
        print("Failed to initialize camera. Check permission or other apps using it.")
        return

    # 入力サイズを実フレームに合わせる
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((actual_w, actual_h))
    print(f"[INFO] Resolution: {actual_w}x{actual_h} / Target FPS: {TARGET_FPS}")

    # 計測
    ts_hist = deque(maxlen=60)
    smoothed_box = None  # [x1,y1,x2,y2]

    help_lines = ["Keys: ESC=Exit  m=Mirror ON/OFF  r=Resolution cycle"]

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame. Trying to continue...")
            continue
        if flip:
            frame = cv2.flip(frame, 1)

        # YuNet 推論
        # detect の返り値: (retval, faces)
        # faces: Nx15 [x, y, w, h, score, l_eye(x,y), r_eye(x,y), nose(x,y), l_mouth(x,y), r_mouth(x,y)]
        retval, faces = detector.detect(frame)

        # もっともスコアの高い顔を選択
        best = None
        if faces is not None and len(faces) > 0:
            faces = faces[np.argsort(faces[:, 4])[::-1]]  # score で降順
            best = faces[0]

        # 平滑化（指数移動平均）
        if best is not None:
            x, y, w, h = best[:4]
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            if smoothed_box is None:
                smoothed_box = box
            else:
                smoothed_box = BOX_SMOOTH_ALPHA * smoothed_box + (1.0 - BOX_SMOOTH_ALPHA) * box
        else:
            smoothed_box = None

        # 表示
        disp = frame.copy()
        warn, pct = highlight_warnings(disp, HIGHLIGHT_WARN_RATIO)
        if smoothed_box is not None:
            x1, y1, x2, y2 = [int(v) for v in smoothed_box]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 255), 2)

        # 5点ランドマーク（任意）
        if SHOW_LANDMARKS and best is not None:
            l_eye = tuple(best[5:7].astype(int))
            r_eye = tuple(best[7:9].astype(int))
            nose  = tuple(best[9:11].astype(int))
            l_m   = tuple(best[11:13].astype(int))
            r_m   = tuple(best[13:15].astype(int))
            for p in [l_eye, r_eye, nose, l_m, r_m]:
                cv2.circle(disp, p, 2, (0, 255, 0), -1, cv2.LINE_AA)

        t1 = time.time()
        ts_hist.append(t1 - t0)
        avg_ms = (sum(ts_hist) / max(1, len(ts_hist))) * 1000.0

        info_lines = [
            f"Device: {device_index}",
            f"Res: {actual_w}x{actual_h}",
            f"Avg Proc: {avg_ms:.1f} ms",
            f"Flip: {'ON' if flip else 'OFF'}",
        ]
        if warn:
            info_lines.append(f"Highlight warn: {pct:.2f}% pixels near 255")

        overlay_text(disp, help_lines + info_lines, pos=(10, 25))
        cv2.imshow("Face Ready - Phase 2 (YuNet)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('m'), ord('M')):
            flip = not flip
        elif key in (ord('r'), ord('R')):
            # 解像度切替：キャプチャを作り直し、YuNet入力サイズも更新
            res_idx = (res_idx + 1) % len(PREFERRED_RES)
            w, h = PREFERRED_RES[res_idx]
            print(f"[INFO] Request resolution -> {w}x{h}")
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
            smoothed_box = None  # リセット

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
