import sys
import time
from collections import deque
import cv2
import numpy as np

# ========= User-tunable defaults =========
PREFERRED_RES = [(1280, 720), (960, 540), (640, 360)]  # press 'r' to cycle
TARGET_FPS = 30
FLIP_HORIZONTAL_DEFAULT = True  # mirror view for self-portrait
HIGHLIGHT_WARN_RATIO = 0.01     # warn if top 1% pixels are saturated

# ========= Utils =========
def try_open(index: int, width: int, height: int, fps: int):
    """Try opening a camera using Media Foundation (MSMF) for Windows stability."""
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    # Request properties (may not be honored by all devices)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def enumerate_devices(max_index: int = 5):
    """Lightweight probe for available device indices using MSMF."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found

def overlay_text(img, lines, pos=(10, 25)):
    x, y = pos
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 20

def highlight_warnings(bgr, ratio_threshold=0.01):
    """Return (warn_flag, percent_saturated) based on grayscale histogram."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Count saturated pixels (255)
    sat = (gray >= 250).sum()
    total = gray.size
    pct = sat / max(1, total)
    return (pct >= ratio_threshold, pct * 100.0)

# ========= Main =========
def main():
    # arg: python capture_preview_en.py [device_index]
    device_index = None
    if len(sys.argv) >= 2:
        try:
            device_index = int(sys.argv[1])
        except ValueError:
            print("device_index must be an integer, e.g., python capture_preview_en.py 0")
            return

    # enumerate devices if not specified
    if device_index is None:
        candidates = enumerate_devices(5)
        if not candidates:
            print("No camera found. Check USB connection or if another app is using it.")
            return
        device_index = candidates[0]
        print(f"[INFO] Candidates: {candidates} -> use: {device_index}")
    else:
        print(f"[INFO] device index={device_index}")

    # open with first workable resolution
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
        print("Failed to initialize camera. Check permissions or close other apps using it.")
        return

    print(f"[INFO] Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} / Target FPS: {TARGET_FPS}")

    ts_hist = deque(maxlen=60)
    frame_count, last_report = 0, time.time()

    help_lines = [
        "Keys: ESC=Exit  m=Mirror ON/OFF  r=Resolution cycle",
    ]

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame. Trying to continue...")
            continue

        if flip:
            frame = cv2.flip(frame, 1)

        warn, pct = highlight_warnings(frame, HIGHLIGHT_WARN_RATIO)

        t1 = time.time()
        ts_hist.append(t1 - t0)
        avg_proc_ms = (sum(ts_hist) / max(1, len(ts_hist))) * 1000.0

        info = [
            f"Device: {device_index}",
            f"Res: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            f"Avg Proc: {avg_proc_ms:.1f} ms",
            f"Flip: {'ON' if flip else 'OFF'}",
        ]
        if warn:
            info.append(f"Highlight warn: {pct:.2f}% pixels near 255")

        disp = frame.copy()
        overlay_text(disp, help_lines + info, pos=(10, 25))

        cv2.imshow("Preview (Phase 1.1 - MSMF)", disp)
        frame_count += 1

        now = time.time()
        if now - last_report >= 60:
            print(f"[INFO] Approx FPS: {frame_count / (now - last_report):.1f}")
            frame_count = 0
            last_report = now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('m'), ord('M')):
            flip = not flip
        elif key in (ord('r'), ord('R')):
            res_idx = (res_idx + 1) % len(PREFERRED_RES)
            w, h = PREFERRED_RES[res_idx]
            print(f"[INFO] Request resolution -> {w}x{h}")
            cap.release()
            cap = try_open(device_index, w, h, TARGET_FPS)
            if cap is None:
                print("[WARN] Resolution not supported. Reverting.")
                res_idx = (res_idx - 1) % len(PREFERRED_RES)
                w, h = PREFERRED_RES[res_idx]
                cap = try_open(device_index, w, h, TARGET_FPS)
                if cap is None:
                    print("[ERROR] Re-init failed. Exiting.")
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
