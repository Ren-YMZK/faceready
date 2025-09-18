import sys
import time
from collections import deque

import cv2
import numpy as np

# =========================
# ユーザー調整しやすい既定値
# =========================
PREFERRED_RES = [(1280, 720), (960, 540), (640, 360)]  # rキーで切替
TARGET_FPS = 30
FLIP_HORIZONTAL_DEFAULT = True  # ミラー表示（自撮り風）

# =========================
# ユーティリティ
# =========================
def try_open(index: int, width: int, height: int, fps: int):
    """指定デバイスindexでOpenCVキャプチャを試す。成功時はcapを返す。"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # WindowsではDirectShowが安定
    if not cap.isOpened():
        return None

    # 解像度とFPSを要求（必ずしも反映されるとは限らない点に注意）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 1フレーム読めるか確認
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def enumerate_devices(max_index: int = 5):
    """0..max_index をざっくり走査して利用可能なデバイス番号を返す。"""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def overlay_text(img, lines, pos=(10, 25)):
    """複数行のテキストを左上に重ねる（日本語可）"""
    x, y = pos
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 20


# =========================
# メイン
# =========================
def main():
    # 引数：python capture_preview.py [device_index]
    device_index = None
    if len(sys.argv) >= 2:
        try:
            device_index = int(sys.argv[1])
        except ValueError:
            print("device_index は整数で指定してください。例: python capture_preview.py 0")
            return

    # デバイス列挙（初回のみ数秒かかることあり）
    if device_index is None:
        candidates = enumerate_devices(5)
        if not candidates:
            print("利用可能なカメラが見つかりませんでした。USB接続や他アプリの占有を確認してください。")
            return
        device_index = candidates[0]
        print(f"[INFO] 使用デバイス候補: {candidates} → 選択: {device_index}")
    else:
        print(f"[INFO] 指定デバイス index={device_index}")

    # 解像度プリセットの先頭から試行
    res_idx = 0
    width, height = PREFERRED_RES[res_idx]
    flip = FLIP_HORIZONTAL_DEFAULT

    cap = try_open(device_index, width, height, TARGET_FPS)
    if cap is None:
        # ほかの解像度でもう一度だけ試す
        for (w, h) in PREFERRED_RES[1:]:
            cap = try_open(device_index, w, h, TARGET_FPS)
            if cap is not None:
                width, height = w, h
                break
    if cap is None:
        print("カメラを初期化できませんでした。別のアプリが使用中の可能性、または権限設定を確認してください。")
        return

    print(f"[INFO] 解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} / 目標FPS: {TARGET_FPS}")

    # FPS計測用（移動平均）
    ts_hist = deque(maxlen=60)
    frame_count = 0
    last_report = time.time()

    help_lines = [
        "キー操作: ESC=終了  m=ミラーON/OFF  r=解像度切替",
    ]

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[WARN] フレーム取得に失敗。継続を試みます…")
            continue

        if flip:
            frame = cv2.flip(frame, 1)

        # FPS・レイテンシ目安（処理時間）を簡易表示
        t1 = time.time()
        ts_hist.append(t1 - t0)
        avg_proc_ms = (sum(ts_hist) / max(1, len(ts_hist))) * 1000.0

        # 画面左上に情報をオーバーレイ
        info = [
            f"Device: {device_index}",
            f"Res: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            f"Avg Proc: {avg_proc_ms:.1f} ms",
            f"Flip: {'ON' if flip else 'OFF'}",
        ]
        disp = frame.copy()
        overlay_text(disp, help_lines + info, pos=(10, 25))

        cv2.imshow("Preview (Phase 1)", disp)
        frame_count += 1

        # 1分間隔で概算fpsをレポート
        now = time.time()
        if now - last_report >= 60:
            print(f"[INFO] 概算FPS（表示側ベース）: {frame_count / (now - last_report):.1f}")
            frame_count = 0
            last_report = now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('m'), ord('M')):
            flip = not flip
        elif key in (ord('r'), ord('R')):
            # 解像度切替：キャプチャを作り直す
            res_idx = (res_idx + 1) % len(PREFERRED_RES)
            w, h = PREFERRED_RES[res_idx]
            print(f"[INFO] 解像度変更要求 → {w}x{h}")
            cap.release()
            cap = try_open(device_index, w, h, TARGET_FPS)
            if cap is None:
                print("[WARN] この解像度は利用できませんでした。元に戻します。")
                res_idx = (res_idx - 1) % len(PREFERRED_RES)
                w, h = PREFERRED_RES[res_idx]
                cap = try_open(device_index, w, h, TARGET_FPS)
                if cap is None:
                    print("[ERROR] カメラの再初期化に失敗しました。終了します。")
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
