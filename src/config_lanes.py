"""
config_lanes.py
Tool interaktif untuk mendefinisikan ROI (Region of Interest) setiap lajur jalan
beserta arah pergerakan yang diizinkan. Hasil disimpan ke config/lanes.json.

Cara pakai:
    python -m src.config_lanes --video traffic.mp4 --out config/lanes.json

Alur interaksi:
    1. Klik 4 titik di gambar untuk membentuk polygon lajur (searah jarum jam).
    2. Klik 2 titik berikutnya: titik awal -> titik akhir panah arah yang diizinkan.
    3. Tekan 'n' untuk menambah lajur baru, 's' untuk simpan & keluar,
       'r' untuk reset lajur yang sedang dibuat, 'q' untuk batal tanpa menyimpan.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# Konstanta warna BGR untuk visualisasi
WARNA_POLYGON_BARU = (0, 255, 255)   # kuning: lajur yang sedang dibuat
WARNA_POLYGON_LAMA = (255, 180, 0)   # biru muda: lajur yang sudah disimpan
WARNA_PANAH = (0, 200, 0)            # hijau: panah arah lajur
WARNA_TITIK = (0, 0, 255)            # merah: titik klik
WARNA_TEKS = (255, 255, 255)         # putih: label


def baca_frame_pertama(path_video: str) -> np.ndarray:
    """Ambil 1 frame pertama dari video sebagai kanvas untuk klik ROI."""
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak bisa dibuka: {path_video}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Gagal membaca frame pertama video.")
    return frame


def gambar_overlay(
    base: np.ndarray,
    lajur_tersimpan: list,
    titik_polygon_baru: list,
    titik_panah_baru: list,
    status_teks: str,
) -> np.ndarray:
    """Render visualisasi lajur (yang sudah disimpan + yang sedang dibuat)."""
    kanvas = base.copy()

    # Gambar lajur-lajur yang sudah disimpan (biru muda) + panahnya
    for i, lajur in enumerate(lajur_tersimpan):
        pts = np.array(lajur["polygon"], dtype=np.int32)
        cv2.polylines(kanvas, [pts], isClosed=True, color=WARNA_POLYGON_LAMA, thickness=2)
        # Isi semi-transparan agar terlihat sebagai "area"
        overlay = kanvas.copy()
        cv2.fillPoly(overlay, [pts], WARNA_POLYGON_LAMA)
        cv2.addWeighted(overlay, 0.15, kanvas, 0.85, 0, kanvas)
        # Panah arah (start -> end)
        p1 = tuple(lajur["arrow"][0])
        p2 = tuple(lajur["arrow"][1])
        cv2.arrowedLine(kanvas, p1, p2, WARNA_PANAH, 2, tipLength=0.3)
        # Label ID lajur
        cx = int(np.mean([p[0] for p in lajur["polygon"]]))
        cy = int(np.mean([p[1] for p in lajur["polygon"]]))
        cv2.putText(kanvas, f"L{i}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNA_TEKS, 2)

    # Gambar polygon yang sedang dibuat (kuning) - titik & garis bertahap
    for p in titik_polygon_baru:
        cv2.circle(kanvas, p, 4, WARNA_TITIK, -1)
    if len(titik_polygon_baru) >= 2:
        for a, b in zip(titik_polygon_baru, titik_polygon_baru[1:]):
            cv2.line(kanvas, a, b, WARNA_POLYGON_BARU, 2)
    if len(titik_polygon_baru) == 4:
        cv2.line(kanvas, titik_polygon_baru[-1], titik_polygon_baru[0],
                 WARNA_POLYGON_BARU, 2)

    # Gambar titik panah yang sedang dibuat
    for p in titik_panah_baru:
        cv2.circle(kanvas, p, 4, (0, 200, 200), -1)
    if len(titik_panah_baru) == 2:
        cv2.arrowedLine(kanvas, titik_panah_baru[0], titik_panah_baru[1],
                        WARNA_PANAH, 2, tipLength=0.3)

    # Bar status di bagian atas
    cv2.rectangle(kanvas, (0, 0), (kanvas.shape[1], 22), (40, 40, 40), -1)
    cv2.putText(kanvas, status_teks, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WARNA_TEKS, 1)
    return kanvas


def jalankan_konfigurator(path_video: str, path_output: str) -> None:
    """Loop utama: tampilkan frame, tangkap klik, simpan ke JSON."""
    frame = baca_frame_pertama(path_video)
    h, w = frame.shape[:2]

    # State global mouse callback
    titik_polygon_baru: list = []   # menampung 4 klik polygon
    titik_panah_baru: list = []     # menampung 2 klik panah
    lajur_tersimpan: list = []      # list of {"polygon": [...], "arrow": [...]}

    def on_mouse(event, x, y, flags, param):
        # Tangani klik kiri sesuai fase: isi polygon dulu, lalu panah
        nonlocal titik_polygon_baru, titik_panah_baru
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(titik_polygon_baru) < 4:
            titik_polygon_baru.append((x, y))
        elif len(titik_panah_baru) < 2:
            titik_panah_baru.append((x, y))

    nama_window = "Konfigurator Lajur - Klik 4 titik polygon, lalu 2 titik panah"
    cv2.namedWindow(nama_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nama_window, max(800, w), max(500, h + 60))
    cv2.setMouseCallback(nama_window, on_mouse)

    print("=" * 64)
    print(" KONFIGURATOR ROI LAJUR")
    print("=" * 64)
    print(" 1) Klik 4 titik polygon (urut: searah jarum jam).")
    print(" 2) Klik 2 titik panah: awal -> akhir = arah yang DIIZINKAN.")
    print(" 3) Tekan 'n' untuk simpan lajur ini & mulai lajur berikutnya.")
    print(" 4) Tekan 's' untuk simpan semua ke JSON dan keluar.")
    print(" 5) Tekan 'r' untuk reset lajur yang sedang dibuat.")
    print(" 6) Tekan 'q' untuk keluar TANPA menyimpan.")
    print("=" * 64)

    while True:
        # Susun teks status agar user tahu fase saat ini
        fase = ("polygon (klik %d/4)" % len(titik_polygon_baru)
                if len(titik_polygon_baru) < 4
                else "panah arah (klik %d/2)" % len(titik_panah_baru))
        status = f"Lajur tersimpan: {len(lajur_tersimpan)} | Fase: {fase} | [n]ext  [s]ave  [r]eset  [q]uit"

        tampilan = gambar_overlay(frame, lajur_tersimpan,
                                  titik_polygon_baru, titik_panah_baru, status)
        cv2.imshow(nama_window, tampilan)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[INFO] Keluar tanpa menyimpan.")
            cv2.destroyAllWindows()
            return

        if key == ord('r'):
            # Reset lajur yang sedang dibuat (tidak menghapus yang sudah disimpan)
            titik_polygon_baru.clear()
            titik_panah_baru.clear()
            print("[INFO] Reset lajur yang sedang dibuat.")

        if key in (ord('n'), ord('s')):
            # Validasi: lajur baru harus punya 4 titik polygon + 2 titik panah
            if len(titik_polygon_baru) == 4 and len(titik_panah_baru) == 2:
                lajur_tersimpan.append({
                    "polygon": [list(p) for p in titik_polygon_baru],
                    "arrow":   [list(p) for p in titik_panah_baru],
                })
                print(f"[OK] Lajur L{len(lajur_tersimpan)-1} disimpan.")
                titik_polygon_baru.clear()
                titik_panah_baru.clear()
            elif (titik_polygon_baru or titik_panah_baru):
                print("[WARN] Lajur belum lengkap (perlu 4 titik polygon + 2 titik panah). "
                      "Tekan 'r' untuk reset atau lanjutkan klik.")

            if key == ord('s'):
                break

    cv2.destroyAllWindows()

    if not lajur_tersimpan:
        print("[WARN] Tidak ada lajur untuk disimpan.")
        return

    # Hitung vektor arah ternormalisasi dari setiap panah (untuk kemudahan konsumsi)
    payload = {
        "video_meta": {"width": w, "height": h},
        "lanes": []
    }
    for i, lj in enumerate(lajur_tersimpan):
        (x1, y1), (x2, y2) = lj["arrow"][0], lj["arrow"][1]
        vx, vy = x2 - x1, y2 - y1
        norma = float(np.hypot(vx, vy))
        # Hindari pembagian nol jika user klik di titik yang sama
        if norma < 1e-6:
            print(f"[WARN] Panah lajur L{i} memiliki panjang 0, dilewati.")
            continue
        payload["lanes"].append({
            "id": i,
            "polygon": lj["polygon"],
            "arrow":   lj["arrow"],
            # Vektor arah unit (dx, dy) - dipakai di pengecekan counterflow
            "direction": [vx / norma, vy / norma],
        })

    # Pastikan direktori output ada lalu tulis JSON dengan indentasi rapi
    Path(os.path.dirname(path_output) or ".").mkdir(parents=True, exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Konfigurasi {len(payload['lanes'])} lajur disimpan ke: {path_output}")


def parse_args():
    ap = argparse.ArgumentParser(description="Definisi ROI lajur via klik mouse.")
    ap.add_argument("--video", default="traffic.mp4", help="Path video sumber.")
    ap.add_argument("--out",   default="config/lanes.json", help="Path JSON output.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    jalankan_konfigurator(args.video, args.out)
