"""
main.py
Pipeline utama deteksi pelanggaran counterflow.

Alur:
    1. Muat konfigurasi lajur (config/lanes.json).
    2. Buka video, inisialisasi detector (MOG2) dan tracker (centroid).
    3. Per frame:
        - Deteksi blob kendaraan (filter area + harus di dalam salah satu lajur).
        - Update tracker.
        - Evaluasi arah tiap objek -> normal / violation / unknown.
        - Render overlay (ROI, bbox berwarna, ID, panah arah).
        - Tulis frame ke output video, log pelanggaran ke CSV.

Cara pakai:
    python -m src.main --video traffic.mp4 \
                       --config config/lanes.json \
                       --out-video output/annotated.mp4 \
                       --out-csv  output/violations.csv \
                       [--show]    # tampilkan window real-time
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .detector import DetektorKendaraan
from .tracker import CentroidTracker
from .violation import cari_lane_id, evaluasi_objek


# Palet warna status (BGR)
WARNA_BBOX = {
    "unknown":   (180, 180, 180),  # abu-abu
    "normal":    (0, 200, 0),      # hijau
    "violation": (0, 0, 255),      # merah
}


def muat_konfigurasi(path: str) -> dict:
    """Baca file JSON konfigurasi lajur."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gambar_overlay_lajur(frame: np.ndarray, lanes: List[dict]) -> None:
    """Render polygon lajur (semi-transparan) dan panah arah di atas frame."""
    overlay = frame.copy()
    for ln in lanes:
        pts = np.array(ln["polygon"], dtype=np.int32)
        # Polygon transparan biar tidak menutupi kendaraan
        cv2.fillPoly(overlay, [pts], (255, 180, 0))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for ln in lanes:
        pts = np.array(ln["polygon"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 180, 0), 2)
        # Panah arah lajur (lebih besar agar mudah dibaca)
        p1 = tuple(ln["arrow"][0])
        p2 = tuple(ln["arrow"][1])
        cv2.arrowedLine(frame, p1, p2, (0, 220, 0), 2, tipLength=0.25)
        # Label lajur di tengah polygon
        cx = int(np.mean([p[0] for p in ln["polygon"]]))
        cy = int(np.mean([p[1] for p in ln["polygon"]]))
        cv2.putText(frame, f"L{ln['id']}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def gambar_objek(frame: np.ndarray, objects: Dict[int, dict]) -> None:
    """Render bounding box, ID, dan panah arah pergerakan tiap objek."""
    for oid, obj in objects.items():
        x, y, w, h = obj["bbox"]
        warna = WARNA_BBOX.get(obj["status"], WARNA_BBOX["unknown"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), warna, 2)

        # Label ID + status singkat
        teks = f"ID{oid}"
        if obj["lane_id"] is not None:
            teks += f"|L{obj['lane_id']}"
        if obj["status"] == "violation":
            teks += " COUNTERFLOW"
        cv2.putText(frame, teks, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1)

        # Panah dari centroid lama -> centroid baru (visualisasi arah pergerakan)
        hist = list(obj["history"])
        if len(hist) >= 2:
            cv2.arrowedLine(frame, hist[0], hist[-1], warna, 2, tipLength=0.4)


def gambar_hud(frame: np.ndarray, frame_idx: int, total_frame: int,
               jumlah_pelanggar: int) -> None:
    """Render header informasi pada frame (heads-up display)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 22), (40, 40, 40), -1)
    teks = (f"Frame {frame_idx}/{total_frame}   "
            f"Pelanggar unik: {jumlah_pelanggar}")
    cv2.putText(frame, teks, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def parse_args():
    ap = argparse.ArgumentParser(description="Deteksi counterflow (classic CV).")
    ap.add_argument("--video",      default="traffic.mp4")
    ap.add_argument("--config",     default="config/lanes.json")
    ap.add_argument("--out-video",  default="output/annotated.mp4")
    ap.add_argument("--out-csv",    default="output/violations.csv")
    ap.add_argument("--min-area",   type=int, default=300,
                    help="Luas kontur minimum (px^2) untuk dianggap kendaraan.")
    ap.add_argument("--show", action="store_true",
                    help="Tampilkan jendela real-time.")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Muat konfigurasi lajur (hasil tool config_lanes.py)
    if not os.path.exists(args.config):
        raise SystemExit(
            f"[ERROR] Konfigurasi tidak ditemukan: {args.config}\n"
            f"        Jalankan dulu: python -m src.config_lanes --video {args.video}"
        )
    cfg = muat_konfigurasi(args.config)
    lanes = cfg["lanes"]
    if not lanes:
        raise SystemExit("[ERROR] Konfigurasi tidak punya lajur. Ulangi konfigurasi.")
    print(f"[INFO] Memuat {len(lanes)} lajur dari {args.config}")

    # 2) Buka video sumber + siapkan writer video output
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Tidak bisa membuka video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {w}x{h} @ {fps:.1f} fps, {total} frame")

    Path(os.path.dirname(args.out_video) or ".").mkdir(parents=True, exist_ok=True)
    # mp4v adalah codec MP4 yang aman di hampir semua build OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))

    # 3) Siapkan CSV log pelanggaran
    Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    f_csv = open(args.out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(["frame", "timestamp_s", "object_id", "lane_id",
                         "centroid_x", "centroid_y"])

    # 4) Inisialisasi detektor & tracker
    detektor = DetektorKendaraan(min_area=args.min_area)
    tracker = CentroidTracker(max_disappeared=15, max_distance=60.0, history_len=12)

    # Lacak set ID yang sudah dilaporkan agar tidak menulis CSV berulang
    sudah_dilaporkan: set = set()

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # 4a) Deteksi blob bergerak di frame ini
            deteksi, _mask = detektor.proses(frame)

            # 4b) Filter: hanya pertahankan deteksi yang centroid-nya berada
            #     di dalam salah satu polygon lajur (mengabaikan trotoar/luar jalan)
            deteksi_dalam_lajur = []
            for d in deteksi:
                lane_id = cari_lane_id(d.centroid, lanes)
                if lane_id is not None:
                    deteksi_dalam_lajur.append(d)

            # 4c) Update tracker dengan deteksi yang relevan
            objects = tracker.update(deteksi_dalam_lajur)

            # 4d) Evaluasi status counterflow untuk setiap objek
            for obj in objects.values():
                evaluasi_objek(obj, lanes)

                # Catat pelanggaran baru ke CSV (sekali per ID)
                if obj["status"] == "violation" and obj["id"] not in sudah_dilaporkan:
                    sudah_dilaporkan.add(obj["id"])
                    ts = frame_idx / fps
                    cx, cy = obj["centroid"]
                    csv_writer.writerow([frame_idx, f"{ts:.2f}", obj["id"],
                                         obj["lane_id"], cx, cy])
                    print(f"[ALERT] Frame {frame_idx} (t={ts:.2f}s) "
                          f"ID{obj['id']} di lajur L{obj['lane_id']} -> COUNTERFLOW")

            # 4e) Visualisasi: overlay lajur, objek, dan HUD
            gambar_overlay_lajur(frame, lanes)
            gambar_objek(frame, objects)
            gambar_hud(frame, frame_idx, total, len(sudah_dilaporkan))

            # Tulis frame annotated ke video output
            writer.write(frame)

            # Tampilkan window jika --show (tekan 'q' untuk berhenti lebih awal)
            if args.show:
                cv2.imshow("Counterflow Detection", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    print("[INFO] Dihentikan oleh user.")
                    break
    finally:
        # Pastikan semua resource ditutup, bahkan jika terjadi error
        cap.release()
        writer.release()
        f_csv.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[OK] Video annotated  : {args.out_video}")
    print(f"[OK] Log pelanggaran  : {args.out_csv}  "
          f"({len(sudah_dilaporkan)} pelanggar unik)")


if __name__ == "__main__":
    main()
