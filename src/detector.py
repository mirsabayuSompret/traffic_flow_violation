"""
detector.py
Modul deteksi objek bergerak (kendaraan) menggunakan Background Subtraction (MOG2).

Pipeline per frame:
    frame -> MOG2 -> threshold (buang bayangan) -> morfologi (open + close)
          -> findContours -> filter area -> list (bbox, centroid)
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# Tipe data ringkas hasil deteksi untuk dipakai modul lain
@dataclass
class Deteksi:
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    centroid: Tuple[int, int]         # (cx, cy)
    area: float                       # luas kontur (px^2)


class DetektorKendaraan:
    """Detektor klasik berbasis MOG2 + operasi morfologi."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 25.0,
        detect_shadows: bool = True,
        min_area: int = 300,
        kernel_open: int = 3,
        kernel_close: int = 7,
    ):
        # MOG2: model background adaptif berbasis Gaussian Mixture.
        # - history: jumlah frame untuk membangun model background
        # - varThreshold: ambang varians; kecil = lebih sensitif terhadap gerakan
        # - detectShadows=True: piksel bayangan diberi nilai 127 (bukan 255)
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._min_area = min_area
        # Kernel struktur untuk operasi morfologi
        # OPEN (erosi -> dilatasi) untuk buang noise titik-titik kecil
        # CLOSE (dilatasi -> erosi) untuk menutup lubang di dalam blob kendaraan
        self._k_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_open, kernel_open))
        self._k_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_close, kernel_close))

    def proses(self, frame: np.ndarray) -> Tuple[List[Deteksi], np.ndarray]:
        """Proses satu frame, kembalikan (list deteksi, mask biner final)."""
        # 1) Jalankan background subtraction; hasil = mask abu-abu
        mask = self._bg.apply(frame)

        # 2) Buang piksel bayangan (nilai 127). Hanya pertahankan piksel "kuat" >= 200.
        #    Ini krusial agar bayangan kendaraan tidak ikut terdeteksi sebagai objek.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # 3) Bersihkan noise (OPEN) lalu satukan blob terpecah (CLOSE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close)

        # 4) Cari kontur eksternal saja (hindari kontur lubang)
        kontur, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5) Filter kontur berdasarkan luas, hitung bbox & centroid
        deteksi: List[Deteksi] = []
        for c in kontur:
            area = cv2.contourArea(c)
            if area < self._min_area:
                # Lewati blob terlalu kecil (kemungkinan noise / daun goyang dll)
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            deteksi.append(Deteksi(bbox=(x, y, w, h),
                                   centroid=(cx, cy),
                                   area=float(area)))
        return deteksi, mask
