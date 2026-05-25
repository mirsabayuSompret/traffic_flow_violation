"""
violation.py
Logika deteksi pelanggaran 'counterflow' (kendaraan melawan arah lajur).

Pendekatan:
    1. Untuk setiap objek yang dilacak, hitung vektor pergerakan v dari
       riwayat centroid: v = centroid_terbaru - centroid_lama (jaraknya N frame).
    2. Skip jika |v| sangat kecil (kendaraan diam, mis. macet/lampu merah)
       agar tidak salah menilai arah dari noise tracking.
    3. Hitung cosine similarity antara v dan vektor arah lajur d:
           cos_theta = (v . d) / (|v| * |d|)
       - cos_theta ~  +1  => searah  (normal)
       - cos_theta ~   0  => tegak lurus (tidak jelas)
       - cos_theta ~  -1  => berlawanan arah (counterflow)
    4. Untuk menghindari false positive 1 frame, butuh streak >= N frame
       berturut-turut yang menunjukkan counterflow baru ditandai 'violation'.
"""

from typing import List, Optional, Tuple

import numpy as np

import cv2


# Ambang konfigurasi - bisa diatur dari main bila diperlukan
COS_THRESHOLD_COUNTERFLOW: float = -0.3   # cos(theta) < -0.3 => sudut > ~108 derajat
COS_THRESHOLD_NORMAL: float = 0.3         # cos(theta) > +0.3 => searah cukup jelas
MIN_PERGERAKAN_PX: float = 2.0            # |v| minimum agar dianggap "bergerak"
STREAK_KONFIRMASI: int = 5                # frame berturut sebelum tandai violation
JARAK_HISTORY_FRAME: int = 5              # bandingkan centroid terbaru vs N frame lalu


def cari_lane_id(centroid: Tuple[int, int], lanes: List[dict]) -> Optional[int]:
    """Cari lajur (berdasarkan id) yang polygon-nya memuat centroid.

    Mengembalikan None bila centroid tidak berada di lajur manapun.
    """
    for ln in lanes:
        polygon = np.array(ln["polygon"], dtype=np.int32)
        # pointPolygonTest mengembalikan +1 di dalam, 0 di tepi, -1 di luar
        if cv2.pointPolygonTest(polygon, (float(centroid[0]), float(centroid[1])), False) >= 0:
            return ln["id"]
    return None


def evaluasi_objek(obj: dict, lanes: List[dict]) -> None:
    """Evaluasi status objek (normal / violation / unknown) berdasarkan arah."""

    # 1) Pastikan objek punya lane_id; bila belum, coba assign dari centroid sekarang
    if obj["lane_id"] is None:
        obj["lane_id"] = cari_lane_id(obj["centroid"], lanes)
        if obj["lane_id"] is None:
            # Masih di luar semua lajur - tidak ada referensi arah, biarkan 'unknown'
            return

    # 2) Ambil vektor arah lajur (sudah ternormalisasi saat config dibuat)
    lane = next((l for l in lanes if l["id"] == obj["lane_id"]), None)
    if lane is None:
        return
    d = np.array(lane["direction"], dtype=np.float32)

    # 3) Butuh history cukup panjang untuk smoothing vektor pergerakan
    history = list(obj["history"])
    if len(history) < JARAK_HISTORY_FRAME + 1:
        return

    p_baru = np.array(history[-1], dtype=np.float32)
    p_lama = np.array(history[-1 - JARAK_HISTORY_FRAME], dtype=np.float32)
    v = p_baru - p_lama
    norma_v = float(np.linalg.norm(v))

    # 4) Skip jika nyaris diam (hindari arah noise)
    if norma_v < MIN_PERGERAKAN_PX:
        return

    # 5) Hitung cosine similarity antara arah pergerakan dan arah lajur
    cos_theta = float(np.dot(v, d) / (norma_v * (np.linalg.norm(d) + 1e-9)))

    # 6) Logika streak: counterflow harus konsisten N frame agar di-flag
    if cos_theta < COS_THRESHOLD_COUNTERFLOW:
        obj["violation_streak"] += 1
        if obj["violation_streak"] >= STREAK_KONFIRMASI:
            obj["status"] = "violation"
    elif cos_theta > COS_THRESHOLD_NORMAL:
        # Searah jelas -> reset streak, status normal
        obj["violation_streak"] = 0
        # Jangan "demote" objek yang sudah dinyatakan violation pada frame ini;
        # status 'violation' bersifat sticky agar laporan akhir konsisten.
        if obj["status"] != "violation":
            obj["status"] = "normal"
    else:
        # Zona abu-abu (tegak lurus / belok) - tahan, jangan ubah status
        pass
