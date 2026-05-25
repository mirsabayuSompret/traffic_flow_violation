"""
tracker.py
Centroid Tracker sederhana berbasis nearest-neighbor (jarak Euclidean greedy).

Setiap objek menyimpan:
    - id            : ID unik
    - centroid      : posisi terakhir (cx, cy)
    - bbox          : bounding box terakhir
    - history       : deque centroid (untuk estimasi arah dengan smoothing)
    - lost          : berapa frame berturut tidak terlihat
    - lane_id       : lajur tempat objek terdeteksi (None jika di luar semua lajur)
    - status        : "unknown" | "normal" | "violation"
    - violation_streak : penghitung frame berturut yang menunjukkan counterflow
"""

from collections import OrderedDict, deque
from typing import Dict, List, Tuple

import numpy as np

from .detector import Deteksi


class CentroidTracker:
    """Tracker greedy: pasangkan deteksi baru ke objek lama via jarak minimum."""

    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: float = 60.0,
        history_len: int = 10,
    ):
        # max_disappeared: setelah berapa frame tanpa terdeteksi, objek dihapus
        # max_distance   : ambang jarak Euclidean (px) untuk dianggap "sama"
        # history_len    : panjang deque centroid (semakin panjang = arah lebih halus)
        self._next_id: int = 0
        self._objects: "OrderedDict[int, dict]" = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_len = history_len

    @property
    def objects(self) -> "OrderedDict[int, dict]":
        # Akses read-only ke dictionary objek (dipakai modul main)
        return self._objects

    def _register(self, det: Deteksi) -> None:
        # Daftarkan deteksi sebagai objek baru dengan ID berurutan
        self._objects[self._next_id] = {
            "id": self._next_id,
            "centroid": det.centroid,
            "bbox": det.bbox,
            "history": deque([det.centroid], maxlen=self.history_len),
            "lost": 0,
            "lane_id": None,
            "status": "unknown",
            "violation_streak": 0,
        }
        self._next_id += 1

    def _deregister(self, oid: int) -> None:
        # Hapus objek (sudah hilang terlalu lama)
        del self._objects[oid]

    def update(self, deteksi: List[Deteksi]) -> "OrderedDict[int, dict]":
        """Update tracker dengan deteksi pada frame saat ini."""

        # Kasus 1: tidak ada deteksi -> seluruh objek bertambah counter 'lost'
        if not deteksi:
            for oid in list(self._objects.keys()):
                self._objects[oid]["lost"] += 1
                if self._objects[oid]["lost"] > self.max_disappeared:
                    self._deregister(oid)
            return self._objects

        # Kasus 2: belum ada objek -> daftarkan semua deteksi sebagai objek baru
        if not self._objects:
            for d in deteksi:
                self._register(d)
            return self._objects

        # Kasus 3: cocokkan secara greedy berdasar jarak terdekat
        oids = list(self._objects.keys())
        old = np.array([self._objects[i]["centroid"] for i in oids], dtype=np.float32)
        new = np.array([d.centroid for d in deteksi], dtype=np.float32)

        # Matriks jarak D[i, j] = jarak Euclidean objek lama-i ke deteksi-j
        D = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)

        baris_terpakai: set = set()
        kolom_terpakai: set = set()

        # Greedy: berulang ambil pasangan dengan jarak terkecil yang belum dipakai
        # (Hungarian akan lebih optimal, tapi greedy cukup untuk pendekatan klasik
        # dan lebih mudah dipahami untuk pembelajaran.)
        jumlah_pasang = min(D.shape)
        for _ in range(jumlah_pasang):
            min_val = np.inf
            min_pos = None
            for i in range(D.shape[0]):
                if i in baris_terpakai:
                    continue
                for j in range(D.shape[1]):
                    if j in kolom_terpakai:
                        continue
                    if D[i, j] < min_val:
                        min_val = D[i, j]
                        min_pos = (i, j)
            # Berhenti jika tidak ada pasangan tersisa atau jaraknya terlalu jauh
            if min_pos is None or min_val > self.max_distance:
                break
            i, j = min_pos
            oid = oids[i]
            d = deteksi[j]
            self._objects[oid]["centroid"] = d.centroid
            self._objects[oid]["bbox"] = d.bbox
            self._objects[oid]["history"].append(d.centroid)
            self._objects[oid]["lost"] = 0
            baris_terpakai.add(i)
            kolom_terpakai.add(j)

        # Objek lama yang tidak ketemu pasangan -> tambah counter 'lost'
        for i, oid in enumerate(oids):
            if i not in baris_terpakai:
                self._objects[oid]["lost"] += 1
                if self._objects[oid]["lost"] > self.max_disappeared:
                    self._deregister(oid)

        # Deteksi yang tidak ketemu pasangan -> daftarkan sebagai objek baru
        for j, d in enumerate(deteksi):
            if j not in kolom_terpakai:
                self._register(d)

        return self._objects
