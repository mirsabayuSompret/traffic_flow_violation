group name :

- Mirsa Bayu Prasetyo
- Gregorius Bugen Jovi Sitindaon
- Nur Ilham Iskandar
- M. Amridhan Mahdi
- Muhana Fawwazy Ilyas

# Title
Traffic Flow Violation Detection Using Object Detection using Gaussian Mixture Model

## Abstract
Traffic surveillance and automated violation detection have become critical components of intelligent transportation systems. Manual monitoring of road traffic is resource-intensive and prone to human error, particularly at complex intersections in high-density urban environments. This paper presents a traffic flow violation detection system designed to identify counterflow incidents, defined as vehicles traveling against the designated direction of a road lane. The system employs a classical computer vision pipeline comprising three main stages: (1) Gaussian Mixture Model-based background subtraction (MOG2) for vehicle blob detection from video frames; (2) a centroid-based greedy nearest-neighbor tracker for maintaining consistent vehicle identities across frames; and (3) cosine similarity analysis between the computed vehicle motion vector and a predefined lane direction vector, combined with a streak-based confirmation mechanism to reduce false positives. The system was evaluated on six video scenarios from the Traffic India Intersection dataset, covering daytime and nighttime conditions, accident and normal traffic situations, high-density uncontrolled intersections, and regulated toll plazas. Experimental results, recorded as CSV violation logs and annotated video outputs, demonstrate that the system successfully detects counterflow violations across all tested conditions. The traffic_day_1 scenario yielded the highest detection count at 23 unique object IDs, while nighttime scenarios showed detections as early as frame 66. The results indicate that the classical approach is viable for counterflow detection while highlighting limitations in complex occlusion and multi-object tracking scenarios.

---

## Tahap 1 — Deteksi Counterflow (Classic CV, tanpa Deep Learning)

Pipeline klasik untuk mendeteksi pelanggaran _counterflow_ (kendaraan melawan arah):

```
Frame  ->  MOG2 Background Subtraction
       ->  Morphological Ops (open + close)
       ->  findContours + filter area
       ->  Centroid Tracker (greedy nearest-neighbor)
       ->  Vektor arah pergerakan vs vektor arah lajur (cosine similarity)
       ->  Streak konfirmasi (>=5 frame) -> tandai pelanggaran
```

### Struktur file

```
src/
  config_lanes.py   # Tool interaktif: klik 4 titik polygon + 2 titik panah per lajur
  detector.py       # Background subtraction MOG2 + morfologi + kontur
  tracker.py        # Centroid tracker sederhana (Euclidean greedy)
  violation.py      # Logika counterflow (dot product + streak)
  main.py           # Pipeline utama + visualisasi
config/lanes.json   # Output config_lanes.py
output/             # annotated.mp4 + violations.csv (di-gitignore)
```

### Instalasi

## Download video dari link ini

```
https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring
```

simpan vide kedalam root folder, ubah namanya menjdai traffic.mp4

## buat virtual env dan install requirement

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Cara pakai

**1. Definisikan ROI lajur** (sekali saja per video):

```bash
python -m src.config_lanes --video traffic.mp4 --out config/lanes.json
```

Klik **4 titik** untuk polygon lajur (searah jarum jam), lalu **2 titik** untuk panah
arah yang diizinkan (titik awal -> titik akhir). Tekan `n` untuk lajur berikutnya,
`s` untuk simpan & keluar, `r` untuk reset lajur yang sedang dibuat, `q` untuk batal.

**2. Jalankan deteksi**:

```bash
python -m src.main --video traffic.mp4 \
                   --config config/lanes.json \
                   --out-video output/annotated.mp4 \
                   --out-csv  output/violations.csv \
                   --show
```

Hapus `--show` untuk mode headless. Kotak hijau = normal, kotak merah = counterflow.
