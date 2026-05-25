group name :

- mirsa bayu prasetyo
- gregorius jovi
- nur ilham iskandar
- m. amridhan mahdi
- muhana

## Description

implementation of template matching algorithm for visual tracking of car in the street.
car detection using YOLO and then it would be tracked using template matching algorithm

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
