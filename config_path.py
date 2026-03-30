"""
config_path.py  
Diselaraskan penuh dengan kontrak app (arrhythmia_detector.py + arrhythmia_parser.py):

    KONTRAK MODEL ↔ APP                                                    
                                                                           
    Input model  : (batch, 12, 2500) float32  →  SKALA mV (bukan [-1,1]) 
    Output model : (batch, 11) logits  →  argmax  →  class index 0–10    
    arrhythmia.bin: int32 per ECG sample = (1 << class_index)             
      • class 0 (Normal) → flag = 1 = 2^0 → log2 = 0 → dikecualikan app 
      • class 1–10       → flag = 2–1024  → log2 = 1–10 → muncul sbg     
                                            region aritmia di app          

  v3   : single-label (CrossEntropy), normalisasi mV, arrhythmia.bin per-sample
  v3.1 : + INCART_ROOT, INCART_FORMAT_DIR, INCART_LABELS_CSV, INCART_STATS_JSON
         + MERGED_LABELS_CSV, MERGED_STATS_JSON
         (semua konstanta lain tidak berubah dari v3)
"""

from pathlib import Path
import numpy as np

# BASE PATHS
PROJECT_ROOT = Path("/home/tmyrza/workspace/ProjectArrythmia")

#  Input (Raw Data) 
PTBXL_ROOT           = PROJECT_ROOT / "RAW DATA" / "ptb-xl"
PTBXL_DATABASE       = PTBXL_ROOT / "ptbxl_database.csv"
PTBXL_SCP_STATEMENTS = PTBXL_ROOT / "scp_statements.csv"
PTBXL_RECORDS        = PTBXL_ROOT / "records500"          # rekaman 500 Hz

#  INCART Input  
# Download : https://physionet.org/content/incartdb/1.0.0/
# Letakkan : RAW DATA/incart/
# Isi      : I01.hea  I01.dat  I01.atr  ...  I75.hea  I75.dat  I75.atr
INCART_ROOT          = PROJECT_ROOT / "RAW DATA" / "incart"

#  Output root 
OUTPUT_ROOT          = PROJECT_ROOT / "OUTPUT"

#  Holter format – PTB-XL binary windows 
HOLTER_FORMAT_DIR    = OUTPUT_ROOT / "HOLTER_V5"
SMOTE_CACHE_DIR      = HOLTER_FORMAT_DIR / "smote_cache"

#  INCART format  
# preprocess_incart.py menulis binary windows ke sini.
# Struktur identik dengan HOLTER_FORMAT_DIR:
#   INCART_FORMAT/batch_00000/incart_000000.bin   (2500 × 12 × int16)
#   INCART_FORMAT/batch_00000/incart_000001.bin
#   ...
INCART_FORMAT_DIR    = OUTPUT_ROOT / "INCART_FORMAT"
INCART_LABELS_CSV    = INCART_FORMAT_DIR / "incart_labels.csv"
INCART_STATS_JSON    = INCART_FORMAT_DIR / "incart_statistics.json"

#  Merged dataset  
# merge_datasets.py menggabungkan PTB-XL + INCART, kemudian MENIMPA
# TRAIN_SPLIT_CSV / VAL_SPLIT_CSV / TEST_SPLIT_CSV di HOLTER_FORMAT_DIR
# → semua training code yang sudah ada TIDAK perlu diubah sama sekali.
MERGED_LABELS_CSV    = HOLTER_FORMAT_DIR / "merged_labels.csv"
MERGED_STATS_JSON    = HOLTER_FORMAT_DIR / "merged_statistics.json"

#  Checkpoints 
CHECKPOINTS_DIR      = OUTPUT_ROOT / "checkpoints"
CNN_CHECKPOINT_DIR   = CHECKPOINTS_DIR / "cnn"
CNN_BEST_MODEL       = CNN_CHECKPOINT_DIR / "best_model.pth"
CNN_LAST_MODEL       = CNN_CHECKPOINT_DIR / "last_model.pth"
CNN_TRAINING_LOG     = CNN_CHECKPOINT_DIR / "training_log.json"

#  Logs 
LOGS_DIR             = OUTPUT_ROOT / "logs"

#  Exported models 
EXPORTED_MODELS_DIR  = OUTPUT_ROOT / "exported_models"
ONNX_MODEL_PATH      = EXPORTED_MODELS_DIR / "arrhythmia_model.onnx"
PKL_MODEL_PATH       = EXPORTED_MODELS_DIR / "arrhythmia_model.pkl"

#  Holter format file paths 
LABELS_CSV           = HOLTER_FORMAT_DIR / "labels.csv"        # output PTB-XL saja
ARRHYTHMIA_BIN       = HOLTER_FORMAT_DIR / "arrhythmia.bin"
STATISTICS_JSON      = HOLTER_FORMAT_DIR / "dataset_statistics.json"

# Split CSV – setelah merge_datasets.py dijalankan = PTB-XL + INCART
TRAIN_SPLIT_CSV      = HOLTER_FORMAT_DIR / "train_split.csv"
VAL_SPLIT_CSV        = HOLTER_FORMAT_DIR / "val_split.csv"
TEST_SPLIT_CSV       = HOLTER_FORMAT_DIR / "test_split.csv"

# HOLTER DEVICE CONSTANTS
HOLTER_SAMPLING_RATE = 500          # Hz
ECG_CHANNELS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
NUM_CHANNELS = len(ECG_CHANNELS)    # 12

BYTES_PER_SAMPLE = 2                # int16
BYTES_PER_RECORD = NUM_CHANNELS * BYTES_PER_SAMPLE  # 24 bytes per timepoint

SECONDS_PER_SPLIT  = 12 * 60 * 60  # 43 200 s
SAMPLES_PER_SPLIT  = SECONDS_PER_SPLIT * HOLTER_SAMPLING_RATE

# Window size harus cocok dengan ArrhythmiaDetector.window_size di app
WINDOW_SIZE = 2500   # 5 detik @ 500 Hz

# ADC GAIN
# Raw Holter ADC → mV : × 0.0025
# mV → int16 (storage) : × 1000
# int16 → mV (load)    : / 1000.0   ← NORMALISASI YANG DIPAKAI MODEL
#
# App (arrhythmia_detector.py) menerima ecg_signal_mv (sudah dalam mV)
# dan TIDAK melakukan normalisasi tambahan sebelum feed ke model.
# Maka model HARUS dilatih dengan data skala mV.
ADC_GAIN_DEVICE = 0.0025    # raw ADC Holter → mV
ADC_GAIN_INT16  = 1000      # mV → int16 (penyimpanan)
ADC_GAIN        = ADC_GAIN_INT16    # alias backward-compat (untuk preprocess PTB-XL)

# Normalisasi saat LOAD ke model: int16 / 1000.0 = mV
INT16_TO_MV = 1.0 / ADC_GAIN_INT16   # = 0.001

# ARRHYTHMIA CLASS MAPPING  –  11 kelas, single-label
# Class index = posisi argmax model = log2(flag_value) di arrhythmia.bin
#
# Class 0 = Normal: flag = 2^0 = 1, log2 = 0
#   → parser mengecualikan (work_data == 0 → bukan region aritmia)
# Class 1–10 = Aritmia: flag = 2^1–2^10 = 2–1024, log2 = 1–10
#   → parser menampilkan sebagai region aritmia

# class_index → nama kelas (cocok dengan ARRHYTHMIA_CODES di arrhythmia_parser.py)
ARRHYTHMIA_CLASSES = {
    0:  'normal',
    1:  'premature_beat',
    2:  'bigeminy',
    3:  'trigeminy',
    4:  'quadrigeminy',
    5:  'couplet',
    6:  'triplet',
    7:  'nsvt',
    8:  'tachycardia',
    9:  'bradycardia',
    10: 'atrial_fibrillation',
}

ARRHYTHMIA_LABELS      = [ARRHYTHMIA_CLASSES[i] for i in range(11)]
NUM_ARRHYTHMIA_CLASSES = len(ARRHYTHMIA_CLASSES)   # 11

# Nama → class index
CLASS_TO_IDX = {v: k for k, v in ARRHYTHMIA_CLASSES.items()}

# Prioritas resolusi multi-label PTB-XL → single label
# (indeks lebih kecil = prioritas lebih tinggi = lebih spesifik/parah)
ARRHYTHMIA_PRIORITY = [
    10,  # atrial_fibrillation   (paling spesifik)
    9,   # bradycardia
    8,   # tachycardia
    7,   # nsvt
    6,   # triplet
    5,   # couplet
    4,   # quadrigeminy
    3,   # trigeminy
    2,   # bigeminy
    1,   # premature_beat
    0,   # normal                (fallback)
]

# Backward-compat aliases (untuk kode lama yang mengimpor nama ini)
ARRHYTHMIA_BIT_MAPPING = CLASS_TO_IDX   # nama → class index
NUM_CLASSES = NUM_ARRHYTHMIA_CLASSES

ARRHYTHMIA_BIN_DTYPE = np.int32    # tipe data per sample
NORMAL_FLAG_VALUE    = 1           # 2^0 = 1 (class 0, dikecualikan parser)