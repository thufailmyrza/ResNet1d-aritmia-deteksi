from pathlib import Path
import numpy as np
# BASE PATHS
PROJECT_ROOT = Path("C:/Users/Myrza/Desktop/project/Project Arrythmia")
#  Input (Raw Data) 
PTBXL_ROOT           = PROJECT_ROOT / "RAW DATA" / "ptb-xl"
PTBXL_DATABASE       = PTBXL_ROOT / "ptbxl_database.csv"
PTBXL_SCP_STATEMENTS = PTBXL_ROOT / "scp_statements.csv"
PTBXL_RECORDS        = PTBXL_ROOT / "records500"         
INCART_ROOT          = PROJECT_ROOT / "RAW DATA" / "incart"
#  Output root 
OUTPUT_ROOT          = PROJECT_ROOT / "OUTPUT"
#  Holter format – PTB-XL binary windows 
HOLTER_FORMAT_DIR    = OUTPUT_ROOT / "HOLTER_V5"
SMOTE_CACHE_DIR      = HOLTER_FORMAT_DIR / "smote_cache"
#  INCART format  
INCART_FORMAT_DIR    = OUTPUT_ROOT / "INCART_FORMAT"
INCART_LABELS_CSV    = INCART_FORMAT_DIR / "incart_labels.csv"
INCART_STATS_JSON    = INCART_FORMAT_DIR / "incart_statistics.json"
# Merged dataset  
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
LABELS_CSV           = HOLTER_FORMAT_DIR / "labels.csv"        
ARRHYTHMIA_BIN       = HOLTER_FORMAT_DIR / "arrhythmia.bin"
STATISTICS_JSON      = HOLTER_FORMAT_DIR / "dataset_statistics.json"
# Split CSV 
TRAIN_SPLIT_CSV      = HOLTER_FORMAT_DIR / "train_split.csv"
VAL_SPLIT_CSV        = HOLTER_FORMAT_DIR / "val_split.csv"
TEST_SPLIT_CSV       = HOLTER_FORMAT_DIR / "test_split.csv"
# HOLTER DEVICE CONSTANTS
HOLTER_SAMPLING_RATE = 500         
ECG_CHANNELS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
NUM_CHANNELS = len(ECG_CHANNELS)

BYTES_PER_SAMPLE = 2               
BYTES_PER_RECORD = NUM_CHANNELS * BYTES_PER_SAMPLE  

SECONDS_PER_SPLIT  = 12 * 60 * 60  
SAMPLES_PER_SPLIT  = SECONDS_PER_SPLIT * HOLTER_SAMPLING_RATE

# Window size harus cocok dengan window_size di app
WINDOW_SIZE = 2500   # 5 detik @ 500 Hz

# ADC gain
ADC_GAIN_DEVICE = 0.0025    
ADC_GAIN_INT16  = 1000      
ADC_GAIN        = ADC_GAIN_INT16    

INT16_TO_MV = 1.0 / ADC_GAIN_INT16   

# ARRHYTHMIA CLASS MAPPING  –  11 kelas, single-label
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

ARRHYTHMIA_PRIORITY = [
    10,  # atrial_fibrillation   
    9,   # bradycardia
    8,   # tachycardia
    7,   # nsvt
    6,   # triplet
    5,   # couplet
    4,   # quadrigeminy
    3,   # trigeminy
    2,   # bigeminy
    1,   # premature_beat
    0,   # normal                
]

# Backward-compat aliases (untuk kode lama yang mengimpor nama ini)
ARRHYTHMIA_BIT_MAPPING = CLASS_TO_IDX   # nama → class index
NUM_CLASSES = NUM_ARRHYTHMIA_CLASSES

ARRHYTHMIA_BIN_DTYPE = np.int32    
NORMAL_FLAG_VALUE    = 1           