# probe_xirka_bin.py
import numpy as np
from pathlib import Path

files = sorted(Path(r"C:\Users\Myrza\Desktop\project\Project Arrythmia\OUTPUT\ECG Record").glob("ecg_*.bin"))

ADC_TO_MV = 0.0025

for f in files:
    data = np.memmap(f, dtype=np.dtype([('leads', np.int16, 12)]), mode='r')
    ecg_mv = data['leads'].astype(np.float32) * ADC_TO_MV

    print(f"\n{f.name}")
    print(f"  Samples : {len(data):,}  ({len(data)/500:.0f}s)")
    print(f"  mV range: [{ecg_mv.min():.3f}, {ecg_mv.max():.3f}]")
    print(f"  Per-lead std (mV):")
    lead_names = ['I','II','III','aVR','aVF','aVL','V1','V2','V3','V4','V5','V6']
    for i, name in enumerate(lead_names):
        std = ecg_mv[:, i].std()
        flag = " ⚠ flat" if std < 0.001 else ""
        print(f"    {name:5s}: {std:.4f}{flag}")