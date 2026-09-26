"""Paths and constants shared by the scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
WEB_DIR = ROOT / "web"

# Dataset: SartajBhuvaji "Brain Tumor Classification (MRI)", published by its
# author on the Hugging Face Hub under the MIT license. Pinned to a revision.
HF_REPO = "sartajbhuvaji/Brain-Tumor-Classification"
HF_REVISION = "011f2917d554afac5c962987e8134e835353726e"
HF_FILES = {
    "Training.zip": "9af6f6a3fb64e6e5bf1017d26586de2eb6fffe20549dfd53712c916b61604a0e",
    "Testing.zip": "b2856ca32613220d6acab10dc45b6a9d24cd11cef37174823cde60b9742ef264",
}

CLASSES_4 = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
HEALTHY = "no_tumor"
CLASSES_2 = ["healthy", "tumor"]  # label 1 = tumor (the positive class)

IMG_SIZE = 224

# Perceptual hash (64-bit pHash) Hamming-distance thresholds.
# <= DUP_THRESHOLD: treated as the same image (one copy kept).
# <= GROUP_THRESHOLD: treated as the same scan series (kept in the same split).
DUP_THRESHOLD = 2
GROUP_THRESHOLD = 6

SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
