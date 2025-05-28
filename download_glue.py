# download_glue.py
from datasets import load_dataset

for task in ["cola", "sst2", "rte"]:
    load_dataset("glue", task, cache_dir="./glue_raw")
