from src.multivalue.Dialects import NigerianDialect, IndianDialect
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

nig = NigerianDialect()
ind = IndianDialect()

# print(ind.transform("Our friends won't buy this analysis, let alone the next one we propose."))
# print(nig.transform("Our friends won't buy this analysis, let alone the next one we propose."))

df = pd.read_csv("GLUE/CoLA/train.tsv", sep="\t", header=None, names=["ref", "label", "n", "sentence"])
success_count = 0
error_count = 0

def safe_transform(text):
    global success_count, error_count
    try:
        result = ind.transform(text)
        success_count += 1
        return result
    except AssertionError:
        error_count += 1
        return None  # 또는 원래 텍스트 유지하려면 return text

df["sentence"] = df["sentence"].progress_apply(safe_transform)

print(f"정상 처리: {success_count}건")
print(f"AssertionError 발생: {error_count}건")