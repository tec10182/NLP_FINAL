from src.multivalue.Dialects import NigerianDialect, IndianDialect
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# 방언 변환기 초기화
nig = NigerianDialect()
ind = IndianDialect()

# 데이터 로드
df = pd.read_csv("GLUE/SST/dev.tsv", sep="\t")

# 성공/에러 카운트 초기화
success_count = 0
error_count = 0

# 안전한 변환 함수 정의
def safe_transform(text):
    global success_count, error_count
    try:
        result = nig.transform(text)
        success_count += 1
        return result
    except AssertionError:
        error_count += 1
        return None

# 문장 변환 적용

df["sentence"] = df["sentence"].progress_apply(safe_transform)
# df["sentence1"] = df["sentence1"].progress_apply(safe_transform)
# df["sentence2"] = df["sentence2"].progress_apply(safe_transform)

# None 값 있는 행 제거
df = df.dropna(subset=["sentence"]).reset_index(drop=True)
# df = df.dropna(subset=["sentence1"]).reset_index(drop=True)
# df = df.dropna(subset=["sentence2"]).reset_index(drop=True)

# 결과 저장
res = "VALUE/SST/val_nig_transformed.tsv"
df.to_csv(res, sep="\t", index=False)

# 처리 결과 출력
print(f"정상 처리: {success_count}건")
print(f"AssertionError 발생: {error_count}건")
print(f"변환 완료 및 파일 저장됨: {res}")