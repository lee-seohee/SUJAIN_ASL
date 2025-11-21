import pandas as pd
from sklearn.model_selection import train_test_split

'''
완성된 dataset_final.csv를 train, test, validation 분할하여
각각의 csv 파일로 저장하는 코드
'''

# 1. 최종 CSV 파일 불러오기
CSV_PATH = 'G:/내 드라이브/수어데이터_공용/metadata/dataset_final.csv'
df = pd.read_csv(CSV_PATH)

# 2. 먼저 전체 데이터를 학습+검증용(90%)과 테스트용(10%)으로 분리
train_val_df, test_df = train_test_split(
    df,
    test_size=0.1,         # 10%를 테스트용으로 사용
    random_state=42,       # 재현 가능하도록 시드 고정
    stratify=df['label']   # 'label' 열의 분포를 유지하며 분리
)

# 3. 다음으로 학습+검증용 데이터를 다시 학습용(80%)과 검증용(10%)으로 분리
# (원래 90% 중 1/9이 검증용이 되면 전체의 10%가 됨)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=1/9,         # 90% 중 11.11...% -> 전체의 10%
    random_state=42,
    stratify=train_val_df['label']
)

# 4. 결과 확인
print(f"총 데이터: {len(df)}개")
print(f"학습 데이터: {len(train_df)}개")
print(f"검증 데이터: {len(val_df)}개")
print(f"테스트 데이터: {len(test_df)}개")

# 5. 분리된 데이터프레임을 각각 CSV 파일로 저장
METADATA_ROOT = 'G:/내 드라이브/수어데이터_공용/metadata'
train_df.to_csv(f'{METADATA_ROOT}/train.csv', index=False)
val_df.to_csv(f'{METADATA_ROOT}/validation.csv', index=False)
test_df.to_csv(f'{METADATA_ROOT}/test.csv', index=False)

print("\n데이터셋 분할 및 저장이 완료되었습니다.")