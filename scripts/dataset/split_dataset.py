import pandas as pd
from sklearn.model_selection import train_test_split

'''
완성된 dataset_final.csv를 train, test, validation 분할하여
각각의 csv 파일로 저장하는 코드
'''

# 랜드마크 경로가 포함된, 증강까지 완료된 전체 데이터셋 CSV
# 이전에 'dataset_final.csv'로 저장함
FULL_DATASET_CSV = './dataset_final.csv'

# 데이터를 나눌 비율
TRAIN_SIZE = 0.8  # 학습용 80%
VALIDATION_SIZE = 0.1 # 검증용 10%
TEST_SIZE = 0.1 # 테스트용 10%

try:
    # 1. 전체 데이터셋 불러오기
    df = pd.read_csv(FULL_DATASET_CSV)
    print(f"전체 데이터 개수: {len(df)}")

    # 2. 라벨(정답) 목록 추출
    labels = df['label']

    # 3. 학습용 데이터와 나머지(검증용+테스트용) 데이터로 1차 분할
    # stratify=labels 옵션은 각 세트마다 단어(라벨)의 비율이 비슷하게 유지되도록 해줍니다. (매우 중요!)
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        random_state=42, # 항상 동일한 결과로 섞이도록 시드 고정
        stratify=labels
    )

    # 4. 나머지 데이터를 검증용과 테스트용으로 2차 분할
    # 남은 데이터 중 절반(0.5)을 테스트용으로 할당 (전체 데이터의 10%에 해당)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        random_state=42,
        stratify=temp_df['label']
    )
    
    # 5. 각 데이터셋을 별도의 CSV 파일로 저장
    train_df.to_csv('train.csv', index=False, encoding='utf-8-sig')
    validation_df.to_csv('validation.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')

    print("\n데이터 분할 완료")
    print(f"학습용 데이터: {len(train_df)}개 -> train.csv")
    print(f"검증용 데이터: {len(validation_df)}개 -> validation.csv")
    print(f"테스트용 데이터: {len(test_df)}개 -> test.csv")
    
except FileNotFoundError:
    print(f"오류: '{FULL_DATASET_CSV}' 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")