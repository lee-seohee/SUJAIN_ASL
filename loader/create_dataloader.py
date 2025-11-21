from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

'''
길이가 다른 랜드마크 시퀀스를 패딩하여 배치 단위로 묶는 DataLoader를 생성하고
collate_fn 함수를 통해 변동 길이 데이터를 처리하는 코드
'''

def collate_fn(batch):
    """
    길이가 다른 시퀀스들을 패딩하여 하나의 배치로 만듦
    """
    # 1. batch는 (데이터, 라벨) 튜플의 리스트 -> 데이터와 라벨을 분리
    sequences, labels = zip(*batch)
    
    # 2. pad_sequence를 이용해 시퀀스들을 패딩
    # batch_first=True 옵션은 텐서의 첫 번째 차원이 배치가 되도록 함 (Batch, SeqLength, Features)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # 3. 라벨들을 하나의 텐서로 합침
    labels = torch.stack(labels, dim=0)
    
    return padded_sequences, labels

# --- 모든 것을 합치기 ---

# 1. 각 데이터셋에 대한 Dataset 객체 생성
DATASET_ROOT_DIR = 'G:/내 드라이브/수어데이터_공용/processed_data/landmark_features'
train_dataset = SignLanguageDataset(f'{METADATA_ROOT}/train.csv', DATASET_ROOT_DIR)
val_dataset = SignLanguageDataset(f'{METADATA_ROOT}/validation.csv', DATASET_ROOT_DIR)
test_dataset = SignLanguageDataset(f'{METADATA_ROOT}/test.csv', DATASET_ROOT_DIR)

# 2. 각 데이터셋에 대한 DataLoader 생성 (collate_fn 지정)
BATCH_SIZE = 32
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # 학습용 데이터는 섞어주는 것이 좋음
    collate_fn=collate_fn
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,      # 검증/테스트용은 섞을 필요 없음
    collate_fn=collate_fn
)
# test_loader도 동일하게 만들기

print("\nDataLoader 생성이 완료되었습니다.")

# train_loader를 for문으로 돌리면 모델 학습에 바로 사용할 수 있는 패딩된 랜드마크 배치와 라벨 배치가 나옴
# for padded_batch, labels_batch in train_loader:
#     print(padded_batch.shape) # ex) torch.Size([32, 150, 1662])
#     print(labels_batch.shape) # ex) torch.Size([32])
#     break