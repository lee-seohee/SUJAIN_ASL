import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

'''
수어 랜드마크(.npy) 데이터를 불러오고 실시간 정규화를 수행하는
PyTorch Dataset 클래스(SignLanguageDataset)를 정의하는 코드
'''

class SignLanguageDataset(Dataset):

    def __init__(self, csv_path, landmark_root_dir):
        """
        데이터셋 초기화
        Args:
            csv_path (str): train.csv, validation.csv 등의 파일 경로
            landmark_root_dir (str): .npy 랜드마크 파일들이 저장된 루트 폴더 경로
        """
        self.df = pd.read_csv(csv_path)
        self.landmark_root_dir = landmark_root_dir
        # 라벨(단어)을 숫자로 변환하는 매핑을 만들 수 있음 (추후 구현)
        # self.label_map = {label: i for i, label in enumerate(self.df['label'].unique())}

    def __len__(self):
        """데이터셋의 총 길이를 반환"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        인덱스(idx)에 해당하는 데이터를 불러오고 전처리하여 반환 (실시간 정규화)
        """
        # 1. CSV에서 파일 경로와 라벨 가져오기
        landmark_relative_path = self.df.iloc[idx]['landmarks_path']
        label = self.df.iloc[idx]['label'] # 실제로는 숫자로 변환된 라벨을 사용함
        
        # 2. 전체 경로 조합하여 .npy 파일 로드
        npy_path = f"{self.landmark_root_dir}/{landmark_relative_path}"
        landmarks = np.load(npy_path)

        # 3. 데이터 정규화 (Normalization)
        # 예시: 모든 프레임의 모든 랜드마크를 '코' (Pose의 0번)를 기준으로 상대 좌표로 변환
        # Pose 랜드마크는 0~131번 인덱스에 저장되어 있음 (x, y, z, vis 순)
        # 0번 랜드마크의 x, y, z는 각각 0, 33, 66번 인덱스에 해당
        
        # 코의 좌표 (모든 프레임에 대해)
        nose_coords = landmarks[:, [0, 33, 66]] 
        
        # 모든 랜드마크 좌표에서 코의 좌표를 빼줌 (Broadcasting 활용)
        # 랜드마크는 (Pose, Face, Hand, Hand) 순으로 저장되어 있으므로
        # 각 부위별로 x, y, z 좌표를 분리해서 정규화를 수행해야 함.
        # (이 부분은 데이터 구조에 맞춰 정교하게 구현 필요)
        # 아래 예시 참고:
        # normalized_landmarks = landmarks - nose_coords.repeat(N) # N은 좌표 개수
        
        # 우선은 정규화 없이 원본을 텐서로 변환
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        label_tensor = torch.tensor(int(label), dtype=torch.long) # 라벨은 정수형

        return landmarks_tensor, label_tensor