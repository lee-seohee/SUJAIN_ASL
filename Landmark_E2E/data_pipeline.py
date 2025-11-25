import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple, Dict

# 작성한 전처리 클래스 (참조용으로 임포트 또는 붙여넣기 했다고 가정)
# from preprocessing import WinstonPreprocessor 

class SimpleTokenizer:
    """
    영어 문장을 토큰화하고 정수 인덱스로 변환하는 간단한 토크나이저.
    Step 5 베이스라인을 위해 복잡한 BPE 대신 공백 기반 분리를 사용함.
    """
    def __init__(self, min_freq: int = 2):
        self.char_to_id = {}
        self.id_to_char = {}
        # 특수 토큰 정의
        self.pad_token = "<pad>" # 0
        self.start_token = "<sos>" # 1
        self.end_token = "<eos>" # 2
        self.unk_token = "<unk>" # 3
        
        self.special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        self.min_freq = min_freq
        self.vocab_size = 0

    def build_vocab(self, sentences: List[str]):
        """학습 데이터의 문장들을 받아 단어 사전을 구축"""
        word_freq = Counter()
        for sentence in sentences:
            # 소문자 변환 및 구두점 일부 제거 (베이스라인용 단순화)
            words = sentence.lower().strip().split()
            word_freq.update(words)
        
        # ID 맵핑 생성
        self.id_to_char = {i: token for i, token in enumerate(self.special_tokens)}
        self.char_to_id = {token: i for i, token in enumerate(self.special_tokens)}
        
        idx = len(self.special_tokens)
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                self.char_to_id[word] = idx
                self.id_to_char[idx] = word
                idx += 1
        
        self.vocab_size = len(self.char_to_id)
        print(f"Tokenizer vocab built: {self.vocab_size} tokens (min_freq={self.min_freq})")

    def encode(self, sentence: str) -> List[int]:
        """문장 -> ID 리스트 변환 (SOS, EOS 포함)"""
        words = sentence.lower().strip().split()
        # Unknown 토큰 처리
        ids = [self.char_to_id.get(w, self.char_to_id[self.unk_token]) for w in words]
        return [self.char_to_id[self.start_token]] + ids + [self.char_to_id[self.end_token]]

    def decode(self, ids: List[int]) -> str:
        """ID 리스트 -> 문장 변환"""
        tokens = []
        for i in ids:
            token = self.id_to_char.get(i, self.unk_token)
            if token in [self.start_token, self.end_token, self.pad_token]:
                continue
            tokens.append(token)
        return " ".join(tokens)


class SignLanguageDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, tokenizer: SimpleTokenizer, 
                 preprocessor, max_seq_len: int = 200, is_train: bool = True):
        """
        Args:
            csv_path: 메타데이터 CSV 파일 경로
            root_dir: NPY 파일들이 있는 최상위 디렉토리 (Colab Mount 경로)
            tokenizer: 학습된 Tokenizer 객체
            preprocessor: WinstonPreprocessor 인스턴스
            max_seq_len: 메모리 보호를 위한 프레임 최대 길이 제한
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_seq_len = max_seq_len
        self.is_train = is_train

        # CSV 컬럼 확인
        # 필요한 컬럼: 'landmarks_path', 'caption'
        self.paths = self.data['landmarks_path'].values
        self.captions = self.data['caption'].values

    def __len__(self):
        return len(self.data)

    def _parse_mediapipe_flat_data(self, flat_data: np.ndarray) -> np.ndarray:
        """
        (Frames, 1662) 형태의 Flat 데이터를 
        (Frames, 543, 2) 형태의 구조화된 데이터로 변환함.
        
        MediaPipe Holistic (Total 543 points) 구조 가정:
        - Pose (33): 4 values (x, y, z, vis) -> 33*4 = 132
        - Face (468): 3 values (x, y, z) -> 468*3 = 1404
        - LHand (21): 3 values (x, y, z) -> 21*3 = 63
        - RHand (21): 3 values (x, y, z) -> 21*3 = 63
        Total flattened length = 132 + 1404 + 63 + 63 = 1662
        """
        frames = flat_data.shape[0]
        
        # Slicing Indices
        pose_end = 132
        face_end = 132 + 1404
        lh_end = 132 + 1404 + 63
        rh_end = 1662 # 132 + 1404 + 63 + 63
        
        # 1. Split by component
        # Pose: (T, 33, 4) -> (x, y)만 필요하므로 (T, 33, 2)
        pose_raw = flat_data[:, 0:pose_end].reshape(frames, 33, 4)
        pose_xy = pose_raw[:, :, :2] 
        
        # Face: (T, 468, 3) -> (T, 468, 2)
        face_raw = flat_data[:, pose_end:face_end].reshape(frames, 468, 3)
        face_xy = face_raw[:, :, :2]
        
        # Left Hand: (T, 21, 3) -> (T, 21, 2)
        lh_raw = flat_data[:, face_end:lh_end].reshape(frames, 21, 3)
        lh_xy = lh_raw[:, :, :2]
        
        # Right Hand: (T, 21, 3) -> (T, 21, 2)
        rh_raw = flat_data[:, lh_end:rh_end].reshape(frames, 21, 3)
        rh_xy = rh_raw[:, :, :2]
        
        # 2. Reorder for WinstonPreprocessor
        # Preprocessor expects: Pose(33) -> LHand(21) -> RHand(21) -> Face(...)
        # (WinstonPreprocessor의 LH_START=33 설정을 맞추기 위함)
        combined = np.concatenate([pose_xy, lh_xy, rh_xy, face_xy], axis=1) # (T, 543, 2)
        
        return combined

    def __getitem__(self, idx):
        # 1. Load NPY Data
        rel_path = self.paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        
        try:
            # (Frames, 1662) 로드
            raw_flat = np.load(full_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # 에러 시 0으로 채워진 더미 데이터 반환 (학습 중단 방지)
            return self.__getitem__((idx + 1) % len(self))

        # 2. Sampling / Truncation (Colab 메모리 보호)
        if raw_flat.shape[0] > self.max_seq_len:
            # 단순 Truncation (앞부분만 사용) or Uniform Sampling
            # 여기서는 간단히 앞부분 사용
            raw_flat = raw_flat[:self.max_seq_len, :]
            
        # 3. Parse & Preprocess
        # (T, 1662) -> (T, 543, 2)
        structured_data = self._parse_mediapipe_flat_data(raw_flat)
        
        # (T, 543, 2) -> (T, 55, 2) [Core 55 points]
        # WinstonPreprocessor.process() 호출
        processed_data = self.preprocessor.process(structured_data)
        
        # Tensor 변환
        src_input = torch.FloatTensor(processed_data) # (T, 55, 2)

        # 4. Tokenize Caption
        caption = self.captions[idx]
        if not isinstance(caption, str): # NaN 처리
            caption = ""
        
        tgt_ids = self.tokenizer.encode(caption)
        tgt_input = torch.LongTensor(tgt_ids) # (L,)

        return {
            "src": src_input,
            "tgt": tgt_input,
            "src_len": src_input.size(0),
            "tgt_len": tgt_input.size(0)
        }


def collate_fn(batch):
    """
    배치 내의 가변 길이 시퀀스를 패딩(Padding)하여 동일한 길이로 맞춤
    """
    # 1. Sort by src length (Optional, for packed_sequence but good practice)
    batch.sort(key=lambda x: x["src_len"], reverse=True)
    
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]
    
    # 2. Padding
    # src: (Batch, Max_Frame, 55, 2)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    
    # tgt: (Batch, Max_Text_Len)
    # Tokenizer의 pad_token_id는 보통 0 (위에서 확인 필요)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)
    
    # 3. Padding Masks 생성
    # Transformer는 패딩된 부분을 보지 않아야 하므로 True/False 마스크 필요
    # (Batch, Time) - True면 무시(Padding)
    
    # Source Mask
    B, T, _, _ = src_padded.shape
    src_lengths = torch.LongTensor([item["src_len"] for item in batch])
    
    # Target Mask
    tgt_lengths = torch.LongTensor([item["tgt_len"] for item in batch])
    
    # Key Padding Mask 생성 함수
    def create_padding_mask(lengths, max_len):
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        return mask # (Batch, Max_Len), True where padded
    
    src_key_padding_mask = create_padding_mask(src_lengths, src_padded.size(1))
    tgt_key_padding_mask = create_padding_mask(tgt_lengths, tgt_padded.size(1))

    return {
        "src": src_padded,
        "tgt": tgt_padded, # <sos> ... <eos> <pad> ...
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask
    }

# 사용법 예시 함수
def get_train_loader(csv_path, root_dir, preprocessor, batch_size=32):
    # 1. Tokenizer 빌드
    print("Building Tokenizer...")
    df = pd.read_csv(csv_path)
    captions = df['caption'].dropna().astype(str).tolist()
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(captions)
    
    # 2. Dataset 생성
    dataset = SignLanguageDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_seq_len=200 # Colab 메모리 고려
    )
    
    # 3. DataLoader 생성
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2, # Colab에서는 2 정도가 적당
        pin_memory=True
    )
    
    return loader, tokenizer