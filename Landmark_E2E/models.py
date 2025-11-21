import torch
import torch.nn as nn
import math

class MultiStreamEmbedding(nn.Module):
    """
    SignMusketeers을 적용한 임베딩 레이어
    전체 55개 키포인트를 Body, Left Hand, Right Hand로 나누어 각각 처리(Stream)한 뒤 합침
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # 1. 각 스트림(부위)별 입력 차원 정의
        # 입력 데이터 형태: (x, y) 좌표이므로 점 개수 * 2
        self.body_in_dim = 13 * 2  # 몸통: 13개 점
        self.l_hand_in_dim = 21 * 2 # 왼손: 21개 점
        self.r_hand_in_dim = 21 * 2 # 오른손: 21개 점

        # 2. 각 스트림을 처리할 작은 신경망 (MLP)
        # DINOv2와 같은 거대 모델 대신, 좌표의 특징을 찾아낼 Linear 레이어를 씀
        # 각 부위의 특징을 d_model의 1/3 정도 크기로 만듦
        hidden_dim = d_model // 3
        
        self.body_embedding = nn.Sequential(
            nn.Linear(self.body_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.l_hand_embedding = nn.Sequential(
            nn.Linear(self.l_hand_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.r_hand_embedding = nn.Sequential(
            nn.Linear(self.r_hand_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. 합쳐진 특징을 최종 d_model 사이즈로 맞춰줄 프로젝션 레이어
        # 3개 스트림을 합치면 크기가 (hidden_dim * 3)이 됨
        self.fusion_layer = nn.Linear(hidden_dim * 3, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (Batch_Size, Frames, 55, 2) - 전처리한 데이터
        """
        # 1. 데이터 Slicing (데이터 쪼개기)
        # x[:, :, :13, :] -> 모든 배치, 모든 프레임, 0~12번 인덱스(Body), (x,y)
        body = x[:, :, :13, :]      # (Batch, Frames, 13, 2)
        l_hand = x[:, :, 13:34, :]  # (Batch, Frames, 21, 2)
        r_hand = x[:, :, 34:, :]    # (Batch, Frames, 21, 2)

        # 2. Flatten (좌표 펼치기)
        # (Batch, Frames, Points, 2) -> (Batch, Frames, Points * 2)
        b, f, _, _ = x.size()
        body = body.flatten(2)      # (Batch, Frames, 26)
        l_hand = l_hand.flatten(2)  # (Batch, Frames, 42)
        r_hand = r_hand.flatten(2)  # (Batch, Frames, 42)

        # 3. Stream Processing (개별 특징 추출)
        body_feat = self.body_embedding(body)     # (Batch, Frames, hidden_dim)
        l_hand_feat = self.l_hand_embedding(l_hand)
        r_hand_feat = self.r_hand_embedding(r_hand)

        # 4. Concatenate & Fuse (합치기)
        combined = torch.cat([body_feat, l_hand_feat, r_hand_feat], dim=2) 
        # 현재 combined 형태: (Batch, Frames, hidden_dim * 3)

        out = self.fusion_layer(combined) # (Batch, Frames, d_model)
        return self.dropout(out)

class PositionalEncoding(nn.Module):
    """
    Transformer는 순서 정보를 모르기 때문에 위치 정보를 더해주는 모듈
    (Pytorch 튜토리얼 표준 구현)
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Frames, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class SignLanguageTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # 1. 우리가 만든 Multi-Stream 임베딩
        self.embedding = MultiStreamEmbedding(d_model, dropout)
        
        # 2. 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. 핵심 Transformer (Encoder-Decoder 구조)
        # batch_first=True로 설정하여 (Batch, Time, Feature) 순서를 유지함
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        # 4. 출력 레이어 (단어 예측)
        # Target(Text)을 위한 임베딩도 필요함
        self.tgt_embedding = nn.Embedding(num_tokens, d_model)
        self.generator = nn.Linear(d_model, num_tokens)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        """
        src: (Batch, Frames, 55, 2) - 전처리된 랜드마크 데이터
        tgt: (Batch, Seq_Len) - 정답 텍스트 토큰 인덱스
        """
        # 1. Source(랜드마크) 처리
        src_emb = self.embedding(src)   # (Batch, Frames, d_model)
        src_emb = self.pos_encoder(src_emb)

        # 2. Target(텍스트) 처리
        tgt_emb = self.tgt_embedding(tgt) # (Batch, Seq_Len, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 3. Transformer 통과
        # nn.Transformer는 내부적으로 Encoder와 Decoder 모두 수행
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 4. 최종 단어 확률 예측
        return self.generator(output)

# 사용 예시 (Dummy Data)
# 배치 크기 2, 프레임 수 100, 55개 점, (x,y) 2좌표
dummy_input = torch.randn(2, 100, 55, 2)
# 배치 크기 2, 문장 길이 10 (정답 토큰)
dummy_target = torch.randint(0, 1000, (2, 10))

# 모델 생성 (단어 집합 크기 1000개 가정)
model = SignLanguageTransformer(num_tokens=1000)

# 추론 (Forward)
output = model(dummy_input, dummy_target)
print("입력 형태:", dummy_input.shape) # torch.Size([2, 100, 55, 2])
print("출력 형태:", output.shape)      # torch.Size([2, 10, 1000]) -> (배치, 문장길이, 단어확률)