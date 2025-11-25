import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

# 1. 유틸리티 함수 (Mask 생성)
def generate_square_subsequent_mask(sz, device):
    """
    Decoder가 미래의 단어를 미리 보지 못하게 가리는 마스크 (Causal Mask)
    형태: 대각선 위쪽은 -inf, 아래쪽은 0
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 2. 학습 및 검증 함수 정의
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # 학습 모드 전환 (Dropout 켜짐)
    total_loss = 0
    
    for batch in dataloader:
        # 데이터 로드
        src = batch["src"].to(device) # (Batch, Frames, 55, 2)
        tgt = batch["tgt"].to(device) # (Batch, Seq_Len)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)
        tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)

        # Transformer 입력 준비
        # tgt_input: <sos> ... <last_word> (모델 입력)
        # tgt_out:   ... <last_word> <eos> (모델이 맞춰야 할 정답)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        # 마스크 길이 재조정 (입력이 하나 줄었으므로)
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        
        # Causal Mask 생성 (Seq_len x Seq_len)
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

        # Forward (예측)
        # src_key_padding_mask는 Encoder가 패딩을 무시하게 함
        # tgt_mask는 Decoder가 미래를 못 보게 함
        # tgt_key_padding_mask는 Decoder가 패딩을 무시하게 함
        output = model(
            src=src, 
            tgt=tgt_input, 
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Loss 계산 (Output: Batch*Seq x Vocab, Target: Batch*Seq)
        # CrossEntropyLoss는 (N, C) 형태를 원하므로 reshape 필요
        output = output.reshape(-1, output.shape[-1])
        tgt_out = tgt_out.reshape(-1)

        loss = criterion(output, tgt_out)

        # Backward (역전파)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval() # 평가 모드 전환 (Dropout 꺼짐)
    total_loss = 0
    
    with torch.no_grad(): # 그래디언트 계산 안 함 (메모리 절약)
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_key_padding_mask = batch["src_key_padding_mask"].to(device)
            tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            tgt_seq_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

            output = model(
                src=src, 
                tgt=tgt_input, 
                src_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            output = output.reshape(-1, output.shape[-1])
            tgt_out = tgt_out.reshape(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# 3. 메인 실행 코드
# 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 1 # epoch 설정
LEARNING_RATE = 0.0001

print(f"Using device: {DEVICE}")

# 데이터 경로 (Colab 경로에 맞게 수정 필요)
# 프로젝트 경로 예시: /content/drive/MyDrive/...
CSV_PATH_TRAIN = '/content/drive/MyDrive/11/dm70_train_with_landmarks_8000.csv'
# Validation용 CSV가 따로 없다면 Train CSV를 쪼개거나, 일단 Train 파일로 테스트(임시)
# 여기서는 일단 Train 파일을 그대로 넣음 (실제로는 Val용 CSV 필요)
CSV_PATH_VAL = CSV_PATH_TRAIN 
ROOT_DIR = '/content/drive/MyDrive/11/' # NPY 파일들이 있는 상위 폴더

# 1. 전처리기 준비
preprocessor = WinstonPreprocessor()

# 2. 데이터 로더 준비 (Train / Val)
# get_train_loader 함수는 위에서 정의했다고 가정
train_loader, tokenizer = get_train_loader(CSV_PATH_TRAIN, ROOT_DIR, preprocessor, batch_size=BATCH_SIZE)
# Val 로더는 셔플을 끔
val_loader, _ = get_train_loader(CSV_PATH_VAL, ROOT_DIR, preprocessor, batch_size=BATCH_SIZE)

# 3. 모델 초기화
# num_tokens는 tokenizer의 vocab_size
model = SignLanguageTransformer(num_tokens=tokenizer.vocab_size).to(DEVICE)

# 4. Loss & Optimizer
# pad_token은 loss 계산에서 제외 (정답이 패딩인 곳은 맞추든 틀리든 상관없음)
pad_token_id = tokenizer.char_to_id[tokenizer.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. 학습 루프 실행
print("Training Started...")
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Validation
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
    
    # Best Model 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'\t>>> Best model saved! (Val Loss: {val_loss:.3f})')

print("Training Finished!")