def translate(model, src_input, tokenizer, max_len=30, device='cuda'):
    """
    Greedy Decoding 방식으로 번역을 수행하는 함수
    """
    model.eval()
    
    # 1. 입력 데이터 준비 (Batch 차원 추가: (T, 55, 2) -> (1, T, 55, 2))
    src = src_input.unsqueeze(0).to(device)
    src_key_padding_mask = torch.zeros(1, src.size(1)).bool().to(device) # 패딩 없음 가정

    # 2. 첫 입력 토큰 <sos> 생성
    sos_id = tokenizer.char_to_id[tokenizer.start_token]
    eos_id = tokenizer.char_to_id[tokenizer.end_token]
    
    # 현재까지 생성된 문장 (처음엔 <sos>만 있음)
    # ys는 무조건 LogTensor(정수)여야 함
    ys = torch.ones(1, 1).fill_(sos_id).long().to(device)

    # 3. 반복해서 다음 단어 예측
    for i in range(max_len):
        # 마스크 생성
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
        
        # 모델 예측
        # 여기서 ys가 float이면 에러가 났음. 따라서 .long()으로 고정해줌.
        out = model(src, ys, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask)
        
        # 마지막 단어의 확률 분포 가져오기
        prob = out[:, -1] 
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # 4. 예측된 단어 추가
        # src.data(실수)를 따라가지 않고 그냥 정수(Long)로 생성해서 붙임
        next_word_tensor = torch.ones(1, 1).fill_(next_word).long().to(device)
        ys = torch.cat([ys, next_word_tensor], dim=1)
        
        # <eos>가 나오면 종료
        if next_word == eos_id:
            break
            
    # 5. ID 리스트를 텍스트로 변환
    final_ids = ys[0].tolist()
    return tokenizer.decode(final_ids)


# 1. 저장된 모델 불러오기 (경로 확인)
# 예: '/content/drive/MyDrive/수어데이터_공용/Daily70/daily70_best_model.pth'
# '/content/drive/MyDrive/수어데이터_공용/LandmarkE2EModel/best_model_train1.pth'
# 이름 확인 잘 하기 (train[횟수])
MODEL_PATH = '/content/drive/MyDrive/수어데이터_공용/LandmarkE2EModel/best_model_train1.pth'

print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
print("Model loaded successfully")

# 2. Validation 데이터 중 하나 꺼내서 번역 시켜보기
print("\n--- Inference Test ---")

# Val Loader에서 첫 번째 배치를 가져옴
batch = next(iter(val_loader))
src_sample = batch["src"][0] # 첫 번째 영상 데이터
tgt_sample = batch["tgt"][0] # 정답 라벨 (ID)

# 정답 텍스트 확인
real_text = tokenizer.decode(tgt_sample.tolist())
print(f"Target(정답): {real_text}")

# 모델 예측 확인
pred_text = translate(model, src_sample, tokenizer, device=DEVICE)
print(f"Model(예측): {pred_text}")

print("\n----------------------")