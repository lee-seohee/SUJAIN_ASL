import random
import os
import sys
from typing import List

# --- 설정 (Configuration) ---

# ASL의 유연한 어순을 모델이 학습하도록 인위적으로 높게 설정 (20% 확률로 인접 단어 교환)
SWAP_PROBABILITY = 0.20 

def augment_reordering(gloss_sentence: str) -> str:

    # 이미 토큰화되어 있으므로, 줄 끝의 개행 문자만 제거 후 공백 분리
    words = gloss_sentence.strip().split(' ')
    if len(words) < 2:
        return gloss_sentence.strip() # 1단어 문장은 증강 불가

    new_words = words[:]
    
    i = 0
    # 인접 단어 쌍(Bigram) 순회
    while i < len(new_words) - 1:
        if random.random() < SWAP_PROBABILITY:
        
            temp = new_words[i]
            new_words[i] = new_words[i+1]
            new_words[i+1] = temp
            # 교환된 쌍은 건너뛰어 중복 교환을 방지
            i += 2
        else:
            # 교환하지 않았으면 다음 단어 쌍으로 이동
            i += 1
            
    # 다시 공백으로 join하여 문자열로 반환
    return " ".join(new_words)


def run_reordering_augmentation(
    input_gloss_path: str, 
    input_text_path: str, 
    output_gloss_path: str, 
    output_text_path: str,
):
    """원본 코퍼스를 읽어 어순 증강을 수행한 후 새로운 파일로 저장합니다."""
    import os 
    
    print(f"--- 🔄 어순 변형 증강 시작 (Swap Probability: {SWAP_PROBABILITY:.2f}) ---")

    try:
        # 파일은 줄 단위로 읽어옵니다. (각 줄 = 토큰화된 문장)
        with open(input_gloss_path, 'r', encoding='utf-8') as f:
            gloss_corpus = f.readlines()
        with open(input_text_path, 'r', encoding='utf-8') as f:
            english_corpus = f.readlines()
    except FileNotFoundError as e:
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {e}")
        return

    augmented_gloss: List[str] = []
    augmented_english: List[str] = []
    swap_count = 0

    for original_gloss_line, original_english_line in zip(gloss_corpus, english_corpus):
        # 1. 원본 쌍 추가
        augmented_gloss.append(original_gloss_line)
        augmented_english.append(original_english_line)
        
        # 2. 증강된 쌍 생성 및 추가
        new_gloss = augment_reordering(original_gloss_line)
        
        # 실제로 변형이 일어났는지 확인 (변형이 없으면 추가하지 않음)
        if new_gloss != original_gloss_line.strip(): 
            augmented_gloss.append(new_gloss + '\n') # 개행 문자 다시 추가
            augmented_english.append(original_english_line) # 영어 문장은 원본 유지
            swap_count += 1

    # 3. 출력 디렉토리 생성 및 파일 저장
    output_dir = os.path.dirname(output_gloss_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_gloss_path, 'w', encoding='utf-8') as f:
        f.writelines(augmented_gloss)
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.writelines(augmented_english)

    print(f"원본 문장 수: {len(gloss_corpus)}")
    print(f"새롭게 증강된 문장 쌍 수: {swap_count}")
    print(f"최종 코퍼스 크기: {len(augmented_gloss)}")
    print("-----------------------------------------")


if __name__ == '__main__':

    # --- 프로젝트 경로 설정 ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 입력 파일 (data/02_processed)
    GLOSS_IN = os.path.join(BASE_DIR, 'data', '02_processed', 'gloss.tok.tagged')
    ENGLISH_IN = os.path.join(BASE_DIR, 'data', '02_processed', 'text.tok.pos')

    # 출력 파일 (data/03_augmented 디렉토리에 저장)
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', '03_augmented')
    GLOSS_OUT = os.path.join(OUTPUT_DIR, 'gloss.reorder_aug')
    ENGLISH_OUT = os.path.join(OUTPUT_DIR, 'text.reorder_aug')
    
    # 0. 출력 디렉토리 생성 확인
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 어순 변형 증강 실행
    print("--- ⚙️ ASL-English 어순 증강 파이프라인 시작 ---")
    run_reordering_augmentation(
        input_gloss_path=GLOSS_IN,
        input_text_path=ENGLISH_IN,
        output_gloss_path=GLOSS_OUT,
        output_text_path=ENGLISH_OUT
    )
    print("--- ✅ 어순 증강 작업 완료 ---")