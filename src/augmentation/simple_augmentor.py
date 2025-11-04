# -*- coding: utf-8 -*-
"""
3. 단순 증강: 소스 노이즈 주입 (Source Noise Injection)

[수정 버전: N-times Augmentation]
각 원본 문장마다 N번씩 노이즈 주입을 시도하여
데이터의 양을 원본의 N배까지 증강합니다.
"""

import os
import random
from typing import List

# --- 1. 파일 경로 설정 (프로젝트 구조 기반) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# [입력]
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
GLOSS_PATH = os.path.join(DATA_DIR, 'gloss.tok.tagged')
TEXT_PATH = os.path.join(DATA_DIR, 'text.tok.pos')

# [출력]
AUG_GLOSS_PATH = os.path.join(DATA_DIR, 'gloss.aug.noise')
AUG_TEXT_PATH = os.path.join(DATA_DIR, 'text.aug.noise')

# --- 2. 하이퍼파라미터 (조정 가능) ---
# [추가] 문장 당 증강 시도 횟수
N_AUGMENTATIONS_PER_SENTENCE = 2

DROPOUT_PROB = 0.1  # 토큰을 삭제할 확률
SWAP_PROB = 0.1     # 인접 토큰과 순서를 바꿀 확률

# --- 3. 헬퍼 함수 ---
def load_lines(path: str) -> List[str]:
    """파일을 읽어 라인 리스트로 반환"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: 입력 파일을 찾을 수 없습니다. {path}")
        return []

def apply_noise(tokens: List[str]) -> (List[str], bool):
    """글로스 토큰 리스트에 노이즈를 적용"""
    if not tokens:
        return [], False

    noisy_tokens = []
    i = 0
    was_modified = False

    while i < len(tokens):
        token = tokens[i]
        roll = random.random()

        if roll < DROPOUT_PROB:
            i += 1
            was_modified = True
            continue
        elif roll < (DROPOUT_PROB + SWAP_PROB) and i + 1 < len(tokens):
            noisy_tokens.append(tokens[i+1])
            noisy_tokens.append(token)
            i += 2
            was_modified = True
            continue
        else:
            noisy_tokens.append(token)
            i += 1

    if not noisy_tokens:
        return tokens, False
        
    return noisy_tokens, was_modified

# --- 4. 메인 증강 로직 ---
def main():
    print("===========================================")
    print(f"--- 단순 증강 (소스 노이즈 주입, x{N_AUGMENTATIONS_PER_SENTENCE}) 시작 ---")
    print(f"(Dropout={DROPOUT_PROB}, Swap={SWAP_PROB})")
    print("===========================================")

    gloss_lines = load_lines(GLOSS_PATH)
    text_lines = load_lines(TEXT_PATH)

    if not gloss_lines or not text_lines or len(gloss_lines) != len(text_lines):
        print("오류: 입력 파일(gloss, text)에 문제가 있어 작업을 중단합니다.")
        return
        
    print(f"로드 완료: {len(gloss_lines)} 개의 병렬 문장")

    augmented_count = 0
    
    with open(AUG_GLOSS_PATH, 'w', encoding='utf-8') as f_gloss_out, \
         open(AUG_TEXT_PATH, 'w', encoding='utf-8') as f_text_out:
        
        total_sentences = len(gloss_lines)
        for i, (gloss_line, text_line) in enumerate(zip(gloss_lines, text_lines)):
            
            original_gloss_tokens = gloss_line.split()
            
            # [수정] 각 문장마다 N번씩 증강 시도
            for _ in range(N_AUGMENTATIONS_PER_SENTENCE):
                noisy_gloss_tokens, was_modified = apply_noise(original_gloss_tokens)
                
                if was_modified:
                    augmented_count += 1
                    f_gloss_out.write(" ".join(noisy_gloss_tokens) + "\n")
                    f_text_out.write(text_line + "\n")
                
            if (i + 1) % 10000 == 0:
                print(f"  ... {i+1} / {total_sentences} 원본 문장 처리 중")

    print("\n--- 단순 노이즈 증강 완료 ---")
    print(f"총 {augmented_count} 개의 문장이 증강되었습니다.")
    print(f"(목표: 약 {len(gloss_lines) * N_AUGMENTATIONS_PER_SENTENCE} 개)")
    print(f"증강된 글로스: {AUG_GLOSS_PATH}")
    print(f"증강된 텍스트: {AUG_TEXT_PATH}")

if __name__ == "__main__":
    main()
