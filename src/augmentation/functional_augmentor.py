# -*- coding: utf-8 -*-
"""
4. 기능어 불일치 기반 증강 (Source-Context)

[최종 수정 버전]
Source-side(글로스) 문맥을 기반으로, 
Source-side(글로스) 문장에 영어 기능어를 삽입하여 노이즈를 줍니다.
Target-side(영어) 문장은 원본을 그대로 유지합니다.
"""

import os
import json
import random
import sys
from collections import defaultdict
from typing import List

# --- 1. 파일 경로 설정 (프로젝트 구조 기반) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# [입력]
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
GLOSS_PATH = os.path.join(PROCESSED_DATA_DIR, 'gloss.tok.tagged')
TEXT_PATH = os.path.join(PROCESSED_DATA_DIR, 'text.tok.pos')

ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'functional_divergence')
PROB_MATRIX_PATH = os.path.join(ANALYSIS_DIR, 'P_insert_Source_matrix.json')

# [출력]
AUG_GLOSS_PATH = os.path.join(PROCESSED_DATA_DIR, 'gloss.aug.functional')
AUG_TEXT_PATH = os.path.join(PROCESSED_DATA_DIR, 'text.aug.functional')


class FunctionalAugmentor:
    def __init__(self, prob_matrix_path):
        self.prob_matrix = self.load_prob_matrix(prob_matrix_path)
        print("FunctionalAugmentor 초기화 완료.")

    def load_prob_matrix(self, path: str) -> dict:
        print(f"\n기능어 삽입 확률 테이블 로딩: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"치명적 오류: 확률 매트릭스 파일을 찾을 수 없습니다! {path}", file=sys.stderr)
            return None

    def augment_gloss_sentence(self, gloss_tokens: List[str]) -> (str, bool):
        """ [핵심 수정 로직] 글로스 문장에 영어 기능어를 삽입 """
        if not self.prob_matrix or not gloss_tokens:
            return " ".join(gloss_tokens), False

        new_gloss_tokens = []
        was_modified = False

        # 1. 문장 시작(BOS) 위치에 기능어 삽입 시도
        context_key = f"BOS_{gloss_tokens[0].split('/')[0]}"
        for feature, contexts in self.prob_matrix.items():
            if context_key in contexts:
                if random.random() < contexts[context_key]:
                    token_to_insert = feature.split('/')[0]
                    new_gloss_tokens.append(token_to_insert)
                    was_modified = True

        # 2. 글로스 단어와 단어 사이에 기능어 삽입 시도
        for i in range(len(gloss_tokens) - 1):
            new_gloss_tokens.append(gloss_tokens[i])
            
            prev_gloss_key = gloss_tokens[i].split('/')[0]
            next_gloss_key = gloss_tokens[i+1].split('/')[0]
            context_key = f"{prev_gloss_key}_{next_gloss_key}"

            for feature, contexts in self.prob_matrix.items():
                if context_key in contexts:
                    if random.random() < contexts[context_key]:
                        token_to_insert = feature.split('/')[0]
                        new_gloss_tokens.append(token_to_insert)
                        was_modified = True
        
        # 3. 마지막 글로스 토큰 추가 및 EOS 위치에 기능어 삽입 시도
        new_gloss_tokens.append(gloss_tokens[-1])
        context_key = f"{gloss_tokens[-1].split('/')[0]}_EOS"
        for feature, contexts in self.prob_matrix.items():
            if context_key in contexts:
                if random.random() < contexts[context_key]:
                    token_to_insert = feature.split('/')[0]
                    new_gloss_tokens.append(token_to_insert)
                    was_modified = True

        return " ".join(new_gloss_tokens), was_modified

def load_lines(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: 입력 파일을 찾을 수 없습니다. {path}", file=sys.stderr)
        return []

def main():
    print("==================================================")
    print("--- 4. 기능어 불일치 기반 (Source-Context) 증강 시작 ---")
    print("==================================================")

    augmentor = FunctionalAugmentor(PROB_MATRIX_PATH)
    if not augmentor.prob_matrix:
        return

    gloss_lines = load_lines(GLOSS_PATH)
    text_lines = load_lines(TEXT_PATH)

    if not gloss_lines or not text_lines or len(gloss_lines) != len(text_lines):
        print("오류: 입력 파일(gloss, text)에 문제가 있어 작업을 중단합니다.", file=sys.stderr)
        return

    print(f"로드 완료: {len(gloss_lines)} 개의 병렬 문장")

    augmented_count = 0
    with open(AUG_GLOSS_PATH, 'w', encoding='utf-8') as f_gloss_out, \
         open(AUG_TEXT_PATH, 'w', encoding='utf-8') as f_text_out:

        total_sentences = len(gloss_lines)
        for i in range(total_sentences):
            gloss_line = gloss_lines[i]
            text_line = text_lines[i]

            augmented_gloss, was_modified = augmentor.augment_gloss_sentence(gloss_line.split())

            if was_modified:
                augmented_count += 1
                f_gloss_out.write(augmented_gloss + "\n")
                f_text_out.write(text_line + "\n") # 영어는 원본 그대로 저장

            if (i + 1) % 10000 == 0:
                print(f"  ... {i+1} / {total_sentences} 문장 처리 중")

    print("\n--- 기능어 기반 증강 완료 ---")
    print(f"총 {augmented_count} 개의 문장이 증강되었습니다.")
    print(f"증강된 글로스: {AUG_GLOSS_PATH}")
    print(f"증강된 텍스트: {AUG_TEXT_PATH}")

if __name__ == "__main__":
    main()
