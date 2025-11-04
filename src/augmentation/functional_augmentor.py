# -*- coding: utf-8 -*-
"""
4. 기능어 불일치 기반 증강 (Source-Context)

[최종 수정 버전 2]
메모리 리스트를 사용하여 파일 쓰기 오류를 수정한 최종 버전
"""

import os
import json
import random
import sys
from collections import defaultdict
from typing import List

# --- 경로 설정 (이전과 동일) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
AUG_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '03_augmented')
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'functional_divergence')

GLOSS_PATH = os.path.join(PROCESSED_DATA_DIR, 'gloss.tok.tagged')
TEXT_PATH = os.path.join(PROCESSED_DATA_DIR, 'text.tok.pos')
ALIGNMENT_PATH = os.path.join(PROCESSED_DATA_DIR, 'alignment_map.txt')
PROB_MATRIX_PATH = os.path.join(ANALYSIS_DIR, 'P_insert_Source_matrix.json')

AUG_GLOSS_PATH = os.path.join(AUG_DATA_DIR, 'gloss.aug.functional')
AUG_TEXT_PATH = os.path.join(AUG_DATA_DIR, 'text.aug.functional')


class FunctionalAugmentor:
    # ... (클래스 로직은 이전과 동일) ...
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
        if not self.prob_matrix or not gloss_tokens:
            return " ".join(gloss_tokens), False
        new_gloss_tokens = []
        was_modified = False
        # ... (이하 augment_gloss_sentence 로직은 이전과 동일) ...
        context_key = f"BOS_{gloss_tokens[0].split('/')[0]}"
        for feature, contexts in self.prob_matrix.items():
            if context_key in contexts:
                if random.random() < contexts[context_key]:
                    new_gloss_tokens.append(feature.split('/')[0])
                    was_modified = True
        for i in range(len(gloss_tokens) - 1):
            new_gloss_tokens.append(gloss_tokens[i])
            prev_gloss_key = gloss_tokens[i].split('/')[0]
            next_gloss_key = gloss_tokens[i+1].split('/')[0]
            context_key = f"{prev_gloss_key}_{next_gloss_key}"
            for feature, contexts in self.prob_matrix.items():
                if context_key in contexts:
                    if random.random() < contexts[context_key]:
                        new_gloss_tokens.append(feature.split('/')[0])
                        was_modified = True
        new_gloss_tokens.append(gloss_tokens[-1])
        context_key = f"{gloss_tokens[-1].split('/')[0]}_EOS"
        for feature, contexts in self.prob_matrix.items():
            if context_key in contexts:
                if random.random() < contexts[context_key]:
                    new_gloss_tokens.append(feature.split('/')[0])
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
    align_lines = load_lines(ALIGNMENT_PATH)

    if not (len(gloss_lines) == len(text_lines) == len(align_lines)):
        print("오류: 입력 파일들의 줄 수가 일치하지 않습니다.", file=sys.stderr)
        return

    print(f"로드 완료: {len(gloss_lines)} 개의 병렬 문장")

    # [핵심 수정] 결과를 메모리 리스트에 먼저 저장
    final_gloss_output = []
    final_text_output = []

    total_sentences = len(gloss_lines)
    for i in range(total_sentences):
        gloss_line = gloss_lines[i]
        text_line = text_lines[i]

        # 1. 원본 데이터 쌍을 먼저 리스트에 추가
        final_gloss_output.append(gloss_line + "\n")
        final_text_output.append(text_line + "\n")

        # 2. 증강을 시도하고, 성공하면 추가로 리스트에 추가
        augmented_gloss, was_modified = augmentor.augment_gloss_sentence(gloss_line.split())

        if was_modified:
            final_gloss_output.append(augmented_gloss + "\n")
            final_text_output.append(text_line + "\n") # 영어는 항상 원본

        if (i + 1) % 10000 == 0:
            print(f"  ... {i+1} / {total_sentences} 문장 처리 중")

    # 3. 모든 작업이 끝난 후, 리스트의 내용을 파일에 한 번에 쓰기
    print("\n결과 파일 저장 중...")
    os.makedirs(os.path.dirname(AUG_GLOSS_PATH), exist_ok=True)
    with open(AUG_GLOSS_PATH, 'w', encoding='utf-8') as f_gloss_out:
        f_gloss_out.writelines(final_gloss_output)
    with open(AUG_TEXT_PATH, 'w', encoding='utf-8') as f_text_out:
        f_text_out.writelines(final_text_output)

    print("\n--- 기능어 기반 증강 완료 ---")
    print(f"총 {len(final_text_output)} 개의 문장이 최종 생성되었습니다.")
    print(f"(원본 {total_sentences}개 포함)")
    print(f"증강된 글로스: {AUG_GLOSS_PATH}")
    print(f"증강된 텍스트: {AUG_TEXT_PATH}")

if __name__ == "__main__":
    main()