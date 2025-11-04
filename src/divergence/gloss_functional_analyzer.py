# -*- coding: utf-8 -*-
"""
ASL Functional Divergence Analyzer (Source-Side Context)

[고도화 버전]
기능어(F) 삽입 확률을 Target-side(영어) 문맥이 아닌,
Source-side(글로스) 문맥을 기준으로 계산합니다. (P(F | Prev_Gloss, Next_Gloss))

이는 트랜스포머 모델의 인코더에 '명시적 피처'로 주입하기에 
더 적합한 통계입니다.
"""
import os
import json
import sys
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# --- 1. 파일 경로 설정 (프로젝트 구조 기반 상대 경로) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 입력 파일 경로
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
TEXT_POS_PATH = os.path.join(PROCESSED_DATA_DIR, 'text.tok.pos')
GLOSS_TAGGED_PATH = os.path.join(PROCESSED_DATA_DIR, 'gloss.tok.tagged')
ALIGNMENT_PATH = os.path.join(PROCESSED_DATA_DIR, 'alignment_map.txt')

# 출력 파일 경로
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'functional_divergence')
COUNT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'F_C_Source_counts.json')
PROB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'P_insert_Source_matrix.json')

# --- 2. 분석할 기능어 품사(POS) 정의 ---
FUNCTION_POS = {
    'ADP', 'DET', 'AUX', 'PRON', 'PART', 'CCONJ', 'SCONJ'
}

# --- 3. 헬퍼 함수 ---
def parse_token_pos(token_pos: str) -> Tuple[str, str]:
    parts = token_pos.rsplit('/', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return token_pos, 'UNK'

def get_gloss_key(gloss_token_pos: str) -> str:
    return parse_token_pos(gloss_token_pos)[0]

class SourceContextAnalyzer:
    # ... (클래스 로직은 원본과 동일) ...
    def __init__(self, text_path, gloss_path, align_path, count_out, prob_out):
        self.text_pos_path = text_path
        self.gloss_tagged_path = gloss_path
        self.align_path = align_path
        self.count_output_path = count_out
        self.prob_output_path = prob_out
        self.text_lines: List[str] = []
        self.gloss_lines: List[str] = []
        self.align_lines: List[str] = []
        self.feature_context_counts = defaultdict(lambda: defaultdict(int))
        self.context_total_counts = defaultdict(int)
        self.p_f_c_table = defaultdict(dict)
        print("SourceContextAnalyzer (고도화 버전) 초기화 완료.")

    def load_data(self) -> bool:
        print("데이터 로딩 시작...")
        try:
            with open(self.text_pos_path, 'r', encoding='utf-8') as f:
                self.text_lines = [line.strip() for line in f if line.strip()]
            print(f"로드 완료: {self.text_pos_path} ({len(self.text_lines)} 라인)")
            with open(self.gloss_tagged_path, 'r', encoding='utf-8') as f:
                self.gloss_lines = [line.strip() for line in f if line.strip()]
            print(f"로드 완료: {self.gloss_tagged_path} ({len(self.gloss_lines)} 라인)")
            with open(self.align_path, 'r', encoding='utf-8') as f:
                self.align_lines = [line.strip() for line in f if line.strip()]
            print(f"로드 완료: {self.align_path} ({len(self.align_lines)} 라인)")
            if not (len(self.text_lines) == len(self.gloss_lines) == len(self.align_lines)):
                print(f"FATAL ERROR: 파일 라인 수가 일치하지 않습니다!", file=sys.stderr)
                print(f"  Text: {len(self.text_lines)}, Gloss: {len(self.gloss_lines)}, Align: {len(self.align_lines)}", file=sys.stderr)
                return False
            print(f"데이터 로딩 성공. 총 {len(self.text_lines)} 문장.")
            return True
        except FileNotFoundError as e:
            print(f"FATAL ERROR: 파일을 찾을 수 없음. {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"FATAL ERROR: 데이터 로딩 중 오류. {e}", file=sys.stderr)
            return False

    def parse_alignment(self, align_str: str) -> Tuple[Dict[int, List[int]], Set[int]]:
        text_to_gloss_map = defaultdict(list)
        aligned_text_indices = set()
        if not align_str:
            return text_to_gloss_map, aligned_text_indices
        try:
            pairs = align_str.split()
            for pair in pairs:
                gloss_idx_str, text_idx_str = pair.split('-')
                text_idx = int(text_idx_str)
                gloss_idx = int(gloss_idx_str)
                text_to_gloss_map[text_idx].append(gloss_idx)
                aligned_text_indices.add(text_idx)
        except ValueError as e:
            print(f"Warning: 정렬 파싱 오류. 건너뜁니다. 라인: '{align_str}'. 오류: {e}")
        return text_to_gloss_map, aligned_text_indices

    def analyze_divergence(self):
        print("\nSource-Context 기반 불일치 분석 시작...")
        total_lines = len(self.text_lines)
        for i, (text_line, gloss_line, align_line) in enumerate(zip(self.text_lines, self.gloss_lines, self.align_lines)):
            if (i + 1) % 10000 == 0:
                print(f"  ... {i+1} / {total_lines} 문장 처리 중")
            text_tokens = text_line.split()
            gloss_tokens = gloss_line.split()
            if not text_tokens or not gloss_tokens:
                continue
            text_to_gloss_map, aligned_text_indices = self.parse_alignment(align_line)
            len_text = len(text_tokens)
            len_gloss = len(gloss_tokens)
            for text_idx in range(len_text):
                token_pos = text_tokens[text_idx]
                token, pos = parse_token_pos(token_pos)
                is_unaligned = text_idx not in aligned_text_indices
                is_function_word = pos in FUNCTION_POS
                if is_unaligned and is_function_word:
                    feature_F = token_pos
                    prev_gloss_key = "BOS"
                    for j in range(text_idx - 1, -1, -1):
                        if j in text_to_gloss_map:
                            aligned_gloss_indices = text_to_gloss_map[j]
                            gloss_idx_prev = max(aligned_gloss_indices)
                            if gloss_idx_prev < len_gloss:
                                prev_gloss_key = get_gloss_key(gloss_tokens[gloss_idx_prev])
                            break
                    next_gloss_key = "EOS"
                    for j in range(text_idx + 1, len_text):
                        if j in text_to_gloss_map:
                            aligned_gloss_indices = text_to_gloss_map[j]
                            gloss_idx_next = min(aligned_gloss_indices)
                            if gloss_idx_next < len_gloss:
                                next_gloss_key = get_gloss_key(gloss_tokens[gloss_idx_next])
                            break
                    context_C = f"{prev_gloss_key}_{next_gloss_key}"
                    self.feature_context_counts[feature_F][context_C] += 1
                    self.context_total_counts[context_C] += 1
        print(f"분석 완료. 총 {sum(len(ctx) for ctx in self.feature_context_counts.values())} 개의 (F, C) 쌍 발견.")

    def calculate_probabilities(self):
        print("\n확률 P(F|C) 계산 시작...")
        num_calculations = 0
        for feature, contexts in self.feature_context_counts.items():
            for context, f_c_count in contexts.items():
                c_count = self.context_total_counts.get(context, 0)
                if c_count > 0:
                    probability = f_c_count / c_count
                    self.p_f_c_table[feature][context] = round(probability, 6)
                    num_calculations += 1
        print(f"확률 계산 완료. 총 {num_calculations} 개의 확률값 계산.")

    def save_results(self):
        if not self.p_f_c_table:
            print("Warning: 저장할 데이터가 없습니다.")
            return
        print(f"\n결과 저장 중... -> {OUTPUT_DIR}")
        os.makedirs(os.path.dirname(self.count_output_path), exist_ok=True)
        try:
            with open(self.count_output_path, 'w', encoding='utf-8') as f_count:
                json.dump(self.feature_context_counts, f_count, indent=2, ensure_ascii=False)
            print(f"빈도 수 저장 완료: {self.count_output_path}")
            with open(self.prob_output_path, 'w', encoding='utf-8') as f_prob:
                json.dump(self.p_f_c_table, f_prob, indent=2, ensure_ascii=False)
            print(f"확률 매트릭스 저장 완료: {self.prob_output_path}")
        except Exception as e:
            print(f"FATAL ERROR: 결과 저장 중 오류. {e}")

# --- 5. 메인 실행 블록 ---
if __name__ == "__main__":
    print("==============================================================")
    print("--- Source-Side Context 기반 기능어 분석기 (고도화 버전) ---")
    print("==============================================================")
    analyzer = SourceContextAnalyzer(
        text_path=TEXT_POS_PATH,
        gloss_path=GLOSS_TAGGED_PATH,
        align_path=ALIGNMENT_PATH,
        count_out=COUNT_OUTPUT_PATH,
        prob_out=PROB_OUTPUT_PATH
    )
    if analyzer.load_data():
        analyzer.analyze_divergence()
        analyzer.calculate_probabilities()
        analyzer.save_results()
        print("\n--- 모든 작업 완료 ---")
    else:
        print("\n--- 오류로 인해 작업을 완료하지 못했습니다 ---")
