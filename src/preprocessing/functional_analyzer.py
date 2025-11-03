#기능어 불일치
import os
import json
from collections import defaultdict
from typing import Dict, Set

# --- 1. 파일 경로 설정 (청사진 기반) ---
TEXT_POS_PATH = r'C:\Users\john9\Desktop\코딩\code\real\text.tok.pos'
ALIGNMENT_PATH = r'C:\Users\john9\Desktop\코딩\code\real\alignment.txt'

OUTPUT_DIR = r'C:\Users\john9\Desktop\코딩\code\real\functional_divergence'
COUNT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'F_C_counts.json')
PROB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'P_insert_matrix.json') # CSV보다 JSON이 더 유연함

# --- 2. 분석할 기능어 품사(POS) 정의 (Universal Dependencies 기준) ---
FUNCTION_POS = {
    'ADP',  # 전치사 (e.g., to, in, on)
    'DET',  # 관사 (e.g., a, an, the)
    'AUX',  # 조동사 (e.g., is, was, have)
    'PRON', # 대명사 (e.g., I, you, it) - 생략되는 경우
    'PART'  # 분사 (e.g., 's, to (in "to go"))
}

class FunctionalDivergenceAnalyzer:
    """
    단어 정렬(Alignment)과 품사 태그(POS)를 기반으로
    기능어 삽입의 조건부 확률 P(F|C)를 계산합니다.
    
    F = 미정렬(unaligned) 기능어 (e.g., "the/DET")
    C = F를 둘러싼 문맥 (e.g., "NOUN_NOUN")
    """
    def __init__(self, text_path: str, align_path: str, count_path: str, prob_path: str):
        self.text_pos_path = text_path
        self.align_path = align_path
        self.count_output_path = count_path
        self.prob_output_path = prob_path

        # P(F|C) = Count(F, C) / Count(C)
        self.context_counts = defaultdict(int) # Count(C)
        self.feature_context_counts = defaultdict(lambda: defaultdict(int)) # Count(F, C)
        
        self.tagged_sentences = []
        self.alignments = []
        self.p_f_c_table = defaultdict(lambda: defaultdict(float))
        
        print("FunctionalDivergenceAnalyzer 초기화 완료")

    def load_data(self) -> bool:
        """text.tok.pos와 alignment_map.txt 파일을 로드"""
        print("\n데이터 로딩 시작...")
        try:
            with open(self.text_pos_path, 'r', encoding='utf-8') as f:
                self.tagged_sentences = [line.strip() for line in f.readlines()]
            
            with open(self.align_path, 'r', encoding='utf-8') as f:
                self.alignments = [line.strip() for line in f.readlines()]

            if len(self.tagged_sentences) != len(self.alignments):
                print(f"오류: 텍스트({len(self.tagged_sentences)})와 정렬({len(self.alignments)}) 파일의 줄 수가 다릅니다.")
                return False
            
            print(f"데이터 로딩 완료: 총 {len(self.tagged_sentences)}개 문장 쌍")
            return True
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다. {e.filename}")
            print("tagger.py와 aligner.py를 먼저 실행해야 합니다.")
            return False

    def analyze_divergence(self):
        """
        P(F|C) = Count(F, C) / Count(C) 계산
        """
        print("\n기능어 불일치 분석 시작...")
        
        for text_line, align_line in zip(self.tagged_sentences, self.alignments):
            
            tagged_tokens = text_line.split()
            if len(tagged_tokens) < 3: # 최소 3단어(prev, F, next)
                continue

            # 1. 정렬된 타겟(영어) 인덱스 집합 생성
            target_aligned_indices = set()
            try:
                for pair in align_line.split():
                    s_idx, t_idx = map(int, pair.split('-'))
                    target_aligned_indices.add(t_idx)
            except ValueError:
                continue # 빈 줄이나 잘못된 정렬 스킵

            # 2. 문장을 순회하며 (i-1) - (i) - (i+1) 구조 탐색
            for i in range(1, len(tagged_tokens) - 1):
                
                # --- 문맥(C) 정의 ---
                try:
                    prev_pos = tagged_tokens[i-1].split('/')[-1]
                    next_pos = tagged_tokens[i+1].split('/')[-1]
                except IndexError:
                    continue # 토큰/품사 형식이 아닐 경우
                
                # 로드맵 정의: C = (직전 단어의 품사) - (직후 단어의 품사)
                context = f"{prev_pos}_{next_pos}"
                
                # --- Count(C) 집계 ---
                self.context_counts[context] += 1
                
                # --- Count(F, C) 집계 ---
                # 현재 토큰(i)이 '미정렬'이고 '기능어'인지 확인
                if i not in target_aligned_indices:
                    try:
                        token_text, pos = tagged_tokens[i].split('/')
                        if pos in FUNCTION_POS:
                            feature = f"{token_text.lower()}/{pos}" # (e.g., "the/DET")
                            self.feature_context_counts[feature][context] += 1
                    except ValueError:
                        continue # "TOKEN/POS" 형식이 아닌 경우

        print("통계 집계 완료. 확률 계산 중...")

        # --- 3. 최종 P(F|C) 계산 ---
        for feature, contexts in self.feature_context_counts.items():
            for context, f_c_count in contexts.items():
                c_count = self.context_counts.get(context, 0)
                if c_count > 0:
                    probability = f_c_count / c_count
                    self.p_f_c_table[feature][context] = round(probability, 4)

        print("확률 계산 완료")

    def save_results(self):
        """계산된 통계치와 확률 테이블을 JSON 파일로 저장"""
        if not self.p_f_c_table:
            print("저장할 데이터가 없습니다.")
            return

        print(f"\n결과 저장 중...")
        os.makedirs(os.path.dirname(self.count_output_path), exist_ok=True)
        
        try:
            # 1. Count(F, C) 저장 (F_C_counts.json)
            with open(self.count_output_path, 'w', encoding='utf-8') as f_count:
                json.dump(self.feature_context_counts, f_count, indent=2, ensure_ascii=False)
            print(f"빈도 수 저장 완료: {self.count_output_path}")

            # 2. P(F|C) 저장 (P_insert_matrix.json)
            with open(self.prob_output_path, 'w', encoding='utf-8') as f_prob:
                json.dump(self.p_f_c_table, f_prob, indent=2, ensure_ascii=False)
            print(f"확률 매트릭스 저장 완료: {self.prob_output_path}")

        except Exception as e:
            print(f"결과 저장 중 오류 발생: {e}")

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    # 이 스크립트는 analysis/functional_divergence/ 폴더에 위치한다고 가정
    analyzer = FunctionalDivergenceAnalyzer(
        TEXT_POS_PATH, 
        ALIGNMENT_PATH, 
        COUNT_OUTPUT_PATH, 
        PROB_OUTPUT_PATH
    )
    
    if analyzer.load_data():
        analyzer.analyze_divergence()
        analyzer.save_results()
        print("\n--- 기능어 불일치 정량화 작업 완료 ---")