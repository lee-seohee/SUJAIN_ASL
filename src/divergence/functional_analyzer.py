import os
import json
import sys
from collections import defaultdict

# --- 경로 설정 (스크립트 위치 기반 상대 경로) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 입력 파일 경로
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed')
TEXT_POS_PATH = os.path.join(PROCESSED_DATA_DIR, 'text.tok.pos')
ALIGNMENT_PATH = os.path.join(PROCESSED_DATA_DIR, 'alignment_map.txt')

# 출력 파일 경로
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'functional_divergence')
COUNT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'F_C_counts.json')
PROB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'P_insert_matrix.json')

# --- 분석할 기능어 품사(POS) 정의 ---
FUNCTION_POS = {
    'ADP', 'DET', 'AUX', 'PRON', 'PART', 'CCONJ', 'SCONJ'
}

class FunctionalDivergenceAnalyzer:
    # ... (클래스의 나머지 코드는 이전과 동일) ...
    def __init__(self, text_path: str, align_path: str, count_path: str, prob_path: str):
        self.text_pos_path = text_path
        self.align_path = align_path
        self.count_output_path = count_path
        self.prob_output_path = prob_path
        self.context_counts = defaultdict(int)
        self.feature_context_counts = defaultdict(lambda: defaultdict(int))
        self.tagged_sentences = []
        self.alignments = []
        self.p_f_c_table = defaultdict(lambda: defaultdict(float))
        print("FunctionalDivergenceAnalyzer 초기화 완료")

    def load_data(self) -> bool:
        print("\n데이터 로딩 시작...")
        try:
            with open(self.text_pos_path, 'r', encoding='utf-8') as f:
                self.tagged_sentences = [line.strip() for line in f.readlines()]
            with open(self.align_path, 'r', encoding='utf-8') as f:
                self.alignments = [line.strip() for line in f.readlines()]
            if len(self.tagged_sentences) != len(self.alignments):
                print(f"오류: 텍스트({len(self.tagged_sentences)})와 정렬({len(self.alignments)}) 파일의 줄 수가 다릅니다.", file=sys.stderr)
                return False
            print(f"데이터 로딩 완료: 총 {len(self.tagged_sentences)}개 문장 쌍")
            return True
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다. {e.filename}", file=sys.stderr)
            return False

    def analyze_divergence(self):
        print("\n기능어 불일치 분석 시작...")
        for text_line, align_line in zip(self.tagged_sentences, self.alignments):
            tagged_tokens = text_line.split()
            if len(tagged_tokens) < 3:
                continue
            target_aligned_indices = set()
            try:
                for pair in align_line.split():
                    s_idx, t_idx = map(int, pair.split('-'))
                    target_aligned_indices.add(t_idx)
            except ValueError:
                continue
            for i in range(1, len(tagged_tokens) - 1):
                try:
                    prev_pos = tagged_tokens[i-1].split('/')[-1]
                    next_pos = tagged_tokens[i+1].split('/')[-1]
                except IndexError:
                    continue
                context = f"{prev_pos}_{next_pos}"
                self.context_counts[context] += 1
                if i not in target_aligned_indices:
                    try:
                        token_text, pos = tagged_tokens[i].split('/')
                        if pos in FUNCTION_POS:
                            feature = f"{token_text.lower()}/{pos}"
                            self.feature_context_counts[feature][context] += 1
                    except ValueError:
                        continue
        print("통계 집계 완료. 확률 계산 중...")
        for feature, contexts in self.feature_context_counts.items():
            for context, f_c_count in contexts.items():
                c_count = self.context_counts.get(context, 0)
                if c_count > 0:
                    probability = f_c_count / c_count
                    self.p_f_c_table[feature][context] = round(probability, 4)
        print("확률 계산 완료")

    def save_results(self):
        if not self.p_f_c_table:
            print("저장할 데이터가 없습니다.")
            return
        print(f"\n결과 저장 중...")
        os.makedirs(os.path.dirname(self.count_output_path), exist_ok=True)
        try:
            with open(self.count_output_path, 'w', encoding='utf-8') as f_count:
                json.dump(self.feature_context_counts, f_count, indent=2, ensure_ascii=False)
            print(f"빈도 수 저장 완료: {self.count_output_path}")
            with open(self.prob_output_path, 'w', encoding='utf-8') as f_prob:
                json.dump(self.p_f_c_table, f_prob, indent=2, ensure_ascii=False)
            print(f"확률 매트릭스 저장 완료: {self.prob_output_path}")
        except Exception as e:
            print(f"결과 저장 중 오류 발생: {e}")

# --- 메인 실행 블록 ---
if __name__ == "__main__":
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
