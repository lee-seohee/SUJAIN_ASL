#어순 불일치
import os
import json
from typing import List, Tuple

# --- 1. 파일 경로 설정 (청사진 기반) ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ALIGNMENT_PATH = os.path.join(PROJECT_DIR, r'C:\Users\john9\Desktop\코딩\code\real\alignment.txt')

OUTPUT_DIR = os.path.join(PROJECT_DIR, r'C:\Users\john9\Desktop\코딩\code\real\reordering_divergence')
METRICS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'D_order_metrics.json')

class ReorderingDivergenceAnalyzer:
    """
    단어 정렬 맵을 기반으로 '정렬 교차(Alignment Crossing)' 수를 계산하여
    어순 불일치(Reordering)를 정량화합니다.
    """
    def __init__(self, align_path: str, metrics_path: str):
        self.align_path = align_path
        self.metrics_output_path = metrics_path
        
        self.alignments = []
        self.reordering_scores = []
        print("ReorderingDivergenceAnalyzer 초기화 완료")

    def load_data(self) -> bool:
        """alignment_map.txt 파일을 로드"""
        print("\n데이터 로딩 시작...")
        try:
            with open(self.align_path, 'r', encoding='utf-8') as f:
                self.alignments = [line.strip() for line in f.readlines()]
            
            print(f"데이터 로딩 완료: 총 {len(self.alignments)}개 문장 쌍")
            return True
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다. {e.filename}")
            print("aligner.py를 먼저 실행해야 합니다.")
            return False

    def analyze_divergence(self):
        """
        문장별 '정렬 교차' 수를 계산합니다.
        교차 수: (s1, t1)과 (s2, t2)에 대해 s1 < s2 이지만 t1 > t2 인 경우
        """
        print("\n어순 불일치 분석 시작 (정렬 교차 계산)...")
        
        total_sentences = len(self.alignments)
        for i, align_line in enumerate(self.alignments):
            
            if (i + 1) % 10000 == 0:
                print(f"  ... {i+1} / {total_sentences} 문장 처리 중")
                
            align_pairs: List[Tuple[int, int]] = []
            try:
                for pair in align_line.split():
                    s_idx, t_idx = map(int, pair.split('-'))
                    align_pairs.append((s_idx, t_idx))
            except ValueError:
                continue # 빈 줄 스킵

            if len(align_pairs) < 2:
                self.reordering_scores.append(0.0) # 정렬이 1개 이하면 교차 0
                continue

            # 1. 소스(gloss) 인덱스 기준으로 정렬
            align_pairs.sort(key=lambda x: x[0])

            # 2. 교차(crossing) 수 계산
            crossings = 0
            num_pairs = len(align_pairs)
            
            for j in range(num_pairs):
                for k in range(j + 1, num_pairs):
                    # s1 < s2 (정렬했으므로 보장됨)
                    # t1 > t2 인지 확인
                    if align_pairs[j][1] > align_pairs[k][1]:
                        crossings += 1
            
            # 3. 정규화 (0.0 ~ 1.0)
            # 최대 교차 수 = nC2 = n * (n-1) / 2
            max_possible_crossings = (num_pairs * (num_pairs - 1)) / 2
            
            if max_possible_crossings > 0:
                score = crossings / max_possible_crossings
                self.reordering_scores.append(score)
            else:
                self.reordering_scores.append(0.0)
                
        print("어순 불일치 분석 완료")

    def save_results(self):
        """전체 데이터셋의 평균 어순 불일치 점수를 JSON으로 저장"""
        if not self.reordering_scores:
            print("저장할 분석 결과가 없습니다.")
            return

        print(f"\n결과 저장 중...")
        os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
        
        # 전체 점수 평균 계산
        average_score = sum(self.reordering_scores) / len(self.reordering_scores)
        
        metrics = {
            "total_sentences": len(self.reordering_scores),
            "average_reordering_score (0.0=Perfectly Monotonic, 1.0=Fully Reversed)": round(average_score, 6),
            "note": "Kendall's Tau (normalized crossing count) based reordering score."
        }
        
        try:
            with open(self.metrics_output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"어순 불일치 지표 저장 완료: {self.metrics_output_path}")
        except Exception as e:
            print(f"결과 저장 중 오류 발생: {e}")

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    # 이 스크립트는 analysis/reordering_divergence/ 폴더에 위치한다고 가정
    analyzer = ReorderingDivergenceAnalyzer(
        ALIGNMENT_PATH, 
        METRICS_OUTPUT_PATH
    )
    
    if analyzer.load_data():
        analyzer.analyze_divergence()
        analyzer.save_results()
        print("\n--- 어순 불일치 정량화 작업 완료 ---")