import json
import math
from collections import defaultdict
import os

# --- 프로젝트 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 입력 파일 경로 ---
ALIGN_FILE = os.path.join(BASE_DIR, 'data/02_processed/alignment_map.txt')
TAGGED_GLOSS_FILE = os.path.join(BASE_DIR, 'data/02_processed/gloss.tok.tagged')

# --- 출력 파일 경로 (JSON으로 통일) ---
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/reordering_divergence')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'ASL_pattern_prob.json')


def calculate_anchor_point(target_indices: list[int]) -> int | None:
    """
    대표 타겟 위치(Anchor Point) J_i를 계산합니다 (가중치 w_ij=1).
    word_order_div.py와 동일한 함수를 사용해 일관성을 유지합니다.
    """
    if not target_indices:
        return None 
    
    mean_index = sum(target_indices) / len(target_indices)
    return int(math.floor(mean_index + 0.5))


def analyze_reordering_patterns(align_lines: list[str], gloss_lines: list[str]) -> dict:
    """
    전체 말뭉치에서 품사 기반의 Bigram 어순 패턴 빈도를 분석합니다.
    """
    # {('NOUN', 'VERB'): {'Monotone': 800, 'Inversion': 200, 'Merge': 0}}
    pattern_counts = defaultdict(lambda: defaultdict(int))
    
    if len(align_lines) != len(gloss_lines):
        raise ValueError("정렬 파일과 태깅된 ASL 파일의 문장 수가 일치하지 않습니다.")

    print(f"🔄 어순 패턴 통계 분석 중... (총 {len(align_lines)} 문장)")

    for align_line, gloss_line in zip(align_lines, gloss_lines):
        align_line = align_line.strip()
        gloss_tokens = gloss_line.strip().split()
        
        if not align_line or not gloss_tokens:
            continue
            
        # 1. 단어 정렬 쌍 파싱 및 중심점 J_map 생성
        source_to_targets = defaultdict(list)
        for pair in align_line.split():
            try:
                src_idx, tgt_idx = map(int, pair.split('-'))
                source_to_targets[src_idx].append(tgt_idx)
            except ValueError:
                continue

        J_map = {i: calculate_anchor_point(targets) 
                 for i, targets in source_to_targets.items()}
        
        # 2. Bigram 순회 및 패턴 카운트
        for i in range(len(gloss_tokens) - 1):
            
            # i번째와 i+1번째 단어의 중심점이 모두 정의되어야 유효한 비교 쌍입니다.
            J_i = J_map.get(i)
            J_i_next = J_map.get(i + 1)
            
            if J_i is None or J_i_next is None:
                continue

            # 3. 품사 태그 추출 (예: "STUDY/VERB" -> "VERB")
            try:
                tag_i = gloss_tokens[i].split('/')[-1]
                tag_i_next = gloss_tokens[i + 1].split('/')[-1]
            except IndexError:
                # 태그 형식이 잘못된 경우 무시
                continue
                
            tag_pattern = f"{tag_i}-{tag_i_next}"
            
            # 4. 패턴 분류
            if J_i_next < J_i:
                pattern_type = 'Inversion'  # 역전 (뒤집힘)
            elif J_i_next > J_i:
                pattern_type = 'Monotone'   # 단조 (순서 유지)
            else:
                pattern_type = 'Merge'      # 병합 (동일 위치)

            # 5. 카운트 누적
            pattern_counts[tag_pattern][pattern_type] += 1
            pattern_counts[tag_pattern]['TotalCount'] += 1

    # 6. 확률 계산 및 최종 JSON 구조 생성
    final_probabilities = {}
    for pattern, counts in pattern_counts.items():
        total = counts['TotalCount']
        if total > 0:
            final_probabilities[pattern] = {
                'TotalCount': total,
                'Monotone': counts['Monotone'] / total,
                'Inversion': counts['Inversion'] / total,
                'Merge': counts['Merge'] / total,
            }

    return final_probabilities


def main():
    """
    메인 실행 함수: 파일 로드, 패턴 분석, JSON 저장
    """
    print("--- ASL 어순 패턴 확률 분석기 ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 입력 파일 로드
    try:
        with open(ALIGN_FILE, 'r', encoding='utf-8') as f:
            align_lines = f.readlines()
        with open(TAGGED_GLOSS_FILE, 'r', encoding='utf-8') as f:
            gloss_lines = f.readlines()
    except FileNotFoundError as e:
        print(f"🚨 오류: 필수 입력 파일이 없습니다. ({e.filename})")
        return

    # 2. 분석 실행
    try:
        probabilities = analyze_reordering_patterns(align_lines, gloss_lines)
    except ValueError as e:
        print(f"🚨 오류: {e}")
        return

    # 3. JSON 파일로 저장
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(probabilities, f, indent=4)
        print(f"\n✅ 완료! 어순 패턴 확률이 {OUTPUT_FILE}에 저장되었습니다.")
    except IOError:
        print(f"🚨 오류: {OUTPUT_FILE}에 파일을 쓸 수 없습니다. 권한을 확인하세요.")

if __name__ == "__main__":
    main()