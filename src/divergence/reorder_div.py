import json
import math
from collections import defaultdict
import os

# --- 프로젝트 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 입력 파일 경로 (수정됨) ---
# .csv 대신 .tok.pos 파일을 사용합니다.
DATA_FILE = os.path.join(BASE_DIR, 'data/02_processed/text.tok.pos') 
ALIGN_FILE = os.path.join(BASE_DIR, 'data/02_processed/alignment_map.txt')

# --- 출력 파일 경로 ---
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/reordering_divergence')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'D_order_metrics.json')


def calculate_anchor_point(target_indices: list[int]) -> int | None:
  
    if not target_indices:
        return None 
    
    sum_of_indices = sum(target_indices)
    count_of_indices = len(target_indices)
    
    mean_index = sum_of_indices / count_of_indices
    
    return int(math.floor(mean_index + 0.5))


def calculate_reordering_metrics(alignment_line: str, target_length: int) -> dict:

    
    # 1. 정렬 쌍 파싱 및 중심점 J_map 생성
    source_to_targets = defaultdict(list)
    
    if alignment_line: # 빈 줄이 아닐 경우에만 파싱
        for pair in alignment_line.split():
            try:
                src_idx, tgt_idx = map(int, pair.split('-'))
                source_to_targets[src_idx].append(tgt_idx)
            except ValueError:
                pass # "NULL" 등 무시

    J_map = {i: calculate_anchor_point(targets) 
             for i, targets in source_to_targets.items()}
    
    # 2. LDIR (Local Dependency Inversion Rate) 계산
    aligned_indices = sorted([i for i, J in J_map.items() if J is not None])
    
    inversions = 0
    total_comparisons = len(aligned_indices) - 1
    
    LDIR = 0.0
    if total_comparisons > 0:
        for k in range(total_comparisons):
            i = aligned_indices[k]
            i_next = aligned_indices[k+1]
            
            if J_map[i_next] < J_map[i]:
                inversions += 1
        
        LDIR = inversions / total_comparisons

    # 3. SNR-D (Sentence Normalized Reordering Distance) 계산
    total_distance = 0
    aligned_count = 0
    
    for i, J in J_map.items():
        if J is not None:
            total_distance += abs(J - i)
            aligned_count += 1
    
    SNR_D = 0.0
    if aligned_count > 0 and target_length > 0:
        mean_distance = total_distance / aligned_count
        SNR_D = mean_distance / target_length
        
    return {'LDIR': LDIR, 'SNR-D': SNR_D}


def main():
    """
    메인 실행 함수: 파일 로드, 지수 계산, JSON 저장
    """
    print("--- 어순 불일치 지수(Reordering Divergence) 계산기 ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📖 로딩 중 (Tokenized Target): {DATA_FILE}")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
    except FileNotFoundError:
        print(f"🚨 오류: {DATA_FILE}을 찾을 수 없습니다. (src/preprocessing/tagger.py를 먼저 실행해야 합니다.)")
        return

    print(f"📖 로딩 중 (Alignment Map): {ALIGN_FILE}")
    try:
        with open(ALIGN_FILE, 'r', encoding='utf-8') as f:
            align_lines = f.readlines()
    except FileNotFoundError:
        print(f"🚨 오류: {ALIGN_FILE}을 찾을 수 없습니다. (fast_align_runner.py 등을 먼저 실행해야 합니다.)")
        return

    if len(target_lines) != len(align_lines):
        print(f"🚨 경고: 타겟 문장 수 ({len(target_lines)})와 정렬 맵 라인 수 ({len(align_lines)})가 다릅니다. 중단합니다.")
        return

    print(f"🔄 어순 불일치 지수(LDIR, SNR-D) 계산 중... (총 {len(target_lines)} 문장)")
    
    all_metrics = {}
    
    for index, (target_line, alignment_line) in enumerate(zip(target_lines, align_lines)):
        # 문장 ID는 0부터 시작하는 인덱스 사용
        sentence_id = index
        
        # 1. 타겟 문장 길이(L_target) 계산
        # .tok.pos 파일의 공백으로 구분된 토큰 수를 셉니다.
        target_length = len(target_line.strip().split())
        
        # 2. 정렬 맵 라인
        alignment_line = alignment_line.strip()
        
        # 3. 지수 계산
        metrics = calculate_reordering_metrics(alignment_line, target_length)
        all_metrics[sentence_id] = metrics

    # 4. JSON 파일로 저장
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"\n✅ 완료! 정량화된 지수가 {OUTPUT_FILE}에 저장되었습니다.")
    except IOError:
        print(f"🚨 오류: {OUTPUT_FILE}에 파일을 쓸 수 없습니다. 권한을 확인하세요.")

if __name__ == "__main__":
    main()