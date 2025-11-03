import os
import sys
import subprocess
from typing import List

# --- 1. 파일 경로 설정 (스크립트 위치 기준 상대 경로) ---
# 이 스크립트의 위치: SUJAIN_ASL/src/preprocessing/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 경로: SUJAIN_ASL/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 데이터 경로
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, '02_processed')
GLOSS_TAGGED_PATH = os.path.join(PROCESSED_DATA_DIR, 'gloss.tok.tagged')
TEXT_POS_PATH = os.path.join(PROCESSED_DATA_DIR, 'text.tok.pos')

# 최종 결과물 경로
ALIGNMENT_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, 'alignment_map.txt')

# --- 2. fastalign 실행 파일 경로 (상대 경로) ---
TOOLS_DIR = os.path.join(PROJECT_ROOT, 'tools')
FASTALIGN_EXEC = os.path.join(TOOLS_DIR, 'fast_align/build/fast_align')
ATOOLS_EXEC = os.path.join(TOOLS_DIR, 'fast_align/build/atools')

# --- 3. fastalign 중간 산출물 경로 ---
TEMP_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'temp') # 중간 파일들은 별도 폴더에 저장
CORPUS_FWD_PATH = os.path.join(TEMP_DIR, 'corpus.fwd')
CORPUS_REV_PATH = os.path.join(TEMP_DIR, 'corpus.rev')
ALIGN_FWD_PATH = os.path.join(TEMP_DIR, 'model.fwd.align')
ALIGN_REV_PATH = os.path.join(TEMP_DIR, 'model.rev.align')

class FastWordAligner:
    def __init__(self):
        print("FastWordAligner 초기화 완료")

    def setup_directories(self):
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

    def load_data(self) -> bool:
        # ... (이전과 동일)
        return True

    def prepare_corpus(self):
        # ... (이전과 동일)
        pass

    def run_command(self, command: List[str], output_file: str, description: str):
        # ... (이전과 동일)
        return True

    def run_alignment_pipeline(self):
        # ... (이전과 동일, ALIGN_FINAL_PATH -> ALIGNMENT_MAP_PATH 로 변수명 변경)
        pass

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    aligner = FastWordAligner()
    aligner.setup_directories()
    if aligner.load_data():
        aligner.prepare_corpus()
        aligner.run_alignment_pipeline()
        print(f"\n--- fastalign 단어 정렬 작업 완료 ---")
