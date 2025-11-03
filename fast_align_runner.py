import os
import sys
import subprocess
from typing import List

# --- 1. 파일 경로 설정 (스크립트 기준 상대 경로) ---
# 이 스크립트가 SUJAIN_ASL 폴더 최상단에 있다고 가정합니다.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로
DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # data 폴더를 사용한다고 가정
GLOSS_TAGGED_PATH = os.path.join(DATA_DIR, 'gloss.tok.tagged')
TEXT_POS_PATH = os.path.join(DATA_DIR, 'text.tok.pos')

# 최종 결과물 경로
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'analysis') # analysis 폴더를 사용한다고 가정
ALIGN_FINAL_PATH = os.path.join(ANALYSIS_DIR, 'alignment.txt')

# --- 2. fastalign 실행 파일 경로 (상대 경로) ---
TOOLS_DIR = os.path.join(PROJECT_ROOT, 'tools')
FASTALIGN_EXEC = os.path.join(TOOLS_DIR, 'fast_align/build/fast_align')
ATOOLS_EXEC = os.path.join(TOOLS_DIR, 'fast_align/build/atools')

# --- 3. fastalign 중간 산출물 경로 ---
TEMP_DIR = os.path.join(ANALYSIS_DIR, 'temp') # 중간 파일들은 별도 폴더에 저장
CORPUS_FWD_PATH = os.path.join(TEMP_DIR, 'corpus.fwd')
CORPUS_REV_PATH = os.path.join(TEMP_DIR, 'corpus.rev')
ALIGN_FWD_PATH = os.path.join(TEMP_DIR, 'model.fwd.align')
ALIGN_REV_PATH = os.path.join(TEMP_DIR, 'model.rev.align')

class FastWordAligner:
    def __init__(self):
        print("FastWordAligner 초기화 완료")

    def setup_directories(self):
        # 필요한 폴더들이 없으면 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

    def load_data(self) -> bool:
        # ... (load_data 내용은 이전과 동일) ...
        return True

    def prepare_corpus(self):
        # ... (prepare_corpus 내용은 이전과 동일) ...
        pass

    def run_command(self, command: List[str], output_file: str, description: str):
        # ... (run_command 내용은 이전과 동일) ...
        return True

    def run_alignment_pipeline(self):
        # ... (run_alignment_pipeline 내용은 이전과 동일) ...
        pass

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    aligner = FastWordAligner()
    aligner.setup_directories() # 폴더 생성 단계 추가
    
    # [사용자 확인 필요] 원본 데이터 파일들을 data/ 폴더로 옮겨주세요.
    print("--- 중요: gloss.tok.tagged 와 text.tok.pos 파일을 data/ 폴더로 옮겼는지 확인해주세요. ---")
    
    if aligner.load_data():
        aligner.prepare_corpus()
        aligner.run_alignment_pipeline()
        
        print(f"\n--- fastalign 단어 정렬 작업 완료 ---")
