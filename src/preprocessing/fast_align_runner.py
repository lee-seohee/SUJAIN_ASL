import os
import sys
import subprocess
from typing import List

# --- 1. 파일 경로 설정 (스크립트 위치 기준 상대 경로) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
TEMP_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'temp')
CORPUS_FWD_PATH = os.path.join(TEMP_DIR, 'corpus.fwd')
CORPUS_REV_PATH = os.path.join(TEMP_DIR, 'corpus.rev')
ALIGN_FWD_PATH = os.path.join(TEMP_DIR, 'model.fwd.align')
ALIGN_REV_PATH = os.path.join(TEMP_DIR, 'model.rev.align')

class FastWordAligner:
    def __init__(self):
        print("FastWordAligner 초기화 완료")
        self.gloss_sentences = []
        self.text_sentences = []

    def setup_directories(self):
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

    def load_data(self) -> bool:
        print("\n데이터 로딩 시작...")
        try:
            print(f"Gloss 파일 로딩: {GLOSS_TAGGED_PATH}")
            with open(GLOSS_TAGGED_PATH, 'r', encoding='utf-8') as f:
                self.gloss_sentences = [line.strip() for line in f.readlines()]
            
            print(f"Text 파일 로딩: {TEXT_POS_PATH}")
            with open(TEXT_POS_PATH, 'r', encoding='utf-8') as f:
                self.text_sentences = [line.strip() for line in f.readlines()]

            if not self.gloss_sentences or not self.text_sentences:
                print("오류: 원본 데이터 파일이 비어있습니다.", file=sys.stderr)
                return False

            if len(self.gloss_sentences) != len(self.text_sentences):
                print(f"오류: 글로스({len(self.gloss_sentences)})와 텍스트({len(self.text_sentences)}) 줄 수가 다릅니다.", file=sys.stderr)
                return False
            
            print(f"데이터 로딩 완료: 총 {len(self.gloss_sentences)}개 문장 쌍")
            return True
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다. {e.filename}", file=sys.stderr)
            print("--- data/02_processed/ 폴더에 gloss.tok.tagged와 text.tok.pos 파일이 있는지 확인하세요. ---", file=sys.stderr)
            return False

    def prepare_corpus(self):
        print("\nfastalign용 코퍼스 파일 생성 중...")
        def extract_tokens(line):
            tokens = [tok.split('/')[0] for tok in line.split() if tok.split('/')[0]]
            return " ".join(tokens)

        with open(CORPUS_FWD_PATH, 'w', encoding='utf-8') as f_fwd:
            for gloss, text in zip(self.gloss_sentences, self.text_sentences):
                gloss_toks = extract_tokens(gloss)
                text_toks = extract_tokens(text)
                if gloss_toks and text_toks:
                    f_fwd.write(f"{gloss_toks} ||| {text_toks}\n")
                
        with open(CORPUS_REV_PATH, 'w', encoding='utf-8') as f_rev:
            for gloss, text in zip(self.gloss_sentences, self.text_sentences):
                gloss_toks = extract_tokens(gloss)
                text_toks = extract_tokens(text)
                if gloss_toks and text_toks:
                    f_rev.write(f"{text_toks} ||| {gloss_toks}\n")
        print(f"코퍼스 생성 완료: {CORPUS_FWD_PATH}, {CORPUS_REV_PATH}")

    def run_command(self, command: List[str], output_file: str, description: str):
        print(f"\n실행: {description}...")
        print(f"명령어: {' '.join(command)}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                process = subprocess.run(command, stdout=f_out, stderr=subprocess.PIPE, check=True, encoding='utf-8')
            print(f"성공: {description}")
            return True
        except FileNotFoundError:
            print(f"!!! 치명적 오류: '{command[0]}' 실행 파일을 찾을 수 없습니다.", file=sys.stderr)
            print("--- tools/fast_align/build 폴더에 fast_align과 atools가 올바르게 설치되었는지 확인하세요. ---", file=sys.stderr)
            return False
        except subprocess.CalledProcessError as e:
            print(f"!!! 치명적 오류: {description} 실패 (종료 코드: {e.returncode})", file=sys.stderr)
            print("--- fast_align 내부 오류 메시지 ---", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            print("------------------------------------", file=sys.stderr)
            return False

    def run_alignment_pipeline(self):
        cmd_fwd = [FASTALIGN_EXEC, '-i', CORPUS_FWD_PATH, '-d', '-v', '-o']
        if not self.run_command(cmd_fwd, ALIGN_FWD_PATH, "정방향 정렬 (gloss -> text)"):
            return

        cmd_rev = [FASTALIGN_EXEC, '-i', CORPUS_REV_PATH, '-d', '-v', '-o']
        if not self.run_command(cmd_rev, ALIGN_REV_PATH, "역방향 정렬 (text -> gloss)"):
            return

        cmd_sym = [ATOOLS_EXEC, '-i', ALIGN_FWD_PATH, '-j', ALIGN_REV_PATH, '-c', 'grow-diag-final-and']
        if not self.run_command(cmd_sym, ALIGNMENT_MAP_PATH, "정렬 대칭화 (Symmetrization)"):
            return
        
        print(f"\n--- 최종 정렬 맵: {ALIGNMENT_MAP_PATH} ---")

if __name__ == "__main__":
    aligner = FastWordAligner()
    aligner.setup_directories()
    if aligner.load_data():
        aligner.prepare_corpus()
        aligner.run_alignment_pipeline()
        print(f"\n--- fastalign 단어 정렬 작업 완료 ---")