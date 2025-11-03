import os
from simalign import SentenceAligner # SimAlign 라이브러리 직접 임포트
from typing import List

# --- 1. 파일 경로 설정 (사용자 지정 절대 경로) ---
GLOSS_TAGGED_PATH = r'C:\Users\john9\Desktop\코딩\code\real\gloss.tok.tagged'
TEXT_POS_PATH = r'C:\Users\john9\Desktop\코딩\code\real\text.tok.pos'
ALIGN_FINAL_PATH = r'C:\Users\john9\Desktop\코딩\code\real\alignment.txt' # 최종 산출물

class NeuralWordAligner:
    
    def __init__(self, gloss_path: str, text_path: str, output_path: str):
        self.gloss_path = gloss_path
        self.text_path = text_path
        self.output_path = output_path
        
        self.gloss_sentences = []
        self.text_sentences = []
        
        # SimAlign 모델 로드
        try:
            self.aligner = SentenceAligner(model="bert", token_type="word", device="cpu")
            print("SimAlign (mBERT) 모델 로드 완료")
        except Exception as e:
            print(f"SimAlign 모델 로드 실패: {e}")
            self.aligner = None

    def load_data(self) -> bool:
        """'최신' 파일들을 로드합니다."""
        try:
            print(f"Gloss 파일 로딩: {self.gloss_path}")
            with open(self.gloss_path, 'r', encoding='utf-8') as f:
                self.gloss_sentences = [line.strip() for line in f.readlines()]
            
            print(f"Text 파일 로딩: {self.text_path}")
            with open(self.text_path, 'r', encoding='utf-8') as f:
                self.text_sentences = [line.strip() for line in f.readlines()]
                
            if not self.gloss_sentences or not self.text_sentences:
                print("!!! 오류: 파일이 비어있습니다.")
                return False

            if len(self.gloss_sentences) != len(self.text_sentences):
                print(f"오류: 글로스({len(self.gloss_sentences)})와 텍스트({len(self.text_sentences)}) 줄 수가 다릅니다.")
                return False

            print(f"데이터 로딩 완료: {len(self.gloss_sentences)}개 문장")
            return True
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다. {e.filename}")
            return False

    def run_alignment(self):
        """[최종 수정] 빈 토큰/빈 리스트를 완벽히 필터링하고 배치(Batch)로 실행"""
        if not self.aligner:
            print("Aligner 모델이 로드되지 않았습니다. 실행을 중단합니다.")
            return

        print("\n신경망 단어 정렬 시작 (빈 문장/토큰 필터링 및 전체 배치 처리)...")
        
        processed_gloss_tokens = []
        processed_text_tokens = []
        original_indices = [] # 유효한 문장의 원래 줄 번호

        total_lines = len(self.gloss_sentences)
        empty_lines_count = 0
        
        for i in range(total_lines):
            gloss_line = self.gloss_sentences[i]
            text_line = self.text_sentences[i]
            
            gloss_tokens = [tok.split('/')[0] for tok in gloss_line.split() if tok.split('/')[0]]
            text_tokens = [tok.split('/')[0] for tok in text_line.split() if tok.split('/')[0]]
            
            if gloss_tokens and text_tokens:
                processed_gloss_tokens.append(gloss_tokens)
                processed_text_tokens.append(text_tokens)
                original_indices.append(i)
            else:
                empty_lines_count += 1

        print(f"총 {total_lines}개 문장 중 {len(processed_gloss_tokens)}개의 유효한 문장 쌍을 찾았습니다.")
        print(f"{empty_lines_count}개의 빈 문장/토큰 쌍은 건너뜁니다.")

        if not processed_gloss_tokens:
             print("오류: 처리할 유효한 문장이 하나도 없습니다.")
             return

        try:
            print("SimAlign 모델이 전체 데이터 정렬을 시작합니다... (몇 분 정도 소요될 수 있습니다)")
            alignments = self.aligner.get_word_aligns(processed_gloss_tokens, processed_text_tokens)
            
            align_results = alignments.get('itermax') 
            
            if not align_results:
                 print(f"오류: 정렬 결과가 없습니다 (None).")
                 return

            print(f"{len(align_results)}개 문장 정렬 완료. 파일 저장 중...")

            final_output_lines = [''] * total_lines
            
            # [최종 오류 수정] 짧은 align_results 리스트(81개)를 기준으로 반복
            for result_idx in range(len(align_results)):
                # 이 결과(result_idx)가 원래 몇 번째 줄이었는지 확인
                original_line_idx = original_indices[result_idx]
                align_pairs = align_results[result_idx]
                
                if isinstance(align_pairs, (list, tuple)):
                    valid_pairs = []
                    for pair in align_pairs:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            valid_pairs.append(f"{pair[0]}-{pair[1]}")
                    line = ' '.join(valid_pairs)
                    final_output_lines[original_line_idx] = line

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for line in final_output_lines:
                    f.write(line + '\n')
            
            print(f"\n최종 정렬 맵 저장 완료: {self.output_path}")
            print(f"({total_lines}개 라인으로 저장됨 - 빈 줄 포함)")

        except Exception as e:
            print(f"!!! 치명적 오류: SimAlign 정렬 실행 중 예외 발생: {e}")

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
     neural_aligner = NeuralWordAligner(
         gloss_path=GLOSS_TAGGED_PATH,
         text_path=TEXT_POS_PATH,
         output_path=ALIGN_FINAL_PATH
     )
     if neural_aligner.load_data():
         neural_aligner.run_alignment()