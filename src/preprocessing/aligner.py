import os
from simalign import SentenceAligner
from typing import List

# --- 1. 파일 경로 설정 (사용자 지정 절대 경로) ---
GLOSS_TAGGED_PATH = r'C:\Users\john9\Desktop\코딩\code\real\gloss.tok.tagged'
TEXT_POS_PATH = r'C:\Users\john9\Desktop\코딩\code\real\text.tok.pos'
ALIGN_FINAL_PATH = r'C:\Users\john9\Desktop\코딩\code\real\alignment.txt'

class NeuralWordAligner:
    
    def __init__(self, gloss_path: str, text_path: str, output_path: str):
        # [수정] 절대 경로를 그대로 사용 (os.path.join 제거)
        self.gloss_path = gloss_path
        self.text_path = text_path
        self.output_path = output_path
        
        self.gloss_sentences = []
        self.text_sentences = []
        
        # [핵심] token_type을 'word'로 설정
        try:
            self.aligner = SentenceAligner(model="bert", token_type="word", device="cpu")
            print("SimAlign (mBERT) 모델 로드 완료 (token_type='word')")
        except Exception as e:
            print(f"SimAlign 모델 로드 실패: {e}")
            self.aligner = None

    def load_data(self) -> bool:
        """'최신' (내용이 꽉 찬) 파일들을 로드합니다."""
        try:
            print(f"Gloss 파일 로딩: {self.gloss_path}")
            with open(self.gloss_path, 'r', encoding='utf-8') as f:
                self.gloss_sentences = [line.strip() for line in f.readlines()]
            
            print(f"Text 파일 로딩: {self.text_path}")
            with open(self.text_path, 'r', encoding='utf-8') as f:
                self.text_sentences = [line.strip() for line in f.readlines()]

            print(f"데이터 로딩 완료: {len(self.gloss_sentences)}개 문장")
            return True
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다. {e.filename}")
            return False

    def run_alignment(self):
        """[최종 수정] '배치' 처리를 포기하고, '한 문장씩' (N+1) 안전하게 처리"""
        if not self.aligner:
            print("Aligner 모델이 로드되지 않았습니다. 실행을 중단합니다.")
            return

        print("\n신경망 단어 정렬 시작 (안전 모드: 한 문장씩 처리 중)...")
        print("!!! 이 작업은 81,922 문장에 대해 개별 실행되므로 매우 오래(수 시간) 걸릴 수 있습니다. !!!")
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        skipped_long_sentences = 0
        failed_sentences = 0
        # BERT의 (BPE) 토큰 제한 512에 대한 안전 버퍼
        MAX_TOKENS_BERT = 500 

        with open(self.output_path, 'w', encoding='utf-8') as f:
            total_sentences = len(self.gloss_sentences)
            for i in range(total_sentences):
                # 5000개마다 진행 상황 출력
                if (i + 1) % 5000 == 0:
                    print(f"  ... {i + 1} / {total_sentences} 문장 처리 중 (실패: {failed_sentences}, 길이초과: {skipped_long_sentences})")

                gloss_line = self.gloss_sentences[i]
                text_line = self.text_sentences[i]

                # "TOKEN/TAG" -> "TOKEN", 빈 문자열("")은 제외
                gloss_tokens = [tok.split('/')[0] for tok in gloss_line.split() if tok.split('/')[0]]
                text_tokens = [tok.split('/')[0] for tok in text_line.split() if tok.split('/')[0]]

                # [안전장치 1] 빈 리스트는 건너뛰기
                if not gloss_tokens or not text_tokens:
                    f.write('\n')
                    continue
                
                # [안전장치 2] BERT의 512 토큰 제한 초과 방지
                # SimAlign은 내부적으로 BPE 토큰화를 다시 수행하므로,
                # 원본 단어 리스트가 아닌 BPE 토큰 길이를 검사해야 합니다.
                try:
                    gloss_bpe_len = len(self.aligner.embed_loader.tokenizer.tokenize(" ".join(gloss_tokens)))
                    text_bpe_len = len(self.aligner.embed_loader.tokenizer.tokenize(" ".join(text_tokens)))
                    
                    if gloss_bpe_len > MAX_TOKENS_BERT or text_bpe_len > MAX_TOKENS_BERT:
                        skipped_long_sentences += 1
                        f.write('\n') # 너무 긴 문장은 빈 줄로 처리
                        continue
                except Exception:
                     # 토큰화 자체에서 오류가 나도 (매우 드묾) 건너뛰기
                    f.write('\n')
                    continue

                # [핵심] 
                # token_type='word'에 맞게 List[List[str]] 형태로 전달
                # [gloss_tokens] => [['GO', 'HOSPITAL']]
                try:
                    alignments = self.aligner.get_word_aligns([gloss_tokens], [text_tokens])
                    align_pairs = alignments.get('itermax', [[]])[0]

                    if isinstance(align_pairs, (list, tuple)):
                        valid_pairs = []
                        for pair in align_pairs:
                            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                valid_pairs.append(f"{pair[0]}-{pair[1]}")
                        line = ' '.join(valid_pairs)
                        f.write(line + '\n')
                    else:
                        failed_sentences += 1
                        f.write('\n') # 정렬 실패 시 빈 줄
                except Exception:
                    # (예: "unhashable type" 또는 기타 SimAlign 내부 오류)
                    failed_sentences += 1
                    f.write('\n') # 오류 발생 시 빈 줄

        print(f"\n최종 정렬 맵 저장 완료: {self.output_path}")
        print(f"({total_sentences}개 라인으로 저장됨 - 빈 줄 포함)")
        print(f"*** 총 {skipped_long_sentences}개 문장이 [최대 토큰 길이 초과]로 건너뛰어졌습니다.")
        print(f"*** 총 {failed_sentences}개 문장이 [정렬 실패]로 건너뛰어졌습니다.")

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    analyzer = NeuralWordAligner(
        gloss_path=GLOSS_TAGGED_PATH,
        text_path=TEXT_POS_PATH,
        output_path=ALIGN_FINAL_PATH
    )
    if analyzer.load_data():
        analyzer.run_alignment()