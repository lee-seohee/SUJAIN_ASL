import pandas as pd
import os
import spacy
from typing import List

# --- 1. 파일 경로 설정 (상대 경로) ---
CLEANED_DATA_PATH = r'C:\Users\john9\Desktop\코딩\code\real\train_cleaned.csv'
TEXT_OUTPUT_PATH = r'C:\Users\john9\Desktop\코딩\code\real\text.tok.pos'
GLOSS_OUTPUT_PATH = r'C:\Users\john9\Desktop\코딩\code\real\gloss.tok.tagged'

class Tagger:
    """
    영어 텍스트는 품사 태깅, 
    ASL 글로스는 마커 태깅 및 토큰화
    """
    def __init__(self, cleaned_path: str, text_out_path: str, gloss_out_path: str):
        
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        
        self.cleaned_data_path = os.path.join(script_dir, cleaned_path)
        self.text_output_path = os.path.join(script_dir, text_out_path)
        self.gloss_output_path = os.path.join(script_dir, gloss_out_path)
        
        self.df = None
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("spacy 모델 로드 완료")
        except OSError:
            print("spacy 모델 로드 실패")
            self.nlp = None

    def load_data(self) -> bool:
       
        if not self.nlp:
            return False

        print(f"\n데이터 로딩 시작: {self.cleaned_data_path}")
        try:
            self.df = pd.read_csv(self.cleaned_data_path, encoding='utf-8')
            print(f"데이터 로딩 완료 : 총 {len(self.df)}개 행")
            return True
        except FileNotFoundError:
            print(f"오류: 클리닝된 파일을 찾을 수 없습니다")
            return False
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            return False

    def tag_text(self) -> List[str]:
        """
        텍스트 토큰화 / 품사태깅
        결과 형식: "Token/POS Token/POS ..."
        POS(일관성 높음) 쓸지 TAG(시제, 상 등 좀 더 자세함) 쓸지 고민해봐야됨
        일단 pos로 함
        """
        print("\n영어 품사 태깅 시작...")
        tagged_text_sentences = []
        
        for t in self.df['text']:
            doc = self.nlp(t)
            
            tagged_tokens = [f"{token.text}/{token.pos_}" for token in doc]
            
            tagged_sentence = ' '.join(tagged_tokens)
            tagged_text_sentences.append(tagged_sentence)
            
        print("영어 품사 태깅 완료")
        return tagged_text_sentences

    def tag_gloss(self) -> List[str]:
        """
        글로스를 토큰화 / 마커 태깅
        결과 형식: "GLOSS_TOKEN/TAG GLOSS_TOKEN/TAG ..."
        --> 일단 휴리스틱으로 처리하긴 하는데 성능 안나오면+결과 빈약하면 수동으로 글로스 마커 태깅 소규모 샘플만들어서 마커 태깅을 위한 모델 학습도 해야됨
        """
        print("\n글로스 마커 태깅 시작...")
        tagged_gloss_sentences = []

        def process_token(token):
            token = token.strip()
            if not token:
                return None
            
            # 지시 대명사: IX-a, IX-loc 등의 패턴을 대명사로
            if token.upper().startswith('IX-'):
                return f"{token.upper()}/PRN"
            # 비수지적 요소: 괄호()로 둘러싸인 마커를 NMM 태그
            elif token.startswith('(') and token.endswith(')'):
                 return f"{token.upper()}/NMM"
            # 일반 품사 글로스는 기본 토큰으로 처리
            else:
                # 글로스는 보통 대문자이므로 UPPER()를 사용
                return f"{token.upper()}/GLOSS"

        for text in self.df['gloss']:
            
            tokens = text.split()
            tagged_tokens = []
            
            for token in tokens:
                result = process_token(token)
                if result:
                    tagged_tokens.append(result)
            
            tagged_sentence = ' '.join(tagged_tokens)
            tagged_gloss_sentences.append(tagged_sentence)
            
        print("글로스 마커 태깅 완료")
        return tagged_gloss_sentences

    def save_results(self, data_list: List[str], output_path: str):
      
        print(f"결과 저장: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # with open 구문을 사용해 안전하게 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in data_list:
                f.write(line + '\n')
                
        print("저장 완료")


# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    tagger = Tagger(CLEANED_DATA_PATH, TEXT_OUTPUT_PATH, GLOSS_OUTPUT_PATH)
    
    if tagger.load_data():
        # 1. 영어 태깅 및 결과 리스트 확보
        text_results = tagger.tag_text()
        # 2. 글로스 태깅 및 결과 리스트 확보
        gloss_results = tagger.tag_gloss()
        
        # 3. 영어 결과 저장
        tagger.save_results(text_results, tagger.text_output_path)
        # 4. 글로스 결과 저장
        tagger.save_results(gloss_results, tagger.gloss_output_path)
        
        print("\n태깅 및 저장 작업 완료")