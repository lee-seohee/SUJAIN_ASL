# gloss.py (이 파일은 수정할 필요 거의 없음)

import pandas as pd
from nltk.tokenize import word_tokenize
import nltk  # nltk.pos_tag를 위해 이 줄을 추가하세요.
import string

class GlossPatternAnalyzer:
    """
    ASL 글로스 문장을 분석하여 핵심 구문 요소 패턴으로 변환하는 클래스.
    (NLTK 리소스가 필요합니다.)
    """
    
    def __init__(self):
        """
        분석기에 필요한 리소스와 키워드를 초기화합니다.
        """
        self._download_nltk_resources()
        
        # 시간 관련 단어들
        self.TIME_WORDS = {'yesterday', 'tomorrow', 'now', 'today', 'morning', 'afternoon', 'week', 'month', 'year', 'day', 'night'}
        
        # 대명사 및 지시(IX) 단어들 (하이픈 제거)
        self.PRONOUNS = {
            'me', 'you', 'he', 'she', 'it', 'we', 'they', 'i',
            'ix1', 'ix2', 'ix3', 'ixme', 'ixyou', 'ixhe', 'ixshe', 'ixit',
            'ixwe', 'ixthey', 'ixloc', 'ixobj', 'me1'
        }
        
        # 인사/감정 등 (하이픈 제거)
        self.GREETINGS = {'hello', 'sorry', 'thankyou', 'please', 'welcome', 'goodbye'}
        
        # 질문
        self.QUESTIONS = {'what', 'where', 'when', 'who', 'why', 'how', 'which'}

        # 동사 (하이픈/아포스트로피 제거 및 '오답 노트' 단어 추가)
        self.VERB_WORDS = {
            'go', 'come', 'eat', 'drink', 'sleep', 'walk', 'run', 'see', 'look',
            'want', 'like', 'love', 'hate', 'need', 'have', 'give', 'take',
            'make', 'do', 'work', 'study', 'play', 'read', 'write', 'say', 'tell',
            'wait', 'help', 'show', 'pay', 'put', 'bring', 'feel', 'send', 
            'register', 'ask', 'forget', 'checkin', 'rest', 'takemedicine', 
            'finish', 'clean',
            'dontwant', 'hurt', 'sit', 'sign' # <-- '오답 노트 2페이지' 단어 추가
        }
        
        # 형용사
        self.ADJ_WORDS = {
            'sick', 'dizzy', 'late', 'positive', 'good', 'bad', 'happy', 'sad'
        }
        # --- (⬆️ 1. 여기까지 추가) ---

    def _download_nltk_resources(self):
        """NLTK 필수 리소스를 다운로드합니다."""
        # ... (이 함수는 수정할 필요 없음) ...
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("NLTK 'punkt' 리소스 다운로드 중...")
            nltk.download('punkt')
        
        RESOURCE_NAME = 'averaged_perceptron_tagger_eng'
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
        except LookupError:
            print(f"NLTK '{RESOURCE_NAME}' 리소스 (패키지: averaged_perceptron_tagger) 다운로드 중...")
            nltk.download('averaged_perceptron_tagger')

    @staticmethod
    def _get_pos_tag(word):
        """NLTK를 사용해 영어 단어의 대략적인 품사를 반환합니다."""
        # ... (이 함수는 수정할 필요 없음) ...
        if not word:
            return 'UNK'
        tag = nltk.pos_tag([word])[0][1]
        
        if tag.startswith('VB'):
            return 'VERB'
        if tag.startswith('NN'):
            return 'NOUN'
        if tag.startswith('JJ'):
            return 'ADJ'
        if tag in ['PRP', 'PRP$']:
            return 'PRON'
        return 'OTHER'

    def tag_gloss_elements(self, gloss_sentence):
        """글로스 문장을 받아서 핵심 요소 패턴으로 변환합니다."""
        if not isinstance(gloss_sentence, str):
            return []
        sentence = gloss_sentence.lower()
        # 2. 문장부호(쉼표 등) 제거
        # string.punctuation은 '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' 입니다.
        translator = str.maketrans('', '', string.punctuation)
        cleaned_sentence = sentence.translate(translator)
        
        words = cleaned_sentence.split()
        pattern = []
        
        for word in words:
            # --- (⬇️ 2. 이 로직 순서 수정!) ---
            # 1. 확장된 키워드 목록을 먼저 확인
            if word in self.TIME_WORDS:
                pattern.append('TIME')
            elif word in self.PRONOUNS:
                pattern.append('PRON')
            elif word in self.GREETINGS:
                pattern.append('GREETING')
            elif word in self.QUESTIONS:
                pattern.append('QUESTION')
            
            # (NEW!) VERB 키워드를 NLTK보다 먼저 확인
            elif word in self.VERB_WORDS:
                pattern.append('VERB')
            elif word in self.ADJ_WORDS:
                pattern.append('ADJ')
            
            else:
                # 2. 키워드에 없으면 NLTK 품사 태깅 활용
                pos = self._get_pos_tag(word)
                if pos == 'NOUN':
                    pattern.append('NOUN/TOPIC') 
                # NLTK가 동사로 판단한 것도 VERB로 인정
                elif pos == 'VERB':
                    pattern.append('VERB')
                elif pos == 'ADJ':
                    pattern.append('ADJ')
                else:
                    pattern.append(pos)
                    
        return pattern

    def analyze_dataframe(self, df, gloss_column_name):
        """
        DataFrame에 'tagged_pattern' 열을 추가하여 반환합니다.
        """
        print(f"'{gloss_column_name}' 열을 기준으로 패턴 분석을 시작합니다...")
        df['tagged_pattern'] = df[gloss_column_name].apply(self.tag_gloss_elements)
        print("패턴 분석 완료.")
        return df