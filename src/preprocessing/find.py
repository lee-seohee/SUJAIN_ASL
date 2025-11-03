import os
import re
from typing import Set

# --- 1. 파일 경로 설정 (청사진 기반) ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 210줄 추가 후, tagger.py가 새로 생성한 바로 그 파일
GLOSS_TAGGED_PATH = os.path.join(PROJECT_DIR, r'C:\Users\john9\Desktop\코딩\code\real\gloss.tok.tagged')

# --- 2. CV팀 77개 단어 목록 (하이픈/언더스코어 포함) ---
CV_VOCAB_LIST = [
    "BAD", "BIG", "BLUE", "BROTHER", "BUT", "CANDY", "CAR", "CLEAN", "CLOTHES", "COLD", 
    "COOK", "COOK_KITCHEN", "DAUGHTER", "DAY", "DOOR", "DRINK", "EAT", "FAMILY", "FATHER", 
    "FEEL", "FINE", "FINISH", "FRIEND", "FULL", "FUTURE", "GIVE", "GO", "GOOD", "GREEN", 
    "HAVE", "HELLO", "HELP", "HOME", "HOT", "HOUSE", "HOW", "KITCHEN", "KNOW", "LIKE", 
    "MAKE", "MAN", "MEET", "MOTHER", "NAME", "NICE", "NOW", "PAST", "PLAY", "RUN", "SCHOOL", 
    "SHORT", "SISTER", "SIT", "SLEEP", "SON", "STUDY", "TABLE", "TALL", "THANK-YOU", 
    "TIME", "TOMORROW", "WALK", "WANT", "WATER", "WEEKEND", "WHAT", "WHERE", "WHO", "WHY", 
    "WITH", "WOMAN", "WORK", "WRONG", "YEAR", "YELLOW", "YES", "YESTERDAY"
]
CV_VOCAB_SET = set(CV_VOCAB_LIST)

class VocabSynchronizerCheck:
    """
    CV팀의 77개 어휘가 전처리 파이프라인을 거친 
    최종 'gloss.tok.tagged' 파일에 100% 포함되었는지 검증합니다.
    """
    def __init__(self, tagged_path: str):
        self.gloss_tagged_path = tagged_path
        self.nlp_vocab_set = set()

    def load_and_extract_vocab(self) -> bool:
        """gloss.tok.tagged 파일을 로드하고 고유 토큰을 추출"""
        print(f"\n데이터 로딩 시작: {self.gloss_tagged_path}")
        try:
            with open(self.gloss_tagged_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens_with_tags = line.strip().split()
                    for token_tag in tokens_with_tags:
                        # "TOKEN/TAG" 형식에서 "TOKEN" 부분만 분리
                        # (wh) 같은 마커는 tagger.py가 이미 처리했어야 함
                        # 혹시 모르니 정규식으로 한번 더 처리 (e.g., NAME(spell))
                        
                        token = token_tag.split('/')[0]
                        
                        # 글로스 마커 (e.g., (wh), (spell)) 제거
                        cleaned_token = re.sub(r"\(.*\)", "", token)
                        
                        if cleaned_token:
                            self.nlp_vocab_set.add(cleaned_token)
            
            print(f"데이터 로딩 및 어휘 추출 완료.")
            print(f"총 {len(self.nlp_vocab_set)}개의 고유 글로스 토큰을 NLP 데이터셋에서 발견했습니다.")
            return True
            
        except FileNotFoundError:
            print(f"!!! 오류: 파일을 찾을 수 없습니다. {self.gloss_tagged_path}")
            print("--- 중요: data_cleaner.py와 tagger.py를 (210줄 추가 후) 재실행했는지 확인하세요. ---")
            return False

    def check_synchronization(self):
        """CV 어휘와 NLP 어휘를 비교하여 동기화 상태를 보고합니다."""
        if not self.nlp_vocab_set:
            print("NLP 어휘가 비어있습니다. 분석을 진행할 수 없습니다.")
            return

        print("\n--- CV-NLP 어휘 동기화 검증 시작 ---")
        
        found_words = CV_VOCAB_SET.intersection(self.nlp_vocab_set)
        missing_words = CV_VOCAB_SET.difference(self.nlp_vocab_set)
        
        num_found = len(found_words)
        num_missing = len(missing_words)
        
        print(f"\n[결과] CV팀 77개 단어 중 {num_found}개 발견, {num_missing}개 누락.")
        
        if num_missing > 0:
            print("\n[!!!] 다음 단어들이 여전히 누락되었습니다:")
            print(", ".join(sorted(list(missing_words))))
            print("--- 중요: 210개 예시 문장이 train.csv에 정확히 추가되었는지, 'DAUGHTTER' 오타 등을 확인하세요. ---")
        else:
            print("\n[성공] CV-NLP 77개 기본 어휘가 100% 완벽하게 동기화되었습니다.")

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    checker = VocabSynchronizerCheck(GLOSS_TAGGED_PATH)
    if checker.load_and_extract_vocab():
        checker.check_synchronization()