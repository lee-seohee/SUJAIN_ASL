# module.py (이 파일을 실행하세요)

import pandas as pd  # <-- 1. 'pd'를 사용하기 때문에 import가 필요합니다!
from gloss import GlossPatternAnalyzer  # <-- 2. 우리가 gloss.py에서 만든 클래스를 가져옵니다.

# --- 1. (수정 완료) ASLG-PC12 데이터 로드 ---
# r"..."을 사용하여 경로 오류를 수정했습니다.
DATA_FILE_PATH = r"C:\Users\john9\Desktop\코딩\code\gloss_dataset_20k.csv"

try:
    df = pd.read_csv(DATA_FILE_PATH)
    
    print(f"'{DATA_FILE_PATH}' 파일 로드 성공.")
    print("데이터 샘플:")
    print(df.head())
    print("\n데이터 열(Columns):", df.columns.tolist())

except FileNotFoundError:
    print(f"오류: '{DATA_FILE_PATH}' 파일을 찾을 수 없습니다.")
    df = pd.DataFrame() # 오류 시 빈 DataFrame 생성


# --- 2. (⚠️ 사용자 수정 필요) 글로스 열 이름 확인 ---
# 위에서 출력된 '데이터 열' 목록을 보고, 글로스가 포함된 열의 실제 이름을 입력하세요.
GLOSS_COLUMN = 'gloss' # ⬅️ 2. 실제 글로스 열 이름으로 수정 (예: 'Gloss_Column' 등)


# --- 3. (수정 불필요) 클래스를 사용해 분석 실행 ---
if not df.empty:
    if GLOSS_COLUMN in df.columns:
        # 3a. 분석기 클래스 생성
        analyzer = GlossPatternAnalyzer()
        
        # 3b. DataFrame에 분석 적용
        df_analyzed = analyzer.analyze_dataframe(df, gloss_column_name=GLOSS_COLUMN)
        
        # 3c. 결과 확인
        print("\n--- 분석 결과 (상위 30개) ---")
        print(df_analyzed[[GLOSS_COLUMN, 'tagged_pattern']].head(100))
    else:
        print(f"오류: DataFrame에 '{GLOSS_COLUMN}' 열이 없습니다.")
        print(f"'{GLOSS_COLUMN}' 변수를 실제 열 이름으로 수정해주세요.")
else:
    print("데이터프레임이 비어있어 분석을 건너뜁니다.")


# --- 4. (NEW!) 위치 빈도 분석 (전체 데이터 대상) ---
if not df_analyzed.empty:
    print("\n--- 4. 위치 빈도 분석 시작 (전체 데이터) ---")
    
    # 4a. 각 패턴의 '첫 번째 요소'만 추출
    df_analyzed['first_element'] = df_analyzed['tagged_pattern'].str[0]

    # 4b. 첫 번째 요소의 빈도수 계산
    first_element_counts = df_analyzed['first_element'].value_counts()
    
    # 4c. 첫 번째 요소의 확률(비율) 계산
    first_element_probs = df_analyzed['first_element'].value_counts(normalize=True)
    
    print("\n--- [결과] 문장 첫 요소 빈도수 (상위 10개) ---")
    print(first_element_counts.head(10))
    
    print("\n--- [결과] 문장 첫 요소 확률 (상위 10개) ---")
    print(first_element_probs.head(10))

    # (옵션) 'TIME' 요소의 확률만 따로 확인
    if 'TIME' in first_element_probs:
        time_prob = first_element_probs['TIME'] * 100
        print(f"\n* 'TIME' 요소가 문장 맨 앞에 올 확률: {time_prob:.2f}%")
    else:
        print("\n* 'TIME' 요소는 문장 맨 앞에서 발견되지 않았습니다.")
        
else:
    print("분석할 데이터가 없어 위치 빈도 분석을 건너뜁니다.")
#상위 n개 어순 패턴 분석 (전체 데이터 대상)
if 'df_analyzed' in locals() and not df_analyzed.empty:
    print("\n--- 5. 상위 N개 어순 패턴 분석 시작 ---")
    
    # 5a. 패턴 리스트를 하나의 문자열로 변환
    # (예: ['GREETING', 'PRON'] -> "GREETING-PRON")
    # .str.join()은 리스트의 각 요소를 '-'로 연결해줍니다.
    df_analyzed['pattern_str'] = df_analyzed['tagged_pattern'].str.join('-')

    # 5b. 패턴 문자열의 빈도수 계산
    pattern_counts = df_analyzed['pattern_str'].value_counts()
    
    # 5c. 패턴 문자열의 확률(비율) 계산
    pattern_probs = df_analyzed['pattern_str'].value_counts(normalize=True)
    
    print("\n--- [결과] 상위 20개 어순 패턴 (빈도수) ---")
    print(pattern_counts.head(20))
    
    print("\n--- [결과] 상위 20개 어순 패턴 (확률) ---")
    print(pattern_probs.head(20))

else:
    print("분석할 데이터가 없어 '상위 어순 패턴 분석'을 건너뜁니다.")


# --- 6. (NEW!) 'NOUN/TOPIC' 태그 상세 분석 ---
import string # <-- 6단계에서도 쉼표 제거를 위해 import

if 'df_analyzed' in locals() and not df_analyzed.empty:
    print("\n--- 6. 'NOUN/TOPIC' 태그 상세 분석 시작 ---")
    
    all_nouns = []
    
    # 6a. 쉼표 등을 제거하기 위한 '번역기' 준비
    translator = str.maketrans('', '', string.punctuation)
    
    # 1. 태거가 'NOUN/TOPIC'이라고 판단한 모든 '실제 단어'들을 수집
    for idx, row in df_analyzed.iterrows():
        # 원본 글로스 문장 가져오기
        original_gloss = row[GLOSS_COLUMN]
        if not isinstance(original_gloss, str):
            continue
            
        # 6b. 'tag_gloss_elements' 함수에서 했던 것과 "똑같은" 클리닝 작업 수행
        sentence = original_gloss.lower()
        cleaned_sentence = sentence.translate(translator)
        gloss_words = cleaned_sentence.split() # 쉼표가 제거된 단어 목록
        
        pattern = row['tagged_pattern'] # 이미 분석된 패턴
        
        for i in range(len(gloss_words)):
            # 태그가 'NOUN/TOPIC'이고,
            if i < len(pattern) and pattern[i] == 'NOUN/TOPIC':
                word = gloss_words[i]
                # 해당 단어가 모든 키워드 목록에 없는지 확인 (NLTK가 추측한 것)
                if (word not in analyzer.TIME_WORDS and
                    word not in analyzer.PRONOUNS and
                    word not in analyzer.GREETINGS and
                    word not in analyzer.QUESTIONS and
                    word not in analyzer.VERB_WORDS and
                    word not in analyzer.ADJ_WORDS): # ADJ 목록도 확인
                    all_nouns.append(word)

    # 2. 가장 많이 등장한 (아마도 잘못 분류된) 'NOUN/TOPIC' 단어들의 빈도수 확인
    noun_counts = pd.Series(all_nouns).value_counts()
    
    print("\n--- [결과] 'NOUN/TOPIC'으로 태깅된 상위 50개 단어 (개선 필요 목록) ---")
    print(noun_counts.head(50))

else:
    print("분석할 데이터가 없어 'NOUN/TOPIC 상세 분석'을 건너뜁니다.")