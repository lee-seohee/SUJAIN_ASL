import pandas as pd
import os

RAW_DATA_PATH = '../../data/01_raw/train.csv'
CLEANED_DATA_PATH = '../../data/02_processed/train_cleaned.csv'

class DataCleaner:
    
    def __init__(self, raw_path, cleaned_path):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.raw_data_path = os.path.join(script_dir, raw_path)
        self.cleaned_data_path = os.path.join(script_dir, cleaned_path)

        self.df = None

    def load_data(self):

        print(f"데이터 로딩 시작: {self.raw_data_path}")

        try:
            self.df = pd.read_csv(self.raw_data_path, encoding = 'utf-8')
            print("데이터 로딩 완료")
            print(f"로드된 초기 행 수: {len(self.df)}")
            return True
        except FileNotFoundError:
            print(f"error: 파일을 찾을 수 없음. 경로를 확인하세요: {self.raw_data_path}")
            return False
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            return False
        
    def initial_clean(self):

        if self.df is None:
            print("데이터를 로드 해야 합니다")
            return False
        
        print("\n--- 1. 누락값(NaN) 처리 ---")
        # NaN이 있는 행 제거
        self.df.dropna(subset=['gloss', 'text'], inplace=True)
        print(f"남은 행: {len(self.df)}개")

        print("\n--- 2. 중복 행 제거 ---")
        self.df.drop_duplicates(subset=['gloss', 'text'], inplace=True)
        print(f"남은 행: {len(self.df)}개")

        print("\n--- 3. 텍스트 필드 정리 및 빈 문자열 제거 ---")
        
        # 3-1. 텍스트 앞뒤 공백 제거
        self.df['gloss'] = self.df['gloss'].astype(str).str.strip()
        self.df['text'] = self.df['text'].astype(str).str.strip()
        
        # 3-2. 길이가 0인 빈 문자열 제거

        self.df = self.df[(self.df['gloss'].str.len() > 0) & (self.df['text'].str.len() > 0)]
        print(f"최종 행: {len(self.df)}개")

        print("\n초기 클리닝 완료")
        return True

    def save_cleaned_data(self):
       
        if self.df is None or self.df.empty:
            print("저장할 데이터가 없거나 비어있습니다")
            return
            
        # 저장 경로의 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(self.cleaned_data_path), exist_ok=True)
        # index=False 옵션으로 불필요한 인덱스 컬럼 저장 방지
        self.df.to_csv(self.cleaned_data_path, index=False, encoding='utf-8')
        print(f"\n클리닝된 데이터 저장 완료: {self.cleaned_data_path}")
        print("--- 최종 데이터 샘플 (5개) ---")
        print(self.df.head())


# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    # DataCleaner 객체 생성
    cleaner = DataCleaner(RAW_DATA_PATH, CLEANED_DATA_PATH)
    
    # 메서드들을 순서대로 호출
    if cleaner.load_data():
        if cleaner.initial_clean():
            cleaner.save_cleaned_data()