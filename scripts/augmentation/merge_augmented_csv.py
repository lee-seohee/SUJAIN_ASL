import pandas as pd
from pathlib import Path

'''
augmented_1.csv와 augmented_2.csv 파일을 합치는 코드
'''

# 1. 작업할 파일들이 있는 기본 경로
METADATA_ROOT = Path('G:/내 드라이브/수어데이터_공용/metadata')

# 2. 합칠 CSV 파일 이름 목록
file_to_merge = ['augmented_1.csv', 'augmented_2.csv']

# 3. 최종적으로 저장될 파일 이름
MERGED_FILE_NAME = 'augmented_all.csv'

def merge_csv_files():
    """지정된 경로의 CSV 파일들을 하나로 합침"""
    
    df_list = []
    print("CSV 파일 병합을 시작합니다...")

    for file_name in file_to_merge:
        file_path = METADATA_ROOT / file_name
        if file_path.exists():
            print(f"-> '{file_name}' 파일을 읽는 중...")
            df_list.append(pd.read_csv(file_path))
        else:
            print(f"[경고] '{file_name}' 파일을 찾을 수 없습니다. 건너뜁니다.")

    if not df_list:
        print("[오류] 병합할 CSV 파일이 하나도 없습니다. 작업을 중단합니다.")
        return

    # 데이터프레임 리스트를 하나로 합치기
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # 중복 데이터가 있을 경우 제거
    merged_df.drop_duplicates(inplace=True)

    # 최종 파일 저장
    output_path = METADATA_ROOT / MERGED_FILE_NAME
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\nCSV 파일 병합 완료")
    print(f"총 {len(merged_df)}개의 데이터가 '{output_path}' 파일에 저장되었습니다.")

if __name__ == '__main__':
    merge_csv_files()