import pandas as pd
import numpy as np
from pathlib import Path

'''
matser.csv 파일을 읽어서 그 안에 있는 전체 단어 목록을 절반으로 나눈 뒤
task_1.csv과 task_2.csv 파일을 생성하는 코드
'''

# 원본 master.csv 파일이 있는 경로
METADATA_DIR = Path(r'C:/Users/seohee/MyProjects/ASL-project/metadata')
MASTER_CSV_PATH = METADATA_DIR / 'master.csv'

# 생성될 task 파일들의 이름
TASK_1_CSV_PATH = METADATA_DIR / 'task_1.csv'
TASK_2_CSV_PATH = METADATA_DIR / 'task_2.csv'


def split_master_csv(master_path, task1_path, task2_path):
    """
    master.csv를 읽어 단어(label) 기준으로 두 개의 task.csv 파일로 분할
    """
    print(f"'{master_path}' 파일을 읽어 작업을 시작합니다...")

    # master.csv 파일이 있는지 확인
    if not master_path.exists():
        print(f"오류: '{master_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # CSV 파일 읽기
    df = pd.read_csv(master_path)

    # 전체 고유 단어(label) 리스트 추출 및 정렬
    unique_labels = sorted(df['label'].unique())
    print(f"총 {len(unique_labels)}개의 고유 단어를 발견했습니다.")

    # 단어 리스트를 두 그룹으로 분할
    # np.array_split은 개수가 홀수일 경우 task_1에 한 개 더 많이 할당
    labels_group_1, labels_group_2 = np.array_split(unique_labels, 2)
    
    print(f"\n그룹 1: {len(labels_group_1)}개 단어")
    print(f"그룹 2: {len(labels_group_2)}개 단어")

    # 각 그룹에 해당하는 데이터프레임 생성
    df_task_1 = df[df['label'].isin(labels_group_1)].reset_index(drop=True)
    df_task_2 = df[df['label'].isin(labels_group_2)].reset_index(drop=True)

    # 각 task CSV 파일로 저장
    df_task_1.to_csv(task1_path, index=False, encoding='utf-8-sig')
    df_task_2.to_csv(task2_path, index=False, encoding='utf-8-sig')

    print("\n작업 완료")
    print(f"'{task1_path}' 파일이 생성되었습니다. (총 {len(df_task_1)}개 행)")
    print(f"'{task2_path}' 파일이 생성되었습니다. (총 {len(df_task_2)}개 행)")

if __name__ == '__main__':
    split_master_csv(MASTER_CSV_PATH, TASK_1_CSV_PATH, TASK_2_CSV_PATH)