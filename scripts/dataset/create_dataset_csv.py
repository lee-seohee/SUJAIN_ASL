import os
import pandas as pd
from pathlib import Path

'''
저장된 데이터셋을 VideoID/Label 형식으로 CSV 파일로 정리하는 코드
(원본 데이터 정보가 담긴 CSV 파일이 생성됨)
'''

# 데이터셋의 루트 폴더 경로
# 예: 'C:/my_project/dataset'
# DATASET_ROOT_DIR = '경로/입력/dataset'
DATASET_ROOT_DIR = r'G:/내 드라이브/수어데이터_공용/raw_videos/dataset_raw'

# 최종 생성될 CSV 파일의 이름과 경로
CSV_OUTPUT_PATH = r'C:\Users\seohee\MyProjects\ASL-project\metadata\master.csv'

def create_dataset_csv(root_dir, output_path):
    """
    지정된 폴더 구조를 탐색하여 filepath와 label이 담긴 CSV 파일을 생성함
    """
    # 파일 경로와 라벨을 저장할 리스트 생성
    file_data = []

    print(f"'{root_dir}' 폴더를 스캔하여 데이터 목록을 생성합니다...")

    # 루트 폴더 안의 모든 하위 폴더(라벨 폴더)를 순회
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)

        # 해당 경로가 폴더가 아니면 건너뜀
        if not os.path.isdir(label_dir):
            continue

        # 라벨 폴더 안의 모든 영상 파일을 순회
        for video_filename in sorted(os.listdir(label_dir)):
            # 영상 파일의 상대 경로 생성 (예: 'you/25721.mp4')
            relative_path = (Path(label) / video_filename).as_posix()

            # 리스트에 [상대경로, 라벨] 형태로 추가
            file_data.append([relative_path, label])
            
    if not file_data:
        print("경고: 데이터셋 폴더에서 영상 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # pandas DataFrame으로 변환
    df = pd.DataFrame(file_data, columns=['filepath', 'label'])
    
    # CSV 파일로 저장 (index=False는 불필요한 번호 열을 만들지 않음)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n작업 완료")
    print(f"총 {len(df)}개의 영상 파일 목록이 '{output_path}' 파일에 성공적으로 저장되었습니다.")
    print("CSV 파일 샘플:")
    print(df.head()) # 상위 5개 행을 샘플로 보여줌

if __name__ == '__main__':
    create_dataset_csv(DATASET_ROOT_DIR, CSV_OUTPUT_PATH)