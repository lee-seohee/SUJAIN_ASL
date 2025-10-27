import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

'''
원본 데이터 정보가 담긴 CSV 파일(master.csv)로부터 데이터 증강을 하는 코드
(증강된 영상과 CSV가 저장될 새로운 폴더가 생성됨)
'''

# TASK 번호
TASK_FILE_NUMBER = 1

# 원본 데이터 정보가 담긴 CSV 파일 경로
SOURCE_CSV_PATH = f'G:/내 드라이브/수어데이터_공용/metadata/task_{TASK_FILE_NUMBER}.csv'

# 원본 영상들이 있는 루트 폴더
DATASET_ROOT_DIR = 'G:/내 드라이브/수어데이터_공용/raw_videos/dataset_raw'

# 증강된 영상과 CSV가 저장될 새로운 폴더
AUGMENTED_ROOT_DIR = 'G:/내 드라이브/수어데이터_공용/processed_data/augmented_videos'

# 공간적 증강 함수
def augment_brightness_contrast(frame):
    """밝기 및 대비 조절"""
    brightness = int(random.uniform(-25, 25))
    contrast = random.uniform(0.85, 1.15)
    frame = np.clip((frame * contrast) + brightness, 0, 255).astype(np.uint8)
    return frame

def augment_horizontal_flip(frame):
    """좌우 반전"""
    return cv2.flip(frame, 1)

def augment_slight_rotation(frame, angle_range=(-7, 7)):
    """미세한 회전"""
    height, width = frame.shape[:2]
    angle = random.uniform(angle_range[0], angle_range[1])
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(frame, M, (width, height))

# 시간적 증강 함수 (영상 전체를 처리)
def augment_temporal_frame_sampling(input_path, output_path, drop_ratio=0.1):
    """영상의 일부 프레임을 무작위로 누락시켜 저장 (속도 변화 대응)"""
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 설정된 확률에 따라 프레임을 건너뜀
        if random.random() > drop_ratio:
            writer.write(frame)
    
    cap.release()
    writer.release()


# 증강 기법 리스트 (프레임 단위)
frame_augmentation_functions = [
    augment_brightness_contrast,
    augment_horizontal_flip,
    augment_slight_rotation,
]

def main():
    df = pd.read_csv(SOURCE_CSV_PATH)
    all_data = []

    print(f"'{SOURCE_CSV_PATH}' 파일을 기반으로 데이터 증강을 시작합니다...")
    
    Path(AUGMENTED_ROOT_DIR).mkdir(parents=True, exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        original_relative_path = Path(row['filepath'])
        label = row['label']
        original_video_path = Path(DATASET_ROOT_DIR) / original_relative_path
        
        augmented_label_dir = Path(AUGMENTED_ROOT_DIR) / label
        augmented_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 원본 영상 복사 및 정보 추가
        new_original_video_path = augmented_label_dir / original_relative_path.name
        if not new_original_video_path.exists():
            shutil.copy2(original_video_path, new_original_video_path)
        all_data.append([new_original_video_path.relative_to(AUGMENTED_ROOT_DIR).as_posix(), label])

        # 2. 프레임 단위 증강 (체계적 적용)
        for aug_func in frame_augmentation_functions:
            cap = cv2.VideoCapture(str(original_video_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            aug_name = aug_func.__name__.replace('augment_', '')
            augmented_filename = f"{original_relative_path.stem}_aug_{aug_name}.mp4"
            augmented_video_path = augmented_label_dir / augmented_filename

            if not augmented_video_path.exists():
                writer = cv2.VideoWriter(str(augmented_video_path), fourcc, fps, (width, height))
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    augmented_frame = aug_func(frame)
                    writer.write(augmented_frame)
                writer.release()
            
            cap.release()
            all_data.append([augmented_video_path.relative_to(AUGMENTED_ROOT_DIR).as_posix(), label])
        
        # 3. 시간적 증강 (프레임 샘플링)
        temp_aug_filename = f"{original_relative_path.stem}_aug_temporal.mp4"
        temp_aug_video_path = augmented_label_dir / temp_aug_filename
        if not temp_aug_video_path.exists():
            augment_temporal_frame_sampling(original_video_path, temp_aug_video_path)
        all_data.append([temp_aug_video_path.relative_to(AUGMENTED_ROOT_DIR).as_posix(), label])

    # 최종 결과 CSV 생성 (개인용 파일)
    augmented_df = pd.DataFrame(all_data, columns=['filepath', 'label']).drop_duplicates().reset_index(drop=True)
    # 최종 결과 CSV 저장 위치
    output_csv_path = rf'C:\Users\seohee\MyProjects\ASL-project\metadata\augmented_{TASK_FILE_NUMBER}.csv'
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n데이터 증강 완료")
    print(f"'{AUGMENTED_ROOT_DIR}' 폴더에 증강된 영상들이 저장되었습니다.")
    print(f"'{output_csv_path}' 파일에 개인 작업 결과가 저장되었습니다.")
    print(f"총 영상 개수: {len(df)} (원본) -> {len(augmented_df)} (증강 후)")

if __name__ == '__main__':
    main()