import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path

'''
증강된 모든 영상 데이터를 바탕으로 MediaPipe Holistic(Pose, Hands, Face) 랜드마크를 추출하고
.npy 파일로 저장한 뒤 최종 학습용 CSV 파일을 생성하는 코드
'''

# 1. 소스 CSV 파일 경로
SOURCE_CSV_PATH = 'G:/내 드라이브/수어데이터_공용/metadata/augmented_all.csv'

# 2. 증강된 영상들이 있는 루트 폴더
DATASET_ROOT_DIR = 'G:/내 드라이브/수어데이터_공용/processed_data/augmented_videos'

# 3. 추출된 랜드마크(.npy)를 저장할 새로운 폴더
LANDMARKS_OUTPUT_DIR = 'G:/내 드라이브/수어데이터_공용/processed_data/landmark_features'

# 4. 랜드마크 경로가 추가된 최종 CSV 파일
FINAL_CSV_PATH = 'G:/내 드라이브/수어데이터_공용/metadata/dataset_final.csv'

mp_holistic = mp.solutions.holistic

def extract_landmarks(video_path):
    """하나의 영상 파일에서 Pose, Hands, Face 랜드마크를 모두 추출함"""
    landmarks_list = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            frame_landmarks = []
            
            # Pose (33개*4 = 132)
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                frame_landmarks.extend([p.x for p in pose] + [p.y for p in pose] + [p.z for p in pose] + [p.visibility for p in pose])
            else:
                frame_landmarks.extend([0] * 132) # 33*4

            # Face (468개*3 = 1404) -> 얼굴 랜드마크 추가
            if results.face_landmarks:
                face = [lm for lm in results.face_landmarks.landmark]
                frame_landmarks.extend([f.x for f in face] + [f.y for f in face] + [f.z for f in face])
            else:
                frame_landmarks.extend([0] * 1404)

            # Left Hand (21개*3 = 63)
            if results.left_hand_landmarks:
                left_hand = [lm for lm in results.left_hand_landmarks.landmark]
                frame_landmarks.extend([lh.x for lh in left_hand] + [lh.y for lh in left_hand] + [lh.z for lh in left_hand])
            else:
                frame_landmarks.extend([0] * 63)

            # Right Hand (21개*3 = 63)
            if results.right_hand_landmarks:
                right_hand = [lm for lm in results.right_hand_landmarks.landmark]
                frame_landmarks.extend([rh.x for rh in right_hand] + [rh.y for rh in right_hand] + [rh.z for rh in right_hand])
            else:
                frame_landmarks.extend([0] * 63)
            
            landmarks_list.append(frame_landmarks)
        
        cap.release()
    return np.array(landmarks_list)

def main():
    """CSV를 읽어 모든 영상의 랜드마크를 추출하고 최종 CSV를 생성함"""
    # 병합을 하지 않았다면 주석 해제:
    # df1 = pd.read_csv('G:/내 드라이브/수어데이터_공용/metadata/augmented_1.csv')
    # df2 = pd.read_csv('G:/내 드라이브/수어데이터_공용/metadata/augmented_2.csv')
    # df_all = pd.concat([df1, df2], ignore_index=True)
    # df_all.to_csv(SOURCE_CSV_PATH, index=False, encoding='utf-8-sig')
    
    df = pd.read_csv(SOURCE_CSV_PATH)
    landmarks_paths = []

    print(f"총 {len(df)}개의 영상에 대해 랜드마크 추출을 시작합니다...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        relative_path = row['filepath']
        video_path = Path(DATASET_ROOT_DIR) / relative_path
        
        # .npy 저장 경로 생성
        npy_relative_path = Path(relative_path).with_suffix('.npy')
        npy_full_path = Path(LANDMARKS_OUTPUT_DIR) / npy_relative_path
        
        # 이미 추출된 파일이 있으면 건너뛰기
        if npy_full_path.exists():
            landmarks_paths.append(npy_relative_path.as_posix())
            continue
        
        landmarks = extract_landmarks(video_path)
        
        if landmarks is None or landmarks.size == 0:
            print(f"경고: {video_path} 처리 중 오류 또는 랜드마크 추출 실패. 건너뜁니다.")
            landmarks_paths.append(None)
            continue

        npy_full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_full_path, landmarks)
        landmarks_paths.append(npy_relative_path.as_posix())

    df['landmarks_path'] = landmarks_paths
    df.dropna(inplace=True)
    df.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8-sig')

    print("\n모든 랜드마크 추출 작업 완료")
    print(f"'{LANDMARKS_OUTPUT_DIR}' 폴더에 모든 랜드마크(.npy) 파일이 저장되었습니다.")
    print(f"'{FINAL_CSV_PATH}' 파일에 최종 데이터셋이 저장되었습니다.")
    print("최종 데이터셋 샘플:")
    print(df.head())


if __name__ == '__main__':
    main()