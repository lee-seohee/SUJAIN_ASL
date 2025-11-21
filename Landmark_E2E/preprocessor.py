import numpy as np

class WinstonPreprocessor:
    """
    Winston(2022) 논문 기반의 수어 데이터 전처리 클래스.
    기능:
    1. Slicing: 543개 랜드마크 -> 55개 핵심 랜드마크 추출
    2. Normalization: Body(Shoulder-based) + Hands(Local Min-Max)
    """
    
    def __init__(self):
        # 1. Slicing을 위한 인덱스 정의
        # MediaPipe Pose(0~32) 중 상체 핵심 13개 포인트
        # 0:Nose, 2:LEye, 5:REye, 7:LEar, 8:REar, 11:LShoulder, 12:RShoulder, 
        # 13:LElbow, 14:RElbow, 15:LWrist, 16:RWrist, 23:LHip, 24:RHip
        self.BODY_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
        
        # MediaPipe 전체 데이터 구조 가정: Pose(33) -> LHand(21) -> RHand(21) -> Face(468)
        # Left Hand (33 ~ 53)
        self.LH_START = 33
        self.LH_END = 33 + 21
        
        # Right Hand (54 ~ 74)
        self.RH_START = 54
        self.RH_END = 54 + 21
        
        # 정규화 시 0으로 나누기 방지용 엡실론
        self.EPSILON = 1e-6

    def process(self, raw_data: np.ndarray) -> np.ndarray:
        """
        메인 처리 함수
        :param raw_data: (Frames, 543, 2) 형태의 원본 데이터
        :return: (Frames, 55, 2) 형태의 정규화된 데이터
        """
        # 0. 결측치(NaN) 처리 - 안전장치
        data = np.nan_to_num(raw_data, nan=0.0)
        
        # 1. 데이터 분리 (Slicing)
        pose_data = data[:, :33, :]      # (Frames, 33, 2)
        lh_data = data[:, self.LH_START:self.LH_END, :] # (Frames, 21, 2)
        rh_data = data[:, self.RH_START:self.RH_END, :] # (Frames, 21, 2)
        
        # Body Slicing (33 -> 13)
        body_selected = pose_data[:, self.BODY_INDICES, :] # (Frames, 13, 2)

        # 2. 부위별 정규화 (Customized Normalization)
        norm_body = self._normalize_body(body_selected, pose_data) # 기준점 계산을 위해 원본 Pose 데이터 전달
        norm_lh = self._normalize_hand(lh_data)
        norm_rh = self._normalize_hand(rh_data)
        
        # 3. 결합 (Concatenation)
        # 순서: Body(13) -> LHand(21) -> RHand(21) = 총 55개
        result = np.concatenate([norm_body, norm_lh, norm_rh], axis=1)
        
        return result

    def _normalize_body(self, body_data, full_pose_data):
        """
        Body Normalization: 어깨 중심 기준, 어깨 너비로 스케일링
        """
        # 기준점 계산을 위해 전체 Pose에서 어깨 인덱스(11, 12) 사용
        l_shoulder = full_pose_data[:, 11, :] # (Frames, 2)
        r_shoulder = full_pose_data[:, 12, :] # (Frames, 2)
        
        # Center (중점)
        center = (l_shoulder + r_shoulder) / 2.0 # (Frames, 2)
        
        # Scale (어깨 너비)
        # np.linalg.norm으로 유클리드 거리 계산 (axis=1: x,y 차원 기준)
        scale = np.linalg.norm(l_shoulder - r_shoulder, axis=1) # (Frames,)
        
        # Broadcasting을 위해 차원 맞춤
        center = center[:, np.newaxis, :] # (Frames, 1, 2)
        scale = scale[:, np.newaxis, np.newaxis] # (Frames, 1, 1)
        
        # 정규화 수행
        normalized = (body_data - center) / (scale + self.EPSILON)
        
        return normalized

    def _normalize_hand(self, hand_data):
        """
        Hand Normalization: Local Min-Max Scaling -> (-0.5 ~ 0.5)
        """
        # 각 프레임별로 Min, Max 계산
        # keepdims=True로 차원 유지 (Frames, 1, 2)
        min_val = np.min(hand_data, axis=1, keepdims=True)
        max_val = np.max(hand_data, axis=1, keepdims=True)
        
        # Min-Max Scaling (0 ~ 1 범위)
        numerator = hand_data - min_val
        denominator = max_val - min_val + self.EPSILON
        normalized = numerator / denominator
        
        # Center to (-0.5 ~ 0.5)
        normalized = normalized - 0.5
        
        return normalized

# 실행 테스트 코드
if __name__ == "__main__":
    # 1. 더미 데이터 생성 (100 프레임, 543 랜드마크, 2 좌표)
    dummy_input = np.random.rand(100, 543, 2)
    
    # 2. 전처리기 초기화 및 실행
    preprocessor = WinstonPreprocessor()
    output = preprocessor.process(dummy_input)
    
    print(f"입력 데이터 크기: {dummy_input.shape}") # (100, 543, 2)
    print(f"출력 데이터 크기: {output.shape}")      # (100, 55, 2)
    
    # 검증: 손 데이터가 -0.5 ~ 0.5 사이에 있는지 확인
    # Body(13) + LHand(21) -> LHand는 인덱스 13부터 33까지
    lh_output = output[:, 13:34, :]
    print(f"왼손 데이터 Min: {lh_output.min():.3f}, Max: {lh_output.max():.3f}")
    
    print("전처리 프로세스 검증 완료")