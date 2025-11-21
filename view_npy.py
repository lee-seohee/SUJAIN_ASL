import numpy as np

path = r"C:\Users\seohee\Downloads\ASLLVD_YES__ASL_2008_03_28_scene47-2300-2330_camera1_aug_brightness_contrast.npy"
data = np.load(path)

# 배열 전체 구조
print("Shape:", data.shape)
print("Dtype:", data.dtype)

# 1번째 프레임 내용 일부
print("\n첫 번째 프레임 데이터 (앞부분 20개만):")
print(data[0][:20])

# 여러 프레임의 평균값, 최소/최대값
print("\n전체 데이터 통계:")
print("min:", np.min(data))
print("max:", np.max(data))
print("mean:", np.mean(data))