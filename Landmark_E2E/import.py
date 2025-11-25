# [Cell 0] 라이브러리 임포트 및 환경 설정
import os
import sys
import random
import time
import math
from typing import List, Tuple, Dict
from collections import Counter

# 데이터 처리를 위한 라이브러리
import numpy as np
import pandas as pd

# 파이토치(Deep Learning) 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 구글 드라이브 마운트 (데이터 로드용)
from google.colab import drive
drive.mount('/content/drive')

# (선택) 실험 결과를 똑같이 재현하기 위한 시드 고정
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything() # 시드 고정 실행
print(f"라이브러리 준비 완료 (사용 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")