sujain_asl/
├── data/
│   ├── 01_raw/
│   │   ├── train.csv       # 데이터 원본은 여기에만 존재
│   │   ├── README_data.md      # 데이터 출처, 라이센스, 다운로드 방법 명시
│   └── 02_processed/
│       ├── english.tok.pos     # 품사 태깅 완료된 영어 텍스트
│       ├── gloss.tok.tagged    # ASL 마커 태깅 완료된 글로스
│       └── alignment_map.txt   # 단어 정렬 결과 맵 (분석용 중간 파일)
├── analysis/
│   ├── functional_divergence/
│   │   ├── P_insert_matrix.csv # 기능어 삽입 확률 최종 결과
│   │   └── F_C_counts.json     # 문맥별 기능어 빈도 카운트
│   └── reordering_divergence/
│       ├── ASL_pattern_prob.csv# ASL 어순 패턴 빈도 및 확률
│       └── D_order_metrics.json# 어순 불일치 지수 측정 결과
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaner.py     # CSV 로딩, 기본 클리닝
│   │   ├── tagger.py           # POS/ASL 마커 태깅, 토큰화
│   │   └── aligner.py          # 단어 정렬 스크립트 실행
│   ├── augmentation/
│   │   ├── functional_augmentor.py  # P_insert 기반 증강 로직
│   │   └── order_augmentor.py       # 어순 패턴 기반 증강 로직
│   └── model/
│       ├── feature_injector.py # 명시적 피처(POS/Topic/Time) 주입 클래스
│       └── training_pipeline.py# 모델 학습 및 실험 설계
├── experiments/
│   ├── config_A.yaml           # 베이스라인 설정 파일
│   └── results.csv             # 최종 실험 결과 비교
└── README.md                   # 프로젝트 개요, 환경 설정, 실행 방법



pip install pandas spacy
python -m spacy download en_core_web_sm