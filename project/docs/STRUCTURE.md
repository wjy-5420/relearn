# 전체 프로젝트 구조

Ubuntu 실행 기준입니다. 최상위는 구현·검증·프로젝트 자료·로컬 작업 영역으로 구분합니다.
아래에서 로컬 영역의 하위 폴더는 해당 기능을 사용할 때 생성됩니다.

```text
relearn/
├── README.md
├── pyproject.toml
├── .gitignore
├── .gitattributes
│
├── src/relearn/
│   ├── __init__.py
│   ├── cli.py                      # models / doctor / validate-csv / smoke
│   ├── data/
│   │   ├── __init__.py
│   │   ├── manifest.py             # 팀 CSV 읽기·헤더 매핑·제품/시점 검사
│   │   ├── dataset.py              # 이미지·마스크를 공통 샘플로 로딩
│   │   ├── labels.py               # 검증 클래스 사전·ROI·BBox 처리
│   │   └── transforms.py           # 448 전처리·Train 증강·좌표 역변환
│   └── adapters/
│       ├── __init__.py
│       ├── common.py
│       ├── registry.py
│       ├── hvq_trans.py
│       ├── dinomaly_small.py
│       ├── u_cat.py                # 공식 구현 확인 전 임시 입력 계약
│       ├── rtdetrv2_s.py
│       ├── yolo11s.py
│       └── ecf_yolov7_tiny.py
│
├── tests/
│   └── test_csv_pipeline.py         # CSV·전처리·6종 배치 검사
│
├── project/
│   ├── configs/
│   │   ├── data.example.json       # 팀 CSV 연결 설정 예시
│   │   ├── class_map.example.json  # 검증 사전 형식, 실제 ID 임의 생성 없음
│   │   ├── experiment.json         # 공통 실험 기준
│   │   ├── models.json             # 6종 출처·커밋·설정·후보 상태
│   │   └── data.local.json         # 개인 PC 경로 설정, Git 제외
│   ├── environments/
│   │   ├── README.md
│   │   ├── common.txt
│   │   ├── hvq_trans.txt
│   │   ├── dinomaly_small.txt
│   │   ├── u_cat.txt
│   │   ├── rtdetrv2_s.txt
│   │   ├── yolo11s.txt
│   │   ├── ecf_yolov7_tiny.txt
│   │   └── records/
│   │       ├── ubuntu-cpu/        # Ubuntu 설치 버전 기록 6개
│   │       └── windows-cpu/       # 과거 Windows 기록
│   ├── scripts/
│   │   ├── setup_ubuntu.sh         # Ubuntu 설치 진입점
│   │   ├── setup_envs.py           # 독립 환경 생성·패키지 설치
│   │   ├── fetch_upstreams.py     # 공식 소스 고정 커밋 확보
│   │   ├── verify_envs.py          # Python 환경 일괄 검사
│   │   └── verify_ubuntu.sh        # Ubuntu 전체/모델별 검사
│   └── docs/
│       ├── STRUCTURE.md
│       ├── USAGE.md
│       ├── CSV_CONTRACT.md
│       ├── IMPLEMENTATION_STATUS.md
│       └── UPSTREAM_STATUS.md
│
├── workspace/                     # README 외 로컬 파일은 Git 제외
│   ├── README.md
│   ├── inputs/metadata/           # 팀 CSV를 둘 경우 선택 위치
│   ├── artifacts/
│   │   ├── environments/
│   │   ├── ubuntu/
│   │   └── logs/
│   ├── runtime/python310/        # Ubuntu 전용 Python 3.10
│   └── third_party/               # 공식 모델 코드
│
└── .venvs/                        # 숨김·Git 제외, Ubuntu에서 새로 생성
    ├── hvq_trans/
    ├── dinomaly_small/
    ├── u_cat/
    ├── rtdetrv2_s/
    ├── yolo11s/
    └── ecf_yolov7_tiny/
```

## 메타데이터 CSV의 역할

팀원이 CSV를 제작합니다. 이 프로젝트는 CSV의 행과 분할을 읽어 모델별 입력으로 변환합니다.
CSV는 `workspace/inputs/metadata/`에 두어도 되고 외부 절대 경로를 참조해도 됩니다.
원본 이미지는 기존 데이터 경로를 참조합니다. CSV 생성·3만 장 추출·분할 생성 코드는 포함하지 않습니다.
헤더가 확정되면 `project/configs/data.local.json`의 매핑을 실제 규격에 맞춥니다.

모델 본학습·퓨전·성능 비교 코드는 아직 구현하지 않았습니다. 현재 존재하는 파일만 구현 영역에 표시했습니다.
