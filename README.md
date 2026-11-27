# Real-IAD 재학습 프로젝트

Ubuntu / Python 3.10 기준의 모델 6종 **가상환경·CSV 로더·입력 어댑터 뼈대**입니다.
메타데이터 CSV는 다른 팀원이 제공하며, 제공된 행·라벨·분할을 읽어 사용합니다...

```text
relearn/
├── README.md                 # 프로젝트 입구
├── pyproject.toml            # Python 패키지 설정
├── src/relearn/              # 공통 CSV 로더·6모델 어댑터
├── tests/                    # 입력 계약·전처리·배치 검사
├── project/                  # 설정·설치 레시피·실행 스크립트·자료
│   ├── configs/
│   ├── environments/
│   ├── scripts/
│   └── docs/
├── workspace/                # 로컬 CSV 참조·로그·외부 소스 (내용은 Git 제외)
└── .venvs/                   # 모델별 독립 환경 (숨김·Git 제외)
```

모델: **HVQ-Trans / Dinomaly-Small / U-CAT / RT-DETRv2-S / YOLO11s / ECF-YOLOv7-tiny**.
U-CAT은 공식 코드 확인 전 임시 입력 계약입니다. 모델 학습·8GB GPU 검증은 후속 단계입니다.

Ubuntu 프로젝트 루트에서:

```bash
bash project/scripts/setup_ubuntu.sh
bash project/scripts/verify_ubuntu.sh
```

현재 작업 경로는 `/home/user1/relearn`입니다. Ubuntu 24.04의 독립 환경 6개에서 패키지 검사와 입력 검사 19개씩 통과했습니다. GPU 학습 검증은 별도입니다.

- [전체 구조](project/docs/STRUCTURE.md)
- [Ubuntu 실행·기능별 검증](project/docs/USAGE.md)
- [팀 메타데이터 CSV 연결 규격](project/docs/CSV_CONTRACT.md)
- [구현·검증 상태](project/docs/IMPLEMENTATION_STATUS.md)
- [공식 모델 소스 확인 사항](project/docs/UPSTREAM_STATUS.md)
