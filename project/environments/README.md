# 환경 레시피와 검증 범위

대상 환경은 **Ubuntu / Python 3.10**입니다. Ubuntu에서 `project/scripts/setup_envs.py`로 각 환경을 새로 생성합니다.
Ubuntu 24.04에서 독립 환경 6개 설치·pip check·환경당 입력 검사 19개를 통과했습니다.
`common.txt`에는 numpy/Pillow/scipy/테스트 도구만 공유 선언하며 설치 위치는 각각의 .venvs입니다.
Torch·torchvision 버전은 project/configs/models.json에서 명시합니다.

| 환경 | Torch / torchvision | 근거·차이 |
|---|---|---|
| hvq_trans | 1.11.0 / 0.12.0 | 공식 버전 계열 유지, CPU wheel로 로더 검증 |
| dinomaly_small | 1.12.0 / 0.13.0 | 공식 버전 계열 유지, Python 3.10용 수치 라이브러리 조정 |
| u_cat | 2.5.1 / 0.20.1 | 공식 의존성 미확인, 임시 어댑터 환경 |
| rtdetrv2_s | 2.5.1 / 0.20.1 | 공식 torch>=2.0.1 요구 범위에서 고정 |
| yolo11s | 2.5.1 / 0.20.1 | Ultralytics 8.3.40, 공통 환경과 독립 |
| ecf_yolov7_tiny | 1.11.0 / 0.12.0 | 공식 tested Torch 1.11 계열, Python 3.10으로 조정 |

HVQ 공식 opencv-python 4.9.0 표기는 배포 버전 접미사가 빠져 있고, Dinomaly 공식 numpy 1.18/scipy 1.4 등은
Windows Python 3.10 wheel과 맞지 않습니다. 이 프로젝트는 numpy 1.23.5/scipy 1.10.1/Pillow 10.4 등으로
**입력 어댑터를 실행하기 위한 호환 레시피**를 제공합니다. 변경된 조합으로 원 논문 전체 학습을 재현했다는 의미는 아닙니다.
예전 upstream의 PIL.Image.ANTIALIAS 등은 모델 연결 단계에서 별도 변경이 필요할 수 있습니다.

설치 완료 후 pip check, pip freeze, doctor로 실제 해석된 버전을 기록합니다.
`records/ubuntu-cpu/*.cpu-ubuntu-py310.txt`에는 실제 Ubuntu 설치 버전을, `records/windows-cpu/`에는 과거 Windows 기록을 보관합니다.
Windows 기록은 Ubuntu 설치에 사용하지 않습니다. Ubuntu에서는 모델별 레시피로 설치하고
`bash project/scripts/verify_ubuntu.sh`로 검사한 뒤 해당 환경의 설치 기록을 따로 남겨야 합니다.
GPU 환경의 잠금 파일이나 해당 모델 전체 학습 재현을 보증하는 파일은 아닙니다.
GPU 패키지는 드라이버·Torch 버전별 배포 wheel과 장비 지원을 확인한 후 별도 검증해야 합니다.
현재 CPU 로더 테스트로 8GB 학습 또는 FP16 안정성을 판정하지 않습니다.

RT-DETRv2 환경은 학습 입력·의존성에 초점을 두며 ONNX/onnxruntime-gpu 등 배포 도구는 설치하지 않습니다.
공식 저장소별 라이선스와 가중치 이용 조건은 각 upstream을 따릅니다. 모델 소스는 Git에서 제외합니다.
