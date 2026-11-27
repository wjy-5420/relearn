# Real-IAD 5시점 재학습 프로젝트

개정된 실험 계획의 모델 6종을 위한 **독립 가상환경과 CSV 기반 데이터로더 어댑터**입니다.
데이터 추출·3만 장 선정·Train/Validation/Test 분할 생성은 하지 않습니다.
팀원이 제공하는 공통 CSV의 행과 분할을 그대로 사용합니다.

| 모델 | 환경 폴더 | 입력 어댑터 | 현재 모델 연결 상태 |
|---|---|---|---|
| HVQ-Trans | `.venvs/hvq_trans` | 이미지·마스크·품목 ID dict | 공식 모델의 448 입력·30품목 변경 필요 |
| Dinomaly-Small | `.venvs/dinomaly_small` | 공식 Real-IAD train/eval tuple | Small 인코더·공통 입력 레시피 연결 예정 |
| U-CAT | `.venvs/u_cat` | 공통 dict, 임시 계약 | 공식 소스·의존성 확인 전 후보 |
| RT-DETRv2-S | `.venvs/rtdetrv2_s` | 이미지 + normalized cxcywh target | v2 공식 구현 기준, Trainer 연결 예정 |
| YOLO11s | `.venvs/yolo11s` | Ultralytics 학습 batch dict | Custom Trainer 연결 예정 |
| ECF-YOLOv7-tiny | `.venvs/ecf_yolov7_tiny` | YOLOv7 이미지·라벨 tuple | 공식 구현 확보, 선정 검증·Trainer 연결 예정 |

이번 범위는 학습 프로젝트의 기반 준비입니다. 로더의 배치 생성 성공과 실제 모델의 Forward/Loss/Backward,
8GB 학습 성공은 별개의 검증입니다. 학습 완료나 성능 측정 결과를 포함하지 않습니다.

## 1. Ubuntu 환경 준비

레거시 모델을 포함하므로 **Python 3.10**으로 실행합니다. 환경 간 패키지는 공유하지 않습니다.

실행 대상은 **Ubuntu**입니다. Windows에서 생성한 `.venvs`는 옮겨 쓰지 않고 Ubuntu에서 새로 만듭니다.
Ubuntu 24.04에서 모델별 Python 3.10 환경 6개 설치와 19개 × 6환경 입력 검사를 통과했습니다.
Windows용 `project/environments/records/windows-cpu/*.cpu-win-py310.txt`도 Ubuntu 설치에 사용하지 않습니다.

```bash
python3.10 --version
bash project/scripts/setup_ubuntu.sh
```

프로젝트의 `workspace/runtime/python310`이 있으면 설치 스크립트가 자동 사용합니다. Conda 등 다른 위치의 Python 3.10을 쓰면 `PYTHON310=/실제/경로/bin/python bash project/scripts/setup_ubuntu.sh`로 지정합니다.
시스템 Python 설치나 OS 패키지 변경은 이 스크립트가 자동 수행하지 않습니다.

기본 PyTorch 설치는 **CPU용**이며 입력 로더 검증을 위한 것입니다.
실제 GPU에서의 CUDA 버전·드라이버·8GB 학습 호환성은 별도로 확인해야 합니다.
확정된 GPU 환경에 맞는 인덱스를 `--torch-index`로 지정할 수 있지만, 구버전 Torch와
신규 Torch의 CUDA wheel 지원 범위가 다르므로 전체 모델에 하나의 CUDA 인덱스를 일괄 적용하지 마세요.
기존 CPU 설치에서 CUDA wheel로 바꿀 때는 `--reinstall-torch`도 지정해야 합니다.

```bash
python3.10 project/scripts/setup_envs.py --model yolo11s --install model
.venvs/yolo11s/bin/python -m relearn.cli doctor
```

`--install none`은 환경 폴더만 만들고, `core`는 공통 로더+Torch,
`model`은 해당 모델의 추가 패키지까지 설치합니다. 실제 설치 버전과 실행 환경은
`workspace/artifacts/environments/<model>/freeze.txt`, `doctor.json`에 기록합니다.
의존성 조정과 OS별 검증 범위는 [환경 안내](../environments/README.md)에 기록했습니다.

공식 소스는 고정한 커밋으로 내려받습니다. Git에는 소스 복사본 대신 출처·커밋을 기록합니다.

```bash
python3.10 project/scripts/fetch_upstreams.py
```

## 2. CSV 도착 후 연결

**CSV 헤더는 아직 미정**입니다. [CSV 계약 안내](CSV_CONTRACT.md)의 컬럼은 권장 예시이며,
추후 실제 헤더를 받으면 `project/configs/data.local.json`의 `columns` 매핑을 수정합니다.

Ubuntu에서 `project/configs/data.example.json`을 복사해 실제 Ubuntu 데이터 경로를 설정합니다.
현재 로컬 설정은 Ubuntu에서 접근 가능한 기존 원본 경로를 참조합니다. 다른 PC에서는 실제 데이터 위치를 지정하세요. 상대 경로는
이미지·마스크에는 `data_root`, CSV·클래스 사전에는 **설정 파일 폴더**를 기준으로 해석합니다.

```bash
cp -n project/configs/data.example.json project/configs/data.local.json
# csv_path, data_root, class_map과 columns를 실제 값으로 수정
.venvs/dinomaly_small/bin/python -m relearn.cli validate-csv --config project/configs/data.local.json --check-files
```

읽기 전용 검사: 이미지 경로 중복, 동일 제품의 분할 중복, 5시점 누락·중복, 제품 라벨 충돌,
UNKNOWN 및 마스크 누락, 선택 시 파일 존재 여부를 확인합니다. 3만 장을 맞추려고 행을 추가하거나 재분할하지 않습니다.

## 3. 모델별 로더 사용

```python
from relearn.data import Manifest, load_config
from relearn.adapters import build_loader

manifest = Manifest.read(load_config("project/configs/data.local.json"))
loader = build_loader("dinomaly_small", manifest, "train", batch_size=2, workers=0)
images, labels = next(iter(loader))

# 학습 루프에서 epoch마다 호출. Train 증강을 사용할 때 epoch별 변환을 공유합니다.
loader.dataset.set_epoch(1)
```

```bash
.venvs/dinomaly_small/bin/python -m relearn.cli smoke --config project/configs/data.local.json --model dinomaly_small --split train
.venvs/yolo11s/bin/python -m relearn.cli smoke --config project/configs/data.local.json --model yolo11s --split val --report workspace/artifacts/yolo11s_batch.json
```

AD Train은 **정상 제품**의 정상 시점만 사용합니다. 불량 제품의 정상 시점을 정상 학습 데이터에 섞지 않습니다.
Validation/Test 분할은 계열에 관계없이 같습니다. UNKNOWN은 검토 전 학습·평가에서 오류로 처리합니다.

검출 3종 및 품목 ID가 필요한 HVQ-Trans/U-CAT은 팀이 검증한 클래스 사전을 요구합니다.
`project/configs/class_map.example.json`은 빈 템플릿이며 실제 141개 클래스 ID를 임의로 생성하지 않습니다.
정상 이미지에는 검증된 제품 ROI 또는 명시적인 정상 객체 annotations가 필요합니다.

공통 입력은 RGB·종횡비 유지·448 letterbox입니다. ROI crop은 검증된 ROI를 받은 후
`crop_to_roi=true`로 켭니다. 현재 기본값 false는 **ROI 미제공 상태에서 원본 입력 로더를 확인하는 용도**이며,
PPT의 공통 ROI를 적용한 최종 기준 실험으로 간주하면 안 됩니다.

Train 증강은 명시적으로 `augment=True`를 지정한 경우 밝기 ±10%, 대비 ±10%, 회전 ±5°만 적용합니다.
좌표·마스크를 함께 변환하고 잘림·소실은 오류로 보고합니다. Validation/Test의 무작위 증강은 금지합니다.
공식 학습기의 Mosaic/Flip/RandomCrop/MultiScale을 다시 적용하지 않도록 Trainer 연결 단계에서 제어해야 합니다.

## 구조

[전체 구조](STRUCTURE.md)에 최상위 구성과 모든 구현 파일을 정리했습니다.

## 검증

```bash
# 6개 Ubuntu 환경 전체 검사
bash project/scripts/verify_ubuntu.sh

# 특정 환경만 검사
bash project/scripts/verify_ubuntu.sh hvq_trans

# 기능별 검사
.venvs/yolo11s/bin/python -m unittest discover -s tests -k ManifestTests -v
.venvs/yolo11s/bin/python -m unittest discover -s tests -k GeometryLabelTests -v
.venvs/yolo11s/bin/python -m unittest discover -s tests -k AdapterTests -v
```

임시 합성 CSV를 사용해 6모델 × Train/Val/Test 배치, DataLoader worker 실행,
좌표 역변환·마스크 정합성, 정상 ROI, 분할 누출 차단, 헤더 매핑을 검증합니다.
팀의 실제 CSV가 도착하면 원본 경로·클래스 사전·ROI와 연결한 검증을 추가합니다.
Ubuntu 실행 결과는 `workspace/artifacts/ubuntu/<model>/`에 저장합니다. Ubuntu 24.04에서 모델별 CPU 입력 검증을 수행했습니다. GPU 학습 검증은 별도입니다.
