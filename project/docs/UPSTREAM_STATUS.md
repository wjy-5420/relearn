# 공식 구현 확인 및 다음 연결 단계

실험 계획은 사용자 원문과 개정 PPT를 기준으로 유지합니다. 아래는 공식 소스 조사 결과이며 후보의 임의 대체를 하지 않습니다.
정확한 커밋은 project/configs/models.json에 기록합니다.

## HVQ-Trans

[공식 저장소](https://github.com/RuiyingLu/HVQ-Trans)에서 custom_dataset의 입력 dict 형식을 확인했습니다.
소스의 feature_size=(14,14), 15개 품목별 codebook/head, MVTec 품목 목록은 448 입력·Real-IAD 30품목에
그대로 쓸 수 없습니다. 어댑터는 공통 품목 ID를 제공하지만 모델 내부 차원·품목 수 변경은 다음 단계입니다.
일부 마스크 누락을 전체 양성 마스크로 대체하는 upstream 동작은 사용하지 않습니다.

## Dinomaly-Small

[공식 저장소](https://github.com/guojiajeremy/Dinomaly)의 Real-IAD 데이터셋 tuple 형식을 사용했습니다.
Small은 dinov2reg_vit_small_14로 설정합니다. 공식 스크립트의 기본 Base, 448 resize 후 392 crop은
공통 실험 계약과 다르므로 실행기를 그대로 호출하지 않습니다. 본 어댑터는 448 letterbox를 제공합니다.

## U-CAT (후보)

[저자 소속 기관 연구 페이지](https://research.knu.ac.kr/en/publications/a-unified-model-utilizing-comprehensive-adapted-tokenization-for-/)에서
논문과 DOI 10.1109/TII.2026.3728535를 확인했습니다. 공식 공개 소스·학습 코드 주소는 현재 확인하지 못했습니다.
공식 코드를 찾았다는 가정으로 URL이나 의존성을 만들지 않습니다. 공통 입력 dict는 임시 계약이고,
실제 U-CAT 입력 API·정규화·가중치·Q1 게재 연도·8GB 실행은 확인 전입니다.

## RT-DETRv2-S

[RT-DETR 공식 저장소의 rtdetrv2_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)를 기준으로 합니다.
R18vd 설정을 Small 변형 연결 후보로 기록하며 최종 실험 변형 명칭은 모델 선정 시 확인합니다.
표준 클래스 ID를 유지하고 COCO 80클래스 자동 remap을 끕니다. 공통 loader가 제공하는 normalized cxcywh를
추가 변환하지 않도록 Trainer에 직접 연결해야 합니다. 기존 학습기 랜덤 crop/flip/multiscale을 다시 적용하면 안 됩니다.
공식 eval postprocessor의 원본 크기 비례 확대 대신 ROI/letterbox Geometry 역변환이 필요합니다.

## YOLO11s

[Ultralytics 공식 안내](https://docs.ultralytics.com/models/yolo11/)의 YOLO11s를 사용합니다.
Ultralytics 8.3.40으로 버전을 고정하고 uint8 이미지·cls·bboxes·batch_idx 계약을 준비했습니다.
일반 YOLO Trainer의 데이터셋 생성 대신 공통 loader를 연결하는 Custom Trainer가 필요합니다.
본 단계에서는 가중치를 다운로드하거나 학습을 시작하지 않습니다.

## ECF-YOLOv7-tiny (후보)

[공식 저장소](https://github.com/dbacea/ecf-yolov7-tiny)의 YOLOv7 학습 tuple을 따릅니다.
[원 논문](https://openaccess.thecvf.com/content/WACV2025/html/Bacea_ECF-YOLOv7-Tiny_Improving_Feature_Fusion_and_the_Receptive_Field_for_Lightweight_WACV_2025_paper.html)은
WACV 2025 학회 논문으로, 계획서의 ‘Q1 저널’ 조건과 일치하지 않습니다. 후보 상태를 유지하고
선정 조건을 통과한 것으로 표시하지 않습니다. 448 학습·가중치·141클래스·Custom Loader 연결은 다음 단계입니다.

## 공통 후속 검증

실제 CSV·141클래스 사전·검증 ROI 수신 → 원본 연결 검사 → Trainer 주입 및 upstream 변환 중복 제거 →
Forward/Loss/Backward/누적/Optimizer Step/Validation → 저장·재개 → 8GB 피크 VRAM 측정 순서입니다.
로더 준비와 학습 가능 여부를 구분합니다.
