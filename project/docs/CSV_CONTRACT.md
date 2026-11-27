# 공통 CSV 연결 계약 (헤더 미확정)

팀원이 3만 장을 선정하고 CSV를 제공합니다. 이 프로젝트는 CSV를 읽기만 하며 데이터셋이나 분할을 생성하지 않습니다.
현재 권장 형태는 **이미지 1장당 1행**입니다. 박스 여러 개는 annotations 셀의 JSON 배열에 담습니다.
최종 CSV가 제품 1개당 1행, 박스 1개당 1행 또는 별도 라벨 파일 참조 방식이면, 실제 파일 수신 후
읽기 계층을 추가해 아래 내부 계약으로 연결합니다. 헤더 설정만으로 행 구조 변경까지 처리하는 것은 아닙니다.

| 내부 필드 | 필요 조건 | 값 |
|---|---|---|
| image_path | 필수 | data_root 기준 상대 경로 또는 절대 경로 |
| item | 필수 | 원본 품목 식별자 |
| product_id | 필수 | 품목 내 실제 물리 제품을 구분하는 고유 키 |
| view | 필수 | C1–C5 또는 1–5 |
| split | 필수 | train / val / test (값 매핑 가능) |
| product_label | 필수 | 제품 정상 0 / 불량 1 / UNKNOWN |
| image_label | 필수 | 해당 시점 정상 0 / 불량 1 / UNKNOWN |
| mask_path | 불량 Pixel 평가·마스크 BBox 시 필요 | 원본 마스크 경로, 없으면 빈칸 |
| defect_class_id | 마스크 → BBox 시 필요 | 검증 사전의 불량 ID 0–110 |
| roi_xyxy | ROI crop·정상 BBox 생성 시 필요 | 원본 픽셀 [x1,y1,x2,y2] JSON |
| roi_verified | ROI 사용 시 필요 | true / false |
| annotations | 명시적 검출 박스가 있을 때 | `[{"class_id":0,"bbox":[10,20,30,40]}]` |

CSV 내부 JSON 셀의 quoting은 표준 CSV 규칙을 따릅니다. 예를 들어 Python `csv.DictWriter`로 저장하면
JSON 문자열의 쉼표·따옴표가 올바르게 처리됩니다. UTF-8 BOM은 기본 지원합니다.

제품 키는 `(item, product_id)`입니다. `S0001`처럼 다른 정상/불량 또는 유형 폴더에서 반복되는 번호만
전달하면 서로 다른 제품이 합쳐질 수 있습니다. 팀원이 원본 식별 체계에 맞춰 충돌 없는 product_id를 제공해야 합니다.

헤더가 `img`, `category`, `sample_id`, `camera`, `partition`으로 정해졌을 때 설정 예시:

```json
{
  "columns": {
    "image_path": "img", "item": "category", "product_id": "sample_id",
    "view": "camera", "split": "partition"
  },
  "split_values": {"training": "train", "validation": "val", "testing": "test"},
  "label_values": {"OK": 0, "NG": 1, "UNKNOWN": -1, "": -1}
}
```

명시하지 않은 필드는 내부 이름과 동일하게 읽습니다. 시점·라벨은 경로나 마스크 부재로 추정하지 않습니다.
불량 제품의 정상 시점은 `product_label=1`, `image_label=0`으로 표현합니다.

마스크 기반 박스는 8연결 성분별 외접 직사각형을 **메모리에서만** 계산합니다.
한 이미지에 여러 결함 클래스가 존재하면 단일 defect_class_id로 표현할 수 없으므로
정확한 클래스별 annotations 또는 추가 클래스별 마스크 규격을 제공해야 합니다.

표준 좌표는 원본 크기의 xyxy이며 오른쪽/아래 경계는 제외하는 픽셀 경계 좌표입니다.
리사이즈·패딩·ROI는 sample.geometry에 기록합니다. 평가 박스의 원본 좌표 복원은
`Geometry(**sample['geometry']).inverse_boxes(boxes)`를 사용합니다.
공식 모델 postprocessor가 448에서 원본으로 단순 비례 확대하면 letterbox/ROI와 맞지 않으므로 이 역변환으로 대체해야 합니다.

## 별도로 필요한 검증 클래스 사전

```json
{
  "status": "verified",
  "version": "team-approved-version",
  "classes": [
    {"id": 0, "name": "기존 검증 불량 클래스명", "item": "실제 품목", "kind": "defect"}
  ],
  "item_ids": {"실제 품목": 0}
}
```

위 코드는 구조 예시로서 그대로 실행할 수 있는 완성 사전이 아닙니다.
실제 파일은 불량 111개 ID 0–110 + 정상 30개 ID 111–140 전체 및 품목 ID 0–29를 포함해야 합니다.
기존 검증 사전의 순서를 보존하며 모델마다 ID를 재번호화하지 않습니다.

실제 CSV가 도착하면 헤더·분할·라벨·마스크·ROI와 사전의 저장 방식을 확인해 연결합니다.
제공된 분할이 없다면 읽기 오류로 알리고 팀원에게 분할 정보를 요청합니다. 이 프로젝트가 임의 분할하지 않습니다.

경로가 다른 동일 이미지의 내용 중복이나 과거 프로젝트 학습 데이터와의 겹침은 경로 검사만으로 판별할 수 없습니다.
원본 식별자·기존 사용 이력에 대한 팀의 검사 기록을 별도로 받아 실험 기록에 연결해야 합니다.
