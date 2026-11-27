# 구현·검증 상태

2026-11-27 · 실제 작업 위치: `/home/user1/relearn` (WSL Ubuntu-24.04).

## 완료

- Windows 프로젝트 파일 851개를 Ubuntu로 복사하고 SHA-256을 대조했습니다. 기존 Ubuntu Git 설정은 보존했습니다.
- 검증 후 Windows 원본의 프로젝트 코드·설정·문서·작업 자료 폴더를 정리했습니다.
- Ubuntu 전용 Python 3.10을 `workspace/runtime/python310`에 준비하고 `.venvs/`에 독립 환경 6개를 새로 생성했습니다.
- 각 Ubuntu 환경에서 pip check와 합성 CSV 입력 검사 19개씩, 총 114개 실행이 모두 통과했습니다.
- 실제 Ubuntu 검증에는 모델별 Train/Validation/Test 입력 배치, DataLoader worker, CSV 매핑·누출 검사, ROI·BBox·마스크·좌표 변환이 포함됩니다.
- Ubuntu 실행 결과는 `workspace/artifacts/ubuntu/`, 환경 요약은 `workspace/artifacts/environments/summary.json`에 있습니다.
- 실제 설치 버전은 `project/environments/records/ubuntu-cpu/`에 기록했습니다. Windows 기록은 별도 하위 폴더에 보관했습니다.

| 환경 | Ubuntu 입력 검사 | 모델 연결 상태 |
|---|---|---|
| HVQ-Trans | 19/19 통과 | 448·30품목 모델 내부 변경은 후속 작업 |
| Dinomaly-Small | 19/19 통과 | Small 레시피·학습기 연결은 후속 작업 |
| U-CAT | 19/19 통과 | 공식 구현 확인 전 임시 입력 계약 |
| RT-DETRv2-S | 19/19 통과 | 학습기·평가 좌표 역변환 연결은 후속 작업 |
| YOLO11s | 19/19 통과 | Custom Trainer 연결은 후속 작업 |
| ECF-YOLOv7-tiny | 19/19 통과 | 선정 검증·학습기 연결은 후속 작업 |

## CSV 수신 후 연결

팀원이 메타데이터 CSV를 제작합니다. 헤더·행 구조가 확정되면 설정과 읽기 계층을 실제 규격에 맞춥니다.
CSV 선정·생성·3만 장 추출·재분할 기능은 포함하지 않습니다. 원본 이미지는 기존 데이터 경로를 참조합니다.
현재 Ubuntu 개인 설정은 `/mnt/c/Users/user1/Desktop/JY_project/team_project_2/data`를 참조합니다.
Windows 절대 경로로 작성된 CSV도 해당 원본 루트의 WSL 경로 매핑을 설정했습니다.
검출 모델·품목 ID가 필요한 어댑터에는 팀이 검증한 141클래스 사전과 ROI 또는 명시적인 객체 라벨이 필요합니다.

## 검증 전

실제 CSV 연결, 후보 모델 최종 선정, 모델 본체·Trainer 연결, Forward/Loss/Backward/누적/Optimizer Step,
체크포인트 저장·재개, 8GB GPU 학습과 모델 성능 비교는 아직 수행하지 않았습니다.
입력 코드·환경 검증 통과를 실제 모델 학습 성공으로 간주하지 않습니다.

GitHub 반영 대상은 코드·설정·문서입니다. CSV·원본 데이터·가상환경·외부 모델 소스·로그는 로컬에서 관리합니다.
