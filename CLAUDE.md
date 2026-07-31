# CLAUDE.md

작고 얇은 산업 부품(나사·너트·볼트·SMD 부품)을 공개 벤치마크에서 정밀 탐지하는
프로젝트. 기술 기획은 `docs/project-plan.md`, 대외용(쉬운 말) 기획서는
`docs/proposal.html`.

## 틀리기 쉬운 전제 — 먼저 확인할 것

**후보 데이터셋 셋 다 "소형 객체" 벤치마크가 아니다.** 이름과 부품 종류만 보면
작을 것 같지만, D1에서 라벨을 직접 열어 실측한 결과 COCO 기준(면적 32×32px 미만)
small 비율이 MVTec Screws 0% / NPU-BOLT 9.5% / PCBsmdComponents 0.2%였다.
물리적으로 작은 부품이라도 **클로즈업으로 찍혔으면 이미지 안에서는 크다** —
"물리적 크기"와 "이미지 안 픽셀 크기"는 다른 축이다. 그래서:

- **주 실험 축은 "작게 찍힌 데이터를 모으는 것"이 아니라 의도적 다운스케일링**이다
  (`docs/project-plan.md` 2.2절). 원본을 단계적으로 축소해 "몇 픽셀부터 무너지는가"를
  통제된 곡선으로 잰다.
- **다운스케일 결과만 단독으로 보고하면 안 된다.** 소프트웨어로 축소한 이미지는
  안티앨리어싱 때문에 실제로 멀리서 찍힌 소형 객체보다 부자연스럽게 깨끗하다.
  항상 **NPU-BOLT 자연 발생 small 서브셋(121개/14장)** 을 대조군으로 같이 본다.
  단 이 서브셋은 표본이 너무 적어 **정량 지표가 아니라 정성 확인용**이다 —
  mAP 하나로 쓰면 안 된다.
- 이 "깨끗한 합성 vs 진짜 노이즈" 격차는 실험 설계의 핵심 리스크다. 두 숫자의
  격차 자체가 이 프로젝트의 결론 중 하나다.

**라벨 형식이 데이터셋마다 다르다.** 전처리 코드를 하나로 뭉뚱그리면 깨진다.
- MVTec Screws: `(row, col, width, height, phi)` **5파라미터 회전 박스**(axis-aligned 아님).
  클래스는 `type_001`~`type_013`으로 익명화돼 실제 나사 종류 이름이 없다. 1920×1440 고정.
- NPU-BOLT: **Supervisely JSON**(COCO 아님). `objects[].geometryType: "rectangle"`,
  `points.exterior`에 좌상단·우하단 두 점. 이미지 크기 편차가 큼(262×230 ~ 4896×3672).
- PCBsmdComponents: 표준 COCO(`_annotations.coco.json`), train/valid/test 분할 완비,
  1280×720으로 통일 리사이즈됨.

**세그멘테이션(마스크) 라벨은 어디에도 없다 — seg 모델로 갈아타는 제안은 이미 검토하고 접었다.**
셋 다 bbox뿐이고, PCBsmd의 COCO `segmentation` 필드는 전부 빈 배열(`[]`)이다.
YOLO26-seg나 Mask2Former/MaskDINO를 쓰려면 마스크를 직접 라벨링해야 하는데,
하필 이 프로젝트의 대상이 "작아서 경계를 그리기 제일 어려운 물체"라 비용 대비
이득이 없고 "새 라벨링 안 함" 원칙과도 충돌한다. 결정 근거는
`docs/proposal.html` 07절 D6. 다시 제안하기 전에 그 항목부터 읽을 것.

## 스코프 밖 (제안하지 말 것)

- **포즈 추정(6D pose)·그립 계획·로봇 시뮬레이션.** 빈피킹은 "이 기술이 쓰일 수 있는
  예시"로만 언급하고 다루지 않는다. 다루는 순간 로보틱스 프로젝트가 된다.
- **새로 촬영하거나 새로 라벨링하는 것.** 능동 학습(3.4절)도 *이미 가진 라벨*을
  꺼내는 순서만 정하는 시뮬레이션이라 이 원칙을 안 깬다.
- **3D 렌더링 기반 합성 데이터.** 증강은 copy-paste로 간다 — 실제 카메라 픽셀
  (노이즈·블러 포함)을 그대로 재사용해 렌더-실사 도메인 갭이 새로 안 생기기 때문.
  선택 근거는 `docs/proposal.html` 04-2절.
- **실시간 처리.** 먼저 정확도를 확인한다. 속도는 D6(양자화)에서 트레이드오프로만 잰다.

## 데이터

`data/` 아래에 이미 받아둔 상태이며 git 추적 안 함(`.gitignore`의 `/data/`).

| 경로 | 내용 |
|---|---|
| `data/mvtec-screws/mvtec_screws{,_train,_val,_test}.json` | COCO 유사 스키마, 회전 bbox |
| `data/npu-bolt/ds/ann/*.json` + `ds/img/` | Supervisely 형식, 이미지당 JSON 1개 |
| `data/pcbsmd/{train,valid,test}/_annotations.coco.json` | 표준 COCO |

경로는 코드에 박지 말고 설정 한 곳(환경변수 또는 `sod/config.py`)에서 읽는다.

## 실행 환경

Windows(`D:\sod-project`)와 WSL Ubuntu-24.04(`/mnt/d/sod-project`) **같은 폴더를
양쪽에서 연다**. 복사본이 아니라 동일 디렉터리라 git 이력도 하나다.

- WSL 쪽 Node는 nvm으로 설치돼 있다(v24.18.1). WSL 안에서 `npm`이
  `/mnt/c/Program Files/nodejs/npm`(Windows용)으로 잡히면 PATH가 잘못 잡힌 것 —
  대화형 셸(`bash -i`)에서는 정상이다.
- `/mnt/d`는 Windows 드라이브 마운트라 파일 I/O가 느리다. 문서 작업은 상관없지만
  D2에서 `data/`(2.1GB) 이미지를 대량으로 읽기 시작하면 병목이 될 수 있다 —
  그때 WSL 네이티브 경로로 옮길지 판단한다.
- Windows 콘솔은 기본 코드페이지가 cp949라 파이썬 스크립트에서 `—`, `→` 같은
  문자를 print하면 `UnicodeEncodeError`가 난다. 출력 문자열은 ASCII로 쓴다.

## git

- 아직 **원격이 없다**(로컬 전용, 브랜치 `master`). 원격을 붙일 때까지 push 하지 않는다.
- `data/`, `runs/`, `outputs/`, `*.pt`, `*.pth`는 추적하지 않는다 — 커밋 전 `git status` 확인.
- 저장소 루트의 `mvtec_screws_v1.1.tar{,.gz}`는 `data/mvtec-screws/`로 이미 압축을
  푼 원본 아카이브(약 2.7GB)라 중복이다. gitignore에도 안 걸려 있으니 **실수로
  커밋하지 말 것.** 삭제는 사용자 확인 후에만.
- 커밋 메시지에는 **무엇을 바꿨는지보다 왜 바꿨는지**를 남긴다. 이 프로젝트의
  산출물은 코드 자체가 아니라 "왜 이 선택인가"에 대한 답이다.

## 기록 규칙

- 수치를 얻으면 `docs/experiments.md`에 누적한다(벤치마크 결과, 파라미터 조정 이력,
  버린 시도까지). 아직 D2를 시작 안 해서 파일이 없다 — 첫 실측 때 만든다.
- 설계 판단을 바꾸면 `docs/proposal.html` 07절 결정 표(D1~D6)에 한 줄 추가한다.
  이 표가 "왜 이렇게 했냐"는 질문에 대한 단일 출처다.
- `docs/proposal.html`을 수정하면 아티팩트를 같은 URL로 재게시한다:
  <https://claude.ai/code/artifact/97ffcc62-6f16-4f06-b65b-0bd728714daf>
  (내용은 전문 용어 없는 쉬운 말로 유지 — 대외용 문서다.)

## 현재 위치

`README.md`의 체크리스트 참조. D1(데이터 확보·실측·방법론 수정) 완료,
D2(다운스케일 파이프라인 + YOLO11n/s baseline)부터가 미착수 — 아직 코드는
한 줄도 없고 기획 문서만 있는 상태다.
