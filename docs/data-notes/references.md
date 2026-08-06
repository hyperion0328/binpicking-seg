# 참고 문헌 — 고르고 버린 근거

`docs/proposal.html` 의 대외용 참고 자료 목록은 링크만 싣는다(v1 형식).
**기법을 새로 들일 때 원 논문이 건 조건까지 옮겼는지 확인하는 곳은 여기다.**


## 적용

### Small Object Detection: A Comprehensive Survey (2025)

소형 객체 탐지 기법의 분류 체계. 다중 스케일 특징 계층(FPN/P2 확장), 데이터 증강, 문맥 활용, 손실 설계를 같은 문제에 대한 서로 다른 대응으로 정리한다. 06절의 P2 헤드 도입과 07절 증강 설계가 이 축을 따른다.

<https://arxiv.org/abs/2503.20516>

### Kisantal et al. — Augmentation for Small Object Detection (2019)

소형 객체 성능 저하를 두 축으로 완화한다 — (1) 소형 객체가 포함된 이미지의 오버샘플링, (2) 이미지 내 소형 객체 인스턴스 복제·재배치. 07절에서 둘을 함께 적용한다. 오버샘플링은 rf-screw-nut-bolt가 인스턴스의 81%를 차지하는 소스 쏠림 완화에도 동시에 작용한다.

<https://arxiv.org/abs/1902.07296>

### Ghiasi et al. — Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation (CVPR 2021)

인스턴스 마스크 단위 copy-paste. 핵심 조건은 large-scale jittering과의 결합으로, 스케일 지터 폭이 좁으면 이득이 크게 줄어든다. ultralytics 기본값 scale: 0.5는 폭이 좁고 (min, max) 튜플을 받으므로 범위를 넓혀 copy_paste와 함께 켠다.

<https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf>

### Domain Randomization for Manufacturing Object Detection (SIP15-OD)

렌더링 기반 합성 데이터와 실촬영 사이의 도메인 갭. 09절에서 3D 렌더·생성 데이터를 배제한 근거 — 이 프로젝트의 측정 대상이 "실촬영 화질 대 인위적 화질"의 격차라, 합성 데이터는 측정 대상 자체를 오염시킨다.

<https://arxiv.org/abs/2506.07539>


## 검토 후 배제

### SAHI — Slicing Aided Hyper Inference (Akyon et al., 2022)

추론 시 이미지를 타일로 분할해 각각 추론 후 병합. 탐지기가 타일 해상도로 학습돼 있다는 전제 위에 있어, 원본을 축소해 학습하는 이번 방법론과 전제가 충돌한다. 적용하려면 타일 단위 재학습이 필요해 별도 실험이 된다.

<https://arxiv.org/abs/2202.06934>

### Solovyev et al. — Weighted Boxes Fusion (2021)

복수 모델의 박스를 신뢰도 가중 평균으로 병합. 박스 좌표 평균이 전제라 마스크 출력에 그대로 옮겨지지 않고, 마스크 병합 규칙을 새로 정의해야 한다.

<https://arxiv.org/abs/1910.13302>

### Chen et al. — Plug and Play Active Learning for Object Detection (2022)

불확실성에 다양성·대표성 신호를 결합해 라벨링 대상을 고르는 능동 학습. 이번 데이터는 정답이 이미 전량 확보돼 있어 "무엇을 먼저 라벨링할까"라는 물음이 성립하지 않는다.

<https://arxiv.org/abs/2211.11612>

### Explaining YOLO — Grad-CAM

활성 맵으로 모델의 근거 영역을 시각화. 정량 지표로 쓰려면 지표 자체의 타당성을 먼저 검증해야 한다는 점이 05절의 SAM 검증 순서 — 도구로 쓰기 전에 측정 대상으로 세우는 것 — 으로 이어졌다.

<https://arxiv.org/abs/2211.12108>


## 데이터셋

### screw-nut-bolt · 1,300장 / 27,265 인스턴스 / RLE 마스크 전량 / CC BY 4.0

<https://universe.roboflow.com/logesh-s-workspace/screw-nut-bolt>

### ARG_FIXINGS_3 · 233장 / 5,541 인스턴스 / 4클래스 / CC BY 4.0

<https://universe.roboflow.com/bolts/arg_fixings_3>

### Nuts and Bolts · 100장 / 579 인스턴스 / 폴리곤 전량 / CC BY 4.0

<https://universe.roboflow.com/jente-kleinh/nuts-and-bolts>

### fasteners · 179장 / 538 인스턴스 / 폴리곤 전량 / CC BY 4.0

<https://universe.roboflow.com/obb-8ehdy/fasteners-vbqng>
