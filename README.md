# SOD — 소형/얇은 산업 부품 정밀 탐지

작고 얇은 산업 부품(나사·너트·볼트)을 공개 벤치마크에서 정밀 탐지하는 프로젝트.
기술 기획은 `docs/project-plan.md`, 쉬운 말로 쓴 기획서는 `docs/proposal.html`
(<https://claude.ai/code/artifact/97ffcc62-6f16-4f06-b65b-0bd728714daf>) 참고.

## 진행 상태

- [x] D1 — 데이터 확보·라벨 스키마 확인. MVTec Screws·NPU-BOLT·PCBsmdComponents
      셋 다 실측한 결과, 원래 계획("MVTec Screws가 주 소형 객체 벤치마크")이
      무효화됨 — 셋 다 COCO 기준 small 객체 비율이 낮았고(0%/9.5%/0.2%), 방법론을
      다운스케일링 중심으로 수정. 자세한 내용은 `docs/project-plan.md` 2장 참고
- [ ] D2 — 다운스케일 파이프라인 구축 + YOLO11n/s baseline
- [ ] D3 — YOLO26(STAL) 벤치마크
- [ ] D4 — SAHI 타일링 추론
- [ ] D5 — copy-paste 증강 (소형 객체 특화)
- [ ] D6 — ONNX 변환·INT8 양자화 (경량화·배포)
- [ ] D7 — Weighted Boxes Fusion (모델 앙상블)
- [ ] D8 — 능동 학습 시뮬레이션 (라벨 효율)
- [ ] D9 — Grad-CAM/Eigen-CAM (설명 가능성)
- [ ] D10 — 결과 정리
