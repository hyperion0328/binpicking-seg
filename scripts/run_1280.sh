#!/usr/bin/env bash
# 성능 축 - 입력 해상도를 640 위로. 탐침에서 test mask50-95 가 0.622 -> 0.716 로 올랐다.
# batch 는 12GB 제약으로 4 까지 내린다(1280 은 640 의 4배 화소).
# ultralytics 는 nbs=64 로 기울기를 누적하므로 유효 배치는 유지된다.
# 이 런은 곡선의 점이 아니라 별도 구성이므로 배치 고정 원칙과 무관하다.
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
for b in 8 4 2; do
  echo "########## s@1280 deg (batch $b) ##########"
  $PY scripts/07_train.py --model yolo11s-seg.pt --imgsz 1280 --epochs 100 --batch $b \
      --degrade --name yolo11s-seg_1280_deg 2>&1 | grep -avE "^\s*$"
  [ -f runs/seg/yolo11s-seg_1280_deg/weights/best.pt ] && { echo "@@@ 성공 (batch $b)"; break; }
  echo "@@@ batch $b 실패 - 낮춰서 재시도"
done
echo "########## 1280 완료 ##########"
