#!/usr/bin/env bash
# 효과가 있었던 것을 모두 결합한다. 지금까지는 축마다 하나씩만 걸었다.
#   s 크기 + P2 헤드 + 열화 증강.  마스크 해상도 2배는 P2 와 대체재라 뺀다(E7).
# 640 은 P2 격자가 160x160 이라 무겁다 - batch 를 낮춰가며 시도한다.
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
while pgrep -f '0[7]_trai[n]' >/dev/null; do sleep 60; done

combo () {  # $1 imgsz
  for b in 8 4 2; do
    echo "########## s+P2+열화 @ $1 (batch $b) ##########"
    $PY scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml --model yolo11s-seg.pt \
        --imgsz $1 --mask-ratio 4 --degrade --epochs 100 --batch $b \
        --name yolo11s-p2_$1_deg 2>&1 | grep -avE "^\s*$"
    [ -f runs/seg/yolo11s-p2_$1_deg/weights/best.pt ] && { echo "@@@ $1 성공 (batch $b)"; return 0; }
    echo "@@@ $1 batch $b 실패 - 낮춰서 재시도"
  done
}
combo 320
combo 640
echo "########## 결합 실험 완료 ##########"
