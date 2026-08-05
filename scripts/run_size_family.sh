#!/usr/bin/env bash
# 크기 축(m)과 계열 축(yolo26s) 각 1회 @640.
# 조건은 주 곡선과 **동일**하게 간다(열화 증강 없음) - 안 그러면 축이 섞인다.
# m 은 12GB 에 batch 16 이 안 들어갈 수 있다. 실패하면 8 로 낮추고 그 사실을 남긴다.
set -u
cd /home/user/sod-project
PY=~/venvs/sod/bin/python
while pgrep -f '0[7]_trai[n]' >/dev/null; do sleep 30; done

try_run () {  # $1 모델, $2 이름
  for b in 16 8; do
    echo "########## $2 (batch $b) ##########"
    $PY scripts/07_train.py --model "$1" --imgsz 640 --epochs 100 --batch $b \
        --name "$2" 2>&1 | grep -avE "^\s*$"
    if [ -f "runs/seg/$2/weights/best.pt" ]; then
      echo "@@@ $2 성공 (batch $b)"
      return 0
    fi
    echo "@@@ $2 batch $b 실패 - 낮춰서 재시도"
  done
  echo "@@@ $2 최종 실패"
}

try_run yolo26s-seg.pt yolo26s-seg_640
try_run yolo11m-seg.pt yolo11m-seg_640
echo "########## 크기·계열 축 완료 ##########"
