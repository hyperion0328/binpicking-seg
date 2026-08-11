#!/usr/bin/env bash
# 640 에서 P2 만 켠 학습. E13 이 "결합이 나쁘다"까지만 갈랐고 "P2 가 나쁜지 결합에서만
# 나쁜지"는 못 갈랐다 - 이 런이 그 칸을 채운다. 채우면 2x2 가 완성된다.
#
#            열화 끔        열화 켬
#   P2 끔    0.3769        0.4181
#   P2 켬    ← 이 런        0.3299
#
# 조건은 나머지 셋에 맞춘다: batch 16 · imgsz 640 · mask_ratio 4 · 100 epoch.
# patience 0 으로 조기 종료를 끄는 이유 - 나머지 셋이 전부 100 을 완주했다. 여기서만
# 중간에 멈추면 학습량도 학습률 일정 위치도 달라져 비교가 깨진다(E12-3).
set -u
cd /home/user/sod-project
while pgrep -f '0[7]_trai[n]' >/dev/null; do sleep 60; done
echo "########## s+P2 @ 640 (열화 없음, batch 16, 100 epoch 강제) ##########"
~/venvs/sod/bin/python scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml \
    --model yolo11s-seg.pt --imgsz 640 --mask-ratio 4 --epochs 100 --batch 16 \
    --patience 0 --name yolo11s-p2_640 2>&1 | grep -avE "^\s*$"
if [ -f runs/seg/yolo11s-p2_640/weights/best.pt ]; then
  echo "@@@ 성공"
else
  echo "@@@ 실패 - 로그 확인"
  exit 1
fi
echo "########## 평가 ##########"
~/venvs/sod/bin/python scripts/08_eval.py --only yolo11s-p2_640 \
    --out outputs/d_eval_native.json 2>&1 | grep -avE "^\s*$"
echo "########## 완료 ##########"
