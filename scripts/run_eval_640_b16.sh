#!/usr/bin/env bash
# 640 결합 batch 16 학습이 끝나면 평가한다. GPU 를 겹쳐 쓰면 메모리 부족으로 죽으므로 기다린다.
# (기다림 대상은 파이썬 학습 프로세스만 본다 - 셸 이름으로 잡으면 자기 자신에 걸린다.)
set -u
cd /home/user/sod-project
while pgrep -f '0[7]_trai[n]' >/dev/null; do sleep 60; done
if [ ! -f runs/seg/yolo11s-p2_640_deg_b16/weights/best.pt ]; then
  echo "@@@ best.pt 가 없다 - 학습이 끝나지 않았거나 실패했다. 평가를 건너뛴다."
  exit 1
fi
echo "########## 640 결합(batch 16) 평가 ##########"
~/venvs/sod/bin/python scripts/08_eval.py \
  --only yolo11s-p2_640_deg_b16 \
  --out outputs/d_eval_native.json 2>&1 | grep -avE "^\s*$"
echo "########## 평가 완료 ##########"
