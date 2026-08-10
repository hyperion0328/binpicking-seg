#!/usr/bin/env bash
# 결합 실험 640 을 batch 16 으로. 지금까지 **시도한 적이 없다** -
# run_combo.sh 의 사다리가 `for b in 8 4 2` 로 시작해 8 에서 바로 성공했을 뿐이고,
# 12GB 에 16 이 들어가는지는 재지 않았다. OOM 기록도 없다.
# 비교 대상(P2@640, s@640)이 batch 16 이라 8 로 돌린 결과는 그대로 비교할 수 없다(1차 D8).
# 들어가면 640 결합이 판정되고, OOM 이면 "안 들어간다"가 비로소 실측이 된다.
set -u
cd /home/user/sod-project
# GPU 를 겹쳐 쓰면 메모리 부족으로 죽는다. 학습 프로세스가 비기를 기다린다.
# (기다림 대상을 셸 이름으로 잡으면 자기 자신에 걸려 영원히 안 끝난다 - 파이썬만 본다.)
while pgrep -f '0[7]_trai[n]' >/dev/null; do sleep 60; done
echo "########## s+P2+열화 @ 640 (batch 16) ##########"
~/venvs/sod/bin/python scripts/07_train.py --cfg configs/yolo11s-seg-p2.yaml \
    --model yolo11s-seg.pt --imgsz 640 --mask-ratio 4 --degrade --epochs 100 --batch 16 \
    --name yolo11s-p2_640_deg_b16 2>&1 | grep -avE "^\s*$"
if [ -f runs/seg/yolo11s-p2_640_deg_b16/weights/best.pt ]; then
  echo "@@@ 640 batch 16 성공"
else
  echo "@@@ 640 batch 16 실패 - OOM 여부를 로그에서 확인할 것"
fi
echo "########## 완료 ##########"
